# NOTE: We are placing all the reactant_overrides here to avoid incompatibilities with
#       Revise.jl. Essentially files that contain reactant_overrides cannot be revised
#       correctly. Once that (https://github.com/timholy/Revise.jl/issues/646) is resolved
#       we should move all the reactant_overrides to relevant files.

# Compiling within a compile should return simply the original function
@reactant_overlay function Compiler.compile(f, args; kwargs...)
    return f
end

@reactant_overlay @noinline function Base.setindex!(
    a::AnyTracedRArray{T,N}, v, indices::Vararg{Any,N}
) where {T,N}
    ancestor_indices = TracedUtils.get_ancestor_indices(a, indices...)
    (Base.inferencebarrier(setindex!))(Reactant.ancestor(a), v, ancestor_indices...)
    return a
end

# Enzyme.jl overlays
@reactant_overlay @noinline function Enzyme.autodiff_deferred(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    return overload_autodiff(rmode, f, rt, args...)
end

@reactant_overlay @noinline function Enzyme.autodiff(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    return overload_autodiff(rmode, f, rt, args...)
end

# Random.jl overlays
@reactant_overlay @noinline function Random.default_rng()
    return call_with_reactant(TracedRandom.default_rng)
end

@reactant_overlay @noinline function TracedRandom.default_rng()
    return TracedRNG(
        TracedUtils.promote_to(TracedRArray{UInt64,1}, TracedRandom.make_seed()), "DEFAULT"
    )
end

## Only problematic edge case here is the direct `<randfun!>(rng, A::AbstractArray)` call
## We can't directly overlay that call without breaking the semantics of inplace update
for randfun in (:rand, :randn, :randexp)
    randfun! = Symbol(randfun, :!)
    overload_randfun = Symbol(:overload_, randfun)
    overload_randfun! = Symbol(:overload_, randfun!)

    @eval begin
        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, ::Type{T}, dims::Dims
        ) where {T}
            if T <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(rng, T, dims)
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return Base.inferencebarrier(Random.$(randfun))(rng, T, dims)
        end

        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, dim1::Integer, dims::Integer...
        )
            return TracedRandom.$(overload_randfun)(rng, dim1, dims...)
        end

        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, ::Type{T}, dim1::Integer, dims::Integer...
        ) where {T}
            if T <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(rng, T, dim1, dims...)
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return Base.inferencebarrier(Random.$(randfun))(rng, T, dim1, dims...)
        end

        # scalars
        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, (::Type{T})=Float64
        ) where {T}
            if T <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(rng, T)
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return Base.inferencebarrier(Random.$(randfun))(rng, T)
        end

        # inplace
        @reactant_overlay @noinline function Random.$(randfun!)(
            rng::AbstractRNG, A::AnyTracedRArray
        )
            return TracedRandom.$(overload_randfun!)(rng, A)
        end
    end
end

# LinearAlgebra.jl overloads
## `mul!` goes through too many layers of abstractions and we aren't able to overload
## without specializing on every possible combination of types
for (cT, aT, bT) in (
    (:AbstractVector, :AbstractMatrix, :AbstractVector),
    (:AbstractMatrix, :AbstractMatrix, :AbstractVecOrMat),
)
    @eval begin
        @reactant_overlay @noinline function LinearAlgebra.mul!(
            C::$cT, A::$aT, B::$bT, α::Number, β::Number
        )
            A, B = aos_to_soa(A), aos_to_soa(B)
            C2 = aos_to_soa(C)
            if use_overlayed_version((C2, A, B))
                TracedLinearAlgebra.overloaded_mul!(C2, A, B, α, β)
                if C2 !== C
                    C .= C2
                end
            else
                # Inference barrier is required when calling function recursively within overload
                # This is required since otherwise type inference will think this is a recursive edge
                # rather than a call to the base method
                Base.inferencebarrier(LinearAlgebra.mul!)(C, A, B, α, β)
            end
            return C
        end

        # Needed mostly for 1.10 where 3-arg mul is often specialized
        @reactant_overlay @noinline function LinearAlgebra.mul!(C::$cT, A::$aT, B::$bT)
            call_with_reactant(LinearAlgebra.mul!, C, A, B, true, false)
            return C
        end
    end
end

# Base overloads
@reactant_overlay @noinline function Base._stack(dims::Union{Integer,Colon}, iter)
    if use_overlayed_version(iter)
        return TracedRArrayOverrides.overloaded_stack(dims, iter)
    else
        iter2 = collect(iter)
        if any(use_overlayed_version, iter2)
            return TracedRArrayOverrides.overloaded_stack(dims, iter2)
        else
            # Inference barrier is required when calling function recursively within overload
            # This is required since otherwise type inference will think this is a recursive edge
            # rather than a call to the base method
            return Base.inferencebarrier(Base._stack)(dims, iter2)
        end
    end
end

## fixes #493
@reactant_overlay @noinline function Base._unique_dims(A::AbstractArray, dims::Colon)
    if use_overlayed_version(A)
        error("Reactant doesn't have a `Base._unique_dims` with the current interpreter.")
    else
        Base.inferencebarrier(Base._unique_dims)(A, dims)
    end
end

# overlay mapreduce since users often do a reduction over empty collections which can have a
# Union{} type. Since Union{} <: TracedRNumber it goes through our dispatch, and here we
# explicitly prevent it from going through our dispatch.
@reactant_overlay @noinline function Base.mapreduce(
    f, op, A::AbstractArray{T}; kwargs...
) where {T}
    if T <: TracedRNumber && T !== Union{}
        return TracedRArrayOverrides.overloaded_mapreduce(f, op, A; kwargs...)
    else
        return Base.inferencebarrier(Base.mapreduce)(f, op, A; kwargs...)
    end
end

@reactant_overlay @noinline function Base._all(f, x::AbstractArray{T}, dims) where {T}
    if T <: TracedRNumber && T !== Union{}
        return TracedRArrayOverrides.overloaded_all(f, x, dims)
    else
        return Base.inferencebarrier(Base._all)(f, x, dims)
    end
end

@reactant_overlay @noinline function Base.any(f, x::AbstractArray{T}, dims) where {T}
    if T <: TracedRNumber && T !== Union{}
        return TracedRArrayOverrides.overloaded_any(f, x, dims)
    else
        return Base.inferencebarrier(Base.any)(f, x, dims)
    end
end
