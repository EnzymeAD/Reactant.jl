# NOTE: We are placing all the reactant_overrides here to avoid incompatibilities with
#       Revise.jl. Essentially files that contain reactant_overrides cannot be revised
#       correctly. Once that (https://github.com/timholy/Revise.jl/issues/646) is resolved
#       we should move all the reactant_overrides to relevant files.

# Compiling within a compile should return simply the original function
@reactant_overlay function Compiler.compile(f, args; kwargs...)
    return f
end

# Enzyme.jl overlays
const WITHIN_AUTODIFF = Ref(false)

@reactant_overlay @noinline function Enzyme.within_autodiff()
    return WITHIN_AUTODIFF[]
end

@reactant_overlay @noinline function Enzyme.autodiff_deferred(
    rmode::Enzyme.Mode, f::FA, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,Nargs}
    original_within_autodiff = WITHIN_AUTODIFF[]
    try
        WITHIN_AUTODIFF[] = true
        return overload_autodiff(rmode, f, args...)
    finally
        WITHIN_AUTODIFF[] = original_within_autodiff
    end
end

@reactant_overlay @noinline function Enzyme.autodiff(
    rmode::Enzyme.Mode, f::FA, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,Nargs}
    original_within_autodiff = WITHIN_AUTODIFF[]
    try
        WITHIN_AUTODIFF[] = true
        return overload_autodiff(rmode, f, args...)
    finally
        WITHIN_AUTODIFF[] = original_within_autodiff
    end
end

@reactant_overlay @noinline function Enzyme.autodiff_deferred(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    original_within_autodiff = WITHIN_AUTODIFF[]
    try
        WITHIN_AUTODIFF[] = true
        return overload_autodiff(rmode, f, rt, args...)
    finally
        WITHIN_AUTODIFF[] = original_within_autodiff
    end
end

@reactant_overlay @noinline function Enzyme.autodiff(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    original_within_autodiff = WITHIN_AUTODIFF[]
    try
        WITHIN_AUTODIFF[] = true
        return overload_autodiff(rmode, f, rt, args...)
    finally
        WITHIN_AUTODIFF[] = original_within_autodiff
    end
end

@reactant_overlay function EnzymeCore.ignore_derivatives(args...)
    res = map(args) do arg
        return Functors.fmap(arg) do argᵢ
            if argᵢ isa AnyTracedRArray && !(argᵢ isa TracedType)
                argᵢ = call_with_reactant(materialize_traced_array, argᵢ)
            end
            argᵢ isa TracedType && return @opcall ignore_derivatives(argᵢ)
            return argᵢ
        end
    end
    length(args) == 1 && return only(res)
    return res
end

# Random.jl overlays
@reactant_overlay @noinline function Random.default_rng()
    return call_with_reactant(TracedRandom.default_rng)
end

@reactant_overlay @noinline function TracedRandom.default_rng()
    return ReactantRNG(
        promote_to(TracedRArray{UInt64,1}, TracedRandom.make_seed()), "DEFAULT"
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
            if unwrapped_eltype(T) <: ReactantPrimitive
                return call_with_native(
                    TracedRandom.$(overload_randfun), rng, unwrapped_eltype(T), dims
                )
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return call_with_native(Random.$(randfun), rng, T, dims)
        end

        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, dim1::Integer, dims::Integer...
        )
            return TracedRandom.$(overload_randfun)(rng, dim1, dims...)
        end

        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, ::Type{T}, dim1::Integer, dims::Integer...
        ) where {T}
            if unwrapped_eltype(T) <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(
                    rng, unwrapped_eltype(T), dim1, dims...
                )
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return call_with_native(Random.$(randfun), rng, T, dim1, dims...)
        end

        # scalars
        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, ::Type{T}=Float64
        ) where {T}
            if unwrapped_eltype(T) <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(rng, unwrapped_eltype(T))
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return call_with_native(Random.$(randfun), rng, T)
        end

        # inplace
        @reactant_overlay @noinline function Random.$(randfun!)(
            rng::AbstractRNG, A::AnyTracedRArray
        )
            return call_with_native(TracedRandom.$(overload_randfun!), rng, A)
        end
        @reactant_overlay @noinline function Random.$(randfun!)(A::AnyTracedRArray)
            return TracedRandom.$(overload_randfun!)(
                call_with_reactant(TracedRandom.default_rng), A
            )
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
            C::CT, A::AT, B::BT, α::Number, β::Number
        ) where {CT<:$cT,AT<:$aT,BT<:$bT}
            A, B = aos_to_soa(A), aos_to_soa(B)
            C2 = aos_to_soa(C)
            if use_overlayed_version((C2, A, B))
                call_with_native(TracedLinearAlgebra.overloaded_mul!, C2, A, B, α, β)
                if C2 !== C
                    C .= C2
                end
            else
                # Inference barrier is required when calling function recursively within
                # overload. This is required since otherwise type inference will think this
                # is a recursive edge rather than a call to the base method
                call_with_native(LinearAlgebra.mul!, C, A, B, α, β)
            end
            return C
        end

        # Needed mostly for 1.10 where 3-arg mul is often specialized
        @reactant_overlay @noinline function LinearAlgebra.mul!(
            C::CT, A::AT, B::BT
        ) where {CT<:$cT,AT<:$aT,BT<:$bT}
            call_with_reactant(LinearAlgebra.mul!, C, A, B, true, false)
            return C
        end
    end
end

# Base overloads
@reactant_overlay @noinline function Base._stack(dims::Union{Integer,Colon}, iter)
    if use_overlayed_version(iter)
        return call_with_native(TracedRArrayOverrides.overloaded_stack, dims, iter)
    else
        iter2 = collect(iter)
        if any(use_overlayed_version, iter2)
            return call_with_native(TracedRArrayOverrides.overloaded_stack, dims, iter2)
        else
            # Inference barrier is required when calling function recursively within
            # overload. This is required since otherwise type inference will think this is
            # a recursive edge rather than a call to the base method
            return call_with_native(Base._stack, dims, iter2)
        end
    end
end

## fixes #493
@reactant_overlay @noinline function Base._unique_dims(A::AbstractArray, dims::Colon)
    if use_overlayed_version(A)
        error("Reactant doesn't have a `Base._unique_dims` with the current interpreter.")
    else
        call_with_native(Base._unique_dims, A, dims)
    end
end

@reactant_overlay @noinline function Base.mapreduce(
    f,
    op,
    A::Union{AbstractArray,Base.Iterators.Zip,Base.Iterators.Enumerate,Base.Generator};
    kwargs...,
)
    if use_overlayed_version(A) || use_overlayed_version(f)
        return call_with_native(
            TracedRArrayOverrides.overloaded_mapreduce, f, op, A; kwargs...
        )
    else
        return call_with_native(
            Base.mapreduce, CallWithReactant(f), CallWithReactant(op), A; kwargs...
        )
    end
end

@reactant_overlay @noinline function Base.map(f, x::AbstractArray, ys::AbstractArray...)
    if (
        use_overlayed_version(x) ||
        use_overlayed_version(f) ||
        looped_any(use_overlayed_version, ys)
    )
        return call_with_native(TracedRArrayOverrides.overloaded_map, f, x, ys...)
    else
        return call_with_native(Base.map, CallWithReactant(f), x, ys...)
    end
end

@reactant_overlay @noinline function Base.map!(
    f, y::AbstractArray, x::AbstractArray, xs::AbstractArray...
)
    if (
        use_overlayed_version(y) ||
        use_overlayed_version(x) ||
        use_overlayed_version(f) ||
        looped_any(use_overlayed_version, xs)
    )
        return call_with_native(TracedRArrayOverrides.overloaded_map!, f, y, x, xs...)
    else
        return call_with_native(Base.map!, CallWithReactant(f), y, x, xs...)
    end
end

@reactant_overlay @noinline function Base._all(f, x::AbstractArray, dims)
    if use_overlayed_version(x) || use_overlayed_version(f)
        return call_with_native(TracedRArrayOverrides.overloaded_mapreduce, f, &, x; dims)
    else
        return call_with_native(Base._all, CallWithReactant(f), x, dims)
    end
end

@reactant_overlay @noinline function Base._any(f, x::AbstractArray, dims)
    if use_overlayed_version(x) || use_overlayed_version(f)
        return call_with_native(TracedRArrayOverrides.overloaded_mapreduce, f, |, x; dims)
    else
        return call_with_native(Base._any, CallWithReactant(f), x, dims)
    end
end

@reactant_overlay @noinline function Base._getindex(
    ::IndexLinear, x::Array{T,N}, idxs::Vararg{Any,N}
) where {T,N}
    if use_overlayed_version(idxs)
        return call_with_native(
            TracedIndexing.overloaded_unsafe_getindex, IndexLinear(), x, idxs...
        )
    else
        return call_with_native(Base._getindex, IndexLinear(), x, idxs...)
    end
end

# LinearAlgebra
## Various factorizations
## TODO(#2258): specialize for `cholesky!` --> cholcopy
factorization_copy(f::F, x, pivot) where {F} = x
factorization_copy(f::F, x) where {F} = x

for (jlop, rop, default_pivot) in (
    (:lu, :overloaded_lu, RowMaximum),
    (:lu!, :overloaded_lu, RowMaximum),
    (:cholesky, :overloaded_cholesky, NoPivot),
    (:cholesky!, :overloaded_cholesky, NoPivot),
)
    @eval begin
        @reactant_overlay @noinline function LinearAlgebra.$(jlop)(
            x::AbstractArray; kwargs...
        )
            if use_overlayed_version(x)
                pivot = $(default_pivot)()
                return call_with_native(
                    TracedLinearAlgebra.$(rop),
                    factorization_copy(LinearAlgebra.$(jlop), x, pivot),
                    pivot;
                    kwargs...,
                )
            else
                return call_with_native(LinearAlgebra.$(jlop), x; kwargs...)
            end
        end

        @reactant_overlay @noinline function LinearAlgebra.$(jlop)(
            x::AbstractArray, pivot::$(default_pivot); kwargs...
        )
            if use_overlayed_version(x)
                return call_with_native(
                    TracedLinearAlgebra.$(rop),
                    factorization_copy(LinearAlgebra.$(jlop), x, pivot),
                    pivot;
                    kwargs...,
                )
            else
                return call_with_native(LinearAlgebra.$(jlop), x, pivot; kwargs...)
            end
        end
    end
end

for (jlop, rop) in ((:svd, :overloaded_svd),)
    @eval begin
        @reactant_overlay @noinline function LinearAlgebra.$(jlop)(
            x::AbstractArray; kwargs...
        )
            if use_overlayed_version(x)
                return call_with_native(
                    TracedLinearAlgebra.$(rop),
                    factorization_copy(LinearAlgebra.$(jlop), x);
                    kwargs...,
                )
            else
                return call_with_native(LinearAlgebra.$(jlop), x; kwargs...)
            end
        end
    end
end

@reactant_overlay @noinline function LinearAlgebra.dot(x::AbstractArray, y::AbstractArray)
    if use_overlayed_version(x) || use_overlayed_version(y)
        return call_with_native(TracedLinearAlgebra.overloaded_dot, x, y)
    else
        return call_with_native(LinearAlgebra.dot, x, y)
    end
end
@reactant_overlay @noinline function LinearAlgebra.dot(
    x::AbstractVector, A::AbstractMatrix, y::AbstractVector
)
    if use_overlayed_version(x) || use_overlayed_version(A) || use_overlayed_version(y)
        return call_with_native(TracedLinearAlgebra.overloaded_dot, x, A, y)
    else
        return call_with_native(LinearAlgebra.dot, x, A, y)
    end
end
