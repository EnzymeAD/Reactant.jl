# NOTE: We are placing all the reactant_overrides here to avoid incompatibilities with
#       Revise.jl. Essentially files that contain reactant_overrides cannot be revised
#       correctly. Once that (https://github.com/timholy/Revise.jl/issues/646) is resolved
#       we should move all the reactant_overrides to relevant files.

# Compiling within a compile should return simply the original function
@reactant_overlay function Compiler.compile(
    f, args; client=nothing, optimize=true, sync=false
)
    return f
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
            return error(
                "Reactant doesn't support sampling of $(T) with the current interpreter."
            )
            # XXX: The following will lead to illegal instruction
            # @warn "Reactant doesn't support sampling of $(T) with the current \
            #        interpreter. Falling back to native interpreter." maxlog = 1
            # return Random.$(randfun)(rng, T, dims)
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
            return error(
                "Reactant doesn't support sampling of $(T) with the current interpreter."
            )
            # XXX: The following will lead to illegal instruction
            # @warn "Reactant doesn't support sampling of $(T) with the current \
            #        interpreter. Falling back to native interpreter." maxlog = 1
            # return Random.$(randfun)(rng, T, dim1, dims...)
        end

        # scalars
        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, ::Type{T}=Float64
        ) where {T}
            if T <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(rng, T)
            end
            return error(
                "Reactant doesn't support sampling of $(T) with the current interpreter."
            )
            # XXX: The following will lead to illegal instruction
            # @warn "Reactant doesn't support sampling of $(T) with the current \
            #        interpreter. Falling back to native interpreter." maxlog = 1
            # return Random.$(randfun)(rng, T)
        end

        # inplace
        @reactant_overlay @noinline function Random.$(randfun!)(
            rng::AbstractRNG, A::AnyTracedRArray
        )
            return TracedRandom.$(overload_randfun!)(rng, A)
        end

        # XXX: Uncomment once AbsInt issues with recursive calls are resolved
        # @reactant_overlay @noinline function Random.$(randfun!)(
        #     rng::AbstractRNG, A::AbstractArray
        # )
        #     @warn "Directly writing to an array using Random.jl functions inside \
        #            ReactantInterpreter will generate a constant array in the IR. Use with \
        #            caution." maxlog = 1
        #     return Random.$(randfun!)(rng, A)
        # end
    end
end

# LinearAlgebra.jl overloads
## `_mul!` goes through too many layers of abstractions and we aren't able to overload
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
                LinearAlgebra.mul!(C, A, B, α, β)
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
   iter2 = collect(iter)
   if use_overlayed_version(iter2) || any(use_overlayed_version, iter2)
        return TracedRArrayOverrides.overloaded_stack(dims, iter2)
    else
        return Base._stack(dims, iter2)
    end
end
