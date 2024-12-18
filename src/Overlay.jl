# NOTE: We are placing all the reactant_overrides here to avoid incompatibilities with
#       Revise.jl. Essentially files that contain reactant_overrides cannot be revised
#       correctly. Once that (https://github.com/timholy/Revise.jl/issues/646) is resolved
#       we should move all the reactant_overrides to relevant files.

# Helper Function to determine if we are inside the ReactantInterpreter
"""
    within_reactant_interpreter()

Returns `true` if we are currently inside the ReactantInterpreter.
"""
@noinline within_reactant_interpreter() = false
@reactant_overlay @noinline within_reactant_interpreter() = true

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
            return Random.$(randfun)(rng, T, dims)
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
            return Random.$(randfun)(rng, T, dim1, dims...)
        end

        # scalars
        @reactant_overlay @noinline function Random.$(randfun)(
            rng::AbstractRNG, ::Type{T}=Float64
        ) where {T}
            if T <: ReactantPrimitive
                return TracedRandom.$(overload_randfun)(rng, T)
            end
            @warn "Reactant doesn't support sampling of $(T) with the current \
                   interpreter. Falling back to native interpreter." maxlog = 1
            return Random.$(randfun)(rng, T)
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
