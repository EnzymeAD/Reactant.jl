module TracedRandom

# Implementation based on the following:
# 1. https://github.com/JuliaGPU/CUDA.jl/blob/master/src/random.jl
# 2. https://github.com/JuliaRandom/Random123.jl/blob/master/src/common.jl

using ..Reactant:
    Reactant, TracedRArray, TracedRNumber, ReactantRNG, AnyTracedRArray, TracedUtils, Ops
using ..Reactant.Ops: @opcall
using Random: Random, AbstractRNG

@noinline make_seed(rng::AbstractRNG=Random.RandomDevice()) =
    Random.rand!(rng, Vector{UInt64}(undef, 2))

@noinline function Random.seed!(rng::ReactantRNG, seed::Number)
    if seed isa TracedRNumber
        error("Passing in `TracedRNumber` as a seed is not supported. Please pass in a \
               `TracedRArray` of the appropriate size instead.")
    end

    seed = reinterpret(UInt64, Random.hash_seed(seed))
    return Random.seed!(rng, seed[1:length(rng.seed)])
end

@noinline function Random.seed!(rng::ReactantRNG, seed::AbstractVector)
    rng.seed .= seed
    return rng
end

Base.copy(rng::ReactantRNG) = ReactantRNG(copy(rng.seed), rng.algorithm)

@noinline function ReactantRNG()
    if Reactant.within_compile()
        return ReactantRNG(Reactant.promote_to(TracedRArray, make_seed()))
    else
        return ReactantRNG(Reactant.to_rarray(make_seed()))
    end
end
@noinline ReactantRNG(seed::AbstractVector) = ReactantRNG(seed, "DEFAULT")

@noinline default_rng() = ReactantRNG()

@noinline rng_algorithm(rng::ReactantRNG) = rng.algorithm
@noinline rng_algorithm(::AbstractRNG) = "DEFAULT"

@noinline function internal_overload_rand!(
    rng::ReactantRNG{<:TracedRArray}, A::AnyTracedRArray{T,N}
) where {T,N}
    length(A) == 0 && return A
    res = @opcall rng_bit_generator(T, rng.seed, [size(A)...]; rng.algorithm)
    copyto!(rng.seed, res.output_state)
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

@noinline function internal_overload_randn!(
    rng::ReactantRNG{<:TracedRArray}, A::AnyTracedRArray{T,N}
) where {T,N}
    length(A) == 0 && return A
    res = @opcall randn(T, rng.seed, [size(A)...]; rng.algorithm)
    copyto!(rng.seed, res.output_state)
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

@noinline function internal_overload_randexp!(
    rng::ReactantRNG{<:TracedRArray}, A::AnyTracedRArray{T,N}
) where {T,N}
    length(A) == 0 && return A
    res = @opcall randexp(T, rng.seed, [size(A)...]; rng.algorithm)
    copyto!(rng.seed, res.output_state)
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

for randfun in (:rand, :randn, :randexp)
    randfun! = Symbol(randfun, :!)
    overload_randfun = Symbol(:internal_overload_, randfun)
    overload_randfun! = Symbol(:internal_overload_, randfun!)

    @eval begin
        @noinline function $(overload_randfun)(
            rng::ReactantRNG{<:TracedRArray}, ::Type{T}, dims::Dims
        ) where {T}
            return $(overload_randfun!)(
                rng, TracedRArray{T,length(dims)}((), nothing, dims)
            )
        end

        @noinline function $(overload_randfun)(rng::ReactantRNG{<:TracedRArray}, dims::Dims)
            return $(overload_randfun)(rng, Float64, dims)
        end

        @noinline function $(overload_randfun)(
            rng::ReactantRNG{<:TracedRArray}, dim1::Integer, dims::Integer...
        )
            return $(overload_randfun)(rng, Dims((dim1, dims...)))
        end

        @noinline function $(overload_randfun)(
            rng::ReactantRNG{<:TracedRArray}, ::Type{T}, dim1::Integer, dims::Integer...
        ) where {T}
            return $(overload_randfun)(rng, T, Dims((dim1, dims...)))
        end

        @noinline function $(overload_randfun!)(A::AnyTracedRArray)
            return $(overload_randfun!)(default_rng(), A)
        end

        # scalars
        @noinline function $(overload_randfun)(
            rng::ReactantRNG{<:TracedRArray}, ::Type{T}=Float64
        ) where {T}
            A = Reactant.promote_to(TracedRArray, fill(T(0)))
            $(overload_randfun!)(rng, A)
            return TracedRNumber{T}((), A.mlir_data)
        end
    end
end

# call from overlay-ed variants. we write this with 2 tiers -- overload_* and
# internal_overload_* -- to avoid method ambiguities
for randfun in (:rand, :randn, :randexp, :rand!, :randn!, :randexp!)
    overload_randfun = Symbol(:overload_, randfun)
    internal_overload_randfun = Symbol(:internal_overload_, randfun)
    @eval begin
        @noinline function $(overload_randfun)(rng::AbstractRNG, args...)
            rng = ReactantRNG(
                Reactant.promote_to(TracedRArray, make_seed(rng)), rng_algorithm(rng)
            )
            return $(internal_overload_randfun)(rng, args...)
        end

        @noinline function $(overload_randfun)(rng::ReactantRNG, args...)
            return $(internal_overload_randfun)(rng, args...)
        end
    end
end

# TODO: At some later point we might want to implement the sampler API as well since it
#       makes all RNG implementation work by default. From the post-optimize IR we need to
#       confirm that the dynamic_update_slice calls are optimized away into a single
#       `stablehlo.rng_bit_generator` call -- confirm that this should be the case based on
#       how the seeding should work?

end
