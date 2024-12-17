module TracedRandom

# Implementation based on the following:
# 1. https://github.com/JuliaGPU/CUDA.jl/blob/master/src/random.jl
# 2. https://github.com/JuliaRandom/Random123.jl/blob/master/src/common.jl#L125

using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    TracedRNG,
    AnyTracedRArray,
    Reactant,
    TracedUtils,
    Ops,
    ConcreteRArray
using Random: Random, AbstractRNG

function Random.seed!(rng::TracedRNG, seed::Number)
    if seed isa TracedRNumber
        error("Passing in `TracedRNumber` as a seed is not supported. Please pass in a \
               `TracedRArray` of the appropriate size instead.")
    end

    seed = reinterpret(UInt64, Random.hash_seed(seed))
    seed = if Reactant.within_reactant_interpreter()
        TracedUtils.promote_to(TracedRArray{UInt64,1}, seed[1:length(rng.seed)])
    else
        ConcreteRArray(seed[1:length(rng.seed)])
    end
    return Random.seed!(rng, seed)
end

function Random.seed!(rng::TracedRNG, seed::AbstractArray{<:Integer,1})
    return Random.seed!(rng, UInt64.(seed))
end

function Random.seed!(rng::TracedRNG, seed::AbstractArray{UInt64,1})
    return Random.seed!(rng, TracedUtils.promote_to(TracedRArray{UInt64,1}, seed))
end

function Random.seed!(
    rng::TracedRNG, seed::Union{ConcreteRArray{UInt64,1},TracedRArray{UInt64,1}}
)
    rng.seed = seed
    return rng
end

make_seed() = rand(Random.RandomDevice(), UInt64, 2)

TracedRNG() = TracedRNG(ConcreteRArray(make_seed()))
TracedRNG(seed::ConcreteRArray{UInt64,1}) = TracedRNG(seed, "DEFAULT")

function default_rng()
    Reactant.within_reactant_interpreter() || return TracedRNG()
    return TracedRNG(TracedUtils.promote_to(TracedRArray{UInt64,1}, make_seed()), "DEFAULT")
end

rng_algorithm(rng::TracedRNG) = rng.algorithm
rng_algorithm(::AbstractRNG) = "DEFAULT"

function internal_overload_rand!(rng::TracedRNG, A::AnyTracedRArray{T,N}) where {T,N}
    length(A) == 0 && return A
    res = Ops.rng_bit_generator(T, rng.seed, [size(A)...]; rng.algorithm)
    rng.seed = res.output_state
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

function internal_overload_randn!(rng::TracedRNG, A::AnyTracedRArray{T,N}) where {T,N}
    length(A) == 0 && return A
    res = Ops.randn(T, rng.seed, [size(A)...]; rng.algorithm)
    rng.seed = res.output_state
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

function internal_overload_randexp!(rng::TracedRNG, A::AnyTracedRArray{T,N}) where {T,N}
    length(A) == 0 && return A
    res = Ops.randexp(T, rng.seed, [size(A)...]; rng.algorithm)
    rng.seed = res.output_state
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

for randfun in (:rand, :randn, :randexp)
    randfun! = Symbol(randfun, :!)
    overload_randfun = Symbol(:internal_overload_, randfun)
    overload_randfun! = Symbol(:internal_overload_, randfun!)

    @eval begin
        function $(overload_randfun)(rng::TracedRNG, ::Type{T}, dims::Dims) where {T}
            return $(overload_randfun!)(
                rng, TracedRArray{T,length(dims)}((), nothing, dims)
            )
        end

        function $(overload_randfun)(rng::TracedRNG, dims::Dims)
            return $(overload_randfun)(rng, Float64, dims)
        end

        function $(overload_randfun)(rng::TracedRNG, dim1::Integer, dims::Integer...)
            return $(overload_randfun)(rng, Dims((dim1, dims...)))
        end

        function $(overload_randfun)(
            rng::TracedRNG, ::Type{T}, dim1::Integer, dims::Integer...
        ) where {T}
            return $(overload_randfun)(rng, T, Dims((dim1, dims...)))
        end

        $(overload_randfun!)(A::AnyTracedRArray) = $(overload_randfun!)(default_rng(), A)

        # scalars
        function $(overload_randfun)(rng::TracedRNG, ::Type{T}=Float64) where {T}
            A = TracedUtils.promote_to(TracedRArray{T,0}, fill(T(0)))
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
        function $(overload_randfun)(rng::AbstractRNG, args...)
            seed_uint64 = Array{UInt64}(undef, 2)
            sampler = Random.Sampler(rng, UInt64, Val(1))
            seed_uint64[1] = rand(rng, sampler)
            seed_uint64[2] = rand(rng, sampler)
            # XXX: Ideally the following should just work but currently it gives an illegal
            #      instruction error. Maybe an issue with Julia's AbsInt?
            # Random.rand!(rng, seed_uint64)
            rng = TracedRNG(
                TracedUtils.promote_to(TracedRArray{UInt64,1}, seed_uint64),
                rng_algorithm(rng),
            )
            return $(internal_overload_randfun)(rng, args...)
        end

        function $(overload_randfun)(rng::TracedRNG, args...)
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
