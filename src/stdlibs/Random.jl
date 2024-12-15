# Implementation based on the following:
# 1. https://github.com/JuliaGPU/CUDA.jl/blob/master/src/random.jl
# 2. https://github.com/JuliaRandom/Random123.jl/blob/master/src/common.jl#L125

mutable struct TracedRNG <: Random.AbstractRNG
    seed::Union{ConcreteRArray{UInt64,1},TracedRArray{UInt64,1}}
    const algorithm::String
end

function Random.seed!(rng::TracedRNG, seed::Number)
    seed = reinterpret(UInt64, Random.hash_seed(seed))
    # TODO: Using `seed!` inside tracing should generate a TracedRArray
    return Random.seed!(rng, ConcreteRArray(seed[1:length(rng.seed)]))
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

default_rng() = TracedRNG()
function default_rng_inside_interpreter()
    return TracedRNG(TracedUtils.promote_to(TracedRArray{UInt64,1}, make_seed()), "DEFAULT")
end

@reactant_override @noinline Random.default_rng() = default_rng_inside_interpreter()
@reactant_override @noinline default_rng() = default_rng_inside_interpreter()

function Random.rand!(rng::TracedRNG, A::AnyTracedRArray{T,N}) where {T,N}
    length(A) == 0 && return A
    res = Ops.rng_bit_generator(T, rng.seed, [size(A)...]; rng.algorithm)
    rng.seed = res.output_state
    TracedUtils.set_mlir_data!(A, res.output.mlir_data)
    return A
end

function Random.randn!(rng::TracedRNG, A::AnyTracedRArray{T,N}) where {T,N}
    length(A) == 0 && return A
    Random.rand!(rng, A)
    scaled_uniform = Ops.subtract(
        Ops.multiply(A, Ops.constant(fill(T(2), size(A)))),
        Ops.constant(fill(T(1), size(A))),
    )
    probit = Ops.erf_inv(scaled_uniform)
    rand_normal = Ops.multiply(probit, Ops.constant(fill(sqrt(T(2)), size(A))))
    TracedUtils.set_mlir_data!(A, rand_normal.mlir_data)
    return A
end

for randfun in (:rand, :randn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        function Random.$(randfun)(rng::TracedRNG, ::Type{T}, dims::Dims) where {T}
            return Random.$(randfun!)(rng, TracedRArray{T,length(dims)}((), nothing, dims))
        end

        function Random.$(randfun)(rng::TracedRNG, dims::Dims)
            return Random.$(randfun)(rng, Float64, dims)
        end

        function Random.$(randfun)(rng::TracedRNG, dim1::Integer, dims::Integer...)
            return Random.$(randfun)(rng, Dims((dim1, dims...)))
        end

        function Random.$(randfun)(
            rng::TracedRNG, ::Type{T}, dim1::Integer, dims::Integer...
        ) where {T}
            return Random.$(randfun)(rng, T, Dims((dim1, dims...)))
        end

        Random.$(randfun!)(A::AnyTracedRArray) = Random.$(randfun!)(default_rng(), A)

        # scalars
        function Random.$(randfun)(rng::TracedRNG, ::Type{T}=Float64) where {T}
            A = promote_to(TracedRArray{T,0}, fill(T(0)))
            Random.$(randfun!)(rng, A)
            return A[]
        end

        # Non-Traced RNGs if used will lead to disastrous performance. We attempt to fix
        # that but with a warning
        function Random.$(randfun!)(rng::Random.AbstractRNG, A::AnyTracedRArray)
            @warn "`rng` is not a `TracedRNG`. We will use this to seed the `TracedRNG` \
                   instead of generating samples from this RNG type." maxlog = 1
            seed = promote_to(TracedRArray{UInt64,1}, rand(rng, UInt64, 2))
            trng = TracedRNG(seed, "DEFAULT")
            return Random.$(randfun!)(trng, A)
        end
    end
end

# resolve ambiguities
function Random.randn(rng::TracedRNG, T::Random.BitFloatType)
    A = promote_to(TracedRArray{T,0}, fill(T(0)))
    Random.randn!(rng, A)
    return A[]
end

# TODO: At some later point we might want to implement the sampler API as well since it
#       makes all RNG implementation work by default. From the post-optimize IR we need to
#       confirm that the dynamic_update_slice calls are optimized away into a single
#       `stablehlo.rng_bit_generator` call -- confirm that this should be the case based on
#       how the seeding should work?
