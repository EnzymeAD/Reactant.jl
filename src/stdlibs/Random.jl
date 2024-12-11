# Implementation based on the following:
# 1. https://github.com/JuliaGPU/CUDA.jl/blob/master/src/random.jl
# 2. https://github.com/JuliaRandom/Random123.jl/blob/master/src/common.jl#L125

mutable struct TracedRNG <: Random.AbstractRNG
    seed::Union{ConcreteRArray{UInt64,1},TracedRArray{UInt64,1}}
    const algorithm::String
end

# TODO: Base.seed!

make_seed() = rand(Random.RandomDevice(), UInt64, 2)

TracedRNG() = TracedRNG(ConcreteRArray(make_seed()))
TracedRNG(seed::ConcreteRArray{UInt64,1}) = TracedRNG(seed, "DEFAULT")

default_rng() = TracedRNG()
function default_rng_inside_interpreter()
    return TracedRNG(promote_to(TracedRArray{UInt64,1}, make_seed()), "DEFAULT")
end

# XXX: Currently we get an illegal instruction if we don't call Random.default_rng()

# TODO: scalar rand functions should return a TracedRNumber

# TODO: Implement `randexp`
# TODO: Implement `randexp!`

function Random.rand!(rng::TracedRNG, A::AnyTracedRArray{T,N}) where {T,N}
    length(A) == 0 && return A
    res = Ops.rng_bit_generator(T, rng.seed, [size(A)...]; rng.algorithm)
    rng.seed = res.output_state
    set_mlir_data!(A, res.output.mlir_data)
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
    set_mlir_data!(A, rand_normal.mlir_data)
    return A
end

function Random.rand(rng::TracedRNG, ::Type{T}, dims::Dims) where {T}
    return Random.rand!(rng, TracedRArray{T,length(dims)}((), nothing, dims))
end
function Random.randn(rng::TracedRNG, ::Type{T}, dims::Dims) where {T}
    return Random.randn!(rng, TracedRArray{T,length(dims)}((), nothing, dims))
end

function Random.rand(rng::TracedRNG, dim1::Integer, dims::Integer...)
    return Random.rand(rng, Dims((dim1, dims...)))
end
function Random.randn(rng::TracedRNG, dim1::Integer, dims::Integer...)
    return Random.randn(rng, Dims((dim1, dims...)))
end

function Random.rand(rng::TracedRNG, ::Type{T}, dim1::Integer, dims::Integer...) where {T}
    return Random.rand(rng, T, Dims((dim1, dims...)))
end
function Random.randn(rng::TracedRNG, ::Type{T}, dim1::Integer, dims::Integer...) where {T}
    return Random.randn(rng, T, Dims((dim1, dims...)))
end

# # CPU arrays
# function Random.rand!(rng::RNG, A::AbstractArray{T}) where {T}
#     B = CuArray{T}(undef, size(A))
#     rand!(rng, B)
#     copyto!(A, B)
# end
# function Random.randn!(rng::RNG, A::AbstractArray{T}) where {T}
#     B = CuArray{T}(undef, size(A))
#     randn!(rng, B)
#     copyto!(A, B)
# end

# # scalars
# Random.rand(rng::RNG, T::Type=Float32) = Random.rand(rng, T, 1)[]
# Random.randn(rng::RNG, T::Type=Float32) = Random.randn(rng, T, 1)[]

# # resolve ambiguities
# Random.randn(rng::RNG, T::Random.BitFloatType) = Random.randn(rng, T, 1)[]

Random.rand!(A::AnyTracedRArray) = Random.rand!(default_rng(), A)
Random.randn!(A::AnyTracedRArray) = Random.randn!(default_rng(), A)

# TODO: At some later point we might want to implement the sampler API as well since it
#       makes all RNG implementation work by default. From the post-optimize IR we need to
#       confirm that the dynamic_update_slice calls are optimized away into a single
#       `stablehlo.rng_bit_generator` call -- confirm that this should be the case based on
#       how the seeding should work?
