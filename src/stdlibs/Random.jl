module TracedRandom

# Implementation based on the following:
# 1. https://github.com/JuliaGPU/CUDA.jl/blob/master/src/random.jl
# 2. https://github.com/JuliaRandom/Random123.jl/blob/master/src/common.jl

using ..Reactant: Reactant, TracedUtils, Ops, TracedRArray, TracedRNumber, AnyTracedRArray
using ..Reactant.Ops: @opcall
import ..Reactant: ReactantRNG

using Random: Random, AbstractRNG
using SHA: SHA

@noinline function should_warn_if_not_natively_supported(rng::AbstractRNG)
    @warn "The RNG $(typeof(rng)) is not natively supported by Reactant. We will convert \
           this to `ReactantRNG` which will have different seed and distribution \
           characteristics." maxlog = 1
    return nothing
end

@noinline function make_seed(rng::AbstractRNG=Random.RandomDevice())
    return Random.rand!(rng, Vector{UInt64}(undef, 2))
end

@noinline function Random.seed!(rng::ReactantRNG, seed::Number)
    if seed isa TracedRNumber
        error("Passing in `TracedRNumber` as a seed is not supported. Please pass in a \
               `TracedRArray` of the appropriate size instead.")
    end

    hashed = @static if isdefined(Random, :hash_seed) && VERSION < v"1.13.0-rc1"
        Random.hash_seed(seed)
    elseif isdefined(Random, :has_seed)
        ctx = SHA.SHA2_256_CTX()
        Random.hash_seed(seed, ctx)
    else
        # Not present < Julia 1.11
        ctx = SHA.SHA2_256_CTX()
        neg = signbit(seed)
        if neg
            seed = ~seed
        end
        @assert seed >= 0
        while true
            word = (seed % UInt32) & 0xffffffff
            seed >>>= 32
            SHA.update!(ctx, reinterpret(NTuple{4,UInt8}, word))
            iszero(seed) && break
        end
        neg && SHA.update!(ctx, (0x01,))
        SHA.digest!(ctx)
    end
    seed = reinterpret(UInt64, hashed)
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

@noinline function default_rng()
    return ReactantRNG()
end

@noinline rng_algorithm(rng::ReactantRNG) = rng.algorithm
@noinline rng_algorithm(::AbstractRNG) = "DEFAULT"

@noinline function internal_overload_rand!(
    rng::ReactantRNG{<:TracedRArray}, A::AnyTracedRArray{Bool,N}
) where {N}
    length(A) == 0 && return A
    shape = collect(Int, size(A))
    res = @opcall rng_bit_generator(UInt8, rng.seed, shape; rng.algorithm)
    copyto!(rng.seed, res.output_state)
    lsb = @opcall and(res.output, @opcall(fill(0x01, shape)))
    bools = @opcall compare(lsb, @opcall(fill(0x00, shape)); comparison_direction="NE")
    TracedUtils.set_mlir_data!(A, bools.mlir_data)
    return A
end

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
                Reactant.promote_to(
                    TracedRArray, Reactant.call_with_native(make_seed, rng)
                ),
                rng_algorithm(rng),
            )
            return $(internal_overload_randfun)(rng, args...)
        end

        @noinline function $(overload_randfun)(rng::ReactantRNG, args...)
            return $(internal_overload_randfun)(rng, args...)
        end
    end
end

# Random.Sampler API support (#2256)

Random.rng_native_52(::ReactantRNG) = UInt64

struct TracedUnitRangeSampler{T<:Reactant.ReactantInt} <: Random.Sampler{TracedRNumber{T}}
    a::TracedRNumber{T}
    b::TracedRNumber{T}
end

function Random.Sampler(
    ::Type{<:ReactantRNG}, r::AbstractUnitRange{TracedRNumber{T}}, ::Random.Repetition
) where {T<:Reactant.ReactantInt}
    return TracedUnitRangeSampler{T}(first(r), last(r))
end

const BuiltinSampler = Union{
    Random.SamplerType,
    Random.SamplerTrivial,
    Random.SamplerSimple,
    Random.SamplerTag,
    Random.SamplerRangeNDL,
    Random.UniformT,
    TracedUnitRangeSampler,
}

@noinline _to_traced_rng(rng::ReactantRNG{<:TracedRArray}) = rng
@noinline function _to_traced_rng(rng::ReactantRNG)
    return ReactantRNG(Reactant.promote_to(TracedRArray, rng.seed), rng.algorithm)
end
@noinline function _to_traced_rng(rng::AbstractRNG)
    return ReactantRNG(
        Reactant.promote_to(TracedRArray, Reactant.call_with_native(make_seed, rng)),
        rng_algorithm(rng),
    )
end

@inline _scalar_from_0d(A::TracedRArray{T,0}) where {T} = TracedRNumber{T}((), A.mlir_data)

_tuple_has_traced(::Tuple{}) = false
function _tuple_has_traced(t::Tuple)
    return (first(t) isa TracedRNumber) || _tuple_has_traced(Base.tail(t))
end

_safe_unwrapped_eltype(::Type{T}) where {T} = T <: Number ? Reactant.unwrapped_eltype(T) : T

_tuple_all_numbers(::Tuple{}) = true
function _tuple_all_numbers(t::Tuple)
    return (first(t) isa Number) && _tuple_all_numbers(Base.tail(t))
end

_tuple_promote_eltype(::Tuple{}) = Union{}
_tuple_promote_eltype(t::Tuple{Any}) = _safe_unwrapped_eltype(typeof(t[1]))
function _tuple_promote_eltype(t::Tuple)
    return promote_type(
        _safe_unwrapped_eltype(typeof(first(t))), _tuple_promote_eltype(Base.tail(t))
    )
end

function _tuple_sampler_eltype(t::Tuple)
    isempty(t) && return Union{}
    GT = _safe_unwrapped_eltype(Random.gentype(typeof(t)))
    if isconcretetype(GT) && (GT <: Reactant.ReactantPrimitive)
        return GT
    end
    if _tuple_all_numbers(t)
        PT = _tuple_promote_eltype(t)
        if isconcretetype(PT) && (PT <: Reactant.ReactantPrimitive)
            return PT
        end
    end
    return Union{}
end

@noinline function _tuple_to_traced_vec(::Type{ET}, t::Tuple) where {ET}
    n = length(t)
    if _tuple_has_traced(t)
        parts = Vector{TracedRArray{ET,1}}(undef, n)
        for i in 1:n
            v = Reactant.promote_to(TracedRNumber{ET}, t[i])
            parts[i] = @opcall broadcast_in_dim(v, Int[], [1])
        end
        return @opcall concatenate(parts, 1)
    else
        buf = Vector{ET}(undef, n)
        for i in 1:n
            buf[i] = convert(ET, t[i])
        end
        return Reactant.promote_to(TracedRArray, buf)
    end
end

_tuple_samplers_supported(::Tuple{}) = true
function _tuple_samplers_supported(sps::Tuple)
    return supports_sampler(first(sps)) && _tuple_samplers_supported(Base.tail(sps))
end

function supports_sampler(::Random.SamplerType{T}) where {T}
    UT = _safe_unwrapped_eltype(T)
    return isconcretetype(UT) && (UT <: Reactant.ReactantPrimitive)
end
function supports_sampler(::Random.UniformT{T}) where {T}
    UT = _safe_unwrapped_eltype(T)
    return isconcretetype(UT) && (UT <: Reactant.ReactantPrimitive)
end
function supports_sampler(::Random.SamplerTrivial{Random.CloseOpen01{T}}) where {T}
    return T <: Reactant.ReactantFloat
end
supports_sampler(::Random.SamplerTrivial{Random.CloseOpen12{Float64}}) = true
supports_sampler(::Random.SamplerTrivial{Random.UInt52Raw{UInt64}}) = true
supports_sampler(::Random.SamplerTrivial{Random.UInt52{UInt64}}) = true
supports_sampler(::Random.SamplerTrivial{Random.UInt23Raw{UInt32}}) = true
supports_sampler(::Random.SamplerTrivial{Random.UInt23{UInt32}}) = true
supports_sampler(::Random.SamplerTrivial{Random.UInt10Raw{UInt16}}) = true
supports_sampler(::Random.SamplerTrivial{Random.UInt10{UInt16}}) = true
function supports_sampler(::Random.SamplerRangeNDL{U,T}) where {U<:Unsigned,T}
    return T <: Reactant.ReactantInt
end
supports_sampler(::TracedUnitRangeSampler) = true
_known_empty(x) = isempty(x)
function _known_empty(x::AbstractArray)
    len = length(x)
    len isa TracedRNumber && return false
    return len == 0
end

function supports_sampler(sp::Random.SamplerSimple{<:AbstractArray,<:Random.Sampler})
    arr = sp[]
    ET = _safe_unwrapped_eltype(eltype(arr))
    return !_known_empty(arr) && isconcretetype(ET) && (ET <: Reactant.ReactantPrimitive)
end
function supports_sampler(sp::Random.SamplerTrivial{Tuple{A}}) where {A}
    return _tuple_sampler_eltype(sp[]) !== Union{}
end
function supports_sampler(sp::Random.SamplerSimple{<:Tuple})
    return _tuple_sampler_eltype(sp[]) !== Union{}
end
function supports_sampler(sp::Random.SamplerSimple{BitSet,<:Random.Sampler})
    return !isempty(sp[])
end
function supports_sampler(sp::Random.SamplerTag{<:Set,<:Random.Sampler})
    ET = _safe_unwrapped_eltype(keytype(sp.data[]))
    return isconcretetype(ET) && (ET <: Reactant.ReactantPrimitive)
end
function supports_sampler(sp::Random.SamplerTrivial{<:AbstractSet})
    s = sp[]
    ET = _safe_unwrapped_eltype(eltype(s))
    return !isempty(s) && isconcretetype(ET) && (ET <: Reactant.ReactantPrimitive)
end
function supports_sampler(sp::Random.SamplerTag{Ref{T}}) where {T<:Tuple}
    return _tuple_samplers_supported(sp.data)
end
function supports_sampler(sp::Random.SamplerTag{<:Ref{<:Pair}})
    return supports_sampler(sp.data.first) && supports_sampler(sp.data.second)
end
supports_sampler(::Any) = false

function supports_sampler_array(sp)
    sp isa Random.SamplerTag{<:Ref{<:Tuple}} && return false
    sp isa Random.SamplerTag{<:Ref{<:Pair}} && return false
    return supports_sampler(sp)
end

@noinline function _sample_unit_range_ndl(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerRangeNDL{U,T}, dims::Dims{N}
) where {U<:Unsigned,T<:Reactant.ReactantInt,N}
    shape = collect(Int, dims)
    if prod(dims) == 0
        z = @opcall fill(zero(T), shape)
        return TracedRArray{T,N}((), z.mlir_data, dims)
    end
    if sp.s == zero(U)
        return internal_overload_rand(rng, T, dims)
    end
    raw = internal_overload_rand(rng, UInt64, dims)
    rem_u64 = @opcall remainder(raw, @opcall(fill(UInt64(sp.s), shape)))
    UT = unsigned(T)
    u_off = UT === UInt64 ? rem_u64 : @opcall(convert(TracedRArray{UT,N}, rem_u64))
    u_a = @opcall fill(reinterpret(UT, sp.a), shape)
    u_sum = @opcall add(u_a, u_off)
    return T === UT ? u_sum : @opcall(bitcast_convert(TracedRArray{T,N}, u_sum))
end

@noinline function _sample_traced_unit_range(
    rng::ReactantRNG{<:TracedRArray}, sp::TracedUnitRangeSampler{T}, dims::Dims{N}
) where {T<:Reactant.ReactantInt,N}
    shape = collect(Int, dims)
    if prod(dims) == 0
        z = @opcall fill(zero(T), shape)
        return TracedRArray{T,N}((), z.mlir_data, dims)
    end
    UT = unsigned(T)
    u_a = T === UT ? sp.a : @opcall(bitcast_convert(UT, sp.a))
    u_b = T === UT ? sp.b : @opcall(bitcast_convert(UT, sp.b))
    span_m1 = u_b - u_a
    raw = internal_overload_rand(rng, UInt64, dims)
    if UT === UInt64
        span_u64 = span_m1 + UInt64(1)
        is_full = span_u64 == UInt64(0)
        safe_div = ifelse(is_full, UInt64(1), span_u64)
        rem_u64 = @opcall remainder(raw, @opcall(broadcast_in_dim(safe_div, Int[], shape)))
        is_full_b = @opcall broadcast_in_dim(is_full, Int[], shape)
        u_off = @opcall select(is_full_b, raw, rem_u64)
    else
        span_u64 = Base.convert(TracedRNumber{UInt64}, span_m1) + UInt64(1)
        rem_u64 = @opcall remainder(raw, @opcall(broadcast_in_dim(span_u64, Int[], shape)))
        u_off = @opcall convert(TracedRArray{UT,N}, rem_u64)
    end
    u_a_b = @opcall broadcast_in_dim(u_a, Int[], shape)
    u_sum = @opcall add(u_a_b, u_off)
    return T === UT ? u_sum : @opcall(bitcast_convert(TracedRArray{T,N}, u_sum))
end

@noinline function _gather_from_1d(
    rng::ReactantRNG{<:TracedRArray}, flat::TracedRArray{ET,1}, dims::Dims{N}
) where {ET,N}
    shape = collect(Int, dims)
    if prod(dims) == 0
        z = @opcall fill(zero(ET), shape)
        return TracedRArray{ET,N}((), z.mlir_data, dims)
    end
    n = length(flat)
    idx_sp = Random.Sampler(ReactantRNG, 1:n)
    if N == 0
        idxs_1d = _sample_unit_range_ndl(rng, idx_sp, (1,))
        picked_1d = flat[idxs_1d]
        return @opcall reshape(picked_1d, Int[])
    else
        idxs = _sample_unit_range_ndl(rng, idx_sp, dims)
        return flat[idxs]
    end
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, ::Random.SamplerType{T}, dims::Dims
) where {T}
    return internal_overload_rand(rng, Reactant.unwrapped_eltype(T), dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, ::Random.UniformT{T}, dims::Dims
) where {T}
    return internal_overload_rand(rng, Reactant.unwrapped_eltype(T), dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.CloseOpen01{T}},
    dims::Dims,
) where {T<:Reactant.ReactantFloat}
    return internal_overload_rand(rng, T, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.CloseOpen12{Float64}},
    dims::Dims,
)
    u = internal_overload_rand(rng, Float64, dims)
    return @opcall add(u, @opcall(fill(1.0, collect(Int, dims))))
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.UInt52Raw{UInt64}},
    dims::Dims,
)
    return internal_overload_rand(rng, UInt64, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.UInt52{UInt64}},
    dims::Dims,
)
    u = internal_overload_rand(rng, UInt64, dims)
    return @opcall shift_right_logical(u, @opcall(fill(UInt64(12), collect(Int, dims))))
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.UInt23Raw{UInt32}},
    dims::Dims,
)
    return internal_overload_rand(rng, UInt32, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.UInt23{UInt32}},
    dims::Dims,
)
    u = internal_overload_rand(rng, UInt32, dims)
    return @opcall shift_right_logical(u, @opcall(fill(UInt32(9), collect(Int, dims))))
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.UInt10Raw{UInt16}},
    dims::Dims,
)
    return internal_overload_rand(rng, UInt16, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    ::Random.SamplerTrivial{Random.UInt10{UInt16}},
    dims::Dims,
)
    u = internal_overload_rand(rng, UInt16, dims)
    return @opcall shift_right_logical(u, @opcall(fill(UInt16(6), collect(Int, dims))))
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerRangeNDL{U,T}, dims::Dims
) where {U<:Unsigned,T<:Reactant.ReactantInt}
    return _sample_unit_range_ndl(rng, sp, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, sp::TracedUnitRangeSampler, dims::Dims
)
    return _sample_traced_unit_range(rng, sp, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    sp::Random.SamplerSimple{<:AbstractArray,<:Random.Sampler},
    dims::Dims{N},
) where {N}
    arr = sp[]
    ET = Reactant.unwrapped_eltype(eltype(arr))
    shape = collect(Int, dims)
    if prod(dims) == 0
        z = @opcall fill(zero(ET), shape)
        return TracedRArray{ET,N}((), z.mlir_data, dims)
    end
    if arr isa StepRange && (eltype(arr) === ET) && (ET <: Signed) && (step(arr) isa ET)
        len = length(arr)
        k_i64 = _sample_unit_range_ndl(rng, Random.Sampler(ReactantRNG, 0:(len - 1)), dims)
        k = ET === Int64 ? k_i64 : @opcall(convert(TracedRArray{ET,N}, k_i64))
        s_arr = @opcall fill(step(arr), shape)
        a_arr = @opcall fill(first(arr), shape)
        return @opcall add(a_arr, @opcall(multiply(k, s_arr)))
    end
    if arr isa Reactant.TracedStepRangeLen ||
        (arr isa AbstractRange && eltype(arr) <: TracedRNumber)
        idxs = _sample_sampler_array(rng, sp.data, dims)
        if arr isa Reactant.TracedStepRangeLen
            return Base.unsafe_getindex.(Ref(arr), idxs)
        else
            k = idxs .- oneunit(eltype(idxs))
            return first(arr) .+ k .* step(arr)
        end
    end
    flat = if arr isa AnyTracedRArray
        vec(Reactant.materialize_traced_array(arr))
    else
        Reactant.promote_to(TracedRArray, collect(ET, vec(arr)))
    end
    return _gather_from_1d(rng, flat, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTrivial{Tuple{A}}, dims::Dims{N}
) where {A,N}
    t = sp[]
    ET = _tuple_sampler_eltype(t)
    shape = collect(Int, dims)
    v = Reactant.promote_to(TracedRNumber{ET}, t[1])
    return @opcall broadcast_in_dim(v, Int[], shape)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerSimple{<:Tuple}, dims::Dims
)
    t = sp[]
    ET = _tuple_sampler_eltype(t)
    flat = _tuple_to_traced_vec(ET, t)
    return _gather_from_1d(rng, flat, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    sp::Random.SamplerSimple{BitSet,<:Random.Sampler},
    dims::Dims,
)
    flat = Reactant.promote_to(TracedRArray, collect(Int, sp[]))
    return _gather_from_1d(rng, flat, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray},
    sp::Random.SamplerTag{<:Set,<:Random.Sampler},
    dims::Dims,
)
    d = sp.data[]
    ET = _safe_unwrapped_eltype(keytype(d))
    buf = Vector{ET}(undef, length(d))
    idx = 0
    for k in keys(d)
        idx += 1
        buf[idx] = convert(ET, k)
    end
    flat = Reactant.promote_to(TracedRArray, buf)
    return _gather_from_1d(rng, flat, dims)
end

@noinline function _sample_sampler_array(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTrivial{<:AbstractSet}, dims::Dims
)
    s = sp[]
    ET = Reactant.unwrapped_eltype(eltype(s))
    buf = Vector{ET}(undef, length(s))
    idx = 0
    for v in s
        idx += 1
        buf[idx] = convert(ET, v)
    end
    flat = Reactant.promote_to(TracedRArray, buf)
    return _gather_from_1d(rng, flat, dims)
end

# Scalar Base.rand methods for ReactantRNG matching Random/src/generation.jl signatures
for T in (Bool, Base.uniontypes(Reactant.ReactantInt)...)
    @eval @noinline function Base.rand(
        rng::ReactantRNG{<:TracedRArray}, ::Random.SamplerType{$(T)}
    )
        return internal_overload_rand(rng, $(T))
    end
end

for FT in Base.uniontypes(Reactant.ReactantFloat)
    @eval begin
        @noinline function Base.rand(
            rng::ReactantRNG{<:TracedRArray}, ::Random.SamplerType{$(FT)}
        )
            return internal_overload_rand(rng, $(FT))
        end
        @noinline function Base.rand(
            rng::ReactantRNG{<:TracedRArray},
            ::Random.SamplerTrivial{Random.CloseOpen01{$(FT)}},
        )
            return internal_overload_rand(rng, $(FT))
        end
    end
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, ::Random.UniformT{T}
) where {T}
    return Base.rand(rng, T)
end

for ST in (
    :(Random.SamplerTrivial{Random.CloseOpen12{Float64}}),
    :(Random.SamplerTrivial{Random.UInt52Raw{UInt64}}),
    :(Random.SamplerTrivial{Random.UInt52{UInt64}}),
    :(Random.SamplerTrivial{Random.UInt23Raw{UInt32}}),
    :(Random.SamplerTrivial{Random.UInt23{UInt32}}),
    :(Random.SamplerTrivial{Random.UInt10Raw{UInt16}}),
    :(Random.SamplerTrivial{Random.UInt10{UInt16}}),
)
    @eval @noinline function Base.rand(rng::ReactantRNG{<:TracedRArray}, sp::$(ST))
        return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
    end
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerRangeNDL{U,T}
) where {U<:Unsigned,T}
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(rng::ReactantRNG{<:TracedRArray}, sp::TracedUnitRangeSampler)
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray},
    sp::Random.SamplerSimple{<:AbstractArray,<:Random.Sampler},
)
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTrivial{Tuple{A}}
) where {A}
    t = sp[]
    ET = _tuple_sampler_eltype(t)
    ET === Union{} && return t[1]
    return Reactant.promote_to(TracedRNumber{ET}, t[1])
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerSimple{Tuple{A,B}}
) where {A,B}
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerSimple{Tuple{A,B,C}}
) where {A,B,C}
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerSimple{T}
) where {T<:Tuple}
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerSimple{BitSet,<:Random.Sampler}
)
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTag{<:Set,<:Random.Sampler}
)
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTrivial{<:AbstractSet}
)
    return _scalar_from_0d(_sample_sampler_array(rng, sp, ()))
end

_sample_tuple_tag_step(rng, ::Val{0}, data::Tuple, ::Int) = ()
function _sample_tuple_tag_step(rng, ::Val{N}, data::Tuple, i::Int) where {N}
    sp_i = data[min(i, length(data))]
    v_i = Base.rand(rng, sp_i)
    return (v_i, _sample_tuple_tag_step(rng, Val(N - 1), data, i + 1)...)
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTag{Ref{T}}
) where {T<:Tuple}
    return _sample_tuple_tag_step(rng, Val(fieldcount(T)), sp.data, 1)
end

@noinline function Base.rand(
    rng::ReactantRNG{<:TracedRArray}, sp::Random.SamplerTag{<:Ref{<:Pair}}
)
    return Base.rand(rng, sp.data.first) => Base.rand(rng, sp.data.second)
end

@noinline function Random.rand!(
    rng::ReactantRNG{<:TracedRArray}, A::AnyTracedRArray{T,N}, sp::Random.Sampler
) where {T,N}
    vals = _sample_sampler_array(rng, sp, size(A))
    converted = eltype(vals) === T ? vals : @opcall(convert(TracedRArray{T,N}, vals))
    TracedUtils.set_mlir_data!(A, converted.mlir_data)
    return A
end

function can_overlay_sampler(X; array::Bool=false)
    if (X isa AbstractArray || X isa Tuple || X isa AbstractSet) && _known_empty(X)
        return true
    end
    sp =
        X isa Random.Sampler ? X : Random.Sampler(ReactantRNG, X, array ? Val(Inf) : Val(1))
    return array ? supports_sampler_array(sp) : supports_sampler(sp)
end

@noinline function overload_rand_sampler(rng::AbstractRNG, X)
    trng = _to_traced_rng(rng)
    sp = X isa Random.Sampler ? X : Random.Sampler(trng, X, Val(1))
    return Base.rand(trng, sp)
end

@noinline function overload_rand_sampler(rng::AbstractRNG, X, dims::Dims)
    trng = _to_traced_rng(rng)
    sp = X isa Random.Sampler ? X : Random.Sampler(trng, X, Val(Inf))
    return _sample_sampler_array(trng, sp, dims)
end

@noinline function overload_rand_sampler!(rng::AbstractRNG, A::AnyTracedRArray, X)
    trng = _to_traced_rng(rng)
    sp = X isa Random.Sampler ? X : Random.Sampler(trng, X, Val(Inf))
    return Random.rand!(trng, A, sp)
end

end
