module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs, fftdims
using LinearAlgebra
using Reactant: Reactant, MLIR, Ops, AnyTracedRArray, TracedRNumber, TracedUtils
using Reactant.Ops: @opcall

# FFTW.jl defines methods on StridedArrays, and we have to be more specific than FFTW to
# catch its methods.
const AnyStridedTracedRArray{T,N} = StridedArray{TracedRNumber{T},N}

# To automatically convert FFT plans to traced versions
# To extend a user needs to extend Reactant.reactant_fftplan for their plan type
# see ReactantFFTWExt.jl for an example implementation
function Reactant.make_tracer(
    seen,
    @nospecialize(prev::AbstractFFTs.Plan{T}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Reactant.Sharding.NoSharding()),
    @nospecialize(runtime),
    kwargs...,
) where {T}
    RT = Reactant.traced_type(typeof(prev), Val(mode), track_numbers, sharding, runtime)
    return reactant_fftplan(RT, prev)
end

function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:AbstractFFTs.Plan}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    RT = reactant_fftplan_type(T)
    return RT
end

abstract type AbstractReactantFFTPlan{T} <: AbstractFFTs.Plan{T} end
AbstractFFTs.fftdims(p::AbstractReactantFFTPlan) = p.dims

function reactant_fftplan(RT, p::AbstractFFTs.Plan)
    rp = make_reactant_fftplan(p)
    @assert RT === typeof(rp) "reactant_fftplan returned type $(typeof(rp)) but expected $RT"
    return rp
end

reactant_fftplan_type(T::Type{<:AbstractReactantFFTPlan}) = T
make_reactant_fftplan(p::AbstractReactantFFTPlan) = p

function reactant_fftplan_type(T::Type{<:AbstractFFTs.ScaledPlan})
    return _scaled_plan_type(T)
end

function _scaled_plan_type(::Type{<:AbstractFFTs.ScaledPlan{T,P,N}}) where {T,P,N}
    PI = _scaled_plan_inner(P)
    NI = _scaled_plan_norm(P, N)
    return AbstractFFTs.ScaledPlan{T,PI,NI}
end

# Recursion to get inner plan type for ScaledPlan since the constructors make it so that you have have
# ScaledPlan{ReactantFFTPlan, N}
function _scaled_plan_inner(::Type{<:AbstractFFTs.ScaledPlan{T,P,N}}) where {T,P,N}
    return _scaled_plan_inner(P)
end

# reactant_fftplan_type may return another ScaledPlan so we need to recurse on that too
function _scaled_plan_inner(P::Type{<:AbstractFFTs.Plan})
    return _scaled_plan_inner(reactant_fftplan_type(P))
end

# Base case
function _scaled_plan_inner(T::Type{<:AbstractReactantFFTPlan})
    return T
end

# Recursion to get scaled type for ScaledPlan since the constructors make it so that you have have
# ScaledPlan{ReactantFFTPlan, N}
function _scaled_plan_norm(::Type{<:AbstractFFTs.ScaledPlan{T,P,N}}, M::Type) where {T,P,N}
    return _scaled_plan_norm(P, promote_type(N, M))
end

# reactant_fftplan_type may return another ScaledPlan so we need to recurse on that too
function _scaled_plan_norm(P::Type{<:AbstractFFTs.Plan}, N::Type)
    return _scaled_plan_norm(reactant_fftplan_type(P), N)
end

# Base case
function _scaled_plan_norm(::Type{<:AbstractReactantFFTPlan}, N::Type)
    return N
end

function make_reactant_fftplan(
    p::AbstractFFTs.ScaledPlan{T,P,N}
) where {T,P<:AbstractFFTs.Plan,N}
    rp = make_reactant_fftplan(p.p)
    return AbstractFFTs.ScaledPlan(rp, p.scale)
end

function __permutation_to_move_dims_to_end(dims, N::Integer)
    perm = [i for i in 1:N if i ∉ Set(dims)]
    append!(perm, reverse(dims))
    return perm
end

__is_valid_stablehlo_fft_dims(dim::Integer, N::Integer) = dim == N

function __is_valid_stablehlo_fft_dims(dims, N::Integer)
    return collect(dims) == collect(N:-1:(N - length(dims) + 1))
end

for op in (:rfft, :fft, :ifft)
    @eval function AbstractFFTs.$(op)(x::AnyTracedRArray, dims)
        @assert maximum(dims) <= ndims(x) "Invalid dimensions for fft: $(dims)"

        (dims isa Union{Integer,Reactant.TracedRNumber{<:Integer}} && (dims = (dims,)))

        fft_lengths = Int64[size(x, dim) for dim in reverse(dims)]
        if __is_valid_stablehlo_fft_dims(dims, ndims(x))
            return @opcall fft(
                TracedUtils.materialize_traced_array(x);
                type=$(uppercase(string(op))),
                length=fft_lengths,
            )
        end
        perm = __permutation_to_move_dims_to_end(dims, ndims(x))
        return permutedims(
            @opcall(
                fft(
                    TracedUtils.materialize_traced_array(permutedims(x, perm));
                    type=$(uppercase(string(op))),
                    length=fft_lengths,
                )
            ),
            invperm(perm),
        )
    end

    # No in-place rfft (different array size)
    if op !== :rfft
        @eval function AbstractFFTs.$(Symbol(op, "!"))(x::AnyTracedRArray, dims)
            y = AbstractFFTs.$(op)(x, dims)
            copyto!(x, y)
            return x
        end
    end

    # Out-of-place plan
    plan_name = Symbol("Reactant", uppercase(string(op)), "Plan")
    plan_f = Symbol("plan_", op)
    @eval struct $(plan_name){T,D} <: AbstractReactantFFTPlan{T}
        dims::D
    end
    @eval $(plan_name){T}(dims) where {T} = $(plan_name){T,typeof(dims)}(dims)

    @eval function AbstractFFTs.$(plan_f)(x::AnyTracedRArray{T}, dims=1:ndims(x)) where {T}
        return $(plan_name){T,typeof(dims)}(dims)
    end

    @eval function Base.:*(p::$(plan_name){T}, x::AnyTracedRArray{T}) where {T}
        return AbstractFFTs.$(op)(x, p.dims)
    end

    @eval function LinearAlgebra.mul!(
        y::AnyTracedRArray, p::$(plan_name), x::AnyTracedRArray
    )
        return copyto!(y, AbstractFFTs.$(op)(x, fftdims(p)))
    end

    # In-place plan
    if op !== :rfft
        plan_name! = Symbol("Reactant", uppercase(string(op)), "InPlacePlan")
        plan_f! = Symbol("plan_", op, "!")
        @eval struct $(plan_name!){T,D} <: AbstractReactantFFTPlan{T}
            dims::D
        end
        @eval $(plan_name!){T}(dims) where {T} = $(plan_name!){T,typeof(dims)}(dims)

        @eval function AbstractFFTs.$(plan_f!)(
            x::AnyTracedRArray{T}, dims=1:ndims(x)
        ) where {T}
            return $(plan_name!){T,typeof(dims)}(dims)
        end
        # This method with `AnyStridedTracedRArray` is needed to extend methods defined in
        # `FFTW.jl` for `StridedArray`s which have keyword arguments (which we'll ignore
        # here).  The body of the method is the same as the above one, but we can't use a
        # `Union` because it'd still be less specific than the FFTW's methods.
        @eval function AbstractFFTs.$(plan_f!)(
            x::AnyStridedTracedRArray{T}, dims=1:ndims(x); _kwargs...
        ) where {T}
            return $(plan_name!){T,typeof(dims)}(dims)
        end
        @eval function Base.:*(p::$(plan_name!){T}, x::AnyTracedRArray{T}) where {T}
            return copyto!(x, AbstractFFTs.$(op)(x, p.dims))
        end

        @eval function LinearAlgebra.mul!(
            y::AnyTracedRArray, p::$(plan_name!), x::AnyTracedRArray
        )
            return copyto!(y, AbstractFFTs.$(op)(x, fftdims(p)))
        end
    end
end

for op in (:irfft,)
    mode = uppercase(string(op))

    @eval function AbstractFFTs.$(op)(x::AnyTracedRArray, d::Integer, dims)
        @assert maximum(dims) <= ndims(x) "Invalid dimensions for irfft: $(dims)"

        (dims isa Union{Integer,Reactant.TracedRNumber{<:Integer}} && (dims = (dims,)))

        fft_lengths = vcat(Int64[size(x, dim) for dim in reverse(dims[2:end])], d)

        if __is_valid_stablehlo_fft_dims(dims, ndims(x))
            return @opcall fft(
                TracedUtils.materialize_traced_array(x);
                type=$(uppercase(string(op))),
                length=fft_lengths,
            )
        end

        perm = __permutation_to_move_dims_to_end(dims, ndims(x))
        return permutedims(
            @opcall(
                fft(
                    TracedUtils.materialize_traced_array(permutedims(x, perm));
                    type=$(uppercase(string(op))),
                    length=fft_lengths,
                )
            ),
            invperm(perm),
        )
    end

    #Inverse plan I need to store the real array length along the first dim in dims
    plan_name = Symbol("Reactant", uppercase(string(op)), "Plan")
    plan_f = Symbol("plan_", op)
    @eval struct $(plan_name){T,D} <: AbstractReactantFFTPlan{T}
        dims::D
        length::Int
    end
    @eval $(plan_name){T}(dims, length) where {T} =
        $(plan_name){T,typeof(dims)}(dims, length)
    @eval function AbstractFFTs.$(plan_f)(
        x::AnyTracedRArray{T}, d::Integer, dims=1:ndims(x)
    ) where {T}
        return $(plan_name){T,typeof(dims)}(dims, d)
    end

    @eval function Base.:*(p::$(plan_name){T}, x::AnyTracedRArray{T}) where {T}
        return AbstractFFTs.$(op)(x, p.length, p.dims)
    end

    @eval function LinearAlgebra.mul!(
        y::AnyTracedRArray{<:Real}, p::$(plan_name){T}, x::AnyTracedRArray{T}
    ) where {T<:Complex}
        return copyto!(y, AbstractFFTs.$(op)(x, p.length, fftdims(p)))
    end
end

# Because XLA defines ifft and irfft directly we need to support bfft by adding a normalization
# factor ifft operations. This is inverse of the usual AbstractFFTs normalization.
function normbfft(::Type{T}, size, dims) where {T}
    return inv(AbstractFFTs.normalization(real(T), size, dims))
end

# Because we override the plan_bfft and plan_brfft functions we actually do not need to define
# AbstractFFTs.bfft functions since they come for free via the plan mechanism.
function AbstractFFTs.plan_bfft(
    x::AnyTracedRArray{T}, dims=1:ndims(x); _kwargs...
) where {T}
    pl = AbstractFFTs.plan_ifft(x, dims)
    return normbfft(real(T), size(x), dims) * pl
end

function AbstractFFTs.plan_bfft!(
    x::AnyTracedRArray{T}, dims=1:ndims(x); _kwargs...
) where {T}
    pl = AbstractFFTs.plan_ifft!(x, dims)
    return normbfft(real(T), size(x), dims) * pl
end

# This must be implemented for bfft
function reallength end

function AbstractFFTs.plan_brfft(
    x::AnyTracedRArray{T}, length::Integer, dims=1:ndims(x)
) where {T}
    y = AbstractFFTs.plan_irfft(x, length, dims)
    sz = AbstractFFTs.brfft_output_size(size(x), length, dims)
    return normbfft(real(T), sz, dims) * y
end

end
