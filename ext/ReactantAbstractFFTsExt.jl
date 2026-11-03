module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs, fftdims
using LinearAlgebra: LinearAlgebra
using Reactant:
    Reactant,
    AbstractConcreteArray,
    AnyTracedRArray,
    TracedRArray,
    TracedRNumber,
    ancestor,
    promote_to,
    unwrapped_eltype
using Reactant.Ops: @opcall
using ReactantCore: ReactantCore, materialize_traced_array

# FFTW.jl defines methods on StridedArrays, and we have to be more specific than FFTW to
# catch its methods.
const AnyStridedTracedRArray{T,N} = StridedArray{<:TracedRNumber{T},N}

# To automatically convert FFT plans to traced versions
# To extend a user needs to extend Reactant.reactant_fftplan for their plan type
# see ReactantFFTWExt.jl for an example implementation
function Reactant.make_tracer(
    seen,
    @nospecialize(prev::AbstractFFTs.Plan{T}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type=Union{}),
    @nospecialize(sharding=Reactant.Sharding.NoSharding()),
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
abstract type AbstractReactantOutOfPlaceFFTPlan{T} <: AbstractReactantFFTPlan{T} end
abstract type AbstractReactantInPlaceFFTPlan{T} <: AbstractReactantFFTPlan{T} end
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

        (dims isa Union{Integer,TracedRNumber{<:Integer}} && (dims = (dims,)))

        fft_lengths = Int64[size(x, dim) for dim in reverse(dims)]
        if __is_valid_stablehlo_fft_dims(dims, ndims(x))
            return @opcall fft(
                materialize_traced_array(x);
                type=($(uppercase(string(op)))),
                length=fft_lengths,
            )
        end
        perm = __permutation_to_move_dims_to_end(dims, ndims(x))
        return permutedims(
            @opcall(
                fft(
                    materialize_traced_array(permutedims(x, perm));
                    type=($(uppercase(string(op)))),
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
    @eval struct $(plan_name){T,D} <: AbstractReactantOutOfPlaceFFTPlan{T}
        dims::D
    end
    @eval $(plan_name){T}(dims) where {T} = $(plan_name){T,typeof(dims)}(dims)

    @eval function AbstractFFTs.$(plan_f)(
        x::AnyTracedRArray{T}, dims=1:ndims(x); _kwargs...
    ) where {T}
        return $(plan_name){T,typeof(dims)}(dims)
    end

    @eval function Base.:*(p::$(plan_name){T}, x::AnyTracedRArray{T}) where {T}
        return AbstractFFTs.$(op)(x, p.dims)
    end

    @eval function LinearAlgebra.mul!(
        y::AnyTracedRArray, p::$(plan_name){T}, x::AnyTracedRArray{T}
    ) where {T}
        return copyto!(y, AbstractFFTs.$(op)(x, fftdims(p)))
    end

    # In-place plan
    if op !== :rfft
        plan_name! = Symbol("Reactant", uppercase(string(op)), "InPlacePlan")
        plan_f! = Symbol("plan_", op, "!")
        @eval struct $(plan_name!){T,D} <: AbstractReactantInPlaceFFTPlan{T}
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
            y::AnyTracedRArray, p::$(plan_name!){T}, x::AnyTracedRArray{T}
        ) where {T}
            return copyto!(y, AbstractFFTs.$(op)(x, fftdims(p)))
        end
    end
end

for op in (:irfft,)
    mode = uppercase(string(op))

    @eval function AbstractFFTs.$(op)(x::AnyTracedRArray, d::Integer, dims)
        @assert maximum(dims) <= ndims(x) "Invalid dimensions for irfft: $(dims)"

        (dims isa Union{Integer,TracedRNumber{<:Integer}} && (dims = (dims,)))

        fft_lengths = vcat(Int64[size(x, dim) for dim in reverse(dims[2:end])], d)

        if __is_valid_stablehlo_fft_dims(dims, ndims(x))
            return @opcall fft(
                materialize_traced_array(x);
                type=($(uppercase(string(op)))),
                length=fft_lengths,
            )
        end

        perm = __permutation_to_move_dims_to_end(dims, ndims(x))
        return permutedims(
            @opcall(
                fft(
                    materialize_traced_array(permutedims(x, perm));
                    type=($(uppercase(string(op)))),
                    length=fft_lengths,
                )
            ),
            invperm(perm),
        )
    end

    #Inverse plan I need to store the real array length along the first dim in dims
    plan_name = Symbol("Reactant", uppercase(string(op)), "Plan")
    plan_f = Symbol("plan_", op)
    @eval struct $(plan_name){T,D} <: AbstractReactantOutOfPlaceFFTPlan{T}
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

# AbstractFFTs defines `*(p::Plan{T}, x::AbstractArray) = p * copy1(T, x)`, assuming
# implementations define `*(::Plan{T}, ::Array{T})`. Since `AbstractReactantFFTPlan`s
# instead dispatch on `AnyTracedRArray{T}`, handle mismatched element types, untraced
# arrays during compilation, and eager concrete/host arrays explicitly rather than
# falling into an infinite `copy1` recursion (#3259).
function _check_fft_input_eltype(p::AbstractFFTs.Plan{T}, x::AbstractArray) where {T}
    if T <: Real &&
        (x isa AnyTracedRArray || ancestor(x) isa AbstractConcreteArray) &&
        unwrapped_eltype(x) <: Complex
        throw(
            ArgumentError(
                "Cannot apply real-input FFT plan of type $(typeof(p)) to complex array of type $(typeof(x))",
            ),
        )
    end
    return nothing
end

function _promote_fft_input(::Type{T}, p::AbstractFFTs.Plan{T}, x::AbstractArray) where {T}
    if x isa AnyTracedRArray || ancestor(x) isa AbstractConcreteArray
        _check_fft_input_eltype(p, x)
        return promote_to(TracedRArray{T,ndims(x)}, x)
    end
    return promote_to(TracedRArray{T,ndims(x)}, AbstractFFTs.copy1(T, x))
end

function _throw_untraced_fft_dest_error(p::AbstractFFTs.Plan, y::AbstractArray)
    if ReactantCore.within_compile()
        throw(
            ArgumentError(
                "Cannot store result of Reactant FFT plan of type $(typeof(p)) into untraced array of type $(typeof(y)); pass the destination array as a traced argument (converted with `Reactant.to_rarray`).",
            ),
        )
    end
    return throw(
        ArgumentError(
            "Cannot apply Reactant FFT plan of type $(typeof(p)) to host array of type $(typeof(y)); convert the array with `Reactant.to_rarray` first.",
        ),
    )
end

function Base.:*(p::AbstractReactantFFTPlan{T}, x::AbstractArray) where {T}
    if x isa AnyTracedRArray
        x isa AnyTracedRArray{T} && throw(MethodError(*, (p, x)))
        return p * _promote_fft_input(T, p, x)
    end
    if ReactantCore.within_compile()
        if p isa AbstractReactantOutOfPlaceFFTPlan
            return p * _promote_fft_input(T, p, x)
        end
        throw(
            ArgumentError(
                "Cannot apply in-place Reactant FFT plan of type $(typeof(p)) to untraced array of type $(typeof(x)); pass the array as a traced argument (converted with `Reactant.to_rarray`) or use an out-of-place FFT plan.",
            ),
        )
    end
    if ancestor(x) isa AbstractConcreteArray
        _check_fft_input_eltype(p, x)
        fn = Reactant.compile(*, (p, x))
        return fn(p, x)
    end
    return throw(
        ArgumentError(
            "Cannot apply Reactant FFT plan of type $(typeof(p)) to host array of type $(typeof(x)); convert the array with `Reactant.to_rarray` first.",
        ),
    )
end

function Base.:*(
    p::AbstractFFTs.ScaledPlan{T,<:AbstractReactantFFTPlan}, x::AbstractArray
) where {T}
    if !ReactantCore.within_compile() &&
        !(x isa AnyTracedRArray) &&
        ancestor(x) isa AbstractConcreteArray
        _check_fft_input_eltype(p, x)
        fn = Reactant.compile(*, (p, x))
        return fn(p, x)
    end
    return LinearAlgebra.rmul!(p.p * x, p.scale)
end

function LinearAlgebra.mul!(
    y::AnyTracedRArray, p::AbstractReactantFFTPlan{T}, x::AbstractArray
) where {T}
    x isa AnyTracedRArray{T} && throw(MethodError(LinearAlgebra.mul!, (y, p, x)))
    return LinearAlgebra.mul!(y, p, _promote_fft_input(T, p, x))
end

function LinearAlgebra.mul!(
    y::AbstractArray, p::AbstractReactantFFTPlan{T}, x::AbstractArray
) where {T}
    if !ReactantCore.within_compile() && ancestor(y) isa AbstractConcreteArray
        _check_fft_input_eltype(p, x)
        fn = Reactant.compile(LinearAlgebra.mul!, (y, p, x))
        return fn(y, p, x)
    end
    return _throw_untraced_fft_dest_error(p, y)
end

function LinearAlgebra.mul!(
    y::AbstractArray,
    p::AbstractFFTs.ScaledPlan{T,<:AbstractReactantFFTPlan},
    x::AbstractArray,
) where {T}
    if y isa AnyTracedRArray
        return LinearAlgebra.lmul!(p.scale, LinearAlgebra.mul!(y, p.p, x))
    end
    if !ReactantCore.within_compile() && ancestor(y) isa AbstractConcreteArray
        _check_fft_input_eltype(p, x)
        fn = Reactant.compile(LinearAlgebra.mul!, (y, p, x))
        return fn(y, p, x)
    end
    return _throw_untraced_fft_dest_error(p, y)
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
