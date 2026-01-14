module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs, fftdims
using LinearAlgebra
using Reactant: Reactant, MLIR, Ops, AnyTracedRArray, TracedRArray, TracedUtils
using Reactant.Ops: @opcall

# To automatically convert FFT plans to traced versions
# To extend a user needs to extend Reactant.reactant_fftplan for their plan type
# see ReactantFFTWExt.jl for an example implementation
function Reactant.make_tracer(
    seen, @nospecialize(prev::AbstractFFTs.Plan{T}), @nospecialize(path), mode; kwargs...
) where {T}
    return reactant_fftplan(prev)
end

abstract type AbstractReactantFFTPlan{T} <: AbstractFFTs.Plan{T} end
AbstractFFTs.fftdims(p::AbstractReactantFFTPlan) = p.dims

reactant_fftplan(p::AbstractReactantFFTPlan) = p
function reactant_fftplan(p::AbstractFFTs.ScaledPlan)
    return AbstractFFTs.ScaledPlan(reactant_fftplan(p.p), p.scale)
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

    @eval function AbstractFFTs.$(plan_f)(
        x::Reactant.TracedRArray{T}, dims=1:ndims(x)
    ) where {T}
        return $(plan_name){T,typeof(dims)}(dims)
    end

    @eval function Base.:*(p::$(plan_name){T}, x::Reactant.TracedRArray{T}) where {T}
        return AbstractFFTs.$(op)(x, p.dims)
    end

    @eval function LinearAlgebra.mul!(
        y::Reactant.TracedRArray, p::$(plan_name), x::Reactant.TracedRArray
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
            x::Reactant.TracedRArray{T}, dims=1:ndims(x)
        ) where {T}
            return $(plan_name!){T,typeof(dims)}(dims)
        end
        @eval function Base.:*(p::$(plan_name!){T}, x::Reactant.TracedRArray{T}) where {T}
            return copyto!(x, AbstractFFTs.$(op)(x, p.dims))
        end

        @eval function LinearAlgebra.mul!(
            y::Reactant.TracedRArray, p::$(plan_name!), x::Reactant.TracedRArray
        )
            return copyto!(y, AbstractFFTs.$(op)(x, fftdims(p)))
        end
    end
end

for op in (:irfft,)
    mode = uppercase(string(op))

    @eval function AbstractFFTs.$(op)(x::AnyTracedRArray, d::Integer, dims)
        @assert maximum(dims) <= ndims(x) "Invalid dimensions for irfft: $(dims)"

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
        x::Reactant.TracedRArray{T}, d::Integer, dims=1:ndims(x)
    ) where {T}
        return $(plan_name){T,typeof(dims)}(dims, d)
    end

    @eval function Base.:*(p::$(plan_name){T}, x::Reactant.TracedRArray{T}) where {T}
        return AbstractFFTs.$(op)(x, p.length, p.dims)
    end

    @eval function LinearAlgebra.mul!(
        y::Reactant.TracedRArray{<:Real}, p::$(plan_name){T}, x::Reactant.TracedRArray{T}
    ) where {T<:Complex}
        return copyto!(y, AbstractFFTs.$(op)(x, p.length, fftdims(p)))
    end
end

# Because XLA defines ifft and irfft directly we need to support bfft by adding a normalization
# factor ifft operations. This is inverse of the usual AbstractFFTs normalization.
function normbfft(::Type{T}, size, dims) where {T}
    return inv(AbstractFFTs.normalization(real(T), size, dims))
end
for op in (:bfft,)
    normop = Symbol("norm", op)
    iop = Symbol(replace(string(op), "b" => "i"))
    @eval function AbstractFFTs.$(op)(x::AnyTracedRArray{T}, dims) where {T}
        y = AbstractFFTs.$(iop)(x, dims)
        y .*= $(normop)(real(T), size(x), dims)
        return y
    end

    planop = Symbol("plan_", op)
    planiop = Symbol("plan_", iop)
    @eval function AbstractFFTs.$(planop)(
        x::Reactant.TracedRArray{T}, dims=1:ndims(x)
    ) where {T}
        y = AbstractFFTs.$(planiop)(x, dims)
        return $(normop)(real(T), size(x), dims) * y # ScaledPlan
    end

    # No in-place brfft (different array size)

    inplaceop = Symbol(op, "!")
    inplaceiop = Symbol(iop, "!")
    @eval function AbstractFFTs.$(inplaceop)(x::AnyTracedRArray{T}, dims) where {T}
        AbstractFFTs.$(inplaceiop)(x, dims)
        x .*= $(normop)(real(T), size(x), dims)
        return x
    end

    planop! = Symbol(planop, "!")
    planiop! = Symbol(planiop, "!")
    @eval function AbstractFFTs.$(planop!)(
        x::Reactant.TracedRArray{T}, dims=1:ndims(x)
    ) where {T}
        return $(normop)(real(T), size(x), dims) * AbstractFFTs.$(planiop!)(x, dims)
    end
end

# This must be implemented for bfft
function reallength end

function AbstractFFTs.brfft(x::AnyTracedRArray{T}, dims) where {T}
    y = AbstractFFTs.$(iop)(x, dims)
    y .*= normbfft(real(T), size(y), dims)
    return y
end

function AbstractFFTs.plan_brfft(
    x::Reactant.TracedRArray{T}, length::Integer, dims=1:ndims(x)
) where {T}
    y = AbstractFFTs.plan_irfft(x, length, dims)
    sz = AbstractFFTs.brfft_output_size(size(x), length, dims)
    return normbfft(real(T), sz, dims) * y # ScaledPlan
end

function LinearAlgebra.mul!(
    y::Reactant.TracedRArray,
    p::AbstractFFTs.ScaledPlan{<:AbstractReactantFFTPlan},
    x::Reactant.TracedRArray,
)
    mul!(y, p.p, x)
    y .*= p.scale
    return y
end

end
