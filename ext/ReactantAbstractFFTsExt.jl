module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using Reactant: Reactant, MLIR, TracedRArray

function check_contiguous_innermost_dims(dims, N)
    @assert sort([dims...]) == [dims...] "un-sorted dims are not supported"
    all(i -> dims[i] == dims[i - 1] + 1, 2:(length(dims))) || return false
    dims[1] != 1 && return false
    return true
end

function compute_correct_pdims(x::AbstractArray, dims::Int)
    counter = 0
    return ntuple(ndims(x)) do i
        i == 1 && return dims
        counter += 1
        return counter
    end
end

function compute_correct_pdims(x::AbstractArray, dims)
    counter = 0
    return ntuple(ndims(x)) do i
        i ≤ length(dims) && return dims[i]
        counter += 1
        while counter ∈ dims
            counter += 1
        end
        return counter
    end
end

for op in (:rfft, :fft, :ifft)
    @eval function AbstractFFTs.$(op)(x::TracedRArray, dims)
        @assert maximum(dims) ≤ ndims(x) "dims out of range"
        if dims isa Integer
            if dims != 1
                pdims = compute_correct_pdims(x, dims)
                return permutedims(
                    AbstractFFTs.$(op)(permutedims(x, pdims), 1), invperm(pdims)
                )
            end
            return generalized_fft(x, $(Meta.quot(op)), nothing, 1)
        end
        if !check_contiguous_innermost_dims(dims, ndims(x))
            pdims = compute_correct_pdims(x, dims)
            return permutedims(
                AbstractFFTs.$(op)(permutedims(x, pdims), 1:length(dims)), invperm(pdims)
            )
        end
        return generalized_fft(x, $(Meta.quot(op)), nothing, length(dims))
    end
end

for op in (:irfft,)
    @eval function AbstractFFTs.$(op)(x::TracedRArray, d::Int, dims)
        @assert maximum(dims) ≤ ndims(x) "dims out of range"
        if dims isa Integer
            if dims != 1
                pdims = compute_correct_pdims(x, dims)
                return permutedims(
                    AbstractFFTs.$(op)(permutedims(x, pdims), d, 1), invperm(pdims)
                )
            end
            return generalized_fft(x, $(Meta.quot(op)), d, 1)
        end
        if !check_contiguous_innermost_dims(dims, ndims(x))
            pdims = compute_correct_pdims(x, dims)
            return permutedims(
                AbstractFFTs.$(op)(permutedims(x, pdims), d, 1:length(dims)), invperm(pdims)
            )
        end
        return generalized_fft(x, $(Meta.quot(op)), d, length(dims))
    end
end

function generalized_fft(x::TracedRArray{T,N}, mode::Symbol, d, first_n::Int) where {T,N}
    @assert mode ∈ (:rfft, :irfft, :fft, :ifft)

    x = permutedims(x, reverse(1:N))
    fft_type_str = uppercase(string(mode))
    fft_type = MLIR.API.stablehloFftTypeAttrGet(MLIR.IR.context(), fft_type_str)

    if d === nothing
        @assert mode ∈ (:rfft, :fft, :ifft)
        if mode == :rfft
            @assert T <: Real
            rT = Complex{T}
            res_size = [size(x)[1:(end - 1)]..., size(x, N) ÷ 2 + 1]
        else
            @assert T <: Complex
            rT = T
            res_size = [size(x)...]
        end
        fft_length = [size(x, i) for i in (ndims(x) - first_n + 1):ndims(x)]
    else
        @assert mode == :irfft
        @assert T <: Complex
        rT = real(T)
        res_size = [size(x)[1:(end - 1)]..., d]
        fft_length = [res_size[i] for i in (ndims(x) - first_n + 1):ndims(x)]
    end

    @assert 1 ≤ length(fft_length) ≤ 3 "stablehlo.fft only supports up to rank 3"
    mlir_type = MLIR.IR.TensorType(res_size, Reactant.MLIR.IR.Type(rT))
    op = MLIR.Dialects.stablehlo.fft(x.mlir_data; fft_type, fft_length, result_0=mlir_type)
    x = TracedRArray{rT,N}((), MLIR.IR.result(op, 1), Tuple(res_size))
    return permutedims(x, reverse(1:N))
end

end
