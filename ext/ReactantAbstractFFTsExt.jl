module ReactantAbstractFFTsExt
using Reactant.MLIR.Dialects: stablehlo
using AbstractFFTs: AbstractFFTs
using Reactant: Reactant, MLIR, Ops, TracedRArray

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

for op in (stablehlo.FftType.RFFT, stablehlo.FftType.FFT, stablehlo.FftType.IFFT)
    name = Symbol(lowercase(string(op)))
    @eval function AbstractFFTs.$(name)(x::TracedRArray, dims)
        @assert maximum(dims) ≤ ndims(x) "dims out of range"
        if dims isa Integer
            if dims != 1
                pdims = compute_correct_pdims(x, dims)
                return permutedims(
                    AbstractFFTs.$(name)(permutedims(x, pdims), 1), invperm(pdims)
                )
            end
            return generalized_fft(x, $(op), nothing, length(dims))
        end
        if !check_contiguous_innermost_dims(dims, ndims(x))
            pdims = compute_correct_pdims(x, dims)
            return permutedims(
                AbstractFFTs.$(name)(permutedims(x, pdims), 1:length(dims)), invperm(pdims)
            )
        end
        return generalized_fft(x, $(op), nothing, length(dims))
    end
end

for op in (stablehlo.FftType.IRFFT,)
    name = Symbol(lowercase(string(op)))
    @eval function AbstractFFTs.$(name)(x::TracedRArray, d::Int, dims)
        @assert maximum(dims) ≤ ndims(x) "dims out of range"
        if dims isa Integer
            if dims != 1
                pdims = compute_correct_pdims(x, dims)
                return permutedims(
                    AbstractFFTs.$(name)(permutedims(x, pdims), d, 1), invperm(pdims)
                )
            end
            return generalized_fft(x, $(op), d, length(dims))
        end
        if !check_contiguous_innermost_dims(dims, ndims(x))
            pdims = compute_correct_pdims(x, dims)
            return permutedims(
                AbstractFFTs.$(name)(permutedims(x, pdims), d, 1:length(dims)),
                invperm(pdims),
            )
        end
        return generalized_fft(x, $(op), d, length(dims))
    end
end

function generalized_fft(
    x::TracedRArray{T,N}, mode::stablehlo.FftType.T, d, first_n::Int
) where {T,N}
    if d === nothing
        @assert mode ∈
            (stablehlo.FftType.RFFT, stablehlo.FftType.FFT, stablehlo.FftType.IFFT)
        fft_length = [size(x, i) for i in 1:first_n]
    else
        @assert mode == stablehlo.FftType.IRFFT
        fft_length = [i == 1 ? d : size(x, i) for i in 1:first_n]
    end

    x = permutedims(x, reverse(1:N))
    reverse!(fft_length)
    x = Ops.fft(x; type=mode, length=fft_length)
    return permutedims(x, reverse(1:N))
end

end
