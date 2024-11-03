module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using Reactant: Reactant, MLIR, TracedRArray

function contiguous_dims(dims, N)
    all(i -> dims[i] == dims[i - 1] + 1, 2:(length(dims))) || return false
    dims[1] != (N - length(dims) + 1) && return false
    return true
end

for op in (:rfft, :fft, :ifft)
    @eval function AbstractFFTs.$(op)(x::TracedRArray, dims)
        if dims isa Integer
            if dims != ndims(x)
                error("Not yet implemented")
            end
            return generalized_fft(x, $(Meta.quot(op)), nothing, ndims(x) - dims + 1)
        end
        contiguous_dims(dims, ndims(x)) &&
            return generalized_fft(x, $(Meta.quot(op)), nothing, ndims(x) - first(dims) + 1)
        return error("Not yet implemented")
    end
end

for op in (:irfft,)
    @eval function AbstractFFTs.$(op)(x::TracedRArray, d::Int, dims)
        if dims isa Integer
            if dims != ndims(x)
                error("Not yet implemented")
            end
            return generalized_fft(x, $(Meta.quot(op)), d, ndims(x) - dims + 1)
        end
        contiguous_dims(dims, ndims(x)) &&
            return generalized_fft(x, $(Meta.quot(op)), d, ndims(x) - first(dims) + 1)
        return error("Not yet implemented")
    end
end

function generalized_fft(x::TracedRArray{T,N}, mode::Symbol, d, last_n::Int) where {T,N}
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
        fft_length = [size(x, i) for i in (ndims(x) - last_n + 1):ndims(x)]
    else
        @assert mode == :irfft
        @assert T <: Complex
        rT = real(T)
        res_size = [size(x)[1:(end - 1)]..., d]
        fft_length = [res_size[i] for i in (ndims(x) - last_n + 1):ndims(x)]
    end

    @assert 1 ≤ length(fft_length) ≤ 3 "stablehlo.fft only supports up to rank 3"
    mlir_type = MLIR.IR.TensorType(res_size, Reactant.MLIR.IR.Type(rT))
    op = MLIR.Dialects.stablehlo.fft(x.mlir_data; fft_type, fft_length, result_0=mlir_type)
    x = TracedRArray{rT,N}((), MLIR.IR.result(op, 1), Tuple(res_size))
    return permutedims(x, reverse(1:N))
end

end
