module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using Reactant: Reactant, MLIR, TracedRArray

function contiguous_dims(dims, N)
    all(i -> dims[i] == dims[i - 1] + 1, 2:(length(dims))) || return false
    dims[1] != (N - length(dims) + 1) && return false
    return true
end

for op in (:rfft, :fft)
    contiguous_op = Symbol(:contiguous_, op)
    @eval function AbstractFFTs.$(op)(x::TracedRArray, dims)
        if dims isa Integer
            if dims != ndims(x)
                error("Not yet implemented")
            end
            return $(contiguous_op)(x, nothing, ndims(x) - dims + 1)
        end
        contiguous_dims(dims, ndims(x)) &&
            return $(contiguous_op)(x, nothing, ndims(x) - first(dims) + 1)
        return error("Not yet implemented")
    end
end

for op in (:irfft, :ifft)
    contiguous_op = Symbol(:contiguous_, op)
    @eval function AbstractFFTs.$(op)(x::TracedRArray, d::Int, dims)
        if dims isa Integer
            if dims != ndims(x)
                error("Not yet implemented")
            end
            return $(contiguous_op)(x, d, ndims(x) - dims + 1)
        end
        contiguous_dims(dims, ndims(x)) &&
            return $(contiguous_op)(x, d, ndims(x) - first(dims) + 1)
        return error("Not yet implemented")
    end
end

for op in (:rfft, :irfft, :fft, :ifft)
    fn_name = Symbol(:contiguous_, op)
    fft_type_str = uppercase(string(op))
    @eval function $(fn_name)(x::TracedRArray{T,N}, d, last_n::Int) where {T,N}
        x = permutedims(x, reverse(1:N))
        fft_type = MLIR.API.stablehloFftTypeAttrGet(MLIR.IR.context(), $(fft_type_str))
        fft_length = [size(x, i) for i in (ndims(x) - last_n + 1):ndims(x)]
        @assert 1 ≤ length(fft_length) ≤ 3 "stablehlo.fft only supports up to rank 3"
        res = MLIR.IR.result(
            Reactant.MLIR.Dialects.stablehlo.fft(x.mlir_data; fft_type, fft_length), 1
        )
        mlir_type = MLIR.IR.type(res)
        x = TracedRArray{MLIR.IR.julia_type(eltype(mlir_type)),N}((), res, size(mlir_type))
        return permutedims(x, reverse(1:N))
    end
end

end
