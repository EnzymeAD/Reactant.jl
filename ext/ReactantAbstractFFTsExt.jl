module ReactantAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using Reactant: Reactant, MLIR, Ops, AnyTracedRArray, TracedRArray, TracedUtils
using Reactant.Ops: @opcall

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

    @eval function AbstractFFTs.$(Symbol(op, "!"))(x::AnyTracedRArray, dims)
        y = AbstractFFTs.$(op)(x, dims)
        copyto!(x, y)
        return x
    end

    # Out-of-place plan
    plan_name = Symbol("Reactant", uppercase(string(op)), "Plan")
    plan_f = Symbol("plan_", op)
    @eval struct $(plan_name){T, D} <: AbstractFFTs.Plan{T} 
        dims::D
    end
    @eval AbstractFFTs.$(plan_f)(x::Reactant.TracedRArray{T}, dims=1:ndims(x)) where {T} = $(plan_name){T}(dims)
    @eval Base.:*(p::$(plan_name){T}, x::Reactant.TracedRArray{T}) where {T} =
        AbstractFFTs.$(op)(x, p.region)

    # In-place plan
    if op !== :rfft
        plan_name! = Symbol("Reactant", uppercase(string(op)), "InPlacePlan")
        plan_f! = Symbol("plan_", op, "!")
        @eval struct $(plan_name!){T, D} <: AbstractFFTs.Plan{T} 
            dims::D
        end
        @eval AbstractFFTs.$(plan_f!)(::Reactant.TracedRArray{T}, dims=1:ndims(x)) where {T} =
            $(plan_name!){T}(dims)
        @eval Base.:*(p::$(plan_name!){T}, x::Reactant.TracedRArray{T}) where {T} =
            copyto!(x, AbstractFFTs.$(op)(x, p.dims))
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

    @eval function AbstractFFTs.$(Symbol(op, "!"))(x::AnyTracedRArray, d::Integer, dims)
        y = AbstractFFTs.$(op)(x, d, dims)
        copyto!(x, y)
        return x
    end

end

end
