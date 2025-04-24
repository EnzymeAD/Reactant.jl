# TODO: move the overload_autodiff here as well

# The default `onehot` will lead to scalar indexing
function Enzyme.onehot(x::TracedRArray{T,N}) where {T,N}
    # TODO: Ideally we do it as a scatter -> slice but we don't implement constant
    #       folding for scatter yet.
    results = Vector{TracedRArray{T,N}}(undef, length(x))
    pad_value = TracedUtils.promote_to(TracedRNumber{T}, 0)
    base_value = TracedUtils.broadcast_to_size(T(1), (1,))
    for i in eachindex(x)
        results[i] = Ops.reshape(
            Ops.pad(base_value, pad_value; low=Int64[i - 1], high=Int64[length(x) - i]),
            collect(Int64, size(x)),
        )
    end
    return Tuple(results)
end
