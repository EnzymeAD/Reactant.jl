# TODO: move the overload_autodiff here as well

# The default `onehot` will lead to scalar indexing
function Enzyme.onehot(x::TracedRArray{T,N}) where {T,N}
    x_arr = zeros(T, size(x))
    return map(Base.Fix1(TracedUtils.promote_to, TracedRArray{T, N}), Enzyme.onehot(x_arr))
end
