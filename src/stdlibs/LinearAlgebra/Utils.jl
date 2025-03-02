## The cartesian version doesn't exist in julia 1.10
function diagonal_indices_zero_indexed(m::Integer, n::Integer, k::Integer=0)
    idx1, idx2 = 1 + max(0, -k), 1 + max(0, k)
    L = max(0, k ≤ 0 ? min(m + k, n) : min(m, n - k))
    indices = Matrix{Int}(undef, (L, 2))
    for i in axes(indices, 1)
        indices[i, 1] = idx1 + i - 2
        indices[i, 2] = idx2 + i - 2
    end
    return indices
end
