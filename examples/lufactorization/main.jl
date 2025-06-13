# LU Factorization implementated in Julia
using Reactant, LinearAlgebra
using Reactant: Ops

function lu_unblocked(x::AbstractMatrix{T}) where {T}
    m, n = size(x)
    idx_len = min(m, n)

    inf_const = similar(x, size(x, 1))
    fill!(inf_const, -Inf)

    m_idx = Ops.iota(Int, [m]; iota_dimension=1)
    n_idx = Ops.iota(Int, [n]; iota_dimension=1)

    pivot = similar(x, Int, idx_len)
    fill!(pivot, 1)

    perm = Ops.iota(Int, [m]; iota_dimension=1) .+ 1

    @trace for k in 1:idx_len
        # written in this way to avoid create dynamically sized tensors
        magnitude = abs.(x[:, k])
        magnitude = ifelse.(m_idx .≥ k, magnitude, inf_const)

        i = argmax(magnitude)
        @allowscalar pivot[k] = i
        x[[k, i], :] = x[[i, k], :]
        perm[[i, k]] = perm[[k, i]]

        den = @allowscalar x[k, k]
        x[:, k] = ifelse.((m_idx .> k) .& (den != 0), x[:, k] ./ den, x[:, k])

        x_outer = x[:, k] * x[k, :]'
        mask = (m_idx .> k) .& (n_idx' .> k)
        x .-= mask .* x_outer
    end

    return x, perm, pivot
end

function lu_blocked(a::AbstractMatrix{T}, block_size=128) where {T}
    m, n = size(a)
    r = min(m, n)

    pivot = similar(a, Int, r)
    perm = Ops.iota(Int, [m]; iota_dimension=1) .+ 1

    for k in 1:block_size:r
        b = min(r - k + 1, block_size) - 1
        lu_block, block_perm, block_pivot = lu_unblocked(a[k:end, k:(k + b)])

        pivot[k:(k + b)] = block_pivot .+ k
        perm[k:m] = perm[block_perm .+ k]

        a[k:end, :] = a[block_perm .+ k, :]
        a[k:end, k:(k + b)] = lu_block

        if k + b < n
            solve_res = Ops.triangular_solve(
                a[k:(k + b), k:(k + b)],
                a[k:(k + b), (k + b + 1):n];
                left_side=true,
                lower=true,
                unit_diagonal=true,
                transpose_a='N',
            )
            a[k:(k + b), (k + b + 1):n] = solve_res
            a[(k + b + 1):m, (k + b + 1):n] .-= a[(k + b + 1):m, k:(k + b)] * solve_res
        end
    end

    return a, pivot, perm
end

x_ra = Reactant.to_rarray(randn(1024, 1024))

@code_hlo lu_blocked(x_ra)

fn_comp = @compile sync = true lu_blocked(x_ra)

@time fn_comp(x_ra)

fn_cusolver = @compile sync = true Ops.lu(x_ra)

@time fn_cusolver(x_ra)
