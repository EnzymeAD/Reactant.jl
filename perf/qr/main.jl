using Reactant
using LinearAlgebra
using BenchmarkTools
using PrettyTables
using Unitful
using Statistics

A = rand(Float64, 4, 4)
Q, R = LinearAlgebra.qr(A)

Are = Reactant.to_rarray(A)

init_tau(A) = zeros(eltype(A), min(size(A)...))
init_tau(A::Reactant.TracedRArray) = Reactant.Ops.fill(zero(eltype(A)), min(size(A)...))

function my_geqrf!(A)
    m, n = @allowscalar size(A)
    k = min(m, n)
    # tau = zeros(eltype(A), k)
    tau = init_tau(A)
    @show typeof(tau)

    # TODO we are loop unrolling here, which might not be ideal
    @allowscalar for i in 1:k
        # compute i-th Householder reflector
        v = A[i:end, i]
        alpha = v[1]
        sigma = norm(v[2:end])^2

        # TODO control-flow is giving problems so we suppose that sigma is never zero
        # if sigma == 0 && alpha >= 0
        #     tau[i] = 0.0
        # elseif sigma == 0 && alpha < 0
        #     tau[i] = 2.0
        # else
        beta = -sign(alpha) * sqrt(alpha^2 + sigma)
        tau[i] = (beta - alpha) / beta
        v[1] = 1.0
        # TODO broadcasting assignment not working: "ERROR: BoundsError: attempt to access 4-element Reactant.TracedRArray{Float64, 1} at index [2, 3, 4]"
        v[2:end] = v[2:end] ./ (alpha - beta)

        # Store Householder vector in A (Packed format)
        A[i:end, i] = v
        # end

        # apply reflector to remaining columns
        if i < n
            # TODO control-flow tracing is giving the following error "ERROR: UndefVarError: `�nonenon` not defined in local scope"
            # so we suppose that tau is never zero
            # @trace if tau[i] != 0.0
            v = A[i:end, i]
            for j in (i + 1):n
                w = dot(v, A[i:end, j])
                A[i:end, j] = A[i:end, j] - tau[i] * v * w
            end
            # end
        end

        A[i, i] = beta
    end

    return A, tau
end

AR, τ = my_geqrf!(copy(A))

@code_hlo optimize = false my_geqrf!(copy(Reactant.to_rarray(A)))
@code_hlo optimize = true my_geqrf!(copy(Reactant.to_rarray(A)))

ARre, τre = @jit my_geqrf!(copy(Reactant.to_rarray(A)))
