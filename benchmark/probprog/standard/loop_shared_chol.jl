# Loop model with shared parameter-dependent Cholesky.
#
# sigma ~ HalfNormal(1)
# Q = sigma^2 * Q_base + I    (loop-invariant, position-dependent)
# x_0 ~ Normal(0, 1)^d
# for t in 1:T:
#     x_t ~ MvNormal(x_{t-1}, Q)    # chol(Q) recomputed per iteration
#
# With LICM before AD: chol(Q) hoisted out of loop → O(d³) adjoint
# Without LICM: chol(Q) inside loop → O(T*d³) adjoint

using Reactant.ReactantCore: @trace
using Reactant: @allowscalar

# Fixed dimensions (compile-time constants)
const LOOP_D = 10
const LOOP_T = 20

function loop_shared_chol(rng, Q_base, y_data)
    @allowscalar begin
        # Shared scale parameter (position-dependent)
        _, sigma = ProbProg.sample(rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:sigma)

        # Build Q (depends on sigma, loop-invariant w.r.t. t)
        sigma_2d = reshape(sigma, 1, 1)
        I_d = Matrix{Float64}(LinearAlgebra.I, LOOP_D, LOOP_D)
        Q = (sigma_2d .* sigma_2d) .* Q_base .+ I_d

        # Initial state
        _, x_prev = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0, (LOOP_D,)); symbol=:x0)

        # Time series loop — each step samples from MvNormal(x_prev, Q)
        # The MvNormal logpdf computes chol(Q) EACH iteration
        @trace track_numbers=false for t in 1:LOOP_T
            _, x_t = ProbProg.sample(
                rng, ProbProg.MultiNormal(x_prev, Q, (LOOP_D,)); symbol=:x
            )
            # Observe
            _, _ = ProbProg.sample(
                rng, ProbProg.Normal(x_t, 0.1, (LOOP_D,)); symbol=:y
            )
            x_prev = x_t
        end

        return nothing
    end
end

function setup(data)
    attrs = data["attrs"]

    Q_base_raw = data["Q_base"]
    Q_base = [Float64(Q_base_raw[i][j]) for i in 1:LOOP_D, j in 1:LOOP_D]
    y_raw = data["Y"]
    y_data = [Float64(y_raw[t][i]) for t in 1:LOOP_T, i in 1:LOOP_D]

    Q_rarray = Reactant.to_rarray(Q_base)
    y_rarray = Reactant.to_rarray(y_data)
    model_args = (Q_rarray, y_rarray)

    selection = ProbProg.select(
        ProbProg.Address(:sigma),
        ProbProg.Address(:x0),
        ProbProg.Address(:x),
    )

    return (
        model_fn = loop_shared_chol,
        model_args = model_args,
        selection = selection,
        position_size = 1 + LOOP_D + LOOP_T * LOOP_D,
        model_name = "Loop Shared Chol",
    )
end

function build_constraint(data, init_params)
    y_raw = data["Y"]
    y_data = [Float64(y_raw[t][i]) for t in 1:LOOP_T, i in 1:LOOP_D]

    return ProbProg.Constraint(
        :sigma => [1.0],
        :x0 => zeros(LOOP_D),
        :x => zeros(LOOP_T, LOOP_D),
        :y => y_data,
    )
end

function extract_samples(trace)
    return Dict{String,Any}(
        "sigma" => collect(vec(trace.choices[:sigma])),
    )
end
