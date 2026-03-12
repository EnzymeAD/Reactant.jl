# Scale-Family MVN (SICM showcase: O(N^3) -> O(1) per iteration)
#
# Model: y ~ MVN(0, tau^2 * R_data)
#   tau ~ HalfNormal(1)
#   R_data is a known N x N correlation matrix (sample-invariant)
#
# SICM cascade (3 patterns fire in fixpoint iteration):
#   1. CholeskyScaleFactorization: chol(tau^2 * R) -> tau * chol(R)
#   2. TriangularSolveScaleFactorization: tri_solve(tau*L, y) -> (1/tau)*tri_solve(L, y)
#   3. LogMultiplyDistribution: log(diag(tau*L)) -> log(tau) + log(diag(L))
# All matrix ops become sample-invariant -> hoisted -> O(1) per iteration.

function model_scale_family_mvn(rng, R_data, y)
    n = Base.length(y)

    _, tau = ProbProg.sample(
        rng, ProbProg.HalfNormal(1.0, (1,)); symbol=:tau
    )

    tau_2d = reshape(tau, 1, 1)
    K = (tau_2d .* tau_2d) .* R_data

    _, obs = ProbProg.sample(
        rng, ProbProg.MultiNormal(zeros(n), K, (n,)); symbol=:y
    )
    return obs
end

function setup(data)
    n = Int(data["attrs"]["n"])
    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    y = Float64.(data["Y"])

    R_rarray = Reactant.to_rarray(R)
    model_args = (R_rarray, y)

    selection = ProbProg.select(ProbProg.Address(:tau))

    return (
        model_fn = model_scale_family_mvn,
        model_args = model_args,
        selection = selection,
        position_size = 1,
        model_name = "Scale-Family MVN",
    )
end

function build_constraint(data, init_params)
    y = Float64.(data["Y"])

    if init_params !== nothing
        init_tau = Float64.(init_params["tau"])
    else
        init_tau = [1.0]
    end

    return ProbProg.Constraint(
        :tau => init_tau,
        :y => y,
    )
end

function extract_samples(trace)
    tau_samples = vec(trace.choices[:tau])
    return Dict{String,Any}(
        "tau" => collect(tau_samples),
    )
end
