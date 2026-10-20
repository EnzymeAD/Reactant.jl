# Leroux CAR with rho ESTIMATED, under CARBayes S.CARleroux's default priors
# -- SICM: EigenLift
#
# Same model and spelling as sicm/car_leroux.jl (which the NYC row still uses);
# only the priors differ, and the variance site is tau2 itself, as in CARBayes.
#
# Model (Leroux, Lei and Breslow 2000), as CARBayes 6.1.1 fits it with
# S.CARleroux(y ~ offset(log(E)), family = "poisson", W = W) and every prior left
# at its default:
#   y_i   ~ Poisson(E_i * exp(alpha + phi_i))
#   phi   ~ MVN(0, P^{-1}),  P = Q(rho) / tau2,  Q(rho) = rho * (D - W) + (1 - rho) * I
#   D     = diag(rowsum W), W the 0/1 adjacency of the areal units (DATA)
#   alpha ~ Normal(0, variance 1e5)        prior.mean.beta = 0, prior.var.beta = 1e5
#   tau2  ~ InverseGamma(1, 0.01)          prior.tau2 = c(1, 0.01) (shape, scale)
#   rho   ~ Uniform(0, 1)                  rho = NULL (estimated)
# Sources: CARBayes 6.1.1 S.CARleroux.Rd (prior.mean.beta, prior.var.beta,
# prior.tau2), the CARBayes vignette (rho ~ Uniform(0, 1)), and the code
# (poisson.lerouxCAR sets the defaults; poisson.lerouxCARMCMC's rho update has
# no prior term; CARBayes.cpp uses prior.var.beta as a variance). Evidence:
# oopsla/evidence/fig9b_priors_2026-10-07/provenance/.
#
# The density targeted here is the Leroux CAR density itself. CARBayes's
# sampler also recentres phi after every update (phi <- phi - mean(phi)) while
# its rho update keeps the full-rank determinant; that is a property of the
# CARBayes sampler, not of the model, and is not reproduced here.
#
# EigenLift shape (unchanged from sicm/car_leroux.jl):
#   P = s * R + t * I,   R = D - W (data),   s = rho / tau2,   t = (1 - rho) / tau2
# The term-by-term spelling below is what CholeskyEigenLift matches.

function _leroux_cb_precision(tau2, rho, R, n)
    t2 = reshape(tau2, 1, 1)
    r2d = reshape(rho, 1, 1)
    tau = 1.0 ./ t2
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    return (tau .* r2d) .* R .+ (tau .* (1.0 .- r2d)) .* I_n
end

# Exact draw from MVN(0, P^{-1}) (used by simulate only).
function _leroux_cb_phi_sampler(rng, tau2, rho, R, n)
    P = _leroux_cb_precision(tau2, rho, R, n)
    L = Reactant.Ops.cholesky(P; lower=true)
    z = randn(rng, n)
    return Reactant.Ops.triangular_solve(
        L, z; left_side=true, lower=true, transpose_a='T', unit_diagonal=false
    )
end

# log N(phi | 0, P^{-1}) = 0.5 log det P - 0.5 phi' P phi - (n/2) log(2 pi).
# Same spelling as sicm/car_leroux.jl (diag(L) by masking with I and
# contracting the row axis), which is the form CholeskyEigenLift recognises.
function _leroux_cb_phi_logpdf(phi, tau2, rho, R, n)
    P = _leroux_cb_precision(tau2, rho, R, n)
    phi_col = reshape(phi, n, 1)
    quad = sum(phi_col .* (P * phi_col))
    L = Reactant.Ops.cholesky(P; lower=true)
    I_n = Matrix{Float64}(LinearAlgebra.I, n, n)
    diagL = transpose(L .* I_n) * ones(n)
    half_logdet = sum(log.(diagL))
    return half_logdet - 0.5 * quad - 0.5 * n * log(2 * pi)
end

# rho ~ Uniform(0, 1). support=:unit_interval: HMC samples logit(rho) and adds
# the sigmoid Jacobian.
_cb_unit_uniform_sampler(rng) = rand(rng, 1)
_cb_unit_uniform_logpdf(rho) = sum(0.0 .* rho)

# tau2 ~ InverseGamma(a, b), a = 1, b = 0.01 (CARBayes: shape and scale,
# tau2 <- 1 / rgamma(shape, scale = 1 / scale_posterior)):
#   log p(tau2) = a log b - lgamma(a) - (a + 1) log tau2 - b / tau2,  lgamma(1) = 0.
# support=:positive: HMC samples log(tau2) and adds the exp Jacobian.
const _CB_IG_A = 1.0
const _CB_IG_B = 0.01
# Exact draw for a = 1: Gamma(1, 1) = Exponential(1) = -log(U), tau2 = b / Gamma.
_cb_tau2_sampler(rng) = _CB_IG_B ./ (-log.(rand(rng, 1)))
function _cb_tau2_logpdf(tau2)
    return sum(_CB_IG_A * log(_CB_IG_B) .- (_CB_IG_A + 1.0) .* log.(tau2) .- _CB_IG_B ./ tau2)
end

function model_car_leroux_carbayes(rng, R, log_E, y, n)
    _, alpha = ProbProg.sample(
        rng, ProbProg.Normal(0.0, sqrt(1.0e5), (1,)); symbol=:alpha
    )

    _, tau2 = ProbProg.sample(
        rng, _cb_tau2_sampler; symbol=:tau2, logpdf=_cb_tau2_logpdf, support=:positive
    )

    _, rho = ProbProg.sample(
        rng, _cb_unit_uniform_sampler;
        symbol=:rho, logpdf=_cb_unit_uniform_logpdf, support=:unit_interval,
    )

    _, phi = ProbProg.sample(
        rng, _leroux_cb_phi_sampler, tau2, rho, R, n;
        symbol=:phi, logpdf=_leroux_cb_phi_logpdf,
    )

    log_rate = log_E .+ reshape(alpha, 1) .+ phi
    _, obs = ProbProg.sample(rng, ProbProg.Poisson(exp.(log_rate), (n,)); symbol=:y)
    return obs
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    R_raw = data["R"]
    R = [Float64(R_raw[i][j]) for i in 1:n, j in 1:n]
    log_E = Float64.(data["log_E"])
    y = Float64.(data["Y"])
    model_args = (Reactant.to_rarray(R), Reactant.to_rarray(log_E), y, n)
    selection = ProbProg.select(
        ProbProg.Address(:alpha),
        ProbProg.Address(:tau2),
        ProbProg.Address(:rho),
        ProbProg.Address(:phi),
    )
    return (
        model_fn=model_car_leroux_carbayes,
        model_args=model_args,
        selection=selection,
        position_size=3 + n,  # alpha(1) + tau2(1) + rho(1) + phi(n)
        model_name="Leroux CAR, rho estimated, CARBayes default priors",
    )
end

# init_params may carry tau2, or sigma_phi (the data files built for
# sicm/car_leroux.jl), in which case tau2 = sigma_phi^2.
function build_constraint(data, init_params)
    n = Int(data["attrs"]["n"])
    y = Float64.(data["Y"])
    if init_params !== nothing
        init_alpha = Float64.(init_params["alpha"])
        init_tau2 = haskey(init_params, "tau2") ? Float64.(init_params["tau2"]) :
                    Float64.(init_params["sigma_phi"]) .^ 2
        init_rho = Float64.(init_params["rho"])
        init_phi = Float64.(init_params["phi"])
    else
        init_alpha = [0.0]
        init_tau2 = [1.0]
        init_rho = [0.5]
        init_phi = zeros(n)
    end
    return ProbProg.Constraint(
        :alpha => init_alpha,
        :tau2 => init_tau2,
        :rho => init_rho,
        :phi => init_phi,
        :y => y,
    )
end

function extract_samples(trace)
    phi_samples = trace.choices[:phi]
    return Dict{String,Any}(
        "alpha" => collect(vec(trace.choices[:alpha])),
        "tau2" => collect(vec(trace.choices[:tau2])),
        "rho" => collect(vec(trace.choices[:rho])),
        "phi" => [collect(phi_samples[i, :]) for i in 1:size(phi_samples, 1)],
    )
end
