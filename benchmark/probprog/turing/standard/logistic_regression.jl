# Turing.jl model spec for Bayesian Logistic Regression
# Matches NumPyro logistic_regression.py and Impulse standard/logistic_regression.jl

using Turing, Distributions, LinearAlgebra

const _log1pexp = Turing.StatsFuns.log1pexp

@model function turing_logistic_regression(X, Y, n, k, alpha_scale, beta_scale, beta_loc)
    alpha ~ Normal(0.0, alpha_scale)
    beta ~ filldist(Normal(beta_loc, beta_scale), k)
    mu = alpha .+ X * beta
    # Bernoulli-logit log likelihood sum_i logpdf(BernoulliLogit(mu_i), Y_i)
    # written out, sum_i y_i mu_i - log(1 + exp(mu_i)), as robust regression's
    # Student-t is: the same log density, and it differentiates under
    # ReverseDiff, where the broadcast logpdf.(BernoulliLogit.(mu), Y) returns
    # -Inf (ReverseDiff's broadcast passes Y as dual numbers, which fail
    # BernoulliLogit's x == 0 / x == 1 support branches).
    Turing.@addlogprob! dot(Y, mu) - sum(_log1pexp.(mu))
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    alpha_scale = Float64(attrs["alpha_scale"])
    beta_scale = Float64(attrs["beta_scale"])
    beta_loc = Float64(attrs["beta_loc"])

    X_raw = data["X"]
    X = zeros(Float64, n, k)
    for i in 1:n, j in 1:k
        X[i, j] = Float64(X_raw[i][j])
    end

    Y = Float64.(data["Y"])

    model = turing_logistic_regression(X, Y, n, k, alpha_scale, beta_scale, beta_loc)

    return (
        turing_model=model,
        model_name="Logistic Regression",
    )
end

function extract_samples(chain, num_samples)
    total = length(chain)
    start = total - num_samples + 1

    alpha = Array(chain[:alpha])[start:total]
    k = size(chain[Symbol("beta[1]")], 1) > 0 ? length([s for s in names(chain, :parameters) if startswith(string(s), "beta[")]) : 0
    beta = hcat([Array(chain[Symbol("beta[$i]")])[start:total] for i in 1:k]...)

    return Dict{String,Any}(
        "alpha" => collect(alpha),
        "beta" => [collect(beta[i, :]) for i in axes(beta, 1)],
    )
end

function get_init_params(data, init_params)
    if init_params === nothing
        return nothing
    end
    alpha = Float64(init_params["alpha"][1])
    beta = Float64.(init_params["beta"])
    # Return a NamedTuple matching Turing's variable names
    return (alpha=alpha, beta=beta)
end

function matched_init_params(data)
    # Deterministic canonical init (matched-init control experiment). Constrained
    # values identical to the Impulse arm's NUTS start: alpha and beta both at
    # MATCHED_INIT_REAL (both real support). DynamicPPL.InitFromParams treats
    # these as constrained (original-space) values. Only used with
    # --matched-init.
    #
    # NOT the origin. Turing's own Bernoulli is differentiable at 0; this arm
    # takes the offset purely to stay matched with the NumPyro arm, which is not
    # (see MATCHED_INIT_REAL in turing_harness.jl). All four arms of
    # logistic_regression_matched_init.json must pin the same literal.
    attrs = data["attrs"]
    k = Int(attrs["k"])
    return (alpha=MATCHED_INIT_REAL, beta=fill(MATCHED_INIT_REAL, k))
end
