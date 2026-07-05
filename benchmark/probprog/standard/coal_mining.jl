# Coal-Mining Disasters change-point model. Two entry points:
#
#  (1) Marginalized NUTS (PRIMARY benchmark). The discrete switch `s` is
#      integrated out with the prefix/suffix cumulative-sum trick, so NUTS
#      runs on the two continuous log-rates (l1, l2) only. Matches the NumPyro
#      `numpyro.factor` + cumsum baseline. Entry: `coal_mining` + `setup`,
#      `build_constraint`, `extract_samples`.
#
#  (2) MH branchy demo (mechanism story, Impulse-only). `s` is explicit and the
#      per-year Poisson likelihood is observed INSIDE each `impulse.if` branch
#      (per-branch addresses `ye_t`/`yl_t`), so nothing sampled/returned by a
#      branch feeds a later sample op (C2 shape constraint). Entry:
#      `coal_mining_branch` + `coal_mining_mh_program`, driven by
#      `coal_mining_mh_demo.jl`.

# =============================================================================
# (1) Marginalized NUTS model
# =============================================================================

# Trivial placeholder sampler for the marginal observed site: the value is
# supplied by the constraint, the marginal enters through `_coal_obs_logpdf`.
function _coal_obs_sampler(rng, l1, l2, lg, MA, MB, T)
    return Reactant.Ops.fill(0.0, T)
end

# Marginal data log-likelihood logsumexp_s(-log(T+1) + A_s + B_s), where
#   a_t = poisson_logpmf(y_t | l1),  b_t = poisson_logpmf(y_t | l2)
#   A_s = sum_{t<s} a_t   (MA is the strictly-lower-triangular ones matrix)
#   B_s = sum_{t>=s} b_t  (MB is the upper-triangular ones matrix, diag incl.)
function _coal_obs_logpdf(y, l1, l2, lg, MA, MB, T)
    a = y .* log.(l1) .- l1 .- lg     # (T,)
    b = y .* log.(l2) .- l2 .- lg     # (T,)
    A = MA * a                        # (T+1,)
    B = MB * b                        # (T+1,)
    lp = A .+ B .- log(T + 1.0)       # (T+1,)
    m = maximum(lp)
    return m + log(sum(exp.(lp .- m)))
end

function coal_mining(rng, lg, MA, MB, T)
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    _, _ = ProbProg.sample(
        rng, _coal_obs_sampler, l1, l2, lg, MA, MB, T;
        symbol=:y, logpdf=_coal_obs_logpdf,
    )
    return nothing
end

_logfactorial(k::Int) = k <= 1 ? 0.0 : sum(log, 2.0:1.0:Float64(k))

function _read_counts(data)
    y_raw = data["y"]
    T = length(y_raw)
    y = zeros(Float64, T)
    for t in 1:T
        y[t] = Float64(y_raw[t])
    end
    return y, T
end

function setup(data)
    y, T = _read_counts(data)

    lg = [_logfactorial(Int(round(y[t]))) for t in 1:T]

    # Cumulative-sum matrices (constant): A_s = MA*a, B_s = MB*b, s = 0..T.
    MA = zeros(Float64, T + 1, T)
    MB = zeros(Float64, T + 1, T)
    for r in 1:(T + 1), c in 1:T
        MA[r, c] = c < r ? 1.0 : 0.0    # t < s
        MB[r, c] = c >= r ? 1.0 : 0.0   # t >= s
    end

    lg_r = Reactant.to_rarray(lg)
    MA_r = Reactant.to_rarray(MA)
    MB_r = Reactant.to_rarray(MB)

    model_args = (lg_r, MA_r, MB_r, T)
    selection = ProbProg.select(ProbProg.Address(:l1), ProbProg.Address(:l2))

    return (
        model_fn = coal_mining,
        model_args = model_args,
        selection = selection,
        position_size = 2,
        model_name = "Coal-Mining Change-Point",
    )
end

function build_constraint(data, init_params)
    y, _ = _read_counts(data)

    if init_params !== nothing
        init_l1 = Float64.(init_params["l1"])
        init_l2 = Float64.(init_params["l2"])
    else
        init_l1 = [1.0]
        init_l2 = [1.0]
    end

    return ProbProg.Constraint(
        :l1 => init_l1,
        :l2 => init_l2,
        :y => y,
    )
end

function extract_samples(trace)
    l1 = vec(trace.choices[:l1])
    l2 = vec(trace.choices[:l2])
    return Dict{String,Any}(
        "l1" => collect(l1),
        "l2" => collect(l2),
    )
end

# =============================================================================
# (2) MH branchy demo model
# =============================================================================

# Cheap observed Poisson site: placeholder sampler, Poisson logpmf.
_pois_obs_sampler(rng, rate) = Reactant.Ops.fill(0.0, 1)
function _pois_obs_logpdf(y, rate)
    return sum(y .* log.(rate) .- rate .- Reactant.Ops.lgamma(y .+ 1.0))
end

# s ~ DiscreteUniform{0..T}; l1,l2 ~ Gamma(3,1); for each year the Poisson
# observation lives inside the taken `impulse.if` branch (per-branch address).
function coal_mining_branch(rng, T)
    _, s = ProbProg.sample(rng, ProbProg.DiscreteUniform(0.0, Float64(T)); symbol=:s)
    _, l1 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l1)
    _, l2 = ProbProg.sample(rng, ProbProg.Gamma(3.0, 1.0, (1,)); symbol=:l2)
    for t in 0:(T - 1)
        cond = sum(s) > Float64(t)   # t < s
        ProbProg.if_(
            rng,
            cond,
            () -> ProbProg.sample(
                rng, _pois_obs_sampler, l1;
                symbol=Symbol("ye_", t), logpdf=_pois_obs_logpdf,
            )[2],
            () -> ProbProg.sample(
                rng, _pois_obs_sampler, l2;
                symbol=Symbol("yl_", t), logpdf=_pois_obs_logpdf,
            )[2],
        )
    end
    return nothing
end

# Constraint over the 2T observed per-branch sites (both branch copies of a
# year carry the same observed count so either branch can be taken) plus the
# initial rate values so MH starts from a sensible point.
function branch_constraint(y::Vector{Float64}; l1_init=[3.0], l2_init=[0.9])
    T = length(y)
    pairs = Pair{Symbol,Any}[:l1 => l1_init, :l2 => l2_init]
    for t in 0:(T - 1)
        push!(pairs, Symbol("ye_", t) => [y[t + 1]])
        push!(pairs, Symbol("yl_", t) => [y[t + 1]])
    end
    return ProbProg.Constraint(pairs...)
end

# Alternating single-address MH (s, then l1, then l2); accumulates the
# posterior histogram over s across `num_iters` sweeps. Column 1 of the trace
# is `s` (first sample in walk order). `model` closes over the constant `T`
# (a Julia Int loop bound, so it must not be threaded as a traced value).
function coal_mining_mh_program(rng, model, bins, obs_tensor, constrained, num_iters)
    trace, weight, _ = ProbProg.generate(
        rng, obs_tensor, model; constrained_addresses=constrained
    )
    hist = zero.(bins)
    @trace for _ in 1:num_iters
        trace, weight, _ = ProbProg.mh(
            rng, trace, weight, model; selection=ProbProg.select(ProbProg.Address(:s))
        )
        trace, weight, _ = ProbProg.mh(
            rng, trace, weight, model; selection=ProbProg.select(ProbProg.Address(:l1))
        )
        trace, weight, _ = ProbProg.mh(
            rng, trace, weight, model; selection=ProbProg.select(ProbProg.Address(:l2))
        )
        s_v = sum(trace[1:1, 1:1])
        hist = hist .+ ifelse.(abs.(bins .- s_v) .< 0.5, 1.0, 0.0)
    end
    return hist
end
