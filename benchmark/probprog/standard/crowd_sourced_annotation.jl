# Crowd-Sourced Annotation (Dawid-Skene), ACL Q14-1025.
# Continuous latents: prev (K-simplex) and confusion_matrix (k*K independent
# K-simplices). The discrete per-item true label is marginalized analytically,
# matching the Stan reference: per item i,
#   logsumexp_c ( sum_j log conf[labeler_ij, c, label_ij] + log prev[c] ).
# The marginalization is vectorized via a precomputed gather index.

function _crowd_obs_sampler(rng, prev, conf, gather_idx, n, k, K, L)
    return Reactant.Ops.fill(0.0, n, L)
end

function _crowd_obs_logpdf(_labels, prev, conf, gather_idx, n, k, K, L)
    # conf :: (k,K,K); gather log conf[labeler_ij, c, label_ij] for all (i,j,c).
    logconf = reshape(log.(conf), k * K * K)          # column-major flat
    g = reshape(logconf[gather_idx], n, L, K)         # (n,L,K)
    per_item = reshape(sum(g; dims=2), n, K)          # (n,K)  sum over labelers
    log_prob = per_item .+ reshape(log.(prev), 1, K)  # (n,K)
    m = maximum(log_prob; dims=2)                     # (n,1)
    lse = m .+ log.(sum(exp.(log_prob .- m); dims=2)) # (n,1)  stable logsumexp
    return sum(lse)
end

function crowd_sourced_annotation(rng, beta, alpha_stacked, gather_idx, n, k, K, L)
    _, prev = ProbProg.sample(
        rng, ProbProg.Dirichlet(beta, (K,)); symbol=:prev
    )
    _, conf = ProbProg.sample(
        rng, ProbProg.Dirichlet(alpha_stacked, (k, K, K)); symbol=:confusion_matrix
    )
    _, _ = ProbProg.sample(
        rng, _crowd_obs_sampler, prev, conf, gather_idx, n, k, K, L;
        symbol=:labels, logpdf=_crowd_obs_logpdf,
    )
    return nothing
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    K = Int(attrs["num_categories"])
    L = Int(attrs["num_labels_per_item"])
    expected_correctness = Float64(attrs["expected_correctness"])
    concentration = Float64(attrs["concentration"])

    # transformed data: Dirichlet concentrations (Stan `alpha`, `beta`).
    beta = fill(1.0 / K, K)
    alpha = fill((1.0 - expected_correctness) / (K - 1), K, K)
    for c in 1:K
        alpha[c, c] = expected_correctness
    end
    alpha .*= concentration
    alpha_stacked = zeros(Float64, k, K, K)
    for l in 1:k
        alpha_stacked[l, :, :] = alpha
    end

    # labelers/labels are 0-based in the dataset (item, item_label).
    labelers_raw = data["labelers"]
    labels_raw = data["labels"]
    who = zeros(Int, n, L)   # 1-based labeler index
    lab = zeros(Int, n, L)   # 1-based observed label
    for i in 1:n, j in 1:L
        who[i, j] = Int(labelers_raw[i][j]) + 1
        lab[i, j] = Int(labels_raw[i][j]) + 1
    end

    # Column-major linear index into vec(conf), conf :: (k,K,K):
    #   idx(l,c,o) = l + (c-1)*k + (o-1)*k*K
    gather_idx = zeros(Int, n, L, K)
    for i in 1:n, j in 1:L, c in 1:K
        gather_idx[i, j, c] = who[i, j] + (c - 1) * k + (lab[i, j] - 1) * k * K
    end

    beta_r = Reactant.to_rarray(beta)
    alpha_r = Reactant.to_rarray(alpha_stacked)
    gather_r = Reactant.to_rarray(vec(gather_idx))

    model_args = (beta_r, alpha_r, gather_r, n, k, K, L)

    selection = ProbProg.select(
        ProbProg.Address(:prev), ProbProg.Address(:confusion_matrix)
    )

    # Unconstrained (stick-breaking) sizes: (K-1) + k*K*(K-1).
    position_size = (K - 1) + k * K * (K - 1)

    return (
        model_fn = crowd_sourced_annotation,
        model_args = model_args,
        selection = selection,
        position_size = position_size,
        inverse_mass_matrix = ones(Float64, position_size),
        model_name = "Crowd-Sourced Annotation",
    )
end

function build_constraint(data, init_params)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    K = Int(attrs["num_categories"])
    L = Int(attrs["num_labels_per_item"])

    labels_raw = data["labels"]
    labels = zeros(Float64, n, L)
    for i in 1:n, j in 1:L
        labels[i, j] = Float64(labels_raw[i][j])
    end

    # No generate_init_params upstream: init at uniform simplices.
    init_prev = fill(1.0 / K, K)
    init_conf = fill(1.0 / K, k, K, K)

    return ProbProg.Constraint(
        :prev => init_prev,
        :confusion_matrix => init_conf,
        :labels => labels,
    )
end

function extract_samples(trace)
    prev_samples = trace.choices[:prev]                # (draws, K)
    # Trace slices are flattened ROW-major (stablehlo.reshape) while
    # unflatten_trace reshapes column-major; the row-major flat index of a
    # (k,K,K) site is o + (c-1)*K + (l-1)*K^2, so the column-major view has
    # dims (draw, obs, true, labeler) -> permute to (draw, labeler, true, obs).
    conf_raw = Array(trace.choices[:confusion_matrix])
    ndraws = size(prev_samples, 1)
    k = size(conf_raw, 2)
    K = size(conf_raw, 3)
    conf_flat = reshape(conf_raw, ndraws, k * K * K)
    conf = permutedims(reshape(conf_flat, ndraws, K, K, k), (1, 4, 3, 2))
    return Dict{String,Any}(
        "prev" => [collect(prev_samples[i, :]) for i in 1:ndraws],
        "confusion_matrix" => [
            [[collect(conf[i, l, c, :]) for c in 1:K] for l in 1:k] for i in 1:ndraws
        ],
    )
end
