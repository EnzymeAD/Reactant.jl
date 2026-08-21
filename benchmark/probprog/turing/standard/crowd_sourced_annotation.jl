# Turing.jl model spec for Crowd-Sourced Annotation (Dawid-Skene, marginalized).
# Matches NumPyro crowd_sourced_annotation.py, Stan crowd_sourced_annotation.py
# and Impulse standard/crowd_sourced_annotation.jl.
#
# Continuous latents: prev (K-simplex prevalence) and confusion_matrix (k*K
# independent K-simplices). The discrete per-item true label is marginalized
# analytically with a logsumexp over categories, so all four frameworks target
# the IDENTICAL continuous posterior under NUTS:
#   per item i,  logsumexp_c ( sum_j log conf[labeler_ij, c, label_ij] + log prev[c] ).
#
# The k*K confusion simplices are sampled as one matrix-variate `conf` via
# arraydist: column m = (c-1)*k + l is the observed-label simplex for labeler l
# under true category c, i.e. conf[o, (c-1)*k+l] = P(obs=o | labeler l, true c),
# with prior Dirichlet(alpha[c,:]). The per-item marginalization is vectorized
# with a precomputed column-major gather index into vec(conf).

using Turing, Distributions, LinearAlgebra

# (n, k, K, L) stashed by setup for extract_samples (single process per run).
const _CROWD_DIMS = Ref{NTuple{4,Int}}((0, 0, 0, 0))

@model function turing_crowd_sourced_annotation(conf_dists, gather_idx, beta, n, K, L)
    prev ~ Dirichlet(beta)                # (K,)
    conf ~ arraydist(conf_dists)          # (K_obs, k*K); col (c-1)*k+l ~ Dir(alpha[c])

    logconf = vec(log.(conf))             # (K_obs*k*K,) column-major
    g = reshape(logconf[gather_idx], n, L, K)         # gather log conf[lab_ij, c-col]
    per_item = reshape(sum(g; dims=2), n, K)          # (n, K) sum over labelers
    log_prob = per_item .+ reshape(log.(prev), 1, K)  # (n, K)
    m = maximum(log_prob; dims=2)                     # (n, 1)
    lse = m .+ log.(sum(exp.(log_prob .- m); dims=2)) # (n, 1) stable logsumexp
    Turing.@addlogprob! sum(lse)
end

function setup(data)
    attrs = data["attrs"]
    n = Int(attrs["n"])
    k = Int(attrs["k"])
    K = Int(attrs["num_categories"])
    L = Int(attrs["num_labels_per_item"])
    ec = Float64(attrs["expected_correctness"])
    conc = Float64(attrs["concentration"])

    # transformed data: Dirichlet concentrations (Stan `alpha`, `beta`).
    beta = fill(1.0 / K, K)
    off = (1.0 - ec) / (K - 1)
    alpha = fill(off, K, K)
    for c in 1:K
        alpha[c, c] = ec
    end
    alpha .*= conc
    # conf_dists[m], m = (c-1)*k + l  (category-major over labelers l=1..k).
    conf_dists = [Dirichlet(alpha[c, :]) for c in 1:K for l in 1:k]

    # labelers/labels are 0-based in the dataset (item, item_label).
    labelers_raw = data["labelers"]
    labels_raw = data["labels"]
    who = zeros(Int, n, L)   # 1-based labeler index
    lab = zeros(Int, n, L)   # 1-based observed label
    for i in 1:n, j in 1:L
        who[i, j] = Int(labelers_raw[i][j]) + 1
        lab[i, j] = Int(labels_raw[i][j]) + 1
    end

    # Column-major linear index into vec(conf), conf :: (K_obs, k*K):
    #   lin(o, m) = o + (m-1)*K,  o = lab_ij,  m = (c-1)*k + who_ij
    gather = zeros(Int, n, L, K)
    for i in 1:n, j in 1:L, c in 1:K
        gather[i, j, c] = lab[i, j] + ((c - 1) * k + who[i, j] - 1) * K
    end

    _CROWD_DIMS[] = (n, k, K, L)
    model = turing_crowd_sourced_annotation(conf_dists, vec(gather), beta, n, K, L)
    return (
        turing_model = model,
        model_name = "Crowd-Sourced Annotation",
    )
end

function extract_samples(chain, num_samples)
    n, k, K, L = _CROWD_DIMS[]
    total = length(chain)
    start = total - num_samples + 1

    prev = hcat([Array(chain[Symbol("prev[$c]")])[start:total] for c in 1:K]...)  # (ndraws, K)
    ndraws = size(prev, 1)

    # MCMCChains names the matrix variable `conf[o, m]` (note the space after the
    # comma), m = (c-1)*k + l. Reassemble confusion[draw, labeler l, true c, obs o].
    confarr = zeros(Float64, ndraws, k, K, K)
    for l in 1:k, c in 1:K, o in 1:K
        col = (c - 1) * k + l
        confarr[:, l, c, o] = Array(chain[Symbol("conf[$o, $col]")])[start:total]
    end

    return Dict{String,Any}(
        "prev" => [collect(prev[d, :]) for d in 1:ndraws],
        "confusion_matrix" => [
            [[collect(confarr[d, l, c, :]) for c in 1:K] for l in 1:k] for d in 1:ndraws
        ],
    )
end

function get_init_params(data, init_params)
    # No generate_init_params upstream for this model; NUTS inits from the prior.
    return nothing
end

function matched_init_params(data)
    # Deterministic uniform-simplex init (matched-init control experiment).
    # Constrained values identical to the Impulse arm's NUTS start: prev is the
    # uniform K-simplex, and every confusion column is the uniform K-simplex.
    # DynamicPPL.InitFromParams treats these as constrained (original-space)
    # values (typed_identity transform). `conf ~ arraydist(conf_dists)` is a
    # (K, k*K) matrix whose columns are the K-simplices. Only used with
    # --matched-init.
    attrs = data["attrs"]
    k = Int(attrs["k"])
    K = Int(attrs["num_categories"])
    return (
        prev = fill(1.0 / K, K),
        conf = fill(1.0 / K, K, k * K),
    )
end
