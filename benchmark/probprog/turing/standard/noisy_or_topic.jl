# Turing.jl model spec for Noisy-Or Topic (Latent Keyphrase Inference / LAKI).
# Matches the upstream Stan noisy_or_topic.py and Impulse standard/noisy_or_topic.jl.
#
# A boolean noisy-OR Bayesian network: node 1 is the always-on leak, then
# `num_topics` topic nodes, then `num_words` word nodes. The inference target is
# the discrete topic on/off vector `active`. As a continuous PPL, Turing (like
# Stan and Impulse here) replaces the discrete model with a Gumbel-softmax
# continuous RELAXATION (temperature tau=0.1): each node is a 2-way softmax whose
# argmax recovers the boolean state. This is the APPROXIMATE relaxation posterior,
# identical to the Stan and Impulse formulation, NOT an exact marginalization of
# the discrete latents.
#
# Two Gumbel(0,1) latent vectors (`true` / `false`) of length 1+num_topics (the
# leak entry is sampled but unused, pinned active). The sequential noisy-OR
# recursion follows the Stan `transformed parameters` block exactly. Per topic j:
#   aw_j        = edge_weight[j, 1:(j-1)] . active_true[1:(j-1)]
#   active_true[j] = softmax([log1m_exp(-aw_j)+g_true[j], -aw_j+g_false[j]] / tau)[1]
#                  = sigmoid(((log1m_exp(-aw_j)+aw_j) + (g_true[j]-g_false[j])) / tau).
# Word logits reuse the solved vector:
#   S_weight = edge_weight[words, :] . active_true
#   S_logit  = log1m_exp(-S_weight) + S_weight;  S ~ Bernoulli(logits = S_logit).

using Turing, Distributions, LinearAlgebra

# Inline helpers (StatsFuns / LogExpFunctions are only transitive deps of
# turing_env; defining these avoids adding packages).
_log1mexpm(x) = log(-expm1(-x))          # log(1 - exp(-x)) for x >= 0
_logistic(z) = inv(one(z) + exp(-z))

# Raw data dict stashed by setup; extract_samples needs edge_weight for the
# argmax recovery (single process per run).
const _NOISY_DATA = Ref{Any}(nothing)

@model function turing_noisy_or_topic(edge_weight, S, P, num_words, tau)
    g_true ~ filldist(Gumbel(0.0, 1.0), P)
    g_false ~ filldist(Gumbel(0.0, 1.0), P)

    T = eltype(g_true)
    active = Vector{T}(undef, P)          # relaxed active_true
    active[1] = one(T)                    # leak pinned to 1
    for j in 2:P
        aw = zero(T)
        @inbounds for l in 1:(j - 1)      # only parents l < j have nonzero weight
            aw += edge_weight[j, l] * active[l]
        end
        # log1m_exp(-aw) = log(1 - exp(-aw)); aw > 0 (leak weight > 0) so it is finite.
        lt = (_log1mexpm(aw) + g_true[j]) / tau
        lf = (-aw + g_false[j]) / tau
        active[j] = _logistic(lt - lf)    # softmax "true" probability
    end

    word_edges = edge_weight[(P + 1):(P + num_words), :]  # (num_words, P)
    S_weight = word_edges * active                        # (num_words,)
    S_logit = _log1mexpm.(S_weight) .+ S_weight
    Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(S_logit), S))
end

function _read_noisy_or(data)
    attrs = data["attrs"]
    num_topics = Int(attrs["num_topics"])
    num_words = Int(attrs["num_words"])
    ew_raw = attrs["edge_weight"]                        # (1+T+W) rows x (1+T)
    P = 1 + num_topics
    nrows = P + num_words
    edge_weight = zeros(Float64, nrows, P)
    for r in 1:nrows, c in 1:P
        edge_weight[r, c] = Float64(ew_raw[r][c])
    end
    return num_topics, num_words, edge_weight
end

function setup(data)
    _NOISY_DATA[] = data
    num_topics, num_words, edge_weight = _read_noisy_or(data)
    tau = 0.1
    P = 1 + num_topics

    # Training sentence(s): n==2 with train_frac 0.5 -> one train sentence.
    S = Float64.(data["S"][1])                           # length num_words

    model = turing_noisy_or_topic(edge_weight, S, P, num_words, tau)
    return (
        turing_model = model,
        model_name = "Noisy-Or Topic (Gumbel-softmax relaxation)",
    )
end

function extract_samples(chain, num_samples)
    num_topics, num_words, edge_weight = _read_noisy_or(_NOISY_DATA[])
    tau = 0.1
    P = 1 + num_topics

    total = length(chain)
    start = total - num_samples + 1
    g_true = hcat([Array(chain[Symbol("g_true[$j]")])[start:total] for j in 1:P]...)   # (ndraws, P)
    g_false = hcat([Array(chain[Symbol("g_false[$j]")])[start:total] for j in 1:P]...)
    ndraws = size(g_true, 1)

    # Recover boolean `active` by the sequential recursion (argmax over the 2-way
    # relaxation), matching the Stan/Impulse extracts. aw uses the soft relaxed
    # active_true; the hard decision lt > lf is tau-invariant.
    active = zeros(Int, ndraws, P)
    for d in 1:ndraws
        a1 = ones(Float64, P)                            # relaxed active_true
        active[d, 1] = 1                                 # leak
        for j in 2:P
            aw = 0.0
            for l in 1:(j - 1)
                aw += edge_weight[j, l] * a1[l]
            end
            lt = _log1mexpm(aw) + g_true[d, j]
            lf = -aw + g_false[d, j]
            a1[j] = _logistic((lt - lf) / tau)
            active[d, j] = lt > lf ? 1 : 0
        end
    end

    return Dict{String,Any}(
        "active" => [collect(active[d, :]) for d in 1:ndraws],
    )
end

function get_init_params(data, init_params)
    # No generate_init_params upstream; NUTS inits from the Gumbel prior.
    return nothing
end

function matched_init_params(data)
    # Deterministic canonical init (matched-init control experiment). Constrained
    # values identical to the Impulse arm's NUTS start: both Gumbel latent vectors
    # at 0. Gumbel support is the whole real line (identity transform), so
    # constrained 0 == unconstrained 0. DynamicPPL.InitFromParams treats these as
    # constrained (original-space) values. Only used with --matched-init.
    num_topics, _, _ = _read_noisy_or(data)
    P = 1 + num_topics
    return (g_true=zeros(P), g_false=zeros(P))
end
