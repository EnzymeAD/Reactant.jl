# Noisy-Or Topic model (Latent Keyphrase Inference / LAKI).
#
# A boolean noisy-OR Bayesian network: node 1 is the always-on leak, then
# `num_topics` topic nodes, then `num_words` word nodes. The inference target is
# the discrete topic on/off vector `active`. Stan (the only continuous-PPL
# reference) cannot sample the discrete nodes, so it replaces the discrete model
# with a Gumbel-softmax continuous RELAXATION (temperature tau=0.1): each topic
# is a 2-way softmax whose argmax recovers the boolean state. This Impulse model
# implements the SAME relaxation, so its posterior is the APPROXIMATE relaxation
# posterior (softmax ~ argmax), exactly as the upstream Stan baseline is; it is
# NOT an exact marginalization of the discrete latents.
#
# active_gumbel[:, 1] ("true") and active_gumbel[:, 2] ("false") are the two
# Gumbel(0, 1) latent vectors (length 1+num_topics). Per topic j:
#   active_weight_j = edge_weight[j, :] . active_true   (parents l >= j have zero
#                     edge weight and active_true[l] = 0, so only l < j count)
#   logit_true  = log1m_exp(-active_weight_j) + gumbel_true[j]
#   logit_false = -active_weight_j            + gumbel_false[j]
#   active_true[j] = softmax([logit_true, logit_false] / tau)[1]
#                  = sigmoid((h(aw_j) + gdiff[j]) / tau),
# with h(aw) = log1m_exp(-aw) + aw and gdiff = gumbel_true - gumbel_false.
#
# The active_true vector is solved as a WHOLE-VECTOR fixed point rather than a
# per-topic slicing loop. The topic dependency edge_weight[1:P, 1:P] is strictly
# lower triangular (parents l < j), so iterating
#   aw = L * active;  active = sigmoid((h(aw) + gdiff) / tau)   (leak pinned to 1)
# P times resolves one triangular level per step and lands EXACTLY on the
# sequential solution. This is not an approximation of the relaxation; it is the
# same relaxation, evaluated without per-element Gumbel indexing. Per-element
# slicing (g_true[j]) is avoided on purpose: its reverse-AD scatters each
# element's gradient into the position vector as a `pad`, which trips a buggy
# `add(pad, y)` HLO rewrite; the fixed point keeps every Gumbel use a whole-
# vector op whose adjoint is a clean matmul/elementwise (the same shape the
# working single/two-site NUTS models use). Word logits reuse the solved vector:
#   S_weight = edge_weight[words, :] . active_true
#   S_logit  = log1m_exp(-S_weight) + S_weight;  S ~ Bernoulli(logits = S_logit)
# Every node has the always-on leak as a parent (edge_weight[j, 1] > 0), so
# aw_j > 0 for real topics; only the leak row (aw_1 = 0, where h is -inf) is
# pinned, so no non-finite value enters the differentiated path.

# log(1 - exp(-x)) for x >= 0, stable branch-free enough for tracing.
_log1mexpm(x) = log(-expm1(-x))

# extract_samples receives only the trace, but the argmax recovery needs
# edge_weight; setup stashes the raw data dict here (single process per run).
const _NOISY_OR_DATA = Ref{Any}(nothing)

function noisy_or_topic(rng, edge_weight, num_topics, num_words, tau)
    P = 1 + num_topics                       # leak + topics
    _, g_true = ProbProg.sample(
        rng, ProbProg.Gumbel(0.0, 1.0, (P,)); symbol=:active_gumbel_true
    )
    _, g_false = ProbProg.sample(
        rng, ProbProg.Gumbel(0.0, 1.0, (P,)); symbol=:active_gumbel_false
    )

    gdiff = g_true .- g_false                            # (P,) whole-vector
    L = edge_weight[1:P, 1:P]                            # (P,P) topic-parent block
    is_leak = Reactant.Ops.constant(collect(1:P)) .== 1  # (P,) leak mask
    active = Reactant.Ops.constant(vcat(1.0, zeros(P - 1)))  # (P,), leak = 1
    for _ in 1:P                                         # P steps: exact fixed point
        aw_raw = L * active                             # (P,) matmul
        aw = ifelse.(is_leak, 1.0, aw_raw)              # avoid -inf at leak (aw_1=0)
        an = 1.0 ./ (1.0 .+ exp.(-((_log1mexpm.(aw) .+ gdiff) ./ tau)))
        active = ifelse.(is_leak, 1.0, an)              # pin leak active to 1
    end

    word_edges = edge_weight[(P + 1):(P + num_words), :]  # (num_words, P)
    S_weight = word_edges * active                       # (num_words,)
    S_logit = _log1mexpm.(S_weight) .+ S_weight
    _, S = ProbProg.sample(
        rng, ProbProg.Bernoulli(S_logit, (num_words,)); symbol=:S
    )
    return S
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
    _NOISY_OR_DATA[] = data
    num_topics, num_words, edge_weight = _read_noisy_or(data)
    tau = 0.1
    P = 1 + num_topics

    edge_r = Reactant.to_rarray(edge_weight)
    model_args = (edge_r, num_topics, num_words, tau)

    selection = ProbProg.select(
        ProbProg.Address(:active_gumbel_true),
        ProbProg.Address(:active_gumbel_false),
    )

    return (
        model_fn = noisy_or_topic,
        model_args = model_args,
        selection = selection,
        position_size = 2 * P,
        inverse_mass_matrix = ones(Float64, 2 * P),
        model_name = "Noisy-Or Topic (Gumbel-softmax relaxation)",
    )
end

function build_constraint(data, init_params)
    num_topics, num_words, _ = _read_noisy_or(data)
    P = 1 + num_topics

    # Training sentence(s): n==2 with train_frac 0.5 -> one train sentence.
    S_raw = data["S"]
    S = Float64.(S_raw[1])                                # length num_words

    # No upstream generate_init_params: start the Gumbel latents at 0.
    init_true = zeros(Float64, P)
    init_false = zeros(Float64, P)

    return ProbProg.Constraint(
        :active_gumbel_true => init_true,
        :active_gumbel_false => init_false,
        :S => S,
    )
end

function extract_samples(trace)
    _, _, edge_weight = _read_noisy_or(_NOISY_OR_DATA[])
    tau = 0.1

    g_true = Array(trace.choices[:active_gumbel_true])   # (draws, P)
    g_false = Array(trace.choices[:active_gumbel_false])
    ndraws = size(g_true, 1)
    P = size(g_true, 2)

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
            a1[j] = 1.0 / (1.0 + exp(-(lt - lf) / tau))
            active[d, j] = lt > lf ? 1 : 0
        end
    end

    return Dict{String,Any}(
        "active" => [collect(active[d, :]) for d in 1:ndraws],
    )
end
