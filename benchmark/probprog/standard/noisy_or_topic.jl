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
#   active_weight_j = edge_weight[j, :j] . active_true[:j]   (parents l >= j have
#                     zero edge weight, so only l < j contribute)
#   logit_true  = log1m_exp(-active_weight_j) + gumbel_true[j]
#   logit_false = -active_weight_j            + gumbel_false[j]
#   active_true[j] = softmax([logit_true, logit_false] / tau)[1]
#                  = sigmoid((h(aw_j) + gdiff[j]) / tau),
# with h(aw) = log1m_exp(-aw) + aw and gdiff = gumbel_true - gumbel_false.
#
# The recursion is evaluated in the SAME sequential form as the NumPyro
# reference (pplbench/pplbench/ppls/numpyro/noisy_or_topic.py) and the upstream
# Stan `transformed parameters` block: a sweep over the topological order in
# which topic j contracts only against the j-1 entries already solved. num_topics
# is static, so the Julia loop traces into a chain of GROWING dots
#   aw_j = edge_weight[j, 1:(j-1)] . active[1:(j-1)]
# totalling sum_j (j-1) multiply-adds, with a reverse-mode adjoint of the same
# triangular shape. The per-topic Gumbel difference is read as a 1-element slice
# gdiff[j:j]; scalar indexing is not traceable, but the 1-element slice and the
# 2-D mixed slice edge_weight[j:j, 1:(j-1)] both trace cleanly.
# The match to NumPyro is STRUCTURAL: the same growing-dot sweep over the same
# topological order. There is ONE deliberate deviation, disclosed here. The
# reference indexes the two Gumbel vectors separately (g_true[j] and g_false[j]);
# this model instead forms gdiff = g_true .- g_false once, as a whole-vector op,
# and takes a single per-element slice gdiff[j:j]. The two are mathematically
# identical. The single-slice spelling is the one that reproduces the NumPyro
# arm pointwise (max |latent diff| ~1e-12 over the validate protocol, `active`
# max_diff exactly 0); the two-slice spelling diverges to O(1) by the first
# drawn sample. The forward log joint is bitwise identical between the two
# spellings at all 25 probe points tested, so the divergence lives in the
# reverse-mode gradient taken across two distinct sample sites, and is under
# separate investigation. Do not switch this loop to two per-element slices
# until that is resolved.
# Word logits reuse the solved vector:
#   S_weight = edge_weight[words, :] . active_true
#   S_logit  = log1m_exp(-S_weight) + S_weight;  S ~ Bernoulli(logits = S_logit)
# Every node has the always-on leak as a parent (edge_weight[j, 1] > 0), so
# aw_j > 0 for every real topic and log1m_exp(-aw_j) stays finite; the leak
# entry is pinned to 1 and is never fed through the transcendentals.

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

    # Julia j == NumPyro's python j + 1; on entry to iteration j the vector
    # `active` holds the j-1 already-solved entries, so edge_weight[j:j, 1:(j-1)]
    # is exactly NumPyro's edge_weight[j, :j].
    gdiff = g_true .- g_false                            # (P,) whole-vector
    active = Reactant.Ops.constant([1.0])                # (1,) leak, pinned to 1
    for j in 2:P
        row = edge_weight[j:j, 1:(j - 1)]                # (1, j-1) parent weights
        aw = row * active                                # (1,) growing dot
        an = 1.0 ./ (1.0 .+ exp.(-((_log1mexpm.(aw) .+ aw .+ gdiff[j:j]) ./ tau)))
        active = vcat(active, an)
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
        # Dense mass-adaptation protocol: fall through to the harness dense-eye
        # default so Impulse adapts a dense Welford mass matching NumPyro/Stan/Turing.
        # inverse_mass_matrix = ones(Float64, 2 * P),
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
