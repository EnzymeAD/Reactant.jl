# Regression: EnzymeHLOOpt BinopPadToConcat on add(pad, y) from AD of per-element slices.

using Reactant, Test, Random, Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

_log1mexpm(x) = log(-expm1(-x))

function _sliced_noisy_or(rng, edge_weight, num_topics, num_words, tau)
    P = 1 + num_topics
    _, g_true = ProbProg.sample(
        rng, ProbProg.Gumbel(0.0, 1.0, (P,)); symbol=:active_gumbel_true
    )
    _, g_false = ProbProg.sample(
        rng, ProbProg.Gumbel(0.0, 1.0, (P,)); symbol=:active_gumbel_false
    )
    active = g_true[1:1] .* 0.0 .+ 1.0
    for j in 2:P
        erow = Reactant.Ops.constant(reshape(edge_weight[j, 1:(j - 1)], (1, j - 1)))
        aw = sum(erow .* active)
        lt = _log1mexpm(aw) .+ g_true[j:j]
        lf = (-aw) .+ g_false[j:j]
        logits = vcat(lt, lf) ./ tau
        m = maximum(logits)
        w = exp.(logits .- m)
        p = w ./ sum(w)
        active = vcat(active, p[1:1])
    end
    word_edges = Reactant.Ops.constant(edge_weight[(P + 1):(P + num_words), :])
    S_weight = word_edges * active
    S_logit = _log1mexpm.(S_weight) .+ S_weight
    _, S = ProbProg.sample(rng, ProbProg.Bernoulli(S_logit, (num_words,)); symbol=:S)
    return S
end

function _ref_active(edge_weight, gt, gf, num_topics, tau)
    P = 1 + num_topics
    a = ones(Float64, P)
    for j in 2:P
        aw = 0.0
        for l in 1:(j - 1)
            aw += edge_weight[j, l] * a[l]
        end
        lt = _log1mexpm(aw) + gt[j]
        lf = -aw + gf[j]
        a[j] = 1.0 / (1.0 + exp(-(lt - lf) / tau))
    end
    return a
end

function _sliced_nuts_program(
    rng, model, ew, nt, nw, tau, ss, mtd, num_warmup, num_samples, imm, otensor, cadr, sel
)
    trace, _, _ = ProbProg.generate(
        rng, otensor, model, ew, nt, nw, tau; constrained_addresses=cadr
    )
    trace, diag, _, _ = ProbProg.mcmc(
        rng,
        trace,
        model,
        ew,
        nt,
        nw,
        tau;
        selection=sel,
        algorithm=:NUTS,
        inverse_mass_matrix=imm,
        step_size=ss,
        max_tree_depth=mtd,
        num_warmup=num_warmup,
        num_samples=num_samples,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
    return trace, diag
end

@testset "gumbel per-topic sliced NUTS (BinopPadToConcat regression)" begin
    num_topics = 3
    num_words = 6
    tau = 0.1
    P = 1 + num_topics
    nrows = P + num_words

    edge_weight = zeros(Float64, nrows, P)
    for r in 2:nrows, c in 1:min(r - 1, P)
        edge_weight[r, c] = 0.4
    end
    S_obs = Float64[1, 0, 1, 0, 1, 0]

    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)
    obs = ProbProg.Constraint(
        :active_gumbel_true => zeros(P), :active_gumbel_false => zeros(P), :S => S_obs
    )
    cadr = ProbProg.extract_addresses(obs)
    otensor = ProbProg.flatten_constraint(obs)
    sel = ProbProg.select(
        ProbProg.Address(:active_gumbel_true), ProbProg.Address(:active_gumbel_false)
    )
    imm = ConcreteRArray(ones(Float64, 2P))
    ss = ConcreteRNumber(0.1)

    num_warmup = 40
    num_samples = 20

    local compiled, tt
    @test begin
        compiled, tt = ProbProg.with_trace() do
            @compile optimize = :probprog _sliced_nuts_program(
                rng,
                _sliced_noisy_or,
                edge_weight,
                num_topics,
                num_words,
                tau,
                ss,
                6,
                num_warmup,
                num_samples,
                imm,
                otensor,
                cadr,
                sel,
            )
        end
        true
    end

    trace_tensor, _ = compiled(
        rng,
        _sliced_noisy_or,
        edge_weight,
        num_topics,
        num_words,
        tau,
        ss,
        6,
        num_warmup,
        num_samples,
        imm,
        otensor,
        cadr,
        sel,
    )
    trace_tensor = Array(trace_tensor)
    entries = ProbProg.filter_entries_by_selection(tt.entries, sel)
    trace = ProbProg.unflatten_trace(trace_tensor, 0.0, entries, nothing)

    gt = Array(trace.choices[:active_gumbel_true])
    gf = Array(trace.choices[:active_gumbel_false])

    @test size(gt) == (num_samples, P)
    @test size(gf) == (num_samples, P)
    @test all(isfinite, gt)
    @test all(isfinite, gf)

    for d in 1:num_samples
        a = _ref_active(edge_weight, gt[d, :], gf[d, :], num_topics, tau)
        @test a[1] == 1.0
        @test all(isfinite, a)
        @test all(0.0 .<= a .<= 1.0)
    end
end
