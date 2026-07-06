# F2 spike F-gate: compile one small model BOTH ways -- traced Normal
# (EMIT_DISTRIBUTIONS off) vs emitted StdNormal+Affine (EMIT_DISTRIBUTIONS on)
# -- through simulate, generate, and one NUTS run, and compare:
#   * the sampler RNG stream (the sampled trace) bitwise
using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

function gate_model(rng, mu, sigma)
    _, s = ProbProg.sample(rng, ProbProg.Normal(mu, sigma, (1,)); symbol=:s)
    _, t = ProbProg.sample(rng, ProbProg.Normal(s, sigma, (1,)); symbol=:t)
    return t
end

# Run `f` once with EMIT_DISTRIBUTIONS off (traced) and once on (emitted).
function both_ways(f)
    ProbProg.EMIT_DISTRIBUTIONS[] = false
    traced = f()
    ProbProg.EMIT_DISTRIBUTIONS[] = true
    emitted = f()
    ProbProg.EMIT_DISTRIBUTIONS[] = false
    return traced, emitted
end

maxabsdiff(a, b) = maximum(abs.(Array(a) .- Array(b)); init=0.0)
bitwise_equal(a, b) = Array(a) == Array(b)

function simulate_program(rng, mu, sigma)
    return ProbProg.simulate(rng, gate_model, mu, sigma)
end

function generate_program(rng, obs_tensor, mu, sigma, constrained_addresses)
    return ProbProg.generate(rng, obs_tensor, gate_model, mu, sigma; constrained_addresses)
end

function nuts_program(
    rng, obs_tensor, mu, sigma, constrained_addresses, selection, imm, step_size
)
    trace, _, _ = ProbProg.generate(
        rng, obs_tensor, gate_model, mu, sigma; constrained_addresses
    )
    new_trace, diagnostics, _, _ = ProbProg.mcmc(
        rng,
        trace,
        gate_model,
        mu,
        sigma;
        selection,
        algorithm=:NUTS,
        inverse_mass_matrix=imm,
        step_size,
        max_tree_depth=5,
        num_warmup=0,
        num_samples=3,
        adapt_step_size=false,
        adapt_mass_matrix=false,
    )
    return new_trace, diagnostics
end

@testset "dist emit-vs-traced gate" begin
    mu = ConcreteRNumber(0.5)
    sigma = ConcreteRNumber(2.0)

    @testset "simulate (seed $s0/$s1)" for (s0, s1) in
                                           ((1, 4), (7, 11), (42, 43), (100, 200))
        function run_simulate()
            seed = Reactant.to_rarray(UInt64[s0, s1])
            rng = ReactantRNG(seed)
            compiled = @compile optimize = :probprog simulate_program(rng, mu, sigma)
            trace, weight, _ = compiled(rng, mu, sigma)
            return Array(trace), Array(weight)
        end
        (t_tr, w_tr), (t_em, w_em) = both_ways(run_simulate)
        wdiff = maxabsdiff(w_tr, w_em)
        tdiff = maxabsdiff(t_tr, t_em)
        @info "simulate" seed = (s0, s1) weight_absdiff = wdiff trace_absdiff = tdiff trace_bitwise = bitwise_equal(
            t_tr, t_em
        )
        @test wdiff <= 1e-12
        @test tdiff <= 1e-9
    end

    @testset "generate importance weight" begin
        obs = ProbProg.Constraint(:t => [0.3])
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)

        function run_generate()
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            compiled = @compile optimize = :probprog generate_program(
                rng, obs_tensor, mu, sigma, constrained_addresses
            )
            trace, weight, _ = compiled(rng, obs_tensor, mu, sigma, constrained_addresses)
            return Array(trace), Array(weight)
        end
        (t_tr, w_tr), (t_em, w_em) = both_ways(run_generate)
        wdiff = maxabsdiff(w_tr, w_em)
        tdiff = maxabsdiff(t_tr, t_em)
        @info "generate" weight_absdiff = wdiff trace_absdiff = tdiff trace_bitwise = bitwise_equal(
            t_tr, t_em
        )
        @test wdiff <= 1e-12
        @test tdiff <= 1e-9
    end

    @testset "generate (s observed, t sampled)" begin
        # Constrains :s only, so the generate lowering scores :s from the
        # constraint (emitDistLogpdf) and DRAWS :t through the emitted sampler
        # (emitDistSampler) -- exercising both generate emit branches.
        obs = ProbProg.Constraint(:s => [0.3])
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)

        function run_generate2()
            seed = Reactant.to_rarray(UInt64[7, 11])
            rng = ReactantRNG(seed)
            compiled = @compile optimize = :probprog generate_program(
                rng, obs_tensor, mu, sigma, constrained_addresses
            )
            trace, weight, _ = compiled(rng, obs_tensor, mu, sigma, constrained_addresses)
            return Array(trace), Array(weight)
        end
        (t_tr, w_tr), (t_em, w_em) = both_ways(run_generate2)
        wdiff = maxabsdiff(w_tr, w_em)
        tdiff = maxabsdiff(t_tr, t_em)
        @info "generate2" weight_absdiff = wdiff trace_absdiff = tdiff trace_bitwise = bitwise_equal(
            t_tr, t_em
        )
        @test wdiff <= 1e-12
        @test tdiff <= 1e-9
    end

    @testset "mh (flat regenerate)" begin
        # A fully-constrained generate yields a deterministic (sampler-free)
        # trace0/weight0; the flat MH step (region=false) lowers to
        # `impulse.regenerate`, whose selected `:s` site is redrawn through the
        # emitted sampler and scored by the emitted logpdf.
        obs = ProbProg.Constraint(:s => [0.3], :t => [0.6])
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)
        selection = ProbProg.select(ProbProg.Address(:s))

        function run_mh()
            seed = Reactant.to_rarray(UInt64[42, 43])
            rng = ReactantRNG(seed)
            function mh_program(rng, obs_tensor, mu, sigma)
                trace0, weight0, _ = ProbProg.generate(
                    rng, obs_tensor, gate_model, mu, sigma; constrained_addresses
                )
                nt, nw, acc, _ = ProbProg.mh(
                    rng, trace0, weight0, gate_model, mu, sigma; selection, region=false
                )
                return nt, nw, acc
            end
            compiled = @compile optimize = :probprog mh_program(rng, obs_tensor, mu, sigma)
            nt, nw, acc = compiled(rng, obs_tensor, mu, sigma)
            return Array(nt), Array(nw), Array(acc)
        end
        (t_tr, w_tr, a_tr), (t_em, w_em, a_em) = both_ways(run_mh)
        wdiff = maxabsdiff(w_tr, w_em)
        tdiff = maxabsdiff(t_tr, t_em)
        @info "mh" weight_absdiff = wdiff trace_absdiff = tdiff accepted = (a_tr, a_em)
        @test wdiff <= 1e-9
        @test tdiff <= 1e-9
        @test a_tr == a_em
    end

    @testset "NUTS" begin
        obs = ProbProg.Constraint(:t => [0.3])
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)
        selection = ProbProg.select(ProbProg.Address(:s))

        function run_nuts()
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            imm = Reactant.to_rarray([1.0])
            step_size = ConcreteRNumber(0.1)
            compiled = @compile optimize = :probprog nuts_program(
                rng, obs_tensor, mu, sigma, constrained_addresses, selection, imm, step_size
            )
            new_trace, diagnostics = compiled(
                rng, obs_tensor, mu, sigma, constrained_addresses, selection, imm, step_size
            )
            return Array(new_trace), Array(diagnostics)
        end
        (tr_tr, d_tr), (tr_em, d_em) = both_ways(run_nuts)
        tdiff = maxabsdiff(tr_tr, tr_em)
        ddiff = maxabsdiff(d_tr, d_em)
        @info "nuts" trace_absdiff = tdiff diag_absdiff = ddiff trace_bitwise = bitwise_equal(
            tr_tr, tr_em
        )
        @test tdiff <= 1e-9
    end
end
