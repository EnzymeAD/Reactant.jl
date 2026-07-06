using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

include(joinpath(@__DIR__, "common.jl"))

function gate_model(rng, mu, sigma)
    _, s = ProbProg.sample(rng, ProbProg.Normal(mu, sigma, (1,)); symbol=:s)
    _, t = ProbProg.sample(rng, ProbProg.Normal(s, sigma, (1,)); symbol=:t)
    return t
end

function gate_model_ref(rng, mu, sigma)
    _, s = ProbProg.sample(
        rng, ref_normal_sampler, mu, sigma, (1,); symbol=:s, logpdf=ref_normal_logpdf
    )
    _, t = ProbProg.sample(
        rng, ref_normal_sampler, s, sigma, (1,); symbol=:t, logpdf=ref_normal_logpdf
    )
    return t
end

function both_ways(f)
    return f(gate_model_ref), f(gate_model)
end

maxabsdiff(a, b) = maximum(abs.(Array(a) .- Array(b)); init=0.0)
bitwise_equal(a, b) = Array(a) == Array(b)

@testset "dist emit-vs-reference gate" begin
    mu = ConcreteRNumber(0.5)
    sigma = ConcreteRNumber(2.0)

    @testset "simulate (seed $s0/$s1)" for (s0, s1) in
                                           ((1, 4), (7, 11), (42, 43), (100, 200))
        function run_simulate(model)
            seed = Reactant.to_rarray(UInt64[s0, s1])
            rng = ReactantRNG(seed)
            prog(rng, mu, sigma) = ProbProg.simulate(rng, model, mu, sigma)
            compiled = @compile optimize = :probprog prog(rng, mu, sigma)
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

        function run_generate(model)
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            function prog(rng, ot, mu, sigma)
                return ProbProg.generate(rng, ot, model, mu, sigma; constrained_addresses)
            end
            compiled = @compile optimize = :probprog prog(rng, obs_tensor, mu, sigma)
            trace, weight, _ = compiled(rng, obs_tensor, mu, sigma)
            return Array(trace), Array(weight)
        end
        (t_tr, w_tr), (t_em, w_em) = both_ways(run_generate)
        wdiff = maxabsdiff(w_tr, w_em)
        tdiff = maxabsdiff(t_tr, t_em)
        @info "generate" weight_absdiff = wdiff trace_absdiff = tdiff
        @test wdiff <= 1e-12
        @test tdiff <= 1e-9
    end

    @testset "generate (s observed, t sampled)" begin
        obs = ProbProg.Constraint(:s => [0.3])
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)

        function run_generate2(model)
            seed = Reactant.to_rarray(UInt64[7, 11])
            rng = ReactantRNG(seed)
            function prog(rng, ot, mu, sigma)
                return ProbProg.generate(rng, ot, model, mu, sigma; constrained_addresses)
            end
            compiled = @compile optimize = :probprog prog(rng, obs_tensor, mu, sigma)
            trace, weight, _ = compiled(rng, obs_tensor, mu, sigma)
            return Array(trace), Array(weight)
        end
        (t_tr, w_tr), (t_em, w_em) = both_ways(run_generate2)
        wdiff = maxabsdiff(w_tr, w_em)
        tdiff = maxabsdiff(t_tr, t_em)
        @info "generate2" weight_absdiff = wdiff trace_absdiff = tdiff
        @test wdiff <= 1e-12
        @test tdiff <= 1e-9
    end

    @testset "NUTS" begin
        obs = ProbProg.Constraint(:t => [0.3])
        constrained_addresses = ProbProg.extract_addresses(obs)
        obs_tensor = ProbProg.flatten_constraint(obs)
        selection = ProbProg.select(ProbProg.Address(:s))

        function run_nuts(model)
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            imm = Reactant.to_rarray([1.0])
            step_size = ConcreteRNumber(0.1)
            function prog(rng, ot, mu, sigma, imm, step_size)
                trace, _, _ = ProbProg.generate(
                    rng, ot, model, mu, sigma; constrained_addresses
                )
                nt, diag, _, _ = ProbProg.mcmc(
                    rng,
                    trace,
                    model,
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
                return nt, diag
            end
            compiled = @compile optimize = :probprog prog(
                rng, obs_tensor, mu, sigma, imm, step_size
            )
            new_trace, diagnostics = compiled(rng, obs_tensor, mu, sigma, imm, step_size)
            return Array(new_trace), Array(diagnostics)
        end
        (tr_tr, d_tr), (tr_em, d_em) = both_ways(run_nuts)
        tdiff = maxabsdiff(tr_tr, tr_em)
        ddiff = maxabsdiff(d_tr, d_em)
        @info "nuts" trace_absdiff = tdiff diag_absdiff = ddiff
        @test tdiff <= 1e-9
    end

    @testset "LogNormal composition (StdNormal + Affine + Exp)" begin
        function lognormal_model(rng, mu, sigma)
            _, x = ProbProg.sample(rng, ProbProg.LogNormal(mu, sigma, (1,)); symbol=:x)
            return x
        end
        function lognormal_model_ref(rng, mu, sigma)
            _, x = ProbProg.sample(
                rng,
                ref_lognormal_sampler,
                mu,
                sigma,
                (1,);
                symbol=:x,
                logpdf=ref_lognormal_logpdf,
            )
            return x
        end

        @testset "simulate (sampler stream)" begin
            function run(model)
                seed = Reactant.to_rarray(UInt64[13, 17])
                rng = ReactantRNG(seed)
                prog(rng, mu, sigma) = ProbProg.simulate(rng, model, mu, sigma)
                compiled = @compile optimize = :probprog prog(rng, mu, sigma)
                t, w, _ = compiled(rng, mu, sigma)
                return Array(t), Array(w)
            end
            (t_tr, w_tr), (t_em, w_em) = ((run(lognormal_model_ref), run(lognormal_model)))
            wdiff = maxabsdiff(w_tr, w_em)
            tdiff = maxabsdiff(t_tr, t_em)
            @info "lognormal simulate" weight_absdiff = wdiff trace_absdiff = tdiff
            @test wdiff <= 1e-12
            @test tdiff <= 1e-9
        end

        @testset "generate (logpdf change-of-variables)" begin
            obs = ProbProg.Constraint(:x => [1.7])
            constrained_addresses = ProbProg.extract_addresses(obs)
            obs_tensor = ProbProg.flatten_constraint(obs)
            function run(model)
                seed = Reactant.to_rarray(UInt64[13, 17])
                rng = ReactantRNG(seed)
                function prog(rng, ot)
                    return ProbProg.generate(
                        rng, ot, model, mu, sigma; constrained_addresses
                    )
                end
                compiled = @compile optimize = :probprog prog(rng, obs_tensor)
                _, w, _ = compiled(rng, obs_tensor)
                return Array(w)
            end
            w_tr, w_em = run(lognormal_model_ref), run(lognormal_model)
            wdiff = maxabsdiff(w_tr, w_em)
            @info "lognormal generate" weight_absdiff = wdiff
            @test wdiff <= 1e-12
        end
    end
end
