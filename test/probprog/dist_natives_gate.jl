# Hard-coded reference values are NumPyro log_prob outputs.
using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

include(joinpath(@__DIR__, "common.jl"))

maxabsdiff(a, b) = maximum(abs.(Array(a) .- Array(b)); init=0.0)
bitwise_equal(a, b) = Array(a) == Array(b)

function emitted_weight(mk, xval, args...)
    obs = ProbProg.Constraint(:x => xval)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    seed = Reactant.to_rarray(UInt64[1, 4])
    rng = ReactantRNG(seed)
    prog(rng, ot, a...) = ProbProg.generate(rng, ot, mk, a...; constrained_addresses)
    compiled = @compile optimize = :probprog prog(rng, obs_tensor, args...)
    _, w, _ = compiled(rng, obs_tensor, args...)
    return Array(w)
end

function ref_weight(mk_ref, xval, args...)
    obs = ProbProg.Constraint(:x => xval)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    seed = Reactant.to_rarray(UInt64[1, 4])
    rng = ReactantRNG(seed)
    prog(rng, ot, a...) = ProbProg.generate(rng, ot, mk_ref, a...; constrained_addresses)
    compiled = @compile optimize = :probprog prog(rng, obs_tensor, args...)
    _, w, _ = compiled(rng, obs_tensor, args...)
    return Array(w)
end

function simulate_pair(mk, mk_ref, args...)
    function run(model)
        seed = Reactant.to_rarray(UInt64[7, 11])
        rng = ReactantRNG(seed)
        prog(rng, a...) = ProbProg.simulate(rng, model, a...)
        compiled = @compile optimize = :probprog prog(rng, args...)
        t, w, _ = compiled(rng, args...)
        return Array(t), Array(w)
    end
    return run(mk_ref), run(mk)
end

@testset "dist natives emit gate" begin
    @testset "Exponential scalar" begin
        exp_model(rng, lam) =
            ProbProg.sample(rng, ProbProg.Exponential(lam, (1,)); symbol=:x)[2]
        exp_ref(rng, lam) = ProbProg.sample(
            rng,
            ref_exponential_sampler,
            lam,
            (1,);
            symbol=:x,
            logpdf=ref_exponential_logpdf,
        )[2]
        lam = ConcreteRNumber(1.5)
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(exp_model, exp_ref, lam)
        @test maxabsdiff(t_tr, t_em) <= 1e-9
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(
            0.5 => -0.3445348918918356,
            1.0 => -1.0945348918918356,
            2.0 => -2.5945348918918354,
        )
        for (x, ref) in numpyro
            we = emitted_weight(exp_model, [x], lam)
            wt = ref_weight(exp_ref, [x], lam)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "Exponential vector (event_dim reduce)" begin
        expv_model(rng, lam) =
            ProbProg.sample(rng, ProbProg.Exponential(lam, (3,)); symbol=:x)[2]
        expv_ref(rng, lam) = ProbProg.sample(
            rng,
            ref_exponential_sampler,
            lam,
            (3,);
            symbol=:x,
            logpdf=ref_exponential_logpdf,
        )[2]
        lam = Reactant.to_rarray([1.5, 1.5, 1.5])
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(expv_model, expv_ref, lam)
        @test maxabsdiff(t_tr, t_em) <= 1e-9
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        ref = -0.3445348918918356 + -1.0945348918918356 + -2.5945348918918354
        we = emitted_weight(expv_model, [0.5, 1.0, 2.0], lam)
        @test isapprox(Array(we)[], ref; atol=1e-9)
    end

    @testset "Bernoulli" begin
        bern_model(rng, lg) =
            ProbProg.sample(rng, ProbProg.Bernoulli(lg, (1,)); symbol=:x)[2]
        bern_ref(rng, lg) = ProbProg.sample(
            rng, ref_bernoulli_sampler, lg, (1,); symbol=:x, logpdf=ref_bernoulli_logpdf
        )[2]
        lg = Reactant.to_rarray([0.3])
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(bern_model, bern_ref, lg)
        @test bitwise_equal(t_tr, t_em)
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(0.0 => -0.854355244468527, 1.0 => -0.5543552444685269)
        for (y, ref) in numpyro
            we = emitted_weight(bern_model, [y], lg)
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "DiscreteUniform" begin
        du_model(rng, lo, hi) =
            ProbProg.sample(rng, ProbProg.DiscreteUniform(lo, hi, (1,)); symbol=:x)[2]
        du_ref(rng, lo, hi) = ProbProg.sample(
            rng,
            ref_discreteuniform_sampler,
            lo,
            hi,
            (1,);
            symbol=:x,
            logpdf=ref_discreteuniform_logpdf,
        )[2]
        lo = ConcreteRNumber(0.0)
        hi = ConcreteRNumber(10.0)
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(du_model, du_ref, lo, hi)
        @test bitwise_equal(t_tr, t_em)
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        ref = -2.3978952727983707
        for x in (2.0, 5.0, 9.0)
            we = emitted_weight(du_model, [x], lo, hi)
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "Gamma logpdf" begin
        gam_model(rng, a, b) =
            ProbProg.sample(rng, ProbProg.Gamma(a, b, (1,)); symbol=:x)[2]
        gam_ref(rng, a, b) = ProbProg.sample(
            rng,
            ProbProg.sampler(ProbProg.Gamma),
            a,
            b,
            (1,);
            symbol=:x,
            logpdf=ref_gamma_logpdf,
        )[2]
        a = ConcreteRNumber(2.0)
        b = ConcreteRNumber(1.5)
        numpyro = Dict(
            0.5 => -0.6322169643436166,
            1.5 => -1.0336046756755068,
            3.0 => -2.5904574951155612,
        )
        for (x, ref) in numpyro
            we = emitted_weight(gam_model, [x], a, b)
            wt = ref_weight(gam_ref, [x], a, b)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "constant (non-traced) params reach emitted logpdf" begin
        gam_const(rng) = ProbProg.sample(rng, ProbProg.Gamma(2.0, 1.5, (1,)); symbol=:x)[2]
        obs = ProbProg.Constraint(:x => [3.0])
        ca = ProbProg.extract_addresses(obs)
        ot = ProbProg.flatten_constraint(obs)
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        prog(rng, t) = ProbProg.generate(rng, t, gam_const; constrained_addresses=ca)
        compiled = @compile optimize = :probprog prog(rng, ot)
        _, w, _ = compiled(rng, ot)
        wv = Array(w)[]
        @test isfinite(wv)
        # NumPyro Gamma(conc=2, rate=1.5).log_prob(3.0) = -2.5904574951155612.
        @test isapprox(wv, -2.5904574951155612; atol=1e-9)
    end

    @testset "StudentT logpdf" begin
        st_model(rng, df, lc, sc) =
            ProbProg.sample(rng, ProbProg.StudentT(df, lc, sc, (1,)); symbol=:x)[2]
        st_ref(rng, df, lc, sc) = ProbProg.sample(
            rng,
            ProbProg.sampler(ProbProg.StudentT),
            df,
            lc,
            sc,
            (1,);
            symbol=:x,
            logpdf=ref_studentt_logpdf,
        )[2]
        df = ConcreteRNumber(3.0)
        lc = ConcreteRNumber(0.5)
        sc = ConcreteRNumber(2.0)
        numpyro = Dict(
            -1.0 => -2.0377365440367736,
            0.0 => -1.7352746045889267,
            2.0 => -2.0377365440367736,
        )
        for (x, ref) in numpyro
            we = emitted_weight(st_model, [x], df, lc, sc)
            wt = ref_weight(st_ref, [x], df, lc, sc)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "Poisson logpdf" begin
        pois_model(rng, r) = ProbProg.sample(rng, ProbProg.Poisson(r, (1,)); symbol=:x)[2]
        pois_ref(rng, r) = ProbProg.sample(
            rng,
            ProbProg.sampler(ProbProg.Poisson),
            r,
            (1,);
            symbol=:x,
            logpdf=ref_poisson_logpdf,
        )[2]
        r = ConcreteRNumber(3.0)
        numpyro = Dict(0.0 => -3.0, 2.0 => -1.4959226032237245, 5.0 => -2.2944302994414976)
        for (k, ref) in numpyro
            we = emitted_weight(pois_model, [k], r)
            wt = ref_weight(pois_ref, [k], r)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "HalfNormal" begin
        hn_model(rng, s) = ProbProg.sample(rng, ProbProg.HalfNormal(s, (1,)); symbol=:x)[2]
        hn_ref(rng, s) = ProbProg.sample(
            rng,
            ref_halfnormal_sampler,
            s,
            (1,);
            symbol=:x,
            logpdf=ref_halfnormal_logpdf,
        )[2]
        s = ConcreteRNumber(1.5)
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(hn_model, hn_ref, s)
        @test maxabsdiff(t_tr, t_em) <= 1e-9
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(
            0.5 => -0.6868120163084473,
            1.0 => -0.8534786829751139,
            2.0 => -1.5201453496417807,
        )
        for (x, ref) in numpyro
            we = emitted_weight(hn_model, [x], s)
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "HalfCauchy" begin
        hc_model(rng, s) = ProbProg.sample(rng, ProbProg.HalfCauchy(s, (1,)); symbol=:x)[2]
        hc_ref(rng, s) = ProbProg.sample(
            rng,
            ref_halfcauchy_sampler,
            s,
            (1,);
            symbol=:x,
            logpdf=ref_halfcauchy_logpdf,
        )[2]
        s = ConcreteRNumber(2.0)
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(hc_model, hc_ref, s)
        @test maxabsdiff(t_tr, t_em) <= 1e-9
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(
            0.5 => -1.205354507665835,
            1.0 => -1.3678734371636097,
            3.0 => -2.3233848821910463,
        )
        for (x, ref) in numpyro
            we = emitted_weight(hc_model, [x], s)
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "MultivariateNormal" begin
        # Non-diagonal covariance so a transposed Cholesky factor changes the draw.
        function mvn_model(rng, mu, Sig)
            return ProbProg.sample(
                rng, ProbProg.MultivariateNormal(mu, Sig, (2,)); symbol=:x
            )[2]
        end
        function mvn_ref(rng, mu, Sig)
            return ProbProg.sample(
                rng, ref_mvn_sampler, mu, Sig, (2,); symbol=:x, logpdf=ref_mvn_logpdf
            )[2]
        end
        mu = Reactant.to_rarray([1.0, -2.0])
        Sig = Reactant.to_rarray([2.0 0.6; 0.6 1.0])
        (t_tr, w_tr), (t_em, w_em) = simulate_pair(mvn_model, mvn_ref, mu, Sig)
        @test maxabsdiff(t_tr, t_em) <= 1e-9
        @test maxabsdiff(w_tr, w_em) <= 1e-9
        for x in ([1.0, -2.0], [1.5, -1.0], [0.0, 0.0])
            we = emitted_weight(mvn_model, x, mu, Sig)
            wt = ref_weight(mvn_ref, x, mu, Sig)
            @test maxabsdiff(wt, we) <= 1e-9
        end
    end

    @testset "Dirichlet logpdf" begin
        dir_model(rng, a) = ProbProg.sample(rng, ProbProg.Dirichlet(a, (3,)); symbol=:x)[2]
        dir_ref(rng, a) = ProbProg.sample(
            rng,
            ProbProg.sampler(ProbProg.Dirichlet),
            a,
            (3,);
            symbol=:x,
            logpdf=ref_dirichlet_logpdf,
        )[2]
        a1 = Reactant.to_rarray([1.0, 1.0, 1.0])
        for (x, ref) in
            ([0.2, 0.3, 0.5] => 0.6931471805599441, [0.1, 0.1, 0.8] => 0.6931471805599441)
            we = emitted_weight(dir_model, x, a1)
            wt = ref_weight(dir_ref, x, a1)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
        a2 = Reactant.to_rarray([2.0, 3.0, 5.0])
        we = emitted_weight(dir_model, [0.2, 0.3, 0.5], a2)
        @test isapprox(Array(we)[], 2.140654225847829; atol=1e-9)
    end

    @testset "Gumbel" begin
        # Sampler not bit-matched to jax.random.gumbel, which clamps u to [tiny, 1).
        gum_model(rng, lc, sc) =
            ProbProg.sample(rng, ProbProg.Gumbel(lc, sc, (1,)); symbol=:x)[2]
        lc0 = ConcreteRNumber(0.0)
        sc0 = ConcreteRNumber(1.0)
        numpyro0 = Dict(
            0.0 => -1.0,
            1.0 => -1.3678794411714423,
            2.0 => -2.135335283236613,
            -1.0 => -1.7182818284590455,
        )
        for (x, ref) in numpyro0
            we = emitted_weight(gum_model, [x], lc0, sc0)
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
        lc1 = ConcreteRNumber(0.5)
        sc1 = ConcreteRNumber(2.0)
        numpyro1 = Dict(
            0.0 => -1.7271725972476868,
            1.0 => -1.7219479636313504,
            2.0 => -1.9155137333009602,
            -1.0 => -2.06014719717262,
        )
        for (x, ref) in numpyro1
            we = emitted_weight(gum_model, [x], lc1, sc1)
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
        seed = Reactant.to_rarray(UInt64[7, 11])
        rng = ReactantRNG(seed)
        sim(rng, lc, sc) = ProbProg.simulate(rng, gum_model, lc, sc)
        compiled = @compile optimize = :probprog sim(rng, lc1, sc1)
        t, w, _ = compiled(rng, lc1, sc1)
        drawn = Array(t)[1]
        @test isfinite(drawn)
        we = emitted_weight(gum_model, [drawn], lc1, sc1)
        @test isapprox(Array(w)[], Array(we)[]; atol=1e-9)
    end
end
