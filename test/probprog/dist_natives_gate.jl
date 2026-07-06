# F4 gate: primitive native distributions emitted by the shared distribution
# builder, each behind the emit-vs-traced flag (EMIT_DISTRIBUTIONS).
#
# For each distribution we check, per the F4 parity rule:
#   * emitted vs traced logpdf pointwise (constrained generate weight), 1e-12;
#   * sampler RNG stream bitwise where emitted (simulate trace);
#   * emitted logpdf vs a hardcoded NumPyro reference on an (x, params) grid.
# DiscreteUniform. Logpdf-only natives (sampler stays traced, logpdf emitted via
# impulse.lgamma): Gamma, StudentT, Poisson -- gated through a fully-constrained
# generate that never draws.
using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

function both_ways(f)
    saved = ProbProg.EMIT_DISTRIBUTIONS[]
    ProbProg.EMIT_DISTRIBUTIONS[] = false
    traced = f()
    ProbProg.EMIT_DISTRIBUTIONS[] = true
    emitted = f()
    ProbProg.EMIT_DISTRIBUTIONS[] = saved
    return traced, emitted
end

maxabsdiff(a, b) = maximum(abs.(Array(a) .- Array(b)); init=0.0)
bitwise_equal(a, b) = Array(a) == Array(b)

# Fully-constrained single-site generate -> weight = logpdf(observed). `mk` builds
# the one-site model closing over its params; `xval` is the observed value vector.
function constrained_weight(mk, xval, args...)
    obs = ProbProg.Constraint(:x => xval)
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)
    function run()
        seed = Reactant.to_rarray(UInt64[1, 4])
        rng = ReactantRNG(seed)
        prog(rng, ot, a...) = ProbProg.generate(rng, ot, mk, a...; constrained_addresses)
        compiled = @compile optimize = :probprog prog(rng, obs_tensor, args...)
        _, w, _ = compiled(rng, obs_tensor, args...)
        return Array(w)
    end
    return both_ways(run)
end

# Simulate a one-site model -> (trace, weight); the site is drawn by the sampler.
function simulate_site(mk, args...)
    function run()
        seed = Reactant.to_rarray(UInt64[7, 11])
        rng = ReactantRNG(seed)
        prog(rng, a...) = ProbProg.simulate(rng, mk, a...)
        compiled = @compile optimize = :probprog prog(rng, args...)
        t, w, _ = compiled(rng, args...)
        return Array(t), Array(w)
    end
    return both_ways(run)
end

@testset "dist natives emit gate" begin
    #==========================================================================#
    # Fully-emittable: Exponential (scalar + vector event_dim)
    #==========================================================================#
    @testset "Exponential scalar" begin
        exp_model(rng, lam) =
            ProbProg.sample(rng, ProbProg.Exponential(lam, (1,)); symbol=:x)[2]
        lam = ConcreteRNumber(1.5)
        # sampler bitwise + logpdf 1e-12 through simulate
        (t_tr, w_tr), (t_em, w_em) = simulate_site(exp_model, lam)
        @test bitwise_equal(t_tr, t_em)
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        # emitted vs traced logpdf, and vs NumPyro, on a grid
        numpyro = Dict(
            0.5 => -0.3445348918918356,
            1.0 => -1.0945348918918356,
            2.0 => -2.5945348918918354,
        )
        for (x, ref) in numpyro
            (wt, we) = constrained_weight(exp_model, [x], lam)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "Exponential vector (event_dim reduce)" begin
        # A length-3 site: the emitted logpdf reduces over the event axis.
        expv_model(rng, lam) =
            ProbProg.sample(rng, ProbProg.Exponential(lam, (3,)); symbol=:x)[2]
        lam = Reactant.to_rarray([1.5, 1.5, 1.5])
        (t_tr, w_tr), (t_em, w_em) = simulate_site(expv_model, lam)
        @test bitwise_equal(t_tr, t_em)         # sampler bitwise
        @test maxabsdiff(w_tr, w_em) <= 1e-12   # summed logpdf matches traced sum
        # NumPyro: sum of per-element logpdfs at x = [0.5, 1.0, 2.0]
        ref = -0.3445348918918356 + -1.0945348918918356 + -2.5945348918918354
        (wt, we) = constrained_weight(expv_model, [0.5, 1.0, 2.0], lam)
        @test maxabsdiff(wt, we) <= 1e-12
        @test isapprox(Array(we)[], ref; atol=1e-9)
    end

    #==========================================================================#
    # Fully-emittable: Bernoulli(logits)
    #==========================================================================#
    @testset "Bernoulli" begin
        # _bernoulli_sampler draws rand(size(logits)...), so logits must carry the
        # site shape (length-1 here) rather than be a bare scalar.
        bern_model(rng, lg) =
            ProbProg.sample(rng, ProbProg.Bernoulli(lg, (1,)); symbol=:x)[2]
        lg = Reactant.to_rarray([0.3])
        (t_tr, w_tr), (t_em, w_em) = simulate_site(bern_model, lg)
        @test bitwise_equal(t_tr, t_em)
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(0.0 => -0.854355244468527, 1.0 => -0.5543552444685269)
        for (y, ref) in numpyro
            (wt, we) = constrained_weight(bern_model, [y], lg)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    #==========================================================================#
    # Fully-emittable: DiscreteUniform(low, high)
    #==========================================================================#
    @testset "DiscreteUniform" begin
        du_model(rng, lo, hi) =
            ProbProg.sample(rng, ProbProg.DiscreteUniform(lo, hi, (1,)); symbol=:x)[2]
        lo = ConcreteRNumber(0.0)
        hi = ConcreteRNumber(10.0)
        (t_tr, w_tr), (t_em, w_em) = simulate_site(du_model, lo, hi)
        @test bitwise_equal(t_tr, t_em)
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        ref = -2.3978952727983707
        for x in (2.0, 5.0, 9.0)
            (wt, we) = constrained_weight(du_model, [x], lo, hi)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    #==========================================================================#
    # Logpdf-only (sampler traced): Gamma, StudentT, Poisson
    #==========================================================================#
    @testset "Gamma logpdf" begin
        gam_model(rng, a, b) =
            ProbProg.sample(rng, ProbProg.Gamma(a, b, (1,)); symbol=:x)[2]
        a = ConcreteRNumber(2.0)
        b = ConcreteRNumber(1.5)
        numpyro = Dict(
            0.5 => -0.6322169643436166,
            1.5 => -1.0336046756755068,
            3.0 => -2.5904574951155612,
        )
        for (x, ref) in numpyro
            (wt, we) = constrained_weight(gam_model, [x], a, b)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "StudentT logpdf" begin
        st_model(rng, df, lc, sc) =
            ProbProg.sample(rng, ProbProg.StudentT(df, lc, sc, (1,)); symbol=:x)[2]
        df = ConcreteRNumber(3.0)
        lc = ConcreteRNumber(0.5)
        sc = ConcreteRNumber(2.0)
        numpyro = Dict(
            -1.0 => -2.0377365440367736,
            0.0 => -1.7352746045889267,
            2.0 => -2.0377365440367736,
        )
        for (x, ref) in numpyro
            (wt, we) = constrained_weight(st_model, [x], df, lc, sc)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "Poisson logpdf" begin
        pois_model(rng, r) = ProbProg.sample(rng, ProbProg.Poisson(r, (1,)); symbol=:x)[2]
        r = ConcreteRNumber(3.0)
        numpyro = Dict(0.0 => -3.0, 2.0 => -1.4959226032237245, 5.0 => -2.2944302994414976)
        for (k, ref) in numpyro
            (wt, we) = constrained_weight(pois_model, [k], r)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    #==========================================================================#
    # Folded natives (sampler + split logpdf): HalfNormal, HalfCauchy
    #==========================================================================#
    @testset "HalfNormal" begin
        hn_model(rng, s) = ProbProg.sample(rng, ProbProg.HalfNormal(s, (1,)); symbol=:x)[2]
        s = ConcreteRNumber(1.5)
        (t_tr, w_tr), (t_em, w_em) = simulate_site(hn_model, s)
        @test maxabsdiff(t_tr, t_em) <= 1e-9   # sampler (sub-ULP StdNormal)
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(
            0.5 => -0.6868120163084473,
            1.0 => -0.8534786829751139,
            2.0 => -1.5201453496417807,
        )
        for (x, ref) in numpyro
            (wt, we) = constrained_weight(hn_model, [x], s)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    @testset "HalfCauchy" begin
        hc_model(rng, s) = ProbProg.sample(rng, ProbProg.HalfCauchy(s, (1,)); symbol=:x)[2]
        s = ConcreteRNumber(2.0)
        (t_tr, w_tr), (t_em, w_em) = simulate_site(hc_model, s)
        @test bitwise_equal(t_tr, t_em)        # uniform draw + tan matches traced
        @test maxabsdiff(w_tr, w_em) <= 1e-12
        numpyro = Dict(
            0.5 => -1.205354507665835,
            1.0 => -1.3678734371636097,
            3.0 => -2.3233848821910463,
        )
        for (x, ref) in numpyro
            (wt, we) = constrained_weight(hc_model, [x], s)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
    end

    #==========================================================================#
    # Distributional native (sampler traced): Dirichlet -- logpdf emitted with a
    # per-block (last-axis) reduction, scored through a constrained generate.
    #==========================================================================#
    @testset "Dirichlet logpdf" begin
        dir_model(rng, a) = ProbProg.sample(rng, ProbProg.Dirichlet(a, (3,)); symbol=:x)[2]
        # alpha = [1,1,1]: uniform on the 2-simplex; log p = log(2) everywhere.
        a1 = Reactant.to_rarray([1.0, 1.0, 1.0])
        for (x, ref) in
            ([0.2, 0.3, 0.5] => 0.6931471805599441, [0.1, 0.1, 0.8] => 0.6931471805599441)
            (wt, we) = constrained_weight(dir_model, x, a1)
            @test maxabsdiff(wt, we) <= 1e-12
            @test isapprox(Array(we)[], ref; atol=1e-9)
        end
        a2 = Reactant.to_rarray([2.0, 3.0, 5.0])
        (wt, we) = constrained_weight(dir_model, [0.2, 0.3, 0.5], a2)
        @test maxabsdiff(wt, we) <= 1e-12
        @test isapprox(Array(we)[], 2.140654225847829; atol=1e-9)
    end
end
