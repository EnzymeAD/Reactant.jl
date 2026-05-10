using Reactant, Test
using Reactant: ProbProg, ConcreteRNumber, ConcreteRArray
using Random
using PythonCall: pyimport, pyconvert, pylist

include(joinpath(@__DIR__, "common.jl"))

# `lp_via_jit(d, xs)` runs the trait-API `logpdf_fn` for `d` under `@jit`
# with traced parameters and returns the scalar Reactant logpdf. We use it
# as the reference path for NumPyro comparisons.
function lp_via_jit(dist, xs::AbstractVector{<:Real}, params::Tuple)
    lp_fn = ProbProg.logpdf_fn(typeof(dist))
    x = ConcreteRArray(xs)
    traced = map(ConcreteRNumber, params)
    lp = @jit lp_fn(x, traced..., nothing)
    return Reactant.to_number(lp)
end

@testset "Distributions" begin
    @testset "Construction" begin
        @testset "Normal" begin
            d = ProbProg.Normal(0.0, 1.0)
            @test ProbProg.params(d) == (0.0, 1.0)
            @test size(d) == ()

            d2 = ProbProg.Normal(2.0, 0.5)
            @test ProbProg.params(d2) == (2.0, 0.5)

            d3 = ProbProg.Normal(0.0, 1.0, (3, 4))
            @test ProbProg.params(d3) == (0.0, 1.0)
            @test size(d3) == (3, 4)

            @test ProbProg.support(typeof(d)) == ProbProg.RealSupport()

            # Per-element parameters
            μ = randn(2, 3)
            σ = abs.(randn(2, 3)) .+ 0.1
            dpe = ProbProg.Normal(μ, σ)
            @test size(dpe) == (2, 3)
            @test ProbProg.params(dpe) == (μ, σ)

        end

        @testset "Exponential" begin
            d = ProbProg.Exponential(1.0)
            @test ProbProg.params(d) == (1.0,)
            @test size(d) == ()

            d2 = ProbProg.Exponential(2.0)
            @test ProbProg.params(d2) == (2.0,)

            d3 = ProbProg.Exponential(0.5, (5,))
            @test ProbProg.params(d3) == (0.5,)
            @test size(d3) == (5,)

            @test ProbProg.support(typeof(d)) == ProbProg.PositiveSupport()

        end

        @testset "Uniform" begin
            d = ProbProg.Uniform(-1.0, 3.0)
            @test ProbProg.params(d) == (-1.0, 3.0)

            @test ProbProg.support(typeof(d)) == ProbProg.RealSupport()

        end

        @testset "InverseGamma" begin
            d = ProbProg.InverseGamma(3.0, 2.0)
            @test ProbProg.params(d) == (3.0, 2.0)

        end

        @testset "TDist" begin
            d = ProbProg.TDist(5.0)
            @test ProbProg.params(d) == (5.0, 0.0, 1.0)

            d2 = ProbProg.TDist(5.0, 0.3, 1.2)
            @test ProbProg.params(d2) == (5.0, 0.3, 1.2)

        end

        @testset "Bernoulli" begin
            d = ProbProg.Bernoulli(0.3)
            @test ProbProg.params(d) == (0.3,)
            @test size(d) == ()

            d2 = ProbProg.Bernoulli(0.5, (4,))
            @test size(d2) == (4,)

            # Per-element logits
            logits = [-1.0, 0.0, 1.0]
            d3 = ProbProg.Bernoulli(logits)
            @test size(d3) == (3,)
            @test ProbProg.params(d3) == (logits,)
        end
    end

    @testset "logpdf (no SpecialFunctions dependency)" begin
        # `_loggamma`, `_erf`, `_erfinv` are routed to MLIR ops via `@opcall`,
        # so anything that touches them needs to go through `@jit`. The
        # closed-form Normal / Exponential / Uniform logpdfs don't, so they
        # stay eager.
        @test ProbProg.logpdf(ProbProg.Normal(0.5, 1.3), 0.3) ≈
            -((0.3 - 0.5) / 1.3)^2 / 2 - log(1.3) - log(2π) / 2
        @test ProbProg.logpdf(ProbProg.Exponential(2.5), 1.0) ≈ -log(2.5) - 1.0 / 2.5
        @test ProbProg.logpdf(ProbProg.Uniform(-1.0, 3.0), 0.0) ≈ -log(4.0)

        # InverseGamma uses the in-tree `_loggamma` shim, so go through `@jit`.
        # Reference: log p(1.5; α=3, θ=2) = α log θ − loggamma(α) − (α+1) log x − θ/x
        @test (@jit ProbProg.logpdf(
            ProbProg.InverseGamma(ConcreteRNumber(3.0), ConcreteRNumber(2.0)),
            ConcreteRNumber(1.5),
        )) ≈ 3 * log(2) - log(2) - 4 * log(1.5) - 2 / 1.5 atol = 1.0e-9

        # unnormed_logpdf + lognorm == logpdf — closed-form bases only.
        for d in (
            ProbProg.Normal(0.5, 1.3),
            ProbProg.Exponential(2.5),
            ProbProg.Uniform(-1.0, 3.0),
        )
            x = ProbProg.Distributions.rand(d)
            @test ProbProg.logpdf(d, x) ≈
                ProbProg.unnormed_logpdf(d, x) + ProbProg.lognorm(d)
        end
    end

    @testset "cdf / quantile for Normal, Exponential, Uniform" begin
        # Normal cdf/quantile use `_erf`/`_erfinv` via `@opcall`, so wrap in
        # `@jit`. Exponential/Uniform are closed-form; their round-trip stays
        # eager.
        for d in (ProbProg.Exponential(2.0), ProbProg.Uniform(-1.0, 1.0))
            for p in (0.1, 0.3, 0.5, 0.7, 0.9)
                q = ProbProg.quantile(d, p)
                @test ProbProg.cdf(d, q) ≈ p atol = 1.0e-6
            end
        end
        # Normal cdf reference values via @jit.
        @test (@jit ProbProg.cdf(
            ProbProg.Normal(ConcreteRNumber(0.0), ConcreteRNumber(1.0)),
            ConcreteRNumber(1.0),
        )) ≈ 0.84134474606854 atol = 1.0e-6
        @test (@jit ProbProg.quantile(
            ProbProg.Normal(ConcreteRNumber(0.0), ConcreteRNumber(1.0)),
            ConcreteRNumber(0.975),
        )) ≈ 1.959963984540054 atol = 1.0e-6

        # cdf/quantile for InverseGamma and TDist are intentionally not
        # implemented (would require regularised incomplete gamma/beta).
        @test_throws MethodError ProbProg.cdf(ProbProg.InverseGamma(3.0, 2.0), 1.0)
        @test_throws MethodError ProbProg.cdf(ProbProg.TDist(5.0), 0.0)
    end

    @testset "trait API works for Modeling.jl" begin
        @test ProbProg.support(typeof(ProbProg.Normal(0.0, 1.0))) == ProbProg.RealSupport()
        @test ProbProg.support(typeof(ProbProg.Exponential(1.0))) == ProbProg.PositiveSupport()
        @test ProbProg.support(typeof(ProbProg.Uniform(0.0, 1.0))) == ProbProg.RealSupport()

        sampler = ProbProg.sampler(typeof(ProbProg.Normal(0.0, 1.0)))
        rng = Random.default_rng()
        @test sampler(rng, 0.0, 1.0, ()) isa Real
        @test size(sampler(rng, 0.0, 1.0, (3, 4))) == (3, 4)

        lp_fn = ProbProg.logpdf_fn(typeof(ProbProg.Normal(0.0, 1.0)))
        @test lp_fn(0.5, 0.0, 1.0, ()) ≈ -0.5^2 / 2 - log(2π) / 2
    end

    @testset "sampling produces in-support values" begin
        rng = Random.default_rng()
        for d in (
            ProbProg.Normal(0.0, 1.0),
            ProbProg.Exponential(1.0),
            ProbProg.Uniform(-1.0, 1.0),
        )
            for _ in 1:50
                x = ProbProg.Distributions.rand(rng, d)
                @test ProbProg.insupport(d, x)
            end
        end
    end

    @testset "Transforms" begin
        Dists = ProbProg.Distributions

        @testset "round-trip" begin
            for (t, z) in (
                (Dists.IdentityTransform(), 1.7),
                (Dists.AffineTransform(0.3, 1.2), 1.7),
                (Dists.LogTransform(), 0.4),
                (Dists.LogitTransform(), -0.2),
            )
                y = Dists.forward(t, z)
                @test Dists.inverse(t, y) ≈ z atol = 1.0e-12
            end
        end

        @testset "logabsdetjac matches finite-difference" begin
            ε = 1.0e-6
            for (t, z) in (
                (Dists.LogTransform(), 0.4),
                (Dists.LogitTransform(), -0.2),
            )
                fd =
                    (Dists.forward(t, z + ε) - Dists.forward(t, z - ε)) / (2ε)
                @test Dists.logabsdetjac(t, z) ≈ log(abs(fd)) atol = 1.0e-5
            end
        end

        @testset "ConstantJacobianTransform compose" begin
            t = Dists.AffineTransform(0.0, 2.0) ∘ Dists.AffineTransform(1.0, 3.0)
            @test t isa Dists.ConstantJacobianTransform
            # f(z) = 2*(1 + 3z) = 2 + 6z; logabsdet = log(6).
            @test Dists.forward(t, 1.0) ≈ 8.0
            @test Dists.logabsdetjac(t) ≈ log(6.0) atol = 1.0e-12
        end

        @testset "Mixed compose is data-dependent" begin
            t = Dists.LogTransform() ∘ Dists.AffineTransform(0.5, 0.7)
            @test !(t isa Dists.ConstantJacobianTransform)
            # f(z) = exp(0.5 + 0.7z); logabsdet = log(0.7) + (0.5 + 0.7z).
            z = 0.3
            @test Dists.forward(t, z) ≈ exp(0.5 + 0.7 * z)
            @test Dists.logabsdetjac(t, z) ≈ log(0.7) + (0.5 + 0.7 * z) atol = 1.0e-12
        end
    end

    @testset "LogNormal" begin
        d = ProbProg.LogNormal(0.0, 1.0)
        @test ProbProg.params(d) == (0.0, 1.0)
        @test ProbProg.support(typeof(d)) == ProbProg.PositiveSupport()
        # logpdf(LogNormal(0, 1), 1) = -log(2π)/2 - 0 - log(1) = -log(2π)/2.
        @test ProbProg.logpdf(d, 1.0) ≈ -log(2π) / 2 atol = 1.0e-12
        # General point: logpdf = -((log y - μ)/σ)²/2 - log σ - log y - log(2π)/2.
        d2 = ProbProg.LogNormal(0.5, 0.7)
        for y in (0.3, 1.0, 2.5)
            ref =
                -((log(y) - 0.5) / 0.7)^2 / 2 - log(0.7) - log(y) - log(2π) / 2
            @test ProbProg.logpdf(d2, y) ≈ ref atol = 1.0e-12
        end
        # unnormed_logpdf + lognorm round-trip.
        for y in (0.3, 1.0, 2.5)
            @test ProbProg.logpdf(d2, y) ≈
                ProbProg.unnormed_logpdf(d2, y) + ProbProg.lognorm(d2)
        end
        # cdf/quantile round-trip — `_erf`/`_erfinv` need `@jit`.
        roundtrip(μ, σ, p) =
            ProbProg.cdf(ProbProg.LogNormal(μ, σ), ProbProg.quantile(ProbProg.LogNormal(μ, σ), p))
        for p in (0.1, 0.5, 0.9)
            r = @jit roundtrip(
                ConcreteRNumber(0.5), ConcreteRNumber(0.7), ConcreteRNumber(p)
            )
            @test r ≈ p atol = 1.0e-6
        end
    end

    @testset "Bernoulli" begin
        # logp(y; ℓ) = y·ℓ − softplus(ℓ).
        for (ℓ, y) in ((0.0, 0), (0.0, 1), (2.5, 1), (-1.7, 0), (3.0, 0))
            d = ProbProg.Bernoulli(ℓ)
            ref = y * ℓ - log1p(exp(ℓ))
            @test ProbProg.logpdf(d, y) ≈ ref atol = 1.0e-12
        end
        # Numerical stability at large |ℓ|.
        @test ProbProg.logpdf(ProbProg.Bernoulli(50.0), 1) ≈ 0 atol = 1.0e-12
        @test isfinite(ProbProg.logpdf(ProbProg.Bernoulli(-50.0), 0))

        # Sampling lands in {0, 1}.
        rng = Random.default_rng()
        d = ProbProg.Bernoulli(0.3, (50,))
        for _ in 1:5
            x = ProbProg.Distributions.rand(rng, d)
            @test all(xi -> xi == 0 || xi == 1, x)
        end

        # Per-element logits.
        logits = [-2.0, 0.0, 2.0]
        d3 = ProbProg.Bernoulli(logits)
        @test ProbProg.logpdf(d3, [0.0, 1.0, 1.0]) ≈
            sum(([0.0, 1.0, 1.0] .* logits) .- log1p.(exp.(logits))) atol = 1.0e-12

        # Mean = sigmoid(logits).
        @test ProbProg.mean(ProbProg.Bernoulli(0.0)) ≈ 0.5 atol = 1.0e-12
        @test ProbProg.var(ProbProg.Bernoulli(0.0)) ≈ 0.25 atol = 1.0e-12
    end

    @testset "LogitNormal" begin
        d = ProbProg.LogitNormal(0.0, 1.0)
        @test ProbProg.params(d) == (0.0, 1.0)
        @test ProbProg.support(typeof(d)) == ProbProg.UnitIntervalSupport()
        # logpdf(LogitNormal(0,1), y) = -logit(y)²/2 - log(2π)/2 - log(y) - log(1-y).
        for y in (0.2, 0.5, 0.8)
            z = log(y / (1 - y))
            ref = -z^2 / 2 - log(2π) / 2 - log(y) - log(1 - y)
            @test ProbProg.logpdf(d, y) ≈ ref atol = 1.0e-12
        end
        # round-trip
        d2 = ProbProg.LogitNormal(0.3, 0.5)
        for y in (0.2, 0.5, 0.8)
            @test ProbProg.logpdf(d2, y) ≈
                ProbProg.unnormed_logpdf(d2, y) + ProbProg.lognorm(d2)
        end
    end

    # NumPyro cross-check. Compares the trait-API `logpdf_fn` (which is what
    # ProbProg's `Modeling.jl` calls under the hood) against NumPyro's
    # `log_prob` summed over a batch of points. Sampling-side comparison is
    # intentionally absent: our Gamma/InvGamma/TDist samplers use the
    # Wilson–Hilferty closed form for Reactant traceability, which doesn't
    # match JAX's rejection sampler bit-for-bit.
    #
    # Gated on `check_numpyro_available()` so this section is silently
    # skipped on non-CPU backends or when CondaPkg can't fetch NumPyro.
    @testset "NumPyro logpdf cross-check" begin
        if !check_numpyro_available()
            @info "Skipping NumPyro comparison tests (NumPyro unavailable)"
        else
            jnp = pyimport("jax.numpy")
            ndist = pyimport("numpyro.distributions")

            np_logpdf(d_py, xs) =
                pyconvert(Float64, d_py.log_prob(jnp.array(pylist(xs))).sum().item())

            @testset "Normal($μ, $σ)" for (μ, σ) in ((0.0, 1.0), (2.0, 0.5))
                xs = [0.5, 1.0, -0.3, 2.1, -1.5]
                lp = lp_via_jit(ProbProg.Normal(μ, σ), xs, (μ, σ))
                ref = np_logpdf(ndist.Normal(μ, σ), xs)
                @test lp ≈ ref atol = 1.0e-10
            end

            # Our `Exponential(θ)` is scale-parameterised: density (1/θ)·exp(-x/θ).
            # NumPyro's `Exponential(rate)` is rate-parameterised, so map θ ↔ 1/rate.
            @testset "Exponential(θ=$θ)" for θ in (0.5, 2.0)
                xs = [0.5, 1.0, 2.0, 0.1]
                lp = lp_via_jit(ProbProg.Exponential(θ), xs, (θ,))
                ref = np_logpdf(ndist.Exponential(; rate=1 / θ), xs)
                @test lp ≈ ref atol = 1.0e-10
            end

            @testset "Uniform($a, $b)" for (a, b) in ((-1.0, 3.0), (0.0, 1.0))
                xs = [a + 0.1, (a + b) / 2, b - 0.1]
                lp = lp_via_jit(ProbProg.Uniform(a, b), xs, (a, b))
                ref = np_logpdf(ndist.Uniform(a, b), xs)
                @test lp ≈ ref atol = 1.0e-10
            end

            # NumPyro's InverseGamma uses (concentration=α, rate=θ); ours uses
            # the same α/θ parameterisation, so the mapping is direct.
            @testset "InverseGamma(α=$α, θ=$θ)" for (α, θ) in ((3.0, 2.0), (5.0, 1.5))
                xs = [0.5, 1.0, 2.0, 5.0]
                lp = lp_via_jit(ProbProg.InverseGamma(α, θ), xs, (α, θ))
                ref = np_logpdf(ndist.InverseGamma(α, θ), xs)
                # Looser tolerance: our `_loggamma` is a single chlo.lgamma op,
                # NumPyro's path is bit-for-bit XLA so they should agree to
                # ~ULP, but allow some slack.
                @test lp ≈ ref atol = 1.0e-10
            end

            @testset "TDist(ν=$ν, μ=$μ, σ=$σ)" for (ν, μ, σ) in
                                                   ((5.0, 0.0, 1.0), (10.0, 1.0, 2.0))
                xs = [-1.5, -0.3, 0.0, 0.7, 2.1]
                lp = lp_via_jit(ProbProg.TDist(ν, μ, σ), xs, (ν, μ, σ))
                ref = np_logpdf(ndist.StudentT(ν, μ, σ), xs)
                @test lp ≈ ref atol = 1.0e-10
            end

            @testset "LogNormal($μ, $σ)" for (μ, σ) in ((0.0, 1.0), (1.0, 0.5))
                xs = [0.5, 1.0, 2.0, 3.0]
                lp = lp_via_jit(ProbProg.LogNormal(μ, σ), xs, (μ, σ))
                ref = np_logpdf(ndist.LogNormal(μ, σ), xs)
                @test lp ≈ ref atol = 1.0e-10
            end

            # NumPyro doesn't ship `LogitNormal` directly; build it explicitly
            # from a Normal base and a SigmoidTransform.
            @testset "LogitNormal($μ, $σ)" for (μ, σ) in ((0.0, 1.0), (0.3, 0.5))
                xs = [0.2, 0.5, 0.8]
                lp = lp_via_jit(ProbProg.LogitNormal(μ, σ), xs, (μ, σ))
                transforms = pyimport("numpyro.distributions.transforms")
                base = ndist.Normal(μ, σ)
                d_py = ndist.TransformedDistribution(base, transforms.SigmoidTransform())
                ref = np_logpdf(d_py, xs)
                @test lp ≈ ref atol = 1.0e-10
            end

            @testset "Bernoulli(logits=$ℓ)" for ℓ in (0.5, -2.0, 3.0)
                ys = [0.0, 1.0, 1.0, 0.0, 1.0]
                lp = lp_via_jit(ProbProg.Bernoulli(ℓ), ys, (ℓ,))
                ref = np_logpdf(ndist.BernoulliLogits(ℓ), ys)
                @test lp ≈ ref atol = 1.0e-10
            end
        end
    end
end
