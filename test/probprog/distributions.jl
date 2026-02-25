using Reactant, Test
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

@testset "Distributions" begin
    @testset "Construction" begin
        @testset "Normal" begin
            d = ProbProg.Normal()
            @test ProbProg.params(d) == (0.0, 1.0, (1,))

            d2 = ProbProg.Normal(2.0, 0.5)
            @test ProbProg.params(d2) == (2.0, 0.5, (1,))

            d3 = ProbProg.Normal(0.0, 1.0, (3, 4))
            @test ProbProg.params(d3) == (0.0, 1.0, (3, 4))

            @test ProbProg.support(ProbProg.Normal) == :real
            @test_throws ArgumentError ProbProg.Normal(0.0, 1.0, ())
        end

        @testset "Exponential" begin
            d = ProbProg.Exponential()
            @test ProbProg.params(d) == (1.0, (1,))

            d2 = ProbProg.Exponential(2.0)
            @test ProbProg.params(d2) == (2.0, (1,))

            d3 = ProbProg.Exponential(0.5, (5,))
            @test ProbProg.params(d3) == (0.5, (5,))

            @test ProbProg.support(ProbProg.Exponential) == :positive
            @test_throws ArgumentError ProbProg.Exponential(1.0, ())
        end

        @testset "LogNormal" begin
            d = ProbProg.LogNormal()
            @test ProbProg.params(d) == (0.0, 1.0, (1,))

            d2 = ProbProg.LogNormal(1.0, 0.5)
            @test ProbProg.params(d2) == (1.0, 0.5, (1,))

            @test ProbProg.support(ProbProg.LogNormal) == :positive
            @test_throws ArgumentError ProbProg.LogNormal(0.0, 1.0, ())
        end

        @testset "Bernoulli" begin
            d = ProbProg.Bernoulli(0.5, (1,))
            @test ProbProg.params(d) == (0.5, (1,))

            @test ProbProg.support(ProbProg.Bernoulli) == :real
            @test_throws ArgumentError ProbProg.Bernoulli(0.5, ())
        end
    end

    @testset "Normal(0, 1)" begin
        sample_fn = ProbProg.sampler(ProbProg.Normal)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.Normal)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[0, 42]))
        μ = ConcreteRNumber(0.0)
        σ = ConcreteRNumber(1.0)

        result = @jit sample_fn(rng, μ, σ, (5,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 5

        x = ConcreteRArray([0.5, 1.0, -0.3, 2.1, -1.5])
        lp = @jit logpdf_fn(x, μ, σ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            jax_vals = pyconvert(
                Vector{Float64}, jax_random.normal(key, (5,); dtype=jnp.float64).tolist()
            )
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Normal(0.0, 1.0)
            x_jnp = jnp.array(pylist([0.5, 1.0, -0.3, 2.1, -1.5]))
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Normal(2, 0.5)" begin
        sample_fn = ProbProg.sampler(ProbProg.Normal)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.Normal)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        μ = ConcreteRNumber(2.0)
        σ = ConcreteRNumber(0.5)

        result = @jit sample_fn(rng, μ, σ, (10,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 10

        x = ConcreteRArray([0.5, 1.0, -0.3])
        lp = @jit logpdf_fn(x, μ, σ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            z = jax_random.normal(key, (10,); dtype=jnp.float64)
            jax_vals = pyconvert(Vector{Float64}, (2.0 + 0.5 * z).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Normal(2.0, 0.5)
            x_jnp = jnp.array(pylist([0.5, 1.0, -0.3]))
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Exponential(2)" begin
        sample_fn = ProbProg.sampler(ProbProg.Exponential)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.Exponential)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        λ = ConcreteRNumber(2.0)

        result = @jit sample_fn(rng, λ, (10,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 10
        @test all(reactant_vals .> 0)

        x = ConcreteRArray([0.5, 1.0, 2.0, 0.1])
        lp = @jit logpdf_fn(x, λ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            exp1 = jax_random.exponential(key, (10,); dtype=jnp.float64)
            jax_vals = pyconvert(Vector{Float64}, (exp1 / 2.0).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Exponential(; rate=2.0)
            x_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 0.1]))
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Exponential(0.5)" begin
        sample_fn = ProbProg.sampler(ProbProg.Exponential)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.Exponential)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[0, 42]))
        λ = ConcreteRNumber(0.5)

        result = @jit sample_fn(rng, λ, (5,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 5
        @test all(reactant_vals .> 0)

        x = ConcreteRArray([1.0, 3.0, 5.0])
        lp = @jit logpdf_fn(x, λ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            exp1 = jax_random.exponential(key, (5,); dtype=jnp.float64)
            jax_vals = pyconvert(Vector{Float64}, (exp1 / 0.5).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Exponential(; rate=0.5)
            x_jnp = jnp.array(pylist([1.0, 3.0, 5.0]))
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "LogNormal(0, 1)" begin
        sample_fn = ProbProg.sampler(ProbProg.LogNormal)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.LogNormal)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        μ = ConcreteRNumber(0.0)
        σ = ConcreteRNumber(1.0)

        result = @jit sample_fn(rng, μ, σ, (10,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 10
        @test all(reactant_vals .> 0)

        x = ConcreteRArray([0.5, 1.0, 2.0, 3.0])
        lp = @jit logpdf_fn(x, μ, σ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            z = jax_random.normal(key, (10,); dtype=jnp.float64)
            jax_vals = pyconvert(Vector{Float64}, jnp.exp(z).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.LogNormal(0.0, 1.0)
            x_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 3.0]))
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "LogNormal(1, 0.5)" begin
        sample_fn = ProbProg.sampler(ProbProg.LogNormal)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.LogNormal)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[0, 42]))
        μ = ConcreteRNumber(1.0)
        σ = ConcreteRNumber(0.5)

        result = @jit sample_fn(rng, μ, σ, (5,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 5
        @test all(reactant_vals .> 0)

        x = ConcreteRArray([1.0, 2.0, 5.0])
        lp = @jit logpdf_fn(x, μ, σ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            z = jax_random.normal(key, (5,); dtype=jnp.float64)
            jax_vals = pyconvert(Vector{Float64}, jnp.exp(1.0 + 0.5 * z).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.LogNormal(1.0, 0.5)
            x_jnp = jnp.array(pylist([1.0, 2.0, 5.0]))
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Bernoulli(logits=0.5)" begin
        sample_fn = ProbProg.sampler(ProbProg.Bernoulli)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.Bernoulli)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        logits = ConcreteRArray(zeros(10))

        result = @jit sample_fn(rng, logits, (10,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 10
        @test all(x -> x == 0.0 || x == 1.0, reactant_vals)

        y = ConcreteRArray([1.0, 0.0, 1.0, 0.0, 1.0])
        logits_scalar = ConcreteRNumber(0.5)
        lp = @jit logpdf_fn(y, logits_scalar, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            probs = jnp.full(10, 0.5; dtype=jnp.float64)
            jax_vals = pyconvert(
                Vector{Float64},
                jax_random.bernoulli(key, probs).astype(jnp.float64).tolist(),
            )
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Bernoulli(; logits=0.5)
            y_jnp = jnp.array(pylist([1.0, 0.0, 1.0, 0.0, 1.0]))
            numpyro_lp = pyconvert(Float64, d.log_prob(y_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Bernoulli(logits=-2)" begin
        sample_fn = ProbProg.sampler(ProbProg.Bernoulli)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.Bernoulli)

        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        logits = ConcreteRArray([-2.0, 0.0, 2.0])

        result = @jit sample_fn(rng, logits, (3,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == 3
        @test all(x -> x == 0.0 || x == 1.0, reactant_vals)

        y = ConcreteRArray([0.0, 0.0, 1.0])
        logits_scalar = ConcreteRNumber(-2.0)
        lp = @jit logpdf_fn(y, logits_scalar, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jax_random = pyimport("jax.random")
            jax_nn = pyimport("jax.nn")
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            logits_jax = jnp.array(pylist([-2.0, 0.0, 2.0]))
            probs = jax_nn.sigmoid(logits_jax)
            jax_vals = pyconvert(
                Vector{Float64},
                jax_random.bernoulli(key, probs).astype(jnp.float64).tolist(),
            )
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Bernoulli(; logits=-2.0)
            y_jnp = jnp.array(pylist([0.0, 0.0, 1.0]))
            numpyro_lp = pyconvert(Float64, d.log_prob(y_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end
end
