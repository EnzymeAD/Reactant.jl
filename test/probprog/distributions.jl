using Reactant, Test, LinearAlgebra
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
        end

        @testset "Exponential" begin
            d = ProbProg.Exponential()
            @test ProbProg.params(d) == (1.0, (1,))

            d2 = ProbProg.Exponential(2.0)
            @test ProbProg.params(d2) == (2.0, (1,))

            d3 = ProbProg.Exponential(0.5, (5,))
            @test ProbProg.params(d3) == (0.5, (5,))

            @test ProbProg.support(ProbProg.Exponential) == :positive
        end

        @testset "LogNormal" begin
            d = ProbProg.LogNormal()
            @test ProbProg.params(d) == (0.0, 1.0, (1,))

            d2 = ProbProg.LogNormal(1.0, 0.5)
            @test ProbProg.params(d2) == (1.0, 0.5, (1,))

            @test ProbProg.support(ProbProg.LogNormal) == :positive
        end

        @testset "Bernoulli" begin
            d = ProbProg.Bernoulli(0.5, (1,))
            @test ProbProg.params(d) == (0.5, (1,))
            @test ProbProg.support(ProbProg.Bernoulli) == :real
        end

        @testset "MultiNormal" begin
            μ = zeros(3)
            Σ = Matrix{Float64}(I, 3, 3)
            d = ProbProg.MultiNormal(μ, Σ, (3,))
            @test ProbProg.params(d) == (μ, Σ, (3,))
            @test ProbProg.support(ProbProg.MultiNormal) == :real
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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            d = dist.Normal(0.0, 1.0)
            jax_vals = pyconvert(Vector{Float64}, d.sample(key, (5,)).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            d = dist.Normal(2.0, 0.5)
            jax_vals = pyconvert(Vector{Float64}, d.sample(key, (10,)).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            d = dist.Exponential(; rate=2.0)
            jax_vals = pyconvert(Vector{Float64}, d.sample(key, (10,)).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            d = dist.Exponential(; rate=0.5)
            jax_vals = pyconvert(Vector{Float64}, d.sample(key, (5,)).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            d = dist.LogNormal(0.0, 1.0)
            jax_vals = pyconvert(Vector{Float64}, d.sample(key, (10,)).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            d = dist.LogNormal(1.0, 0.5)
            jax_vals = pyconvert(Vector{Float64}, d.sample(key, (5,)).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            d = dist.Bernoulli(; logits=jnp.zeros(10; dtype=jnp.float64))
            jax_vals = pyconvert(
                Vector{Float64}, d.sample(key).astype(jnp.float64).tolist()
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
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            d = dist.Bernoulli(; logits=jnp.array(pylist([-2.0, 0.0, 2.0])))
            jax_vals = pyconvert(
                Vector{Float64}, d.sample(key).astype(jnp.float64).tolist()
            )
            @test reactant_vals ≈ jax_vals atol = 1e-10

            d = dist.Bernoulli(; logits=-2.0)
            y_jnp = jnp.array(pylist([0.0, 0.0, 1.0]))
            numpyro_lp = pyconvert(Float64, d.log_prob(y_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "MultiNormal (identity cov)" begin
        sample_fn = ProbProg.sampler(ProbProg.MultiNormal)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.MultiNormal)

        n = 3
        rng = ReactantRNG(Reactant.to_rarray(UInt64[0, 42]))
        μ = ConcreteRArray(zeros(n))
        Σ = ConcreteRArray(Matrix{Float64}(LinearAlgebra.I, n, n))

        result = @jit sample_fn(rng, μ, Σ, (n,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == n

        x = ConcreteRArray([0.5, -0.3, 1.2])
        lp = @jit logpdf_fn(x, μ, Σ, (n,))
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[0, 42])
            d = dist.MultivariateNormal(
                jnp.zeros(n; dtype=jnp.float64), jnp.eye(n; dtype=jnp.float64)
            )
            jax_vals = pyconvert(Vector{Float64}, d.sample(key).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            x_jnp = jnp.array(pylist([0.5, -0.3, 1.2]); dtype=jnp.float64)
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "MultiNormal (non-trivial cov)" begin
        sample_fn = ProbProg.sampler(ProbProg.MultiNormal)
        logpdf_fn = ProbProg.logpdf_fn(ProbProg.MultiNormal)

        n = 3
        rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
        μ = ConcreteRArray([1.0, -0.5, 2.0])
        A = [0.5 0.1 0.0; 0.2 0.8 0.3; 0.0 0.1 0.6]
        Σ_val = A' * A + Matrix{Float64}(LinearAlgebra.I, n, n)
        Σ = ConcreteRArray(Σ_val)

        result = @jit sample_fn(rng, μ, Σ, (n,))
        reactant_vals = Array(result)
        @test length(reactant_vals) == n

        x = ConcreteRArray([0.8, 0.2, 1.5])
        lp = @jit logpdf_fn(x, μ, Σ, (n,))
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            key = seed_to_rbg_key(UInt64[1, 4])
            μ_jnp = jnp.array(pylist([1.0, -0.5, 2.0]); dtype=jnp.float64)
            d = dist.MultivariateNormal(μ_jnp, jnp.array(Σ_val; dtype=jnp.float64))
            jax_vals = pyconvert(Vector{Float64}, d.sample(key).tolist())
            @test reactant_vals ≈ jax_vals atol = 1e-10

            x_jnp = jnp.array(pylist([0.8, 0.2, 1.5]); dtype=jnp.float64)
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end
end
