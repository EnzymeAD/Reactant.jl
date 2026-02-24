using Reactant, Test
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

function sample_normal(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, ProbProg.Normal(μ, σ, shape))
    return s
end

function sample_exponential(rng, λ, shape)
    _, s = ProbProg.sample(rng, ProbProg.Exponential(λ, shape))
    return s
end

function sample_lognormal(rng, μ, σ, shape)
    _, s = ProbProg.sample(rng, ProbProg.LogNormal(μ, σ, shape))
    return s
end

function sample_bernoulli(rng, logits, shape)
    _, s = ProbProg.sample(rng, ProbProg.Bernoulli(logits, shape))
    return s
end

function compute_normal_logpdf(x, μ, σ)
    fn = ProbProg.logpdf_fn(ProbProg.Normal)
    return fn(x, μ, σ, nothing)
end

function compute_exponential_logpdf(x, λ)
    fn = ProbProg.logpdf_fn(ProbProg.Exponential)
    return fn(x, λ, nothing)
end

function compute_lognormal_logpdf(x, μ, σ)
    fn = ProbProg.logpdf_fn(ProbProg.LogNormal)
    return fn(x, μ, σ, nothing)
end

function compute_bernoulli_logpdf(y, logits)
    fn = ProbProg.logpdf_fn(ProbProg.Bernoulli)
    return fn(y, logits, nothing)
end

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

    @testset "Samplers" begin
        @testset "Normal(2, 0.5)" begin
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            μ = ConcreteRNumber(2.0)
            σ = ConcreteRNumber(0.5)
            shape = (10,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_normal, μ, σ, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 10

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[1, 4])
                z = jax_random.normal(key, (10,); dtype=jnp.float64)
                jax_vals = pyconvert(Vector{Float64}, (2.0 + 0.5 * z).tolist())

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "Normal(0, 1)" begin
            seed = Reactant.to_rarray(UInt64[0, 42])
            rng = ReactantRNG(seed)
            μ = ConcreteRNumber(0.0)
            σ = ConcreteRNumber(1.0)
            shape = (5,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_normal, μ, σ, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 5

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[0, 42])
                jax_vals = pyconvert(
                    Vector{Float64},
                    jax_random.normal(key, (5,); dtype=jnp.float64).tolist(),
                )

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "Exponential(2)" begin
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            λ = ConcreteRNumber(2.0)
            shape = (10,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_exponential, λ, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 10
            @test all(reactant_vals .> 0)

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[1, 4])
                exp1 = jax_random.exponential(key, (10,); dtype=jnp.float64)
                jax_vals = pyconvert(Vector{Float64}, (exp1 / 2.0).tolist())

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "Exponential(0.5)" begin
            seed = Reactant.to_rarray(UInt64[0, 42])
            rng = ReactantRNG(seed)
            λ = ConcreteRNumber(0.5)
            shape = (5,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_exponential, λ, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 5
            @test all(reactant_vals .> 0)

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[0, 42])
                exp1 = jax_random.exponential(key, (5,); dtype=jnp.float64)
                jax_vals = pyconvert(Vector{Float64}, (exp1 / 0.5).tolist())

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "LogNormal(0, 1)" begin
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            μ = ConcreteRNumber(0.0)
            σ = ConcreteRNumber(1.0)
            shape = (10,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_lognormal, μ, σ, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 10
            @test all(reactant_vals .> 0)

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[1, 4])
                z = jax_random.normal(key, (10,); dtype=jnp.float64)
                jax_vals = pyconvert(Vector{Float64}, jnp.exp(z).tolist())

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "LogNormal(1, 0.5)" begin
            seed = Reactant.to_rarray(UInt64[0, 42])
            rng = ReactantRNG(seed)
            μ = ConcreteRNumber(1.0)
            σ = ConcreteRNumber(0.5)
            shape = (5,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_lognormal, μ, σ, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 5
            @test all(reactant_vals .> 0)

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[0, 42])
                z = jax_random.normal(key, (5,); dtype=jnp.float64)
                jax_vals = pyconvert(Vector{Float64}, jnp.exp(1.0 + 0.5 * z).tolist())

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "Bernoulli(logits=0)" begin
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            logits = ConcreteRArray(zeros(10))
            shape = (10,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_bernoulli, logits, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 10
            @test all(x -> x == 0.0 || x == 1.0, reactant_vals)

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[1, 4])
                probs = jnp.full(10, 0.5; dtype=jnp.float64)
                jax_vals = pyconvert(
                    Vector{Float64},
                    jax_random.bernoulli(key, probs).astype(jnp.float64).tolist(),
                )

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end

        @testset "Bernoulli(logits=[-2, 0, 2])" begin
            seed = Reactant.to_rarray(UInt64[1, 4])
            rng = ReactantRNG(seed)
            logits = ConcreteRArray([-2.0, 0.0, 2.0])
            shape = (3,)

            _, result = @jit optimize = :probprog ProbProg.untraced_call(
                rng, sample_bernoulli, logits, shape
            )
            reactant_vals = Array(result)
            @test length(reactant_vals) == 3
            @test all(x -> x == 0.0 || x == 1.0, reactant_vals)

            if check_numpyro_available()
                jax_random = pyimport("jax.random")
                jax_nn = pyimport("jax.nn")
                jnp = pyimport("jax.numpy")

                key = seed_to_rbg_key(UInt64[1, 4])
                logits_jax = jnp.array(pylist([-2.0, 0.0, 2.0]))
                probs = jax_nn.sigmoid(logits_jax)
                jax_vals = pyconvert(
                    Vector{Float64},
                    jax_random.bernoulli(key, probs).astype(jnp.float64).tolist(),
                )

                @test reactant_vals ≈ jax_vals atol = 1e-10
            end
        end
    end

    @testset "LogPDF" begin
        @testset "Normal(0, 1)" begin
            x = ConcreteRArray([0.5, 1.0, -0.3, 2.1, -1.5])
            μ = ConcreteRNumber(0.0)
            σ = ConcreteRNumber(1.0)

            result = @jit compute_normal_logpdf(x, μ, σ)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.Normal(0.0, 1.0)
                x_jnp = jnp.array(pylist([0.5, 1.0, -0.3, 2.1, -1.5]))
                numpyro_val = pyconvert(Float64, d.log_prob(x_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "Normal(2, 0.5)" begin
            x = ConcreteRArray([0.5, 1.0, -0.3])
            μ = ConcreteRNumber(2.0)
            σ = ConcreteRNumber(0.5)

            result = @jit compute_normal_logpdf(x, μ, σ)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.Normal(2.0, 0.5)
                x_jnp = jnp.array(pylist([0.5, 1.0, -0.3]))
                numpyro_val = pyconvert(Float64, d.log_prob(x_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "Exponential(2)" begin
            x = ConcreteRArray([0.5, 1.0, 2.0, 0.1])
            λ = ConcreteRNumber(2.0)

            result = @jit compute_exponential_logpdf(x, λ)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.Exponential(; rate=2.0)
                x_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 0.1]))
                numpyro_val = pyconvert(Float64, d.log_prob(x_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "Exponential(0.5)" begin
            x = ConcreteRArray([1.0, 3.0, 5.0])
            λ = ConcreteRNumber(0.5)

            result = @jit compute_exponential_logpdf(x, λ)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.Exponential(; rate=0.5)
                x_jnp = jnp.array(pylist([1.0, 3.0, 5.0]))
                numpyro_val = pyconvert(Float64, d.log_prob(x_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "LogNormal(0, 1)" begin
            x = ConcreteRArray([0.5, 1.0, 2.0, 3.0])
            μ = ConcreteRNumber(0.0)
            σ = ConcreteRNumber(1.0)

            result = @jit compute_lognormal_logpdf(x, μ, σ)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.LogNormal(0.0, 1.0)
                x_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 3.0]))
                numpyro_val = pyconvert(Float64, d.log_prob(x_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "LogNormal(1, 0.5)" begin
            x = ConcreteRArray([1.0, 2.0, 5.0])
            μ = ConcreteRNumber(1.0)
            σ = ConcreteRNumber(0.5)

            result = @jit compute_lognormal_logpdf(x, μ, σ)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.LogNormal(1.0, 0.5)
                x_jnp = jnp.array(pylist([1.0, 2.0, 5.0]))
                numpyro_val = pyconvert(Float64, d.log_prob(x_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "Bernoulli(logits=0.5)" begin
            y = ConcreteRArray([1.0, 0.0, 1.0, 0.0, 1.0])
            logits = ConcreteRNumber(0.5)

            result = @jit compute_bernoulli_logpdf(y, logits)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.Bernoulli(; logits=0.5)
                y_jnp = jnp.array(pylist([1.0, 0.0, 1.0, 0.0, 1.0]))
                numpyro_val = pyconvert(Float64, d.log_prob(y_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end

        @testset "Bernoulli(logits=-2.0)" begin
            y = ConcreteRArray([0.0, 0.0, 1.0])
            logits = ConcreteRNumber(-2.0)

            result = @jit compute_bernoulli_logpdf(y, logits)
            reactant_val = Reactant.to_number(result)

            if check_numpyro_available()
                jnp = pyimport("jax.numpy")
                dist = pyimport("numpyro.distributions")

                d = dist.Bernoulli(; logits=-2.0)
                y_jnp = jnp.array(pylist([0.0, 0.0, 1.0]))
                numpyro_val = pyconvert(Float64, d.log_prob(y_jnp).sum().item())

                @test reactant_val ≈ numpyro_val atol = 1e-10
            end
        end
    end
end
