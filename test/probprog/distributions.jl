using Reactant, Test
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
using SpecialFunctions: loggamma

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "studentt_numpyro_reference_values.jl"))

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
        sample_fn = ref_normal_sampler
        logpdf_fn = ref_normal_logpdf

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
        sample_fn = ref_normal_sampler
        logpdf_fn = ref_normal_logpdf

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
        sample_fn = ref_exponential_sampler
        logpdf_fn = ref_exponential_logpdf

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
        sample_fn = ref_exponential_sampler
        logpdf_fn = ref_exponential_logpdf

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
        sample_fn = ref_lognormal_sampler
        logpdf_fn = ref_lognormal_logpdf

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
        sample_fn = ref_lognormal_sampler
        logpdf_fn = ref_lognormal_logpdf

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

    @testset "Normal(0, vector σ) [heteroscedastic]" begin
        logpdf_fn = ref_normal_logpdf

        x = ConcreteRArray([0.5, 1.0, -0.3, 2.1, -1.5])
        μ = ConcreteRNumber(0.0)
        σ = ConcreteRArray([0.5, 1.0, 2.0, 0.3, 1.5])

        lp = @jit logpdf_fn(x, μ, σ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            x_jnp = jnp.array(pylist([0.5, 1.0, -0.3, 2.1, -1.5]))
            σ_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 0.3, 1.5]))
            d = dist.Normal(0.0, σ_jnp)
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "LogNormal(0, vector σ) [heteroscedastic]" begin
        logpdf_fn = ref_lognormal_logpdf

        x = ConcreteRArray([0.5, 1.0, 1.5, 2.0, 3.0])
        μ = ConcreteRNumber(0.0)
        σ = ConcreteRArray([0.5, 1.0, 2.0, 0.3, 1.5])

        lp = @jit logpdf_fn(x, μ, σ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            x_jnp = jnp.array(pylist([0.5, 1.0, 1.5, 2.0, 3.0]))
            σ_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 0.3, 1.5]))
            d = dist.LogNormal(0.0, σ_jnp)
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Exponential(vector λ) [heteroscedastic]" begin
        logpdf_fn = ref_exponential_logpdf

        x = ConcreteRArray([0.5, 1.0, 2.0, 0.1, 3.0])
        λ = ConcreteRArray([2.0, 0.5, 1.0, 3.0, 0.25])

        lp = @jit logpdf_fn(x, λ, nothing)
        reactant_lp = Reactant.to_number(lp)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            dist = pyimport("numpyro.distributions")

            x_jnp = jnp.array(pylist([0.5, 1.0, 2.0, 0.1, 3.0]))
            λ_jnp = jnp.array(pylist([2.0, 0.5, 1.0, 3.0, 0.25]))
            d = dist.Exponential(; rate=λ_jnp)
            numpyro_lp = pyconvert(Float64, d.log_prob(x_jnp).sum().item())
            @test reactant_lp ≈ numpyro_lp atol = 1e-10
        end
    end

    @testset "Bernoulli(logits=0.5)" begin
        sample_fn = ref_bernoulli_sampler
        logpdf_fn = ref_bernoulli_logpdf

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
        sample_fn = ref_bernoulli_sampler
        logpdf_fn = ref_bernoulli_logpdf

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

    # Reference values come from studentt_numpyro_reference.py.
    @testset "StudentT / HalfStudentT pointwise NumPyro parity" begin
        ulps(a, b) =
            if a == b
                0
            else
                (
                    if signbit(a) == signbit(b)
                        abs(reinterpret(Int64, a) - reinterpret(Int64, b))
                    else
                        typemax(Int64)
                    end
                )
            end

        function emitted_logpdf(model, nparams)
            obs = ProbProg.Constraint(:x => [1.0])
            ca = ProbProg.extract_addresses(obs)
            rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 4]))
            function prog(rng, ot, a...)
                return ProbProg.generate(rng, ot, model, a...; constrained_addresses=ca)
            end
            ps = ntuple(_ -> ConcreteRNumber(1.0), nparams)
            compiled = @compile optimize = :probprog prog(
                rng, ProbProg.flatten_constraint(obs), ps...
            )
            return (x, p...) -> begin
                ot = ProbProg.flatten_constraint(ProbProg.Constraint(:x => [x]))
                _, w, _ = compiled(rng, ot, map(ConcreteRNumber, p)...)
                Array(w)[]
            end
        end

        @testset "HalfStudentT construction" begin
            d = ProbProg.HalfStudentT(3.0, 10.0)
            @test ProbProg.params(d) == (3.0, 10.0, (1,))
            @test ProbProg.support(ProbProg.HalfStudentT) == :positive
            @test ProbProg.sampler_traced(ProbProg.HalfStudentT)
            @test ProbProg.dist_spec(ProbProg.HalfStudentT) ==
                "#impulse.dist<HALF_STUDENT_T>"
            @test_throws ArgumentError ProbProg.HalfStudentT(3.0, 10.0, ())
        end

        @testset "HalfStudentT == TruncatedDistribution(StudentT(df, 0, scale), low=0)" begin
            hst(rng, df, sc) =
                ProbProg.sample(rng, ProbProg.HalfStudentT(df, sc, (1,)); symbol=:x)[2]
            f = emitted_logpdf(hst, 2)
            nbits = 0
            for (df, sc, x, ref) in HALF_STUDENT_T_NUMPYRO_REF
                v = f(x, df, sc)
                nbits += v === ref
                v === ref || @info "HalfStudentT mismatch" df sc x ref v ulps(v, ref)
            end
            @test nbits == length(HALF_STUDENT_T_NUMPYRO_REF)
        end

        @testset "StudentT == numpyro StudentT(df, loc, scale)" begin
            st(rng, df, lc, sc) =
                ProbProg.sample(rng, ProbProg.StudentT(df, lc, sc, (1,)); symbol=:x)[2]
            f = emitted_logpdf(st, 3)
            nbits = 0
            for (df, lc, sc, x, ref) in STUDENT_T_NUMPYRO_REF
                v = f(x, df, lc, sc)
                nbits += v === ref
                v === ref || @info "StudentT mismatch" df lc sc x ref v ulps(v, ref)
            end
            @test nbits == length(STUDENT_T_NUMPYRO_REF)
        end

        @testset "HalfStudentT sampler" begin
            n = 200_000
            df, sc = 10.0, 2.0
            rng = ReactantRNG(Reactant.to_rarray(UInt64[3, 17]))
            draws = Array(
                @jit ProbProg.sampler(ProbProg.HalfStudentT)(
                    rng, ConcreteRNumber(df), ConcreteRNumber(sc), (n,)
                )
            )
            rng2 = ReactantRNG(Reactant.to_rarray(UInt64[3, 17]))
            t_draws = Array(
                @jit ProbProg.sampler(ProbProg.StudentT)(
                    rng2,
                    ConcreteRNumber(df),
                    ConcreteRNumber(0.0),
                    ConcreteRNumber(sc),
                    (n,),
                )
            )
            @test draws == abs.(t_draws)
            @test all(>(0.0), draws)
            m1 =
                2 * sc * sqrt(df / π) * exp(loggamma((df + 1) / 2) - loggamma(df / 2)) /
                (df - 1)
            m2 = sc^2 * df / (df - 2)
            m4 = sc^4 * 3 * df^2 / ((df - 2) * (df - 4))
            se1 = sqrt((m2 - m1^2) / n)
            se2 = sqrt((m4 - m2^2) / n)
            @test abs(sum(draws) / n - m1) < 5 * se1
            @test abs(sum(draws .^ 2) / n - m2) < 5 * se2
        end
    end
end
