using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# Reference stick-breaking, term-for-term with numpyro StickBreakingTransform.
_log_sigmoid(t) = min(t, 0.0) - log1p(exp(-abs(t)))  # jax.nn.log_sigmoid

function _sb_constrain(x::Vector{Float64})
    n = length(x)
    K = n + 1
    shift = log.(Float64.(n:-1:1))
    z = 1.0 ./ (1.0 .+ exp.(-(x .- shift)))
    y = zeros(K)
    remaining = 1.0
    for k in 1:n
        y[k] = remaining * z[k]
        remaining *= (1.0 - z[k])
    end
    y[K] = remaining
    return y
end

function _sb_unconstrain(y::Vector{Float64})
    K = length(y)
    n = K - 1
    x = zeros(n)
    remaining = 1.0
    for k in 1:n
        zk = y[k] / remaining
        x[k] = (log(zk) - log(1.0 - zk)) + log(Float64(n - k + 1))
        remaining -= y[k]
    end
    return x
end

function _sb_logdet(x::Vector{Float64})
    n = length(x)
    shift = log.(Float64.(n:-1:1))
    xsh = x .- shift
    y = _sb_constrain(x)
    return sum(log.(y[1:n]) .+ (_log_sigmoid.(xsh) .- xsh))
end

@testset "Dirichlet + SIMPLEX" begin
    @testset "Construction" begin
        d = ProbProg.Dirichlet([1.0, 1.0, 1.0])
        @test ProbProg.params(d) == ([1.0, 1.0, 1.0], (3,))
        @test ProbProg.support(ProbProg.Dirichlet) == :simplex
        @test_throws ArgumentError ProbProg.Dirichlet([1.0, 1.0], ())
    end

    @testset "logpdf vs NumPyro / SciPy" begin
        logpdf_fn = ref_dirichlet_logpdf
        # (x, alpha, numpyro Dirichlet.log_prob, scipy.stats.dirichlet.logpdf)
        cases = [
            ([0.2, 0.3, 0.5], [1.0, 1.0, 1.0], 0.6931471805599441, 0.6931471805599453),
            ([0.1, 0.2, 0.7], [2.0, 3.0, 4.0], 1.5282105032788094, 1.5282105032788076),
            (
                [0.25, 0.25, 0.5],
                [0.5, 0.5, 0.5],
                -0.10500911500948296,
                -0.10500911500948185,
            ),
        ]
        for (xv, av, numpyro_ref, scipy_ref) in cases
            x = ConcreteRArray(xv)
            alpha = ConcreteRArray(av)
            lp = Reactant.to_number(@jit logpdf_fn(x, alpha, nothing))
            @test lp ≈ numpyro_ref atol = 1e-10
            @test lp ≈ scipy_ref atol = 1e-10
        end
    end

    @testset "batched (m>1) logpdf vs NumPyro / SciPy" begin
        logpdf_fn = ref_dirichlet_logpdf
        # (x (m,K), alpha (m,K), numpyro total, scipy total)
        cases = [
            (
                [0.2 0.3 0.5; 0.1 0.2 0.7; 0.25 0.25 0.5],
                [1.0 1.0 1.0; 2.0 3.0 4.0; 0.5 0.5 0.5],
                2.1163485688292702,
                2.116348568829271,
            ),
            (
                [0.3 0.3 0.4; 0.6 0.1 0.3],
                [1.0 1.0 1.0; 1.0 1.0 1.0],
                1.3862943611198881,
                1.3862943611198906,
            ),
            (
                [0.1 0.2 0.3 0.4; 0.4 0.3 0.2 0.1],
                [2.0 2.0 2.0 2.0; 1.0 2.0 3.0 4.0],
                1.479191741554379,
                1.4791917415543736,
            ),
        ]
        for (xv, av, numpyro_ref, scipy_ref) in cases
            x = ConcreteRArray(xv)
            alpha = ConcreteRArray(av)
            lp = Reactant.to_number(@jit logpdf_fn(x, alpha, nothing))
            @test lp ≈ numpyro_ref atol = 1e-10
            @test lp ≈ scipy_ref atol = 1e-10
        end
    end

    @testset "stick-breaking vs NumPyro" begin
        # numpyro StickBreakingTransform (JAX x64): x -> (T(x), logdet, T.inv(T(x))).
        fwd_cases = [
            (
                [0.0, 0.0],
                [0.3333333333333333, 0.33333333333333337, 0.33333333333333337],
                -3.295836866004329,
            ),
            (
                [1.3, -0.7],
                [0.6472225538269676, 0.11705587034349875, 0.23572157582953368],
                -4.0252729332527455,
            ),
            (
                [-2.0, 0.5],
                [0.06337893833303762, 0.5830085196347884, 0.35361254203217396],
                -4.337730634278863,
            ),
        ]
        for (xv, yref, ldref) in fwd_cases
            @test _sb_constrain(xv) ≈ yref atol = 1e-12
            @test _sb_logdet(xv) ≈ ldref atol = 1e-12
            @test _sb_unconstrain(_sb_constrain(xv)) ≈ xv atol = 1e-12
        end
        # T.inv on simplex points (numpyro references).
        inv_cases = [
            ([0.2, 0.3, 0.5], [-0.6931471805599453, -0.5108256237659907]),
            ([0.1, 0.2, 0.7], [-1.5040773967762737, -1.2527629684953678]),
        ]
        for (yv, xref) in inv_cases
            @test _sb_unconstrain(yv) ≈ xref atol = 1e-12
            @test _sb_constrain(_sb_unconstrain(yv)) ≈ yv atol = 1e-12
        end
    end

    @testset "sampler (distributional)" begin
        sample_fn = ProbProg.sampler(ProbProg.Dirichlet)
        rng = ReactantRNG(Reactant.to_rarray(UInt64[7, 11]))
        alpha = ConcreteRArray([2.0, 3.0, 5.0])
        draws = [Array(@jit sample_fn(rng, alpha, (3,))) for _ in 1:1]
        @test length(draws[1]) == 3
        @test all(draws[1] .> 0.0)
        @test sum(draws[1]) ≈ 1.0 atol = 1e-12
    end

    @testset "batched (m>1) sampler normalizes per block" begin
        sample_fn = ProbProg.sampler(ProbProg.Dirichlet)
        rng = ReactantRNG(Reactant.to_rarray(UInt64[7, 11]))
        alpha = ConcreteRArray([2.0 3.0 5.0; 1.0 1.0 1.0; 4.0 2.0 1.0])
        draw = Array(@jit sample_fn(rng, alpha, (3, 3)))
        @test size(draw) == (3, 3)
        @test all(draw .> 0.0)
        @test vec(sum(draw; dims=2)) ≈ ones(3) atol = 1e-12
    end
end

function _simplex_obs_sampler(rng, prev, counts)
    return Reactant.Ops.fill(0.0, 1)
end
function _simplex_obs_logpdf(y, prev, counts)
    return sum(counts .* log.(prev))
end
function simplex_model(rng, alpha0, counts)
    _, prev = ProbProg.sample(rng, ProbProg.Dirichlet(alpha0, (3,)); symbol=:prev)
    _, _ = ProbProg.sample(
        rng, _simplex_obs_sampler, prev, counts; symbol=:obs, logpdf=_simplex_obs_logpdf
    )
    return prev
end

function simplex_nuts_program(
    rng,
    model,
    alpha0,
    counts,
    step_size,
    inverse_mass_matrix,
    num_warmup,
    num_samples,
    constraint_tensor,
    constrained_addresses,
    selection,
)
    trace, _, _ = ProbProg.generate(
        rng, constraint_tensor, model, alpha0, counts; constrained_addresses
    )
    trace, diagnostics, _, _ = ProbProg.mcmc(
        rng,
        trace,
        model,
        alpha0,
        counts;
        selection,
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
        adapt_step_size=false,
        adapt_mass_matrix=false,
    )
    return trace, diagnostics
end

@testset "simplex NUTS vs NumPyro" begin
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)

    alpha0 = [1.0, 1.0, 1.0]
    counts = [10.0, 20.0, 30.0]
    post_alpha = alpha0 .+ counts
    post_mean = post_alpha ./ sum(post_alpha)

    obs = ProbProg.Constraint(:prev => [1 / 3, 1 / 3, 1 / 3], :obs => [0.0])
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)

    step_size = ConcreteRNumber(0.1)
    inverse_mass_matrix = ConcreteRArray([1.0, 1.0])  # unconstrained dim = K-1 = 2
    num_warmup = 0
    num_samples = 5
    selection = ProbProg.select(ProbProg.Address(:prev))

    alpha0_r = Reactant.to_rarray(alpha0)
    counts_r = Reactant.to_rarray(counts)

    compiled_fn, tt = ProbProg.with_trace() do
        @compile optimize = :probprog simplex_nuts_program(
            rng,
            simplex_model,
            alpha0_r,
            counts_r,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples,
            obs_tensor,
            constrained_addresses,
            selection,
        )
    end

    trace_tensor, diagnostics = compiled_fn(
        rng,
        simplex_model,
        alpha0_r,
        counts_r,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
        obs_tensor,
        constrained_addresses,
        selection,
    )
    trace_tensor = Array(trace_tensor)

    selected_entries = ProbProg.filter_entries_by_selection(tt.entries, selection)
    trace = ProbProg.unflatten_trace(trace_tensor, 0.0, selected_entries, nothing)

    prev_samples = trace.choices[:prev]
    @test size(prev_samples) == (num_samples, 3)
    @test all(abs.(vec(sum(prev_samples; dims=2)) .- 1.0) .< 1e-12)
    @test all(prev_samples .> 0.0)

    if check_numpyro_available()
        jnp = pyimport("jax.numpy")
        numpyro_mod = pyimport("numpyro")
        dist = pyimport("numpyro.distributions")
        np = pyimport("numpy")
        numpyro_infer = pyimport("numpyro.infer")

        numpyro_model = pyfunc(
            function (alpha0_arg, counts_arg)
                prev = numpyro_mod.sample("prev", dist.Dirichlet(alpha0_arg))
                return numpyro_mod.factor("obs", jnp.sum(counts_arg * jnp.log(prev)))
            end;
            name="model",
        )

        kernel = numpyro_infer.NUTS(
            numpyro_model;
            step_size=0.1,
            max_tree_depth=10,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            dense_mass=false,
            inverse_mass_matrix=jnp.array(pylist([1.0, 1.0])),
            find_heuristic_step_size=false,
            init_strategy=numpyro_infer.init_to_value(;
                values=pydict(; prev=jnp.array(pylist([1 / 3, 1 / 3, 1 / 3])))
            ),
        )
        mcmc = numpyro_infer.MCMC(
            kernel; num_warmup=0, num_samples=5, num_chains=1, progress_bar=false
        )
        mcmc.run(
            seed_to_rbg_key(UInt64[1, 5]),
            jnp.array(pylist(alpha0)),
            jnp.array(pylist(counts)),
        )
        numpyro_prev = pyconvert(
            Matrix{Float64}, np.asarray(mcmc.get_samples()["prev"]).reshape(5, 3)
        )
        @test prev_samples ≈ numpyro_prev atol = 1e-8 rtol = 1e-6
    else
        est_mean = vec(mean(prev_samples; dims=1))
        @test all(isfinite, est_mean)
    end
end
