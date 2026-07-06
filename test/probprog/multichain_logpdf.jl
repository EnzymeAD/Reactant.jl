using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

standard_normal_logpdf(x) = -0.5 * sum(x .^ 2)

function logpdf_multichain_program(
    rng,
    logpdf_fn,
    init_pos,
    step_size,
    imm,
    num_warmup::Int,
    num_samples::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
)
    samples, _, _, _, states = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        init_pos;
        num_chains=2,
        chain_method=:sequential,
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix=imm,
        max_tree_depth=10,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
    )
    keys = ProbProg._stack_chain_major([s.rng for s in states])
    return samples, keys
end

function logpdf_multichain_split_program(
    rng, logpdf_fn, init_pos, step_size, imm, M::Int, N::Int
)
    sA, _, _, _, statesA = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        init_pos;
        num_chains=2,
        chain_method=:sequential,
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix=imm,
        max_tree_depth=10,
        num_warmup=0,
        num_samples=M,
        adapt_step_size=false,
        adapt_mass_matrix=false,
    )
    sB, _, _, _, statesB = ProbProg.mcmc_logpdf(
        statesA, logpdf_fn; num_samples=N, adapt_step_size=false, adapt_mass_matrix=false
    )
    keysB = ProbProg._stack_chain_major([s.rng for s in statesB])
    return sA, sB, keysB
end

@testset "multichain_logpdf" begin
    # The thunk donates and mutates rng.seed, so each call needs a fresh key.
    fresh_rng() = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))
    pos_size = 2
    num_chains = 2
    init_pos = Reactant.to_rarray([0.5 -0.5; 0.5 -0.5])
    step_size = ConcreteRNumber(0.1)
    imm = ConcreteRArray([0.5 0.0; 0.0 0.5])

    @testset "parity adapt=($ass,$amm)" for (ass, amm) in [(false, false), (true, true)]
        num_warmup = 200
        num_samples = 5
        compiled = @compile optimize = :probprog logpdf_multichain_program(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos,
            step_size,
            imm,
            num_warmup,
            num_samples,
            ass,
            amm,
        )
        samples, keys = compiled(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos,
            step_size,
            imm,
            num_warmup,
            num_samples,
            ass,
            amm,
        )
        samples = Array(samples)
        keys = Array(keys)
        @test size(samples) == (num_chains, num_samples, pos_size)
        @test size(keys) == (num_chains, 2)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            numpyro_infer = pyimport("numpyro.infer")
            jax_random = pyimport("jax.random")
            np = pyimport("numpy")

            ns = pydict()
            pybuiltins.exec(
                "import jax.numpy as jnp\n" *
                "def standard_normal_potential(z):\n" *
                "    x = z['x']\n" *
                "    return 0.5 * jnp.sum(x ** 2)\n",
                ns,
            )
            potential_fn = ns["standard_normal_potential"]
            kernel = numpyro_infer.NUTS(;
                potential_fn=potential_fn,
                step_size=0.1,
                max_tree_depth=10,
                adapt_step_size=ass,
                adapt_mass_matrix=amm,
                dense_mass=true,
                inverse_mass_matrix=jnp.array(
                    pylist([pylist([0.5, 0.0]), pylist([0.0, 0.5])])
                ),
                find_heuristic_step_size=false,
            )
            mcmc_runner = numpyro_infer.MCMC(
                kernel;
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method="sequential",
                progress_bar=false,
            )
            rng_key = seed_to_rbg_key(UInt64[1, 5])
            init_params = pydict(;
                x=jnp.array(pylist([pylist([0.5, -0.5]), pylist([0.5, -0.5])]))
            )
            mcmc_runner.run(rng_key; init_params=init_params)
            np_samples = pyconvert(
                Array{Float64,3},
                np.asarray(mcmc_runner.get_samples(; group_by_chain=true)["x"]),
            )
            key_data = np.asarray(jax_random.key_data(mcmc_runner.last_state.rng_key))
            np_keys = pyconvert(
                Matrix{UInt64}, key_data.view(np.uint64).reshape(num_chains, 2)
            )

            @testset "chain $c" for c in 1:num_chains
                @test samples[c, :, :] ≈ np_samples[c, :, :] atol = 1e-8 rtol = 1e-6
                @test keys[c, :] == np_keys[c, :]
            end
        end
    end

    @testset "resumption M then N == monolithic M+N" begin
        M = 3
        N = 4
        mono = @compile optimize = :probprog logpdf_multichain_program(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos,
            step_size,
            imm,
            0,
            M + N,
            false,
            false,
        )
        mono_samples, mono_keys = mono(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos,
            step_size,
            imm,
            0,
            M + N,
            false,
            false,
        )
        mono_samples = Array(mono_samples)
        mono_keys = Array(mono_keys)

        split = @compile optimize = :probprog logpdf_multichain_split_program(
            fresh_rng(), standard_normal_logpdf, init_pos, step_size, imm, M, N
        )
        sA, sB, keysB = split(
            fresh_rng(), standard_normal_logpdf, init_pos, step_size, imm, M, N
        )
        sA = Array(sA)
        sB = Array(sB)
        keysB = Array(keysB)

        resumed = cat(sA, sB; dims=2)
        @test size(resumed) == size(mono_samples)
        @test maximum(abs.(resumed .- mono_samples)) == 0.0
        @test keysB == mono_keys
    end
end
