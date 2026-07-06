# Run standalone with julia -t4 for concurrent chains and a 2-device NumPyro pmap.
if !haskey(ENV, "XLA_FLAGS")
    ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
end

using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

standard_normal_logpdf(x) = -0.5 * sum(x .^ 2)

const SEED = UInt64[1, 5]
fresh_rng() = ReactantRNG(Reactant.to_rarray(copy(SEED)))

@testset "multichain_parallel" begin
    num_chains = 2
    pos_size = 2
    step_size = ConcreteRNumber(0.1)
    imm = ConcreteRArray([0.5 0.0; 0.0 0.5])
    init_pos() = Reactant.to_rarray([0.5 -0.5; 0.5 -0.5])

    @testset "chain_method validation" begin
        @test_throws ErrorException ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos();
            num_chains,
            chain_method=:vectorized,
            num_samples=1,
            progress_bar=false,
        )
        @test_throws ErrorException ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos();
            num_chains,
            chain_method=:bogus,
            num_samples=1,
            progress_bar=false,
        )
    end

    @testset "split matches sequential unroll key derivation" begin
        k1 = ProbProg.split_chain_rngs(fresh_rng(), num_chains)
        k2 = ProbProg.split_chain_rngs(fresh_rng(), num_chains)
        a1 = [Array(k.seed) for k in k1]
        a2 = [Array(k.seed) for k in k2]
        @test a1 == a2
        @test a1[1] != a1[2]
    end

    @testset "parallel == sequential bitwise, adapt=($ass,$amm)" for (ass, amm) in [
        (false, false), (true, true)
    ]
        num_warmup = 200
        num_samples = 5

        t_seq = @elapsed seq_samples, seq_states = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos();
            num_chains,
            chain_method=:sequential,
            algorithm=:NUTS,
            step_size,
            inverse_mass_matrix=imm,
            max_tree_depth=10,
            num_warmup,
            num_samples,
            adapt_step_size=ass,
            adapt_mass_matrix=amm,
            progress_bar=false,
        )
        t_par = @elapsed par_samples, par_states = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            init_pos();
            num_chains,
            chain_method=:parallel,
            algorithm=:NUTS,
            step_size,
            inverse_mass_matrix=imm,
            max_tree_depth=10,
            num_warmup,
            num_samples,
            adapt_step_size=ass,
            adapt_mass_matrix=amm,
            progress_bar=false,
        )

        @test size(par_samples) == (num_chains, num_samples, pos_size)
        @test size(par_samples) == size(seq_samples)
        @test maximum(abs.(par_samples .- seq_samples)) == 0.0
        for c in 1:num_chains
            @test Array(par_states[c].rng) == Array(seq_states[c].rng)
        end
        @info "multichain_parallel wall-clock adapt=($ass,$amm): sequential=$(round(t_seq; digits=3))s parallel=$(round(t_par; digits=3))s (nthreads=$(Threads.nthreads()); includes compile)"

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            numpyro_infer = pyimport("numpyro.infer")
            jax = pyimport("jax")
            jax_random = pyimport("jax.random")
            np = pyimport("numpy")

            ndev = pyconvert(Int, jax.local_device_count())
            @info "NumPyro chain_method=parallel with local_device_count=$ndev (genuine pmap requires >= $num_chains)"

            ns = pydict()
            pybuiltins.exec(
                "import jax.numpy as jnp\n" *
                "def standard_normal_potential(z):\n" *
                "    x = z['x']\n" *
                "    return 0.5 * jnp.sum(x ** 2)\n",
                ns,
            )
            kernel = numpyro_infer.NUTS(;
                potential_fn=ns["standard_normal_potential"],
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
                chain_method="parallel",
                progress_bar=false,
            )
            rng_key = seed_to_rbg_key(SEED)
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

            @testset "chain $c vs NumPyro parallel" for c in 1:num_chains
                @test par_samples[c, :, :] ≈ np_samples[c, :, :] atol = 1e-8 rtol = 1e-6
                @test Array(par_states[c].rng) == np_keys[c, :]
            end
        end
    end
end
