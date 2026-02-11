using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

function standard_normal_logpdf(x)
    return -0.5 * sum(x .^ 2)
end

function logpdf_nuts_program(
    rng,
    logpdf_fn,
    initial_position,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
)
    samples, diagnostics, rng = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        initial_position;
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        max_tree_depth=10,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
    )
    return samples, diagnostics
end

@testset "mcmc_logpdf" begin
    @testset "adapt_step_size=$ass, adapt_mass_matrix=$amm" for ass in [false, true],
        amm in [false, true]

        seed = Reactant.to_rarray(UInt64[1, 5])
        rng = ReactantRNG(seed)

        pos_size = 2
        initial_position = Reactant.to_rarray(reshape([0.5, -0.5], 1, pos_size))
        step_size = ConcreteRNumber(0.1)
        inverse_mass_matrix = ConcreteRArray([0.5 0.0; 0.0 0.5])

        num_warmup = 200
        num_samples = 5

        compile_time_s = @elapsed begin
            compiled = @compile optimize = :probprog logpdf_nuts_program(
                rng,
                standard_normal_logpdf,
                initial_position,
                step_size,
                inverse_mass_matrix,
                num_warmup,
                num_samples,
                ass,
                amm,
            )
        end
        run_time_s = @elapsed begin
            samples, diagnostics = compiled(
                rng,
                standard_normal_logpdf,
                initial_position,
                step_size,
                inverse_mass_matrix,
                num_warmup,
                num_samples,
                ass,
                amm,
            )
            samples_arr = Array(samples)
            diagnostics_arr = Array(diagnostics)
        end
        @test size(samples_arr) == (num_samples, pos_size)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            numpyro_infer = pyimport("numpyro.infer")
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
                kernel; num_warmup=num_warmup, num_samples=num_samples, progress_bar=false
            )

            rng_key = seed_to_rbg_key(UInt64[1, 5])
            init_params = pydict(; x=jnp.array(pylist([0.5, -0.5])))

            mcmc_runner.run(rng_key; init_params=init_params)

            numpyro_samples = pyconvert(
                Matrix{Float64}, np.asarray(mcmc_runner.get_samples()["x"])
            )

            @testset "NumPyro pointwise comparison" begin
                max_abs_diff = maximum(abs.(samples_arr .- numpyro_samples))
                max_rel_diff = maximum(
                    abs.(samples_arr .- numpyro_samples) ./
                    max.(abs.(numpyro_samples), 1e-300),
                )
                @test samples_arr ≈ numpyro_samples atol = 1e-8 rtol = 1e-6
            end
        end
    end
end
