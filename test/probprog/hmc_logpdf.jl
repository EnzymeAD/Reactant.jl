using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
using Reactant.ReactantCore: @trace

include(joinpath(@__DIR__, "common.jl"))

const HL_TAU = 2.0
const HL_Y = [
    1.3 -0.4
    0.7 2.1
    1.9 -1.8
    1.1 0.6
    0.4 1.5
]
const HL_S = [1.0, 2.0]
const HL_V = [1 / (1 / HL_TAU^2 + size(HL_Y, 1) / HL_S[j]^2) for j in 1:2]
const HL_M = [HL_V[j] * sum(HL_Y[:, j]) / HL_S[j]^2 for j in 1:2]
const HL_N_SIGMA = 5

# Data are passed transposed (Yt = Y', 2 x N) to broadcast against the length-2 position.
function hl_gauss_logpdf(theta, Yt, s)
    r = (Yt .- theta) ./ s
    return -0.5 * sum(theta .^ 2) / HL_TAU^2 - 0.5 * sum(r .* r)
end

function hl_gauss_logpdf_loop(theta, Yt, s)
    acc = -0.5 * sum(theta .^ 2) / HL_TAU^2
    @trace for i in 1:size(Yt, 2)
        r = (Yt[:, i] .- theta) ./ s
        acc = acc - 0.5 * sum(r .* r)
    end
    return acc
end

function hl_hmc_program(
    rng,
    logpdf_fn,
    initial_position,
    Y,
    s,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
)
    samples, diagnostics, _, rng, _ = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        initial_position,
        Y,
        s;
        algorithm=:HMC,
        step_size,
        inverse_mass_matrix,
        trajectory_length=1.0,
        num_warmup,
        num_samples,
        adapt_step_size,
        adapt_mass_matrix,
    )
    return samples, diagnostics
end

function hl_run(logpdf_fn, num_warmup, num_samples, ass, amm)
    rng = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))
    initial_position = Reactant.to_rarray(reshape([0.5, -0.5], 1, 2))
    Y = Reactant.to_rarray(permutedims(HL_Y))
    s = Reactant.to_rarray(HL_S)
    step_size = ConcreteRNumber(0.1)
    inverse_mass_matrix = ConcreteRArray([0.5 0.0; 0.0 0.5])
    compiled = @compile optimize = :probprog hl_hmc_program(
        rng,
        logpdf_fn,
        initial_position,
        Y,
        s,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
        ass,
        amm,
    )
    samples, diagnostics = compiled(
        rng,
        logpdf_fn,
        initial_position,
        Y,
        s,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
        ass,
        amm,
    )
    return Array(samples), Array(diagnostics)
end

function hl_batch_means_se(v::AbstractVector; nbatch::Int=50)
    b = length(v) ÷ nbatch
    bm = [mean(@view v[((k - 1) * b + 1):(k * b)]) for k in 1:nbatch]
    return std(bm) / sqrt(nbatch)
end

@testset "hmc_logpdf" begin
    @testset "adapt_step_size=$ass, adapt_mass_matrix=$amm" for ass in [false, true],
        amm in [false, true]

        num_warmup = 200
        num_samples = 5
        samples_arr, _ = hl_run(hl_gauss_logpdf, num_warmup, num_samples, ass, amm)
        @test size(samples_arr) == (num_samples, 2)
        @test all(isfinite, samples_arr)

        if check_numpyro_available()
            jnp = pyimport("jax.numpy")
            numpyro_infer = pyimport("numpyro.infer")
            np = pyimport("numpy")

            ns = pydict()
            pybuiltins.exec(
                "import jax.numpy as jnp\n" *
                "Y = jnp.array([[1.3, -0.4], [0.7, 2.1], [1.9, -1.8], [1.1, 0.6], [0.4, 1.5]])\n" *
                "S = jnp.array([[1.0, 2.0]])\n" *
                "def gauss_potential(z):\n" *
                "    x = z['x']\n" *
                "    r = (Y - x) / S\n" *
                "    return 0.5 * jnp.sum(x ** 2) / 4.0 + 0.5 * jnp.sum(r * r)\n",
                ns,
            )
            kernel = numpyro_infer.HMC(;
                potential_fn=ns["gauss_potential"],
                step_size=0.1,
                trajectory_length=1.0,
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
            mcmc_runner.run(
                seed_to_rbg_key(UInt64[1, 5]);
                init_params=pydict(; x=jnp.array(pylist([0.5, -0.5]))),
            )
            numpyro_samples = pyconvert(
                Matrix{Float64}, np.asarray(mcmc_runner.get_samples()["x"])
            )

            @testset "NumPyro pointwise comparison" begin
                @test samples_arr ≈ numpyro_samples atol = 1e-8 rtol = 1e-6
            end
        end
    end

    @testset "loop logpdf matches the closed form" begin
        closed, _ = hl_run(hl_gauss_logpdf, 200, 5, true, true)
        loop, _ = hl_run(hl_gauss_logpdf_loop, 200, 5, true, true)
        @test size(loop) == (5, 2)
        @test loop ≈ closed atol = 1e-8 rtol = 1e-6
    end

    # ESS floor: a chain that barely moves must not pass on an inflated standard error.
    @testset "posterior moments ($name)" for (name, f) in (
        ("closed form", hl_gauss_logpdf), ("loop", hl_gauss_logpdf_loop)
    )
        num_samples = 5000
        samples_arr, _ = hl_run(f, 500, num_samples, true, true)
        @test size(samples_arr) == (num_samples, 2)
        for j in 1:2
            x = samples_arr[:, j]
            se_m = hl_batch_means_se(x)
            @test var(x) / se_m^2 >= num_samples / 10
            @test abs(mean(x) - HL_M[j]) <= HL_N_SIGMA * se_m
            d2 = (x .- HL_M[j]) .^ 2
            @info "hmc_logpdf $name theta_$j" mean_hat = mean(x) m_true = HL_M[j] var_hat = mean(
                d2
            ) v_true = HL_V[j] ess = var(x) / se_m^2
            @test abs(mean(d2) - HL_V[j]) <= HL_N_SIGMA * hl_batch_means_se(d2)
        end
    end
end
