using Reactant, Test
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

function logpdf_divinf(x)
    return -sum(min.(1.0, 1.0 ./ (x .^ 2)))
end

function run_nuts_sz(
    rng,
    logpdf_fn,
    initial_position,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
    strong_zero::Bool,
)
    samples, diagnostics, rng, _ = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        initial_position;
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        max_tree_depth=5,
        num_warmup,
        num_samples,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        strong_zero,
    )
    return samples, diagnostics
end

pos_size = 2
initial_position = Reactant.to_rarray([0.0, 0.0])
step_size = ConcreteRNumber(0.5)
inverse_mass_matrix = ConcreteRArray([1.0 0.0; 0.0 1.0])

@testset "mcmc_logpdf strong_zero" begin
    @testset "strong_zero=true: samples move away from singularity" begin
        rng = ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
        compiled = @compile optimize = :probprog run_nuts_sz(
            rng, logpdf_divinf, initial_position, step_size, inverse_mass_matrix, 0, 3, true
        )
        samples, diagnostics = compiled(
            rng, logpdf_divinf, initial_position, step_size, inverse_mass_matrix, 0, 3, true
        )
        samples_arr = Array(samples)
        @test size(samples_arr) == (3, pos_size)
        @test all(isfinite, samples_arr)
        @test !all(samples_arr .== 0.0)
    end

    @testset "strong_zero=false: stuck at singularity" begin
        rng = ReactantRNG(Reactant.to_rarray(UInt64[42, 0]))
        compiled = @compile optimize = :probprog run_nuts_sz(
            rng,
            logpdf_divinf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            0,
            3,
            false,
        )
        samples, diagnostics = compiled(
            rng,
            logpdf_divinf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            0,
            3,
            false,
        )
        samples_arr = Array(samples)
        @test size(samples_arr) == (3, pos_size)
        @test all(samples_arr .== 0.0)
    end
end
