using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

function standard_normal_logpdf(x)
    return -0.5 * sum(x .^ 2)
end

function warmup_program(
    rng,
    logpdf_fn,
    initial_position,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
)
    samples, diagnostics, _, state = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        initial_position;
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        max_tree_depth=10,
        num_warmup,
        num_samples,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
    return samples, diagnostics, state
end

function continuation_program(
    state_rng,
    logpdf_fn,
    state_position,
    state_gradient,
    state_potential_energy,
    state_step_size,
    state_inverse_mass_matrix,
    num_samples::Int,
)
    samples, diagnostics, _, state = ProbProg.mcmc_logpdf(
        ReactantRNG(state_rng),
        logpdf_fn,
        state_position;
        algorithm=:NUTS,
        step_size=state_step_size,
        inverse_mass_matrix=state_inverse_mass_matrix,
        initial_gradient=state_gradient,
        initial_potential_energy=state_potential_energy,
        max_tree_depth=10,
        num_warmup=0,
        num_samples,
        adapt_step_size=false,
        adapt_mass_matrix=false,
    )
    return samples, diagnostics
end

function combined_program(
    rng,
    logpdf_fn,
    initial_position,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
)
    samples, diagnostics, _, _ = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        initial_position;
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        max_tree_depth=10,
        num_warmup,
        num_samples,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
    return samples, diagnostics
end

@testset "mcmc_state" begin
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)
    fresh_rng() = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))

    pos_size = 2
    initial_position = Reactant.to_rarray(reshape([0.5, -0.5], 1, pos_size))
    step_size = ConcreteRNumber(0.1)
    inverse_mass_matrix = ConcreteRArray([0.5 0.0; 0.0 0.5])

    num_warmup = 200
    num_samples_warmup = 3
    num_samples_continue = 5

    @testset "state fields have correct shapes" begin
        compiled_warmup = @compile optimize = :probprog warmup_program(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        samples, diagnostics, state = compiled_warmup(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        @test size(Array(samples)) == (num_samples_warmup, pos_size)
        @test size(Array(state.position)) == (1, pos_size)
        @test size(Array(state.gradient)) == (1, pos_size)
        @test size(Array(state.potential_energy)) == ()
        @test size(Array(state.step_size)) == ()
        @test size(Array(state.inverse_mass_matrix)) == (pos_size, pos_size)
        @test size(Array(state.rng)) == (2,)

        adapted_ss = Array(state.step_size)[]
        @test adapted_ss > 0.0
        @test isfinite(adapted_ss)

        @test isfinite(Array(state.potential_energy)[])
    end

    @testset "continuation produces valid samples" begin
        compiled_warmup = @compile optimize = :probprog warmup_program(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        _, _, state = compiled_warmup(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        compiled_continue = @compile optimize = :probprog continuation_program(
            state.rng,
            standard_normal_logpdf,
            state.position,
            state.gradient,
            state.potential_energy,
            state.step_size,
            state.inverse_mass_matrix,
            num_samples_continue,
        )

        samples2, diagnostics2 = compiled_continue(
            state.rng,
            standard_normal_logpdf,
            state.position,
            state.gradient,
            state.potential_energy,
            state.step_size,
            state.inverse_mass_matrix,
            num_samples_continue,
        )

        samples2_arr = Array(samples2)
        @test size(samples2_arr) == (num_samples_continue, pos_size)
        @test all(isfinite, samples2_arr)
    end

    @testset "num_samples prefix determinism" begin
        total_samples = num_samples_warmup + num_samples_continue

        compiled_all = @compile optimize = :probprog combined_program(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            total_samples,
        )

        compiled_short = @compile optimize = :probprog combined_program(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        all_samples, _ = compiled_all(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            total_samples,
        )

        short_samples, _ = compiled_short(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        all_arr = Array(all_samples)
        short_arr = Array(short_samples)

        @test short_arr == all_arr[1:num_samples_warmup, :]
    end

    @testset "chunked continuation matches single-shot" begin
        total_samples = num_samples_warmup + num_samples_continue

        compiled_warmup = @compile optimize = :probprog warmup_program(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        samples1, _, state = compiled_warmup(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples_warmup,
        )

        compiled_continue = @compile optimize = :probprog continuation_program(
            state.rng,
            standard_normal_logpdf,
            state.position,
            state.gradient,
            state.potential_energy,
            state.step_size,
            state.inverse_mass_matrix,
            num_samples_continue,
        )

        samples2, _ = compiled_continue(
            state.rng,
            standard_normal_logpdf,
            state.position,
            state.gradient,
            state.potential_energy,
            state.step_size,
            state.inverse_mass_matrix,
            num_samples_continue,
        )

        compiled_combined = @compile optimize = :probprog combined_program(
            rng,
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            total_samples,
        )

        combined_samples, _ = compiled_combined(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            total_samples,
        )

        samples1_arr = Array(samples1)
        samples2_arr = Array(samples2)
        combined_arr = Array(combined_samples)

        @test samples1_arr == combined_arr[1:num_samples_warmup, :]
        @test samples2_arr == combined_arr[(num_samples_warmup + 1):total_samples, :]
    end
end
