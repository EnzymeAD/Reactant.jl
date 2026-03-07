using Reactant, Test, Random
using Statistics, Serialization
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

function continuation_with_state_program(
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
    return samples, diagnostics, state
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

    @testset "save and load state" begin
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

        tmpfile = tempname()
        try
            ProbProg.save_state(tmpfile, state)
            loaded = ProbProg.load_state(tmpfile)

            @test Array(loaded.position) == Array(state.position)
            @test Array(loaded.gradient) == Array(state.gradient)
            @test Array(loaded.potential_energy) == Array(state.potential_energy)
            @test Array(loaded.step_size) == Array(state.step_size)
            @test Array(loaded.inverse_mass_matrix) == Array(state.inverse_mass_matrix)
            @test Array(loaded.rng) == Array(state.rng)

            compiled_continue = @compile optimize = :probprog continuation_program(
                loaded.rng,
                standard_normal_logpdf,
                loaded.position,
                loaded.gradient,
                loaded.potential_energy,
                loaded.step_size,
                loaded.inverse_mass_matrix,
                num_samples_continue,
            )

            samples_from_loaded, _ = compiled_continue(
                loaded.rng,
                standard_normal_logpdf,
                loaded.position,
                loaded.gradient,
                loaded.potential_energy,
                loaded.step_size,
                loaded.inverse_mass_matrix,
                num_samples_continue,
            )

            samples_from_orig, _ = compiled_continue(
                state.rng,
                standard_normal_logpdf,
                state.position,
                state.gradient,
                state.potential_energy,
                state.step_size,
                state.inverse_mass_matrix,
                num_samples_continue,
            )

            @test Array(samples_from_loaded) == Array(samples_from_orig)
        finally
            rm(tmpfile; force=true)
        end
    end

    @testset "chunked sampling loop" begin
        num_chunks = 3
        chunk_size = 2
        total_samples = num_samples_warmup + num_chunks * chunk_size

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

        compiled_chunk = @compile optimize = :probprog continuation_with_state_program(
            state.rng,
            standard_normal_logpdf,
            state.position,
            state.gradient,
            state.potential_energy,
            state.step_size,
            state.inverse_mass_matrix,
            chunk_size,
        )

        all_chunks = [Array(samples1)]
        for i in 1:num_chunks
            chunk_samples, _, state = compiled_chunk(
                state.rng,
                standard_normal_logpdf,
                state.position,
                state.gradient,
                state.potential_energy,
                state.step_size,
                state.inverse_mass_matrix,
                chunk_size,
            )
            push!(all_chunks, Array(chunk_samples))
        end
        chunked_result = vcat(all_chunks...)

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

        @test chunked_result == Array(combined_samples)
    end

    @testset "run_chain fresh start" begin
        total_samples = 8

        samples, state = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position;
            algorithm=:NUTS,
            num_warmup,
            num_samples=total_samples,
            chunk_size=3,
            step_size,
            inverse_mass_matrix,
            progress_bar=false,
            max_tree_depth=10,
        )

        @test size(samples) == (total_samples, pos_size)
        @test all(isfinite, samples)
        @test size(Array(state.position)) == (1, pos_size)

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

        @test samples == Array(combined_samples)
    end

    @testset "run_chain continuation from state" begin
        _, warmup_state = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position;
            num_warmup,
            num_samples=num_samples_warmup,
            chunk_size=num_samples_warmup,
            step_size,
            inverse_mass_matrix,
            progress_bar=false,
        )

        samples2, state2 = ProbProg.run_chain(
            warmup_state,
            standard_normal_logpdf;
            num_samples=num_samples_continue,
            chunk_size=2,
            progress_bar=false,
        )

        @test size(samples2) == (num_samples_continue, pos_size)
        @test all(isfinite, samples2)
        @test size(Array(state2.position)) == (1, pos_size)
    end

    @testset "run_chain save/load round-trip" begin
        _, state = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position;
            num_warmup,
            num_samples=num_samples_warmup,
            chunk_size=num_samples_warmup,
            step_size,
            inverse_mass_matrix,
            progress_bar=false,
        )

        tmpfile = tempname()
        try
            ProbProg.save_state(tmpfile, state)
            loaded = ProbProg.load_state(tmpfile)

            samples_orig, _ = ProbProg.run_chain(
                state,
                standard_normal_logpdf;
                num_samples=num_samples_continue,
                chunk_size=num_samples_continue,
                progress_bar=false,
            )
            samples_loaded, _ = ProbProg.run_chain(
                loaded,
                standard_normal_logpdf;
                num_samples=num_samples_continue,
                chunk_size=num_samples_continue,
                progress_bar=false,
            )

            @test samples_orig == samples_loaded
        finally
            rm(tmpfile; force=true)
        end
    end

    @testset "chunked disk workflow" begin
        tmpdir = mktempdir()
        try
            num_chunks = 3
            chunk_sz = 2

            _, state = ProbProg.run_chain(
                fresh_rng(),
                standard_normal_logpdf,
                initial_position;
                num_warmup,
                num_samples=num_samples_warmup,
                chunk_size=num_samples_warmup,
                step_size,
                inverse_mass_matrix,
                progress_bar=false,
            )

            for i in 1:num_chunks
                chunk_samples, state = ProbProg.run_chain(
                    state,
                    standard_normal_logpdf;
                    num_samples=chunk_sz,
                    chunk_size=chunk_sz,
                    progress_bar=false,
                )
                open(
                    io -> Serialization.serialize(io, chunk_samples),
                    joinpath(tmpdir, "chunk_$i.jls"),
                    "w",
                )
                ProbProg.save_state(joinpath(tmpdir, "state.jls"), state)
            end

            loaded_chunks = [
                open(Serialization.deserialize, joinpath(tmpdir, "chunk_$i.jls")) for
                i in 1:num_chunks
            ]
            all_samples = vcat(loaded_chunks...)

            loaded_state = ProbProg.load_state(joinpath(tmpdir, "state.jls"))
            more_samples, _ = ProbProg.run_chain(
                loaded_state,
                standard_normal_logpdf;
                num_samples=2,
                chunk_size=2,
                progress_bar=false,
            )

            @test size(all_samples) == (num_chunks * chunk_sz, pos_size)
            @test all(isfinite, all_samples)
            @test size(more_samples) == (2, pos_size)
            @test all(isfinite, more_samples)
        finally
            rm(tmpdir; recursive=true)
        end
    end
end
