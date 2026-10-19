using Reactant, Test, Random
using Statistics, Serialization
using LinearAlgebra: diag
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
using Reactant.ProbProg: MCMCState

standard_normal_logpdf(x) = -0.5 * sum(x .^ 2)

function combined_program(rng, logpdf_fn, pos, ss, imm, num_warmup::Int, num_samples::Int)
    samples, _, _, _, _ = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        pos;
        algorithm=:NUTS,
        step_size=ss,
        inverse_mass_matrix=imm,
        max_tree_depth=10,
        num_warmup,
        num_samples,
        adapt_step_size=true,
        adapt_mass_matrix=true,
    )
    return samples
end

@testset "mcmc_warmup_resumable" begin
    fresh_rng() = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))
    initial_position = Reactant.to_rarray(reshape([0.5, -0.5], 1, 2))
    pos_size = length(initial_position)
    step_size = ConcreteRNumber(0.1)
    inverse_mass_matrix = ConcreteRArray([0.5 0.0; 0.0 0.5])
    num_warmup = 60
    num_samples = 5

    compiled_combined = @compile optimize = :probprog combined_program(
        fresh_rng(),
        standard_normal_logpdf,
        initial_position,
        step_size,
        inverse_mass_matrix,
        num_warmup,
        num_samples,
    )
    combined_samples = Array(
        compiled_combined(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            num_samples,
        ),
    )

    warmup_only(rng) = ProbProg.run_chain(
        rng,
        standard_normal_logpdf,
        initial_position;
        algorithm=:NUTS,
        num_warmup=num_warmup,
        num_samples=0,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_tree_depth=10,
        progress_bar=false,
    )[2]

    @testset "warmup then sample from state == single-shot" begin
        state = warmup_only(fresh_rng())
        @test state isa MCMCState

        samples, _ = ProbProg.run_chain(
            state, standard_normal_logpdf; num_samples=num_samples, progress_bar=false
        )
        @test size(samples) == (num_samples, pos_size)
        @test samples ≈ combined_samples
    end

    @testset "run_chain warmup_callback == single-shot, fires per chunk" begin
        nsamp = 8
        compiled_combined8 = @compile optimize = :probprog combined_program(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            num_warmup,
            nsamp,
        )
        combined8 = Array(
            compiled_combined8(
                fresh_rng(),
                standard_normal_logpdf,
                initial_position,
                step_size,
                inverse_mass_matrix,
                num_warmup,
                nsamp,
            ),
        )

        events = NamedTuple[]
        cb = function (info)
            posterior = if info.phase === :sampling
                (; mean=vec(mean(info.samples; dims=1)), std=vec(std(info.samples; dims=1)))
            else
                nothing
            end
            push!(
                events,
                (;
                    phase=info.phase,
                    step=info.step,
                    total=info.total,
                    progress=info.step / info.total,
                    step_size=info.step_size,
                    mass_scales=diag(info.inverse_mass_matrix),
                    acceptance_rate=info.acceptance_rate,
                    position=vec(Array(info.state.position)),
                    posterior=posterior,
                ),
            )
            return nothing
        end
        samples, state = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position;
            algorithm=:NUTS,
            num_warmup=num_warmup,
            num_samples=nsamp,
            chunk_size=4,
            warmup_chunk_size=20,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            callback=cb,
            max_tree_depth=10,
        )

        @test size(samples) == (nsamp, pos_size)
        @test samples ≈ combined8
        @test state isa MCMCState

        warmup_events = filter(e -> e.phase === :warmup, events)
        sampling_events = filter(e -> e.phase === :sampling, events)
        @test length(warmup_events) == cld(num_warmup, 20)
        @test length(sampling_events) == cld(nsamp, 4)
        @test warmup_events[end].step == num_warmup
        @test sampling_events[end].step == nsamp

        @test all(e -> isfinite(e.step_size) && e.step_size > 0, events)
        @test all(e -> length(e.mass_scales) == pos_size, events)
        @test all(e -> length(e.position) == pos_size, events)
        @test all(e -> e.progress ≈ e.step / e.total, events)
        @test all(e -> e.acceptance_rate === nothing, warmup_events)
        @test all(e -> e.posterior === nothing, warmup_events)
        @test all(e -> e.acceptance_rate isa Real, sampling_events)
        @test all(e -> length(e.posterior.mean) == pos_size, sampling_events)
        @test all(e -> length(e.posterior.std) == pos_size, sampling_events)
    end

    @testset "run_chain(state) continuation is pure and bit-exact" begin
        state = warmup_only(fresh_rng())
        rng_before = Array(state.rng)

        mono, _ = ProbProg.run_chain(
            state, standard_normal_logpdf; num_samples=num_samples, progress_bar=false
        )
        @test mono ≈ combined_samples
        @test Array(state.rng) == rng_before

        chunked, _ = ProbProg.run_chain(
            state,
            standard_normal_logpdf;
            num_samples=num_samples,
            chunk_size=2,
            progress_bar=true,
        )
        @test chunked ≈ combined_samples
        @test Array(state.rng) == rng_before
    end

    @testset "MCMCState carries inference config; continuation reuses it" begin
        state = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position;
            algorithm=:NUTS,
            num_warmup=num_warmup,
            num_samples=0,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_tree_depth=4,
            progress_bar=false,
        )[2]

        @test state.config.algorithm == :NUTS
        @test state.config.max_tree_depth == 4

        tmp = tempname()
        try
            ProbProg.save_state(tmp, state)
            loaded = ProbProg.load_state(tmp)
            @test loaded.config.algorithm == :NUTS
            @test loaded.config.max_tree_depth == 4
        finally
            rm(tmp; force=true)
        end

        from_state, _ = ProbProg.run_chain(
            state, standard_normal_logpdf; num_samples=num_samples, progress_bar=false
        )
        explicit, _ = ProbProg.run_chain(
            state,
            standard_normal_logpdf;
            num_samples=num_samples,
            progress_bar=false,
            max_tree_depth=4,
        )
        @test from_state == explicit
    end

    @testset "warmup callback observes adaptation each chunk" begin
        trace = NamedTuple[]
        debug_cb = function (info)
            if info.phase === :warmup
                push!(
                    trace,
                    (;
                        step=info.step,
                        step_size=info.step_size,
                        inverse_mass_matrix=info.inverse_mass_matrix,
                        has_adaptation=info.state.adaptation !== nothing,
                    ),
                )
            end
            return nothing
        end

        samples, state = ProbProg.run_chain(
            fresh_rng(),
            standard_normal_logpdf,
            initial_position;
            algorithm=:NUTS,
            num_warmup=num_warmup,
            num_samples=num_samples,
            warmup_chunk_size=20,
            chunk_size=num_samples,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            callback=debug_cb,
            max_tree_depth=10,
            progress_bar=false,
        )

        @test length(trace) == cld(num_warmup, 20)
        @test [e.step for e in trace] == [20, 40, 60]
        @test all(e -> isfinite(e.step_size) && e.step_size > 0, trace)
        @test all(e -> all(isfinite, e.inverse_mass_matrix), trace)
        @test all(e -> e.has_adaptation, trace)
        @test size(samples) == (num_samples, pos_size)
        @test samples ≈ combined_samples
        @test state isa MCMCState
    end
end
