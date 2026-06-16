using Reactant, Test, Random
using Statistics, Serialization
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray
using Reactant.ProbProg: MCMCState, AdaptationState

standard_normal_logpdf(x) = -0.5 * sum(x .^ 2)

function combined_program(rng, logpdf_fn, pos, ss, imm, num_warmup::Int, num_samples::Int)
    samples, _, _, _ = ProbProg.mcmc_logpdf(
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

function fresh_warmup_program(rng, logpdf_fn, pos, ss, imm, offset, nsteps::Int, total::Int)
    _, _, _, st = ProbProg.mcmc_logpdf(
        ReactantRNG(rng),
        logpdf_fn,
        pos;
        algorithm=:NUTS,
        step_size=ss,
        inverse_mass_matrix=imm,
        max_tree_depth=10,
        num_warmup=nsteps,
        num_samples=0,
        adapt_step_size=true,
        adapt_mass_matrix=true,
        total_warmup=total,
        warmup_offset=offset,
        expose_adaptation=true,
    )
    return st::MCMCState
end

function resume_warmup_program(st::MCMCState, logpdf_fn, offset, nsteps::Int, total::Int)
    _, _, _, st2 = ProbProg.mcmc_logpdf(
        st,
        logpdf_fn;
        algorithm=:NUTS,
        max_tree_depth=10,
        num_warmup=nsteps,
        num_samples=0,
        adapt_step_size=true,
        adapt_mass_matrix=true,
        total_warmup=total,
        warmup_offset=offset,
        expose_adaptation=true,
    )
    return st2::MCMCState
end

function sample_program(state::MCMCState, logpdf_fn, num_samples::Int)
    samples, _, _, _ = ProbProg.mcmc_logpdf(
        state, logpdf_fn; algorithm=:NUTS, max_tree_depth=10, num_samples
    )
    return samples
end

@testset "mcmc_warmup_resumable" begin
    fresh_rng() = ReactantRNG(Reactant.to_rarray(UInt64[1, 5]))
    raw_seed() = Reactant.to_rarray(UInt64[1, 5])
    pos_size = 2
    initial_position = Reactant.to_rarray(reshape([0.5, -0.5], 1, pos_size))
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

    @testset "expose full warmup, then sample == single-shot" begin
        cw = @compile optimize = :probprog fresh_warmup_program(
            raw_seed(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            ConcreteRNumber(Int64(0)),
            num_warmup,
            num_warmup,
        )
        st = cw(
            raw_seed(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            ConcreteRNumber(Int64(0)),
            num_warmup,
            num_warmup,
        )
        @test st isa MCMCState
        @test st.adaptation !== nothing

        cs = @compile optimize = :probprog sample_program(
            st, standard_normal_logpdf, num_samples
        )
        samples = Array(cs(st, standard_normal_logpdf, num_samples))
        @test size(samples) == (num_samples, pos_size)
        @test samples ≈ combined_samples
    end

    @testset "chunked warmup (resume + offset) == single-shot" begin
        k = 24
        cf = @compile optimize = :probprog fresh_warmup_program(
            raw_seed(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            ConcreteRNumber(Int64(0)),
            k,
            num_warmup,
        )
        st1 = cf(
            raw_seed(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            ConcreteRNumber(Int64(0)),
            k,
            num_warmup,
        )

        cr = @compile optimize = :probprog resume_warmup_program(
            st1,
            standard_normal_logpdf,
            ConcreteRNumber(Int64(k)),
            num_warmup - k,
            num_warmup,
        )
        st2 = cr(
            st1,
            standard_normal_logpdf,
            ConcreteRNumber(Int64(k)),
            num_warmup - k,
            num_warmup,
        )

        cs = @compile optimize = :probprog sample_program(
            st2, standard_normal_logpdf, num_samples
        )
        chunked_samples = Array(cs(st2, standard_normal_logpdf, num_samples))
        @test chunked_samples ≈ combined_samples
    end

    @testset "save/load mid-warmup MCMCState resumes identically" begin
        cf = @compile optimize = :probprog fresh_warmup_program(
            raw_seed(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            ConcreteRNumber(Int64(0)),
            24,
            num_warmup,
        )
        st1 = cf(
            raw_seed(),
            standard_normal_logpdf,
            initial_position,
            step_size,
            inverse_mass_matrix,
            ConcreteRNumber(Int64(0)),
            24,
            num_warmup,
        )
        tmp = tempname()
        try
            ProbProg.save_state(tmp, st1)
            loaded = ProbProg.load_state(tmp)
            @test loaded isa MCMCState
            @test loaded.adaptation !== nothing

            cr = @compile optimize = :probprog resume_warmup_program(
                st1,
                standard_normal_logpdf,
                ConcreteRNumber(Int64(24)),
                num_warmup - 24,
                num_warmup,
            )
            from_orig = cr(
                st1,
                standard_normal_logpdf,
                ConcreteRNumber(Int64(24)),
                num_warmup - 24,
                num_warmup,
            )
            from_loaded = cr(
                loaded,
                standard_normal_logpdf,
                ConcreteRNumber(Int64(24)),
                num_warmup - 24,
                num_warmup,
            )
            cs = @compile optimize = :probprog sample_program(
                from_orig, standard_normal_logpdf, num_samples
            )
            s_orig = Array(cs(from_orig, standard_normal_logpdf, num_samples))
            s_loaded = Array(cs(from_loaded, standard_normal_logpdf, num_samples))
            @test s_orig == s_loaded
        finally
            rm(tmp; force=true)
        end
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
        cb =
            info -> begin
                push!(events, (; phase=info.phase, step=info.step, total=info.total))
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
    end
end
