using Reactant, Test, Random
using Statistics
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# Warm restart: M iters, then N resumed via initial_state, must match M+N iters bitwise.
function model(rng)
    _, x = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0); symbol=:x)
    _, _y = ProbProg.sample(rng, ProbProg.Normal(x, 0.5); symbol=:y)
    return x
end

function svi_full(
    rng,
    model,
    step_size,
    num_steps::Int,
    selection,
    obs_tensor,
    constrained_addresses,
    init_loc,
)
    trace0, _, _ = ProbProg.generate(rng, obs_tensor, model; constrained_addresses)
    trace, elbo_traj, _, state = ProbProg.svi(
        rng,
        trace0,
        model;
        selection,
        guide=:AutoNormal,
        optimizer=:Adam,
        step_size,
        num_steps,
        num_particles=1,
        initial_position=init_loc,
    )
    return trace, elbo_traj, state.loc, state.scale, state.slots, state.rng
end

function svi_resume(
    rng,
    model,
    step_size,
    num_steps::Int,
    selection,
    obs_tensor,
    constrained_addresses,
    init_slots,
)
    trace0, _, _ = ProbProg.generate(rng, obs_tensor, model; constrained_addresses)
    trace, elbo_traj, _, state = ProbProg.svi(
        rng,
        trace0,
        model;
        selection,
        guide=:AutoNormal,
        optimizer=:Adam,
        step_size,
        num_steps,
        num_particles=1,
        initial_state=init_slots,
    )
    return trace, elbo_traj, state.loc, state.scale, state.slots, state.rng
end

@testset "vi_init_state_resumption" begin
    obs_y = 1.0
    obs = ProbProg.Constraint(:x => [0.0], :y => [obs_y])
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_flat = Float64[]
    for addr in constrained_addresses
        append!(obs_flat, vec(obs[addr]))
    end
    obs_tensor = Reactant.to_rarray(reshape(obs_flat, 1, :))

    selection = ProbProg.select(ProbProg.Address(:x))
    lr = 0.01
    M = 3
    N = 4

    step_size = ConcreteRNumber(lr)
    init_loc = Reactant.to_rarray(zeros(Float64, 1, 1))

    seed_full = Reactant.to_rarray(UInt64[0, 0])
    rng_full = ReactantRNG(seed_full)

    compiled_full, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_full(
            rng_full,
            model,
            step_size,
            M + N,
            selection,
            obs_tensor,
            constrained_addresses,
            init_loc,
        )
    end

    trace_full, elbo_full, loc_full, scale_full, slots_full, rng_full_out = compiled_full(
        rng_full,
        model,
        step_size,
        M + N,
        selection,
        obs_tensor,
        constrained_addresses,
        init_loc,
    )

    loc_traj_full = Array(trace_full)
    elbo_traj_full = Array(elbo_full)
    loc_final_full = Array(loc_full)
    scale_final_full = Array(scale_full)

    seed_part = Reactant.to_rarray(UInt64[0, 0])
    rng_part = ReactantRNG(seed_part)

    compiled_partM, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_full(
            rng_part,
            model,
            step_size,
            M,
            selection,
            obs_tensor,
            constrained_addresses,
            init_loc,
        )
    end
    trace_M, elbo_M, loc_M, scale_M, slots_M, rng_M_out = compiled_partM(
        rng_part,
        model,
        step_size,
        M,
        selection,
        obs_tensor,
        constrained_addresses,
        init_loc,
    )

    rng_resumed = ReactantRNG(rng_M_out)
    compiled_partN, _ = ProbProg.with_trace() do
        @compile optimize = :probprog svi_resume(
            rng_resumed,
            model,
            step_size,
            N,
            selection,
            obs_tensor,
            constrained_addresses,
            slots_M,
        )
    end
    trace_N, elbo_N, loc_N, scale_N, slots_N, rng_N_out = compiled_partN(
        rng_resumed,
        model,
        step_size,
        N,
        selection,
        obs_tensor,
        constrained_addresses,
        slots_M,
    )

    loc_traj_N = Array(trace_N)
    elbo_traj_N = Array(elbo_N)
    loc_final_N = Array(loc_N)
    scale_final_N = Array(scale_N)

    loc_traj_M = Array(trace_M)
    elbo_traj_M = Array(elbo_M)
    @info "First-phase loc max diff vs full[1:M]:" maxabs = maximum(
        abs.(loc_traj_M .- loc_traj_full[1:M, :])
    )
    @test isapprox(loc_traj_M, loc_traj_full[1:M, :]; atol=1e-12, rtol=1e-12)

    @info "Resumed loc max diff vs full[M+1:M+N]:" maxabs = maximum(
        abs.(loc_traj_N .- loc_traj_full[(M + 1):(M + N), :])
    )
    @test isapprox(loc_traj_N, loc_traj_full[(M + 1):(M + N), :]; atol=1e-12, rtol=1e-12)

    @info "Final loc diff:" maxabs = maximum(abs.(loc_final_N .- loc_final_full))
    @test isapprox(loc_final_N, loc_final_full; atol=1e-12, rtol=1e-12)
    @test isapprox(scale_final_N, scale_final_full; atol=1e-12, rtol=1e-12)

    @test isapprox(elbo_traj_M, elbo_traj_full[1:M]; atol=1e-12, rtol=1e-12)
    @test isapprox(elbo_traj_N, elbo_traj_full[(M + 1):(M + N)]; atol=1e-12, rtol=1e-12)
end
