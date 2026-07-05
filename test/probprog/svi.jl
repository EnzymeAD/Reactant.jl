using Reactant, Test, Random
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "common.jl"))

# Smoke test: svi() builds and runs for each config; NumPyro parity is in vi_*.jl.

function model(rng)
    _, x = ProbProg.sample(rng, ProbProg.Normal(0.0, 1.0); symbol=:x)
    return x
end

function svi_program(
    rng,
    model,
    step_size,
    num_steps,
    num_particles,
    guide::Symbol,
    optimizer::Symbol,
    selection,
    obs_tensor,
    constrained_addresses,
)
    trace0, _, _ = ProbProg.generate(rng, obs_tensor, model; constrained_addresses)
    trace, elbo_traj, _, state = ProbProg.svi(
        rng, trace0, model; selection, guide, optimizer, step_size, num_steps, num_particles
    )
    return trace, elbo_traj, state.loc
end

@testset "svi smoke test" begin
    rng = ReactantRNG(ConcreteRArray(UInt64[0, 0]))
    selection = ProbProg.select(ProbProg.Address(:x))
    step_size = ConcreteRNumber(0.05)
    # Empty constraint: generate builds the trace without consuming rng.
    obs = ProbProg.Constraint()
    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = Reactant.to_rarray(reshape(Float64[], 1, 0))

    @testset "AutoNormal + Adam + K=1" begin
        compiled, _ = ProbProg.with_trace() do
            @compile optimize = :probprog svi_program(
                rng,
                model,
                step_size,
                5,
                1,
                :AutoNormal,
                :Adam,
                selection,
                obs_tensor,
                constrained_addresses,
            )
        end
        trace, elbo_traj, loc_final = compiled(
            rng,
            model,
            step_size,
            5,
            1,
            :AutoNormal,
            :Adam,
            selection,
            obs_tensor,
            constrained_addresses,
        )
        @test size(Array(trace)) == (5, 1)
        @test size(Array(elbo_traj)) == (5,)
        @test size(Array(loc_final)) == (1, 1)
    end

    @testset "AutoNormal + SGD + K=4" begin
        compiled, _ = ProbProg.with_trace() do
            @compile optimize = :probprog svi_program(
                rng,
                model,
                step_size,
                3,
                4,
                :AutoNormal,
                :SGD,
                selection,
                obs_tensor,
                constrained_addresses,
            )
        end
        trace, elbo_traj, _ = compiled(
            rng,
            model,
            step_size,
            3,
            4,
            :AutoNormal,
            :SGD,
            selection,
            obs_tensor,
            constrained_addresses,
        )
        @test size(Array(trace)) == (3, 1)
        @test size(Array(elbo_traj)) == (3,)
    end

    @testset "AutoMVN + Adam + K=1" begin
        compiled, _ = ProbProg.with_trace() do
            @compile optimize = :probprog svi_program(
                rng,
                model,
                step_size,
                3,
                1,
                :AutoMultivariateNormal,
                :Adam,
                selection,
                obs_tensor,
                constrained_addresses,
            )
        end
        trace, elbo_traj, _ = compiled(
            rng,
            model,
            step_size,
            3,
            1,
            :AutoMultivariateNormal,
            :Adam,
            selection,
            obs_tensor,
            constrained_addresses,
        )
        @test size(Array(trace)) == (3, 1)
        @test size(Array(elbo_traj)) == (3,)
    end
end
