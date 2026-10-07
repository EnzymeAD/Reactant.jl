using Reactant, Test, Enzyme
import CheckpointingCore

# A loop marked for checkpointing with CheckpointingCore's loop annotation,
# inside @trace: Reactant traces it once, with Enzyme-MLIR's checkpointing
# for the schedule it names, and a while loop keeps its condition.

mutable struct MarkerState{T}
    x::T
end

function marker_for(x, n)
    s = MarkerState(x)
    @trace track_numbers = false CheckpointingCore.@ad_checkpoint Revolve(3) for i in 1:n
        s.x = sin.(s.x)
    end
    return sum(s.x)
end

function marker_periodic(x, n)
    s = MarkerState(x)
    @trace track_numbers = false CheckpointingCore.@ad_checkpoint Periodic(3) for i in 1:n
        s.x = sin.(s.x)
    end
    return sum(s.x)
end

function marker_while(x, n)
    y = x
    i = zero(n)
    @trace track_numbers = false CheckpointingCore.@ad_checkpoint Binomial(3) while i < n
        y = sin.(y)
        i += one(i)
    end
    return sum(y)
end

function grad(f, x, n)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, f, Active, Duplicated(x, dx), Const(n))
    return dx
end

sin_reference(x, n) = Enzyme.gradient(
    Reverse, x -> sum(foldl((v, _) -> sin.(v), 1:n; init=x)), copy(x)
)[1]

const N = 10

@testset "@ad_checkpoint $(nameof(f)) under @trace" for (f, attrs) in (
    (marker_for, ("enzyme.enable_checkpointing", "enzyme.binomial_checkpointing")),
    (marker_periodic, ("enzyme.enable_checkpointing", "enzyme.checkpoint_period = 3")),
)
    x = Float32[1.0, 0.5, -0.5]
    @test @jit(f(Reactant.to_rarray(x), N)) ≈ sum(foldl((v, _) -> sin.(v), 1:N; init=x))
    @test Array(@jit(grad(f, Reactant.to_rarray(x), N))) ≈ sin_reference(x, N)
    ir = sprint(show, @code_hlo optimize = false f(Reactant.to_rarray(x), N))
    @test count("stablehlo.while", ir) == 1
    for attr in attrs
        @test occursin(attr, ir)
    end
end

@testset "@ad_checkpoint while loop under @trace" begin
    x = Float32[1.0, 0.5, -0.5]
    n = Reactant.ConcreteRNumber(N)
    @test Array(@jit(grad(marker_while, Reactant.to_rarray(x), n))) ≈ sin_reference(x, N)
end
