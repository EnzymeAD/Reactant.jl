using Reactant, Test, Enzyme, Checkpointing

# A time loop as Checkpointing.jl users write it: the state is mutated in place
# through a closure.
mutable struct SinState{T}
    x::T
end

function sin_loop(x, n, scheme)
    state = SinState(x)
    @ad_checkpoint scheme for i in 1:n
        state.x = sin.(state.x)
    end
    return sum(state.x)
end

function sin_loop_grad(x, n, scheme)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, sin_loop, Active, Duplicated(x, dx), Const(n), Const(scheme))
    return dx
end

mutable struct WhileState{T,I}
    x::T
    i::I
end

function sin_while(x, n, scheme)
    state = WhileState(x, zero(n))
    @ad_checkpoint scheme while state.i < n
        state.x = sin.(state.x)
        state.i += one(state.i)
    end
    return sum(state.x)
end

function sin_while_grad(x, n, scheme)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(
        Reverse, sin_while, Active, Duplicated(x, dx), Const(n), Const(scheme)
    )
    return dx
end

sin_reference(x, n) = Enzyme.gradient(
    Reverse, x -> sum(foldl((v, _) -> sin.(v), 1:n; init=x)), copy(x)
)[1]

const N = 10

@testset "@ad_checkpoint $(nameof(typeof(scheme))) becomes a checkpointed loop" for (
    scheme, attrs
) in (
    (Revolve(3), ("enzyme.enable_checkpointing", "enzyme.binomial_checkpointing")),
    (Periodic(3), ("enzyme.enable_checkpointing", "enzyme.checkpoint_period = 3")),
)
    x = Float32[1.0, 0.5, -0.5]
    x_ra = Reactant.to_rarray(x)

    @test @jit(sin_loop(x_ra, N, scheme)) ≈ sin_loop(x, N, scheme)
    @test Array(@jit(sin_loop_grad(x_ra, N, scheme))) ≈ sin_reference(x, N)

    # The loop is traced once, not unrolled.
    ir = sprint(show, @code_hlo optimize = false sin_loop(x_ra, N, scheme))
    @test count("stablehlo.while", ir) == 1
    for attr in attrs
        @test occursin(attr, ir)
    end
end

@testset "@ad_checkpoint Online_r2 while loop" begin
    x = Float32[1.0, 0.5, -0.5]
    x_ra = Reactant.to_rarray(x)
    scheme = Online_r2(3)

    n_ra = Reactant.ConcreteRNumber(N)

    @test Array(@jit(sin_while_grad(x_ra, n_ra, scheme))) ≈ sin_reference(x, N)
end

struct OtherStorage <: Checkpointing.AbstractStorage end

@testset "unsupported storage is an error" begin
    scheme = Revolve(3)
    scheme.storage = OtherStorage()
    x_ra = Reactant.to_rarray(Float32[1.0, 0.5, -0.5])
    @test_throws ErrorException @jit(sin_loop(x_ra, N, scheme))
end
