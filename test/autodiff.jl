using Enzyme, Reactant, Test

square(x) = x * 2

fwd(Mode, RT, x, y) = Enzyme.autodiff(Mode, square, RT, Duplicated(x, y))

@testset "Basic Forward Mode" begin
    ores1 = fwd(Forward, Duplicated, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == NamedTuple{(Symbol("1"),),Tuple{Array{Float64,2}}}
    @test ores1[1] ≈ 6.2 * ones(3, 2)

    res1 = @jit(
        fwd(
            set_abi(Forward, Reactant.ReactantABI),
            Duplicated,
            ConcreteRArray(ones(3, 2)),
            ConcreteRArray(3.1 * ones(3, 2)),
        )
    )

    @test typeof(res1) == Tuple{ConcreteRArray{Float64,2}}
    @test res1[1] ≈ ores1[1]

    ores1 = fwd(ForwardWithPrimal, Duplicated, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) ==
        NamedTuple{(Symbol("1"), Symbol("2")),Tuple{Array{Float64,2},Array{Float64,2}}}

    res1 = @jit(
        fwd(
            set_abi(ForwardWithPrimal, Reactant.ReactantABI),
            Duplicated,
            ConcreteRArray(ones(3, 2)),
            ConcreteRArray(3.1 * ones(3, 2)),
        )
    )

    @test typeof(res1) == Tuple{ConcreteRArray{Float64,2},ConcreteRArray{Float64,2}}
    @test res1[1] ≈ ores1[1]
    @test res1[2] ≈ ores1[2]

    ores1 = fwd(Forward, Const, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == Tuple{}

    res1 = @jit(
        fwd(
            set_abi(Forward, Reactant.ReactantABI),
            Const,
            ConcreteRArray(ones(3, 2)),
            ConcreteRArray(3.1 * ones(3, 2)),
        )
    )

    @test typeof(res1) == Tuple{}

    ores1 = fwd(ForwardWithPrimal, Const, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == NamedTuple{(Symbol("1"),),Tuple{Array{Float64,2}}}

    res1 = @jit(
        fwd(
            set_abi(ForwardWithPrimal, Reactant.ReactantABI),
            Const,
            ConcreteRArray(ones(3, 2)),
            ConcreteRArray(3.1 * ones(3, 2)),
        )
    )

    @test typeof(res1) == Tuple{ConcreteRArray{Float64,2}}
    @test res1[1] ≈ ores1[1]
end

function gw(z)
    return Enzyme.gradient(Forward, sum, z; chunk=Val(1))
end

@testset "Forward Gradient" begin
    x = Reactant.ConcreteRArray(3.1 * ones(2, 2))
    res = @jit gw(x)
    # TODO we should probably override https://github.com/EnzymeAD/Enzyme.jl/blob/5e6a82dd08e74666822b9d7b2b46c36b075668ca/src/Enzyme.jl#L2132
    # to make sure this gets merged as a tracedrarray
    @test typeof(res) == Tuple{Enzyme.TupleArray{ConcreteRNumber{Float64},(2, 2),4,2}}
    @test res[1] ≈ ones(2, 2)
end
