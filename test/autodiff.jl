using Enzyme, Reactant, Test

square(x) = x * 2

fwd(Mode, RT, x, y) = Enzyme.autodiff(Mode, square, RT, Duplicated(x, y))

@testset "Basic Forward Mode" begin
    ores1 = fwd(Forward, Duplicated, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == NamedTuple{(Symbol("1"),),Tuple{Array{Float64,2}}}
    @test ores1[1] ≈ 6.2 * ones(3, 2)

    res1 = @jit(
        fwd(
            Forward,
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
        fwd(Forward, Const, ConcreteRArray(ones(3, 2)), ConcreteRArray(3.1 * ones(3, 2)))
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

mutable struct StateReturn
    st::Any
end

mutable struct StateReturn1
    st1::Any
    st2::Any
end

function cached_return(x, stret::StateReturn)
    loss = sum(x)
    stret.st = x .+ 1
    return loss
end

function cached_return(x, stret::StateReturn1)
    loss = sum(x)
    tmp = x .+ 1
    stret.st1 = tmp
    stret.st2 = tmp
    return loss
end

@testset "Cached Return: Issue #416" begin
    x = rand(10)
    x_ra = Reactant.to_rarray(x)

    stret = StateReturn(nothing)
    ret = @jit Enzyme.gradient(Reverse, cached_return, x_ra, Const(stret))

    @test @allowscalar all(isone, ret[1])
    @test stret.st isa ConcreteRArray
    @test stret.st ≈ x .+ 1

    stret = StateReturn1(nothing, nothing)
    ret = @jit Enzyme.gradient(Reverse, cached_return, x_ra, Const(stret))

    @test @allowscalar all(isone, ret[1])
    @test stret.st1 isa ConcreteRArray
    @test stret.st1 ≈ x .+ 1
    @test stret.st2 isa ConcreteRArray
    @test stret.st2 ≈ x .+ 1
    @test stret.st1 === stret.st2
end

@testset "Nested AD" begin
    x = ConcreteRNumber(3.1)
    f(x) = x * x * x * x
    df(x) = Enzyme.gradient(Reverse, f, x)[1]
    res1 = @jit df(x)
    @test res1 ≈ 4 * 3.1^3
    ddf(x) = Enzyme.gradient(Reverse, df, x)[1]
    res2 = @jit ddf(x)
    @test res2 ≈ 4 * 3 * 3.1^2
end

@testset "Seed initialization of Complex arrays on matmul: Issue #593" begin
    a = ones(ComplexF64, 2, 2)
    b = 2.0 * ones(ComplexF64, 2, 2)
    a_re = Reactant.to_rarray(a)
    b_re = Reactant.to_rarray(b)
    df(x, y) = Enzyme.gradient(ReverseWithPrimal, *, x, y)
    res = @jit df(a_re, b_re) # before, this segfaulted
    @test res.val ≈ 4ones(2, 2)
    @test res.derivs[1] ≈ 4ones(2, 2)
    @test res.derivs[2] ≈ 2ones(2, 2)
end
