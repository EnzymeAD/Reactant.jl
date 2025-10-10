using Enzyme, Reactant, Test, Random

square(x) = x * 2

fwd(Mode, RT, x, y) = Enzyme.autodiff(Mode, square, RT, Duplicated(x, y))

@testset "Activity" begin
    @test Enzyme.guess_activity(
        Reactant.ConcretePJRTArray{
            Float32,2,1,Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding,Nothing}
        },
        Enzyme.Reverse,
    ) <: Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTArray{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(
        Reactant.ConcreteIFRTArray{
            Float32,2,Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding,Nothing}
        },
        Enzyme.Reverse,
    ) <: Enzyme.Duplicated

    @test Enzyme.guess_activity(
        Reactant.ConcretePJRTNumber{
            Float32,1,Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding,Nothing}
        },
        Enzyme.Reverse,
    ) <: Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTNumber{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(
        Reactant.ConcretePJRTNumber{
            Float32,1,Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding,Nothing}
        },
        Enzyme.Reverse,
    ) <: Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTNumber{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(
        Reactant.ConcreteIFRTNumber{
            Float32,Reactant.Sharding.ShardInfo{Reactant.Sharding.NoSharding,Nothing}
        },
        Enzyme.Reverse,
    ) <: Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcreteIFRTNumber{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.TracedRArray{Float32,2}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.TracedRArray{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.TracedRNumber{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated
end

@testset "Basic Forward Mode" begin
    ores1 = fwd(Forward, Duplicated, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == NamedTuple{(Symbol("1"),),Tuple{Array{Float64,2}}}
    @test ores1[1] ≈ 6.2 * ones(3, 2)

    res1 = @jit(
        fwd(
            Forward,
            Duplicated,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test res1 isa Tuple{<:ConcreteRArray{Float64,2}}
    @test res1[1] ≈ ores1[1]

    ores1 = fwd(ForwardWithPrimal, Duplicated, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) ==
        NamedTuple{(Symbol("1"), Symbol("2")),Tuple{Array{Float64,2},Array{Float64,2}}}

    res1 = @jit(
        fwd(
            set_abi(ForwardWithPrimal, Reactant.ReactantABI),
            Duplicated,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test res1 isa Tuple{<:ConcreteRArray{Float64,2},<:ConcreteRArray{Float64,2}}
    @test res1[1] ≈ ores1[1]
    @test res1[2] ≈ ores1[2]

    ores1 = fwd(Forward, Const, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == Tuple{}

    res1 = @jit(
        fwd(
            Forward,
            Const,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test typeof(res1) == Tuple{}

    ores1 = fwd(ForwardWithPrimal, Const, ones(3, 2), 3.1 * ones(3, 2))
    @test typeof(ores1) == NamedTuple{(Symbol("1"),),Tuple{Array{Float64,2}}}

    res1 = @jit(
        fwd(
            set_abi(ForwardWithPrimal, Reactant.ReactantABI),
            Const,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test res1 isa Tuple{<:ConcreteRArray{Float64,2}}
    @test res1[1] ≈ ores1[1]
end

function gw(z)
    return Enzyme.gradient(Forward, sum, z; chunk=Val(1))
end

@testset "Forward Gradient" begin
    x = Reactant.to_rarray(3.1 * ones(2, 2))
    res = @jit gw(x)
    # TODO we should probably override https://github.com/EnzymeAD/Enzyme.jl/blob/5e6a82dd08e74666822b9d7b2b46c36b075668ca/src/Enzyme.jl#L2132
    # to make sure this gets merged as a tracedrarray
    @test res isa Tuple{<:Enzyme.TupleArray{<:ConcreteRNumber{Float64},(2, 2),4,2}}
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
    df(x, y) = Enzyme.gradient(ReverseWithPrimal, *, x, y)
    @test begin
        a = ones(ComplexF64, 2, 2)
        b = 2.0 * ones(ComplexF64, 2, 2)
        a_re = Reactant.to_rarray(a)
        b_re = Reactant.to_rarray(b)
        res = @jit df(a_re, b_re) # before, this segfaulted
        (res.val ≈ 4ones(2, 2)) &&
            (res.derivs[1] ≈ 4ones(2, 2)) &&
            (res.derivs[2] ≈ 2ones(2, 2))
    end skip = contains(string(Reactant.devices()[1]), "TPU")
end

@testset "onehot" begin
    x = Reactant.to_rarray(rand(3, 4))
    hlo = @code_hlo optimize = false Enzyme.onehot(x)
    @test !contains("stablehlo.constant", repr(hlo))
end

fn(x) = sum(abs2, x)
vector_forward_ad(x) = Enzyme.autodiff(Forward, fn, BatchDuplicated(x, Enzyme.onehot(x)))

@testset "Vector Mode AD" begin
    x = reshape(collect(Float32, 1:6), 3, 2)
    x_ra = Reactant.to_rarray(x)
    res = @jit vector_forward_ad(x_ra)
    res_enz = vector_forward_ad(x)

    @test x_ra ≈ x # See https://github.com/EnzymeAD/Reactant.jl/issues/1733
    @test res[1][1] ≈ res_enz[1][1]
    @test res[1][2] ≈ res_enz[1][2]
    @test res[1][3] ≈ res_enz[1][3]
    @test res[1][4] ≈ res_enz[1][4]
end

function fn2!(y, x)
    copyto!(y, x .^ 2)
    return nothing
end

@testset "Vector Mode AD (Reverse)" begin
    x = [2.0, 3.0]
    x_ra = Reactant.to_rarray(x)
    y = [0.0, 0.0]
    y_ra = Reactant.to_rarray(y)

    dx1 = zeros(2)
    dx2 = zeros(2)
    dx3 = zeros(2)
    dx4 = zeros(2)
    dx1_ra = Reactant.to_rarray(dx1)
    dx2_ra = Reactant.to_rarray(dx2)
    dx3_ra = Reactant.to_rarray(dx3)
    dx4_ra = Reactant.to_rarray(dx4)

    dy1 = zeros(2)
    dy2 = zeros(2)
    dy3 = zeros(2)
    dy4 = zeros(2)
    dy1_ra = Reactant.to_rarray(dy1)
    dy2_ra = Reactant.to_rarray(dy2)
    dy3_ra = Reactant.to_rarray(dy3)
    dy4_ra = Reactant.to_rarray(dy4)

    autodiff(
        ReverseWithPrimal,
        fn2!,
        BatchDuplicated(y, (dy1, dy2, dy3, dy4)),
        BatchDuplicated(x, (dx1, dx2, dx3, dx4)),
    )

    @jit autodiff(
        Reverse,
        fn2!,
        BatchDuplicated(y_ra, (dy1_ra, dy2_ra, dy3_ra, dy4_ra)),
        BatchDuplicated(x_ra, (dx1_ra, dx2_ra, dx3_ra, dx4_ra)),
    )

    @test y ≈ y_ra
    @test dy1 ≈ dy1_ra
    @test dy2 ≈ dy2_ra
    @test dy3 ≈ dy3_ra
    @test dy4 ≈ dy4_ra
end

@testset "make_zero!" begin
    x = Reactant.to_rarray([3.1])
    @jit Enzyme.make_zero!(x)

    @test @allowscalar x[1] ≈ 0.0
end

function simple_forward(x, st)
    rng = copy(st.rng)
    y = similar(x)
    rand!(rng, y)
    return x .+ y, (; rng)
end

function gradient_fn(x, st)
    stₙ = Ref{Any}(nothing)
    function lfn(x, st_old)
        y, st_new = simple_forward(x, st_old)
        stₙ[] = st_new
        return sum(abs2, y)
    end
    return Enzyme.gradient(Reverse, lfn, x, Const(st)), stₙ[]
end

@testset "seed" begin
    x = Reactant.to_rarray(rand(2, 2))
    st = (; rng=Reactant.ReactantRNG())

    @test begin
        hlo = @code_hlo gradient_fn(x, st)
        contains(repr(hlo), "stablehlo.rng_bit_generator")
    end
end

function divinf(x)
    return min(1.0, 1 / x)
end

function grad_divinf(x)
    return Enzyme.gradient(Reverse, divinf, x)
end

function grad_divinf_sz(x)
    return Enzyme.gradient(Enzyme.set_strong_zero(Reverse), divinf, x)
end

@testset "Strong zero" begin
    x = ConcreteRNumber(0.0)
    @test isnan((@jit grad_divinf(x))[1])
    @test iszero((@jit grad_divinf_sz(x))[1])
end

function simple_grad_without_ignore(x::AbstractArray{T}) where {T}
    return (sum(x; dims=1), x .- 1, (x, x .+ 2)), sum(abs2, x)
end

function simple_grad_with_ignore(x::AbstractArray{T}) where {T}
    return Enzyme.ignore_derivatives(sum(x; dims=1), x .- 1, (x, x .+ 2)), sum(abs2, x)
end

function zero_grad(x)
    return Enzyme.ignore_derivatives(sum(x))
end

function zero_grad2(x)
    return Enzyme.ignore_derivatives(sum(x), x)
end

@testset "ignore_derivatives" begin
    x = Reactant.to_rarray(rand(Float32, 4, 4))

    res1 = @jit Enzyme.gradient(Reverse, simple_grad_without_ignore, x)
    @test res1[1] ≈ (2 .* Array(x) .+ 4)

    res2 = @jit Enzyme.gradient(Reverse, simple_grad_with_ignore, x)
    @test res2[1] ≈ (2 .* Array(x))

    ∂x, result = @jit Enzyme.gradient(ReverseWithPrimal, zero_grad, x)
    @test result isa ConcreteRNumber{Float32}
    @test ∂x[1] ≈ zeros(Float32, 4, 4)

    ∂x2, result2 = @jit Enzyme.gradient(ReverseWithPrimal, zero_grad2, x)
    @test result2 isa Tuple{<:ConcreteRNumber{Float32},<:ConcreteRArray{Float32,2}}
    @test ∂x2[1] ≈ zeros(Float32, 4, 4)
end
