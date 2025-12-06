using Enzyme, Reactant, Test, Random

square(x) = x * 2

fwd(Mode, RT, x, y) = Enzyme.autodiff(Mode, square, RT, Duplicated(x, y))

@testset "Activity" begin
    @test Enzyme.guess_activity(Reactant.ConcretePJRTArray{Float32,2,1}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTArray{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcreteIFRTArray{Float32,2}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTNumber{Float32,1}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTNumber{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTNumber{Float32,1}, Enzyme.Reverse) <:
        Enzyme.Duplicated

    @test Enzyme.guess_activity(Reactant.ConcretePJRTNumber{Float32}, Enzyme.Reverse) <:
        Enzyme.Duplicated

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
    res1 = @jit(
        fwd(
            Forward,
            Duplicated,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test res1 isa Tuple{<:ConcreteRArray{Float64,2}}
    @test res1[1] ≈ fill(6.2, 3, 2)

    res1 = @jit(
        fwd(
            set_abi(ForwardWithPrimal, Reactant.ReactantABI),
            Duplicated,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test res1 isa Tuple{<:ConcreteRArray{Float64,2},<:ConcreteRArray{Float64,2}}
    @test res1[1] ≈ fill(6.2, 3, 2)
    @test res1[2] ≈ fill(2.0, 3, 2)

    res1 = @jit(
        fwd(
            Forward,
            Const,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test typeof(res1) == Tuple{}

    res1 = @jit(
        fwd(
            set_abi(ForwardWithPrimal, Reactant.ReactantABI),
            Const,
            Reactant.to_rarray(ones(3, 2)),
            Reactant.to_rarray(3.1 * ones(3, 2)),
        )
    )

    @test res1 isa Tuple{<:ConcreteRArray{Float64,2}}
    @test res1[1] ≈ fill(2.0, 3, 2)
end

function error_not_within_autodiff()
    !Enzyme.within_autodiff() && error("Not within autodiff")
    return nothing
end

fwd_within_autodiff(Mode, RT) = Enzyme.autodiff(Mode, error_not_within_autodiff, RT)

@testset "within_autodiff" begin
    @test_throws ErrorException error_not_within_autodiff()
    @test fwd_within_autodiff(Forward, Const) == ()

    @test_throws ErrorException @jit error_not_within_autodiff()
    @test (@jit fwd_within_autodiff(Forward, Const)) == ()
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
    x = Reactant.TestUtils.construct_test_array(Float64, 10)
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
    x = Reactant.to_rarray(ones(3, 4))
    hlo = @code_hlo optimize = false Enzyme.onehot(x)
    @test !contains("stablehlo.constant", repr(hlo))
end

fn(x) = sum(abs2, x)
vector_forward_ad(x) = Enzyme.autodiff(Forward, fn, BatchDuplicated(x, Enzyme.onehot(x)))
function vector_forward_ad2(x, dx)
    return Enzyme.autodiff(Forward, fn, StackedBatchDuplicated(x, dx))
end

@testset "Vector Mode AD" begin
    x = reshape(collect(Float32, 1:6), 3, 2)
    x_ra = Reactant.to_rarray(x)
    res = @jit vector_forward_ad(x_ra)

    @test x_ra ≈ x # See https://github.com/EnzymeAD/Reactant.jl/issues/1733
    @test res[1][1] ≈ 2
    @test res[1][2] ≈ 4
    @test res[1][3] ≈ 6
    @test res[1][4] ≈ 8
    @test res[1][5] ≈ 10
    @test res[1][6] ≈ 12

    oh = Enzyme.onehot(x)
    oh_stacked = stack(oh)
    oh_ra = Reactant.to_rarray(oh_stacked)
    res2 = @jit vector_forward_ad2(x_ra, oh_ra)

    @test res2[1][1] ≈ 2
    @test res2[1][2] ≈ 4
    @test res2[1][3] ≈ 6
    @test res2[1][4] ≈ 8
    @test res2[1][5] ≈ 10
    @test res2[1][6] ≈ 12
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

    dy1 = ones(2) .* 1
    dy2 = ones(2) .* 2
    dy3 = ones(2) .* 3
    dy4 = ones(2) .* 4
    dy1_ra = Reactant.to_rarray(dy1)
    dy2_ra = Reactant.to_rarray(dy2)
    dy3_ra = Reactant.to_rarray(dy3)
    dy4_ra = Reactant.to_rarray(dy4)

    @jit autodiff(
        Reverse,
        fn2!,
        BatchDuplicated(y_ra, (dy1_ra, dy2_ra, dy3_ra, dy4_ra)),
        BatchDuplicated(x_ra, (dx1_ra, dx2_ra, dx3_ra, dx4_ra)),
    )

    @test y_ra ≈ x .^ 2
    @test dx1_ra ≈ 2 .* x .* dy1
    @test dx2_ra ≈ 2 .* x .* dy2
    @test dx3_ra ≈ 2 .* x .* dy3
    @test dx4_ra ≈ 2 .* x .* dy4
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
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 2, 2))
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
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 4))

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

cubic(x) = x .^ 3

function vjp_cubic(x, lambdas)
    vjps = similar(lambdas)
    for i in 1:size(lambdas, 2)
        lambda = lambdas[:, i]
        eval_jac_T_v(x) = sum(cubic(x) .* lambda)
        vjps[:, i] .= Enzyme.gradient(Reverse, Const(eval_jac_T_v), x)[1]
    end
    return vjps
end

function jvp_vjp_cubic(v, x, lambdas)
    vjp_cubic_inline(x) = vjp_cubic(x, lambdas)
    return Enzyme.autodiff(Forward, Const(vjp_cubic_inline), Duplicated(x, v))[1]
end

@testset "Nested Forward over Reverse AD" begin
    x = ones(3)
    x_r = Reactant.to_rarray(x)
    v = ones(3)
    v_r = Reactant.to_rarray(x)
    lambdas = ones(3, 2)
    lambdas_r = Reactant.to_rarray(lambdas)

    @test @jit(jvp_vjp_cubic(v_r, x_r, lambdas_r)) ≈ fill(6, (3, 2))
end

@testset "Finite Difference Gradient" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float16, 2, 2))
    res = @jit Reactant.TestUtils.finite_difference_gradient(sum, x)
    @test res isa Reactant.ConcreteRArray{Float16,2}
end

function fdiff_multiple_args(f, nt, x)
    return sum(abs2, f(nt.y .+ x .- nt.x))
end

struct WrapperFunc{T}
    x::T
end

(f::WrapperFunc)(x) = x .^ 3 .+ f.x

@testset "Finite Difference Gradient (non vector inputs)" begin
    nt = (;
        x=Reactant.TestUtils.construct_test_array(Float64, 3, 4),
        y=Reactant.TestUtils.construct_test_array(Float64, 3, 4),
    )
    fn = WrapperFunc(Reactant.TestUtils.construct_test_array(Float64, 3, 4))
    x = Reactant.TestUtils.construct_test_array(Float64, 3, 4)

    nt_ra = Reactant.to_rarray(nt)
    fn_ra = Reactant.to_rarray(fn)
    x_ra = Reactant.to_rarray(x)

    results_fd = @jit Reactant.TestUtils.finite_difference_gradient(
        fdiff_multiple_args, fn_ra, nt_ra, x_ra
    )
    @test results_fd isa typeof((fn_ra, nt_ra, x_ra))

    results_enz = @jit Enzyme.gradient(Reverse, fdiff_multiple_args, fn_ra, nt_ra, x_ra)

    @test results_fd[1].x ≈ results_enz[1].x
    @test results_fd[2].x ≈ results_enz[2].x
    @test results_fd[2].y ≈ results_enz[2].y
    @test results_fd[3] ≈ results_enz[3]
end

@testset "Correct return tuple" begin
    # issue 1875
    x = ones(2)
    xr = Reactant.to_rarray(x)
    res = autodiff(Reverse, sum, Duplicated(x, zero(x)))
    res_reactant = @jit autodiff(Reverse, sum, Duplicated(xr, zero(xr)))
    @test length(res) == length(res_reactant)
end
