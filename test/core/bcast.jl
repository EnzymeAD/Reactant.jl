using Test, Reactant, Enzyme, NNlib, Statistics
using Reactant.MLIR

@noinline function no(@nospecialize(x))
    x = @ccall $(Base.@cfunction(identity, Any, (Any,)))(x::Any)::Any
    return x[]::Any
end

mutable struct Data
    v::Reactant.TracedRArray{Float64,1}
end
@noinline function tmp(a, b, d)
    c = d.v

    return reshape(a, (4,)) ./ sqrt.(b .+ a)
end

function test()
    MLIR.IR.with_context() do ctx
        mod = MLIR.IR.Module(MLIR.IR.Location())
        modbody = MLIR.IR.body(mod)

        in_tys = [MLIR.IR.TensorType([4], MLIR.IR.Type(Float64))]

        func = MLIR.Dialects.func.func_(;
            sym_name="main_tmp",
            function_type=MLIR.IR.FunctionType(in_tys, []),
            body=MLIR.IR.Region(),
        )

        fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for _ in in_tys])
        push!(MLIR.IR.region(func, 1), fnbody)

        GC.@preserve mod func fnbody begin
            MLIR.IR.with_block(fnbody) do
                a = ones(4)
                b = ones(4)
                d = Data(
                    Reactant.TracedRArray{Float64,1}((), MLIR.IR.argument(fnbody, 1), (4,))
                )

                return tmp(a, b, d)
            end
        end

        return string(mod)
    end
end
@test test() == "module {\n}"

@testset "ConcretePJRTArray broadcasting" begin
    x = ones(10, 10)
    y = ones(10, 10)

    x_ca = Reactant.to_rarray(x)
    y_ca = Reactant.to_rarray(y)

    @testset "Broadcasting" begin
        @test x .+ y ≈ @jit x_ca .+ y_ca
        @test x .- y ≈ @jit x_ca .- y_ca
        @test x .* y ≈ @jit x_ca .* y_ca
        @test x ./ y ≈ @jit x_ca ./ y_ca
    end
end

function scalar_bcast(x)
    sc = sum(x)
    return sc .+ x
end

@testset "Scalar broadcasting" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(scalar_bcast(x_ra)) ≈ scalar_bcast(x)
end

function custom_ln(x)
    mu = mean(x; dims=1)
    sigma = var(x; dims=3)
    return (x .- mu) ./ sqrt.(sigma)
end

@testset "Custom layernorm" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 3, 3, 4, 2)
    x_ra = Reactant.to_rarray(x)
    @test @jit(custom_ln(x_ra)) ≈ custom_ln(x)
end

pow(x, n) = x .^ n

@testset "Pow" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(pow(x_ra, 2)) ≈ pow(x, 2)
end

struct CustomBCastFunction{X}
    x::X
end

(f::CustomBCastFunction)(x::Number) = f.x + x

function custombcast(x)
    fn = CustomBCastFunction(3.0)
    return fn.(x)
end

@testset "Broadcasting closures / functors" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(custombcast(x_ra)) ≈ custombcast(x)
end

function bcast_scalar_with_jlarray(jlarr, x)
    return jlarr .+ x
end

@testset "Broadcasting with scalars" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3)
    a = ConcreteRNumber(0.3f0)
    hlo = @code_hlo bcast_scalar_with_jlarray(x, a)
    @test !occursin("stablehlo.slice", repr(hlo))
    @test occursin("stablehlo.broadcast_in_dim", repr(hlo))

    @test @jit(bcast_scalar_with_jlarray(x, a)) ≈
        bcast_scalar_with_jlarray(Array(x), Float32(a))
end

literal_pow_bcast(x) = x .^ 2

@testset "Literal pow bcast" begin
    x = ConcreteRNumber(2.0f0)
    @test @jit(literal_pow_bcast(x)) ≈ 4
end

function DRF_whilebody_ifbody(XN, YN, ZN)
    XNROOT = sqrt(XN)
    YNROOT = sqrt(YN)
    ZNROOT = sqrt(ZN)
    YNROOTZNROOT = YNROOT * ZNROOT
    LAMDA = muladd(XNROOT, (YNROOT + ZNROOT), YNROOTZNROOT)
    return (XN + LAMDA) / 4, (YN + LAMDA) / 4, (ZN + LAMDA) / 4
end

function DRF_whilebody(XN, YN, ZN, ERRTOL)
    MU = (XN + YN + ZN) / 3
    ninvMU = -1 / MU
    XNDEV = muladd(ninvMU, (MU + XN), 2)
    YNDEV = muladd(ninvMU, (MU + YN), 2)
    ZNDEV = muladd(ninvMU, (MU + ZN), 2)
    EPSLON = max(abs(XNDEV), abs(YNDEV), abs(ZNDEV))
    XN, YN, ZN = Base.ifelse(
        EPSLON >= ERRTOL, DRF_whilebody_ifbody(XN, YN, ZN), (XN, YN, ZN)
    )

    return (XN, YN, ZN, XNDEV, YNDEV, ZNDEV, MU, (EPSLON >= ERRTOL))
end

function DRF_ifbody(X::A, Y::B, Z::C, ERRTOL::D) where {A,B,C,D}
    T = promote_type(A, B, C, D)
    C1 = T(1 / 24)
    C2 = T(3 / 44)
    C3 = T(1 / 14)

    XN = X
    YN = Y
    ZN = Z
    MU = zero(T)
    XNDEV = zero(T)
    YNDEV = zero(T)
    ZNDEV = zero(T)
    CONTINUE = true

    @trace while CONTINUE
        XN, YN, ZN, XNDEV, YNDEV, ZNDEV, MU, CONTINUE = DRF_whilebody(XN, YN, ZN, ERRTOL)
    end
    XNDEVYNDEV = XNDEV * YNDEV
    E2 = muladd(-ZNDEV, ZNDEV, XNDEVYNDEV)
    E3 = XNDEVYNDEV * ZNDEV
    S = 1 + muladd(E2, muladd(-C2, E3, muladd(C1, E2, -T(1 / 10))), C3 * E3)
    return S / sqrt(MU)
end

function DRF(X::A, Y::B, Z::C) where {A,B,C}
    T = promote_type(A, B, C)

    ERRTOL = (4 * eps(T) / 2)^T(1 / 6)
    LOLIM = 5floatmin(T)
    UPLIM = floatmax(T) / 5

    ans = zero(T)
    ierr = 0
    ans, ierr = Base.ifelse(
        min(X, Y, Z) < zero(T),
        (ans, 1),
        Base.ifelse(
            max(X, Y, Z) > UPLIM,
            (ans, 3),
            Base.ifelse(min(X + Y, X + Z, Y + Z) < LOLIM, (ans, 2), (zero(T), 0)),
        ),
    )

    ans = Base.ifelse(ierr == 0, DRF_ifbody(X, Y, Z, ERRTOL), ans)

    return (ans, ierr)
end

function K(m::T) where {T}
    drf, ierr = Base.ifelse(
        m < 1,
        DRF(zero(T), 1 - m, one(T)),
        Base.ifelse(m == 1, (T(Inf), 0), Base.ifelse(isnan(m), (T(NaN), 0), (T(NaN), 4))),
    )
    if ierr isa Int
        @assert ierr == 0
    end
    return drf
end

@testset "Non-concrete inferred type for broadcasting: #2467" begin
    x_inner = Vector{Float64}(LinRange(0.0, 1.0, 1000))
    x = Reactant.to_rarray(x_inner)

    res_ra = @jit K.(x)
    res_jl = K.(x_inner)
    @test res_ra ≈ res_jl
end
