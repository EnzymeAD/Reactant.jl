using LogExpFunctions, Reactant, Test

@testset "xlogx" begin
    @test LogExpFunctions.xlogx(0.0) ≈ @jit(LogExpFunctions.xlogx(ConcreteRNumber(0.0)))
    @test LogExpFunctions.xlogx(0.5) ≈ @jit(LogExpFunctions.xlogx(ConcreteRNumber(0.5)))
    @test LogExpFunctions.xlogx(1.0) ≈ @jit(LogExpFunctions.xlogx(ConcreteRNumber(1.0)))
    @test LogExpFunctions.xlogx(2.0) ≈ @jit(LogExpFunctions.xlogx(ConcreteRNumber(2.0)))
    @test LogExpFunctions.xlogx(Float32(0.5)) ≈
        @jit(LogExpFunctions.xlogx(ConcreteRNumber(Float32(0.5))))
end

@testset "xlogy" begin
    @test LogExpFunctions.xlogy(0.0, 0.0) ≈
        @jit(LogExpFunctions.xlogy(ConcreteRNumber(0.0), ConcreteRNumber(0.0)))
    @test LogExpFunctions.xlogy(0.0, 1.0) ≈
        @jit(LogExpFunctions.xlogy(ConcreteRNumber(0.0), ConcreteRNumber(1.0)))
    @test LogExpFunctions.xlogy(2.0, 3.0) ≈
        @jit(LogExpFunctions.xlogy(ConcreteRNumber(2.0), ConcreteRNumber(3.0)))
    @test LogExpFunctions.xlogy(Float32(2.0), Float32(3.0)) ≈ @jit(
        LogExpFunctions.xlogy(ConcreteRNumber(Float32(2.0)), ConcreteRNumber(Float32(3.0)))
    )
end

@testset "xlog1py" begin
    @test LogExpFunctions.xlog1py(0.0, -1.0) ≈
        @jit(LogExpFunctions.xlog1py(ConcreteRNumber(0.0), ConcreteRNumber(-1.0)))
    @test LogExpFunctions.xlog1py(0.0, 0.0) ≈
        @jit(LogExpFunctions.xlog1py(ConcreteRNumber(0.0), ConcreteRNumber(0.0)))
    @test LogExpFunctions.xlog1py(2.0, 3.0) ≈
        @jit(LogExpFunctions.xlog1py(ConcreteRNumber(2.0), ConcreteRNumber(3.0)))
end

@testset "xexpx" begin
    @test LogExpFunctions.xexpx(-Inf) ≈ @jit(LogExpFunctions.xexpx(ConcreteRNumber(-Inf)))
    @test LogExpFunctions.xexpx(0.0) ≈ @jit(LogExpFunctions.xexpx(ConcreteRNumber(0.0)))
    @test LogExpFunctions.xexpx(1.0) ≈ @jit(LogExpFunctions.xexpx(ConcreteRNumber(1.0)))
    @test LogExpFunctions.xexpx(2.0) ≈ @jit(LogExpFunctions.xexpx(ConcreteRNumber(2.0)))
end

@testset "xexpy" begin
    @test LogExpFunctions.xexpy(1.0, -Inf) ≈
        @jit(LogExpFunctions.xexpy(ConcreteRNumber(1.0), ConcreteRNumber(-Inf)))
    @test LogExpFunctions.xexpy(0.0, 1.0) ≈
        @jit(LogExpFunctions.xexpy(ConcreteRNumber(0.0), ConcreteRNumber(1.0)))
    @test LogExpFunctions.xexpy(2.0, 3.0) ≈
        @jit(LogExpFunctions.xexpy(ConcreteRNumber(2.0), ConcreteRNumber(3.0)))
end

@testset "logistic" begin
    # Test Float64
    @test LogExpFunctions.logistic(0.0) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(0.0)))
    @test LogExpFunctions.logistic(1.0) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(1.0)))
    @test LogExpFunctions.logistic(-1.0) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.logistic(-1000.0) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(-1000.0)))
    @test LogExpFunctions.logistic(1000.0) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(1000.0)))
    # Test Float32
    @test LogExpFunctions.logistic(Float32(0.0)) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(Float32(0.0))))
    @test LogExpFunctions.logistic(Float32(-200.0)) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(Float32(-200.0))))
    @test LogExpFunctions.logistic(Float32(200.0)) ≈
        @jit(LogExpFunctions.logistic(ConcreteRNumber(Float32(200.0))))
end

@testset "logit" begin
    @test LogExpFunctions.logit(0.5) ≈ @jit(LogExpFunctions.logit(ConcreteRNumber(0.5)))
    @test LogExpFunctions.logit(0.1) ≈ @jit(LogExpFunctions.logit(ConcreteRNumber(0.1)))
    @test LogExpFunctions.logit(0.9) ≈ @jit(LogExpFunctions.logit(ConcreteRNumber(0.9)))
end

@testset "logcosh" begin
    @test LogExpFunctions.logcosh(0.0) ≈ @jit(LogExpFunctions.logcosh(ConcreteRNumber(0.0)))
    @test LogExpFunctions.logcosh(1.0) ≈ @jit(LogExpFunctions.logcosh(ConcreteRNumber(1.0)))
    @test LogExpFunctions.logcosh(-1.0) ≈
        @jit(LogExpFunctions.logcosh(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.logcosh(10.0) ≈
        @jit(LogExpFunctions.logcosh(ConcreteRNumber(10.0)))
end

@testset "logabssinh" begin
    @test LogExpFunctions.logabssinh(1.0) ≈
        @jit(LogExpFunctions.logabssinh(ConcreteRNumber(1.0)))
    @test LogExpFunctions.logabssinh(-1.0) ≈
        @jit(LogExpFunctions.logabssinh(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.logabssinh(10.0) ≈
        @jit(LogExpFunctions.logabssinh(ConcreteRNumber(10.0)))
end

@testset "log1psq" begin
    @test LogExpFunctions.log1psq(0.0) ≈ @jit(LogExpFunctions.log1psq(ConcreteRNumber(0.0)))
    @test LogExpFunctions.log1psq(1.0) ≈ @jit(LogExpFunctions.log1psq(ConcreteRNumber(1.0)))
    @test LogExpFunctions.log1psq(10.0) ≈
        @jit(LogExpFunctions.log1psq(ConcreteRNumber(10.0)))
    @test LogExpFunctions.log1psq(1e10) ≈
        @jit(LogExpFunctions.log1psq(ConcreteRNumber(1e10))) rtol = 1e-10
    @test LogExpFunctions.log1psq(Float32(10.0)) ≈
        @jit(LogExpFunctions.log1psq(ConcreteRNumber(Float32(10.0))))
end

@testset "log1pexp" begin
    @test LogExpFunctions.log1pexp(-1000.0) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(-1000.0)))
    @test LogExpFunctions.log1pexp(-50.0) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(-50.0)))
    @test LogExpFunctions.log1pexp(0.0) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(0.0)))
    @test LogExpFunctions.log1pexp(10.0) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(10.0)))
    @test LogExpFunctions.log1pexp(50.0) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(50.0)))
    # Test Float32
    @test LogExpFunctions.log1pexp(Float32(-50.0)) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(Float32(-50.0))))
    @test LogExpFunctions.log1pexp(Float32(10.0)) ≈
        @jit(LogExpFunctions.log1pexp(ConcreteRNumber(Float32(10.0))))
end

@testset "log1mexp" begin
    @test LogExpFunctions.log1mexp(-1.0) ≈
        @jit(LogExpFunctions.log1mexp(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.log1mexp(-0.5) ≈
        @jit(LogExpFunctions.log1mexp(ConcreteRNumber(-0.5)))
    @test LogExpFunctions.log1mexp(-0.1) ≈
        @jit(LogExpFunctions.log1mexp(ConcreteRNumber(-0.1)))
    @test LogExpFunctions.log1mexp(-10.0) ≈
        @jit(LogExpFunctions.log1mexp(ConcreteRNumber(-10.0)))
end

@testset "log2mexp" begin
    @test LogExpFunctions.log2mexp(0.0) ≈
        @jit(LogExpFunctions.log2mexp(ConcreteRNumber(0.0)))
    @test LogExpFunctions.log2mexp(-1.0) ≈
        @jit(LogExpFunctions.log2mexp(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.log2mexp(-10.0) ≈
        @jit(LogExpFunctions.log2mexp(ConcreteRNumber(-10.0)))
end

@testset "logexpm1" begin
    @test LogExpFunctions.logexpm1(1.0) ≈
        @jit(LogExpFunctions.logexpm1(ConcreteRNumber(1.0)))
    @test LogExpFunctions.logexpm1(10.0) ≈
        @jit(LogExpFunctions.logexpm1(ConcreteRNumber(10.0)))
    @test LogExpFunctions.logexpm1(20.0) ≈
        @jit(LogExpFunctions.logexpm1(ConcreteRNumber(20.0)))
    @test LogExpFunctions.logexpm1(50.0) ≈
        @jit(LogExpFunctions.logexpm1(ConcreteRNumber(50.0)))
    # Test Float32
    @test LogExpFunctions.logexpm1(Float32(5.0)) ≈
        @jit(LogExpFunctions.logexpm1(ConcreteRNumber(Float32(5.0))))
    @test LogExpFunctions.logexpm1(Float32(15.0)) ≈
        @jit(LogExpFunctions.logexpm1(ConcreteRNumber(Float32(15.0))))
end

@testset "softplus" begin
    @test LogExpFunctions.softplus(0.0) ≈
        @jit(LogExpFunctions.softplus(ConcreteRNumber(0.0)))
    @test LogExpFunctions.softplus(1.0) ≈
        @jit(LogExpFunctions.softplus(ConcreteRNumber(1.0)))
    @test LogExpFunctions.softplus(-10.0) ≈
        @jit(LogExpFunctions.softplus(ConcreteRNumber(-10.0)))
end

@testset "invsoftplus" begin
    @test LogExpFunctions.invsoftplus(1.0) ≈
        @jit(LogExpFunctions.invsoftplus(ConcreteRNumber(1.0)))
    @test LogExpFunctions.invsoftplus(10.0) ≈
        @jit(LogExpFunctions.invsoftplus(ConcreteRNumber(10.0)))
end

@testset "logaddexp" begin
    @test LogExpFunctions.logaddexp(1.0, 2.0) ≈
        @jit(LogExpFunctions.logaddexp(ConcreteRNumber(1.0), ConcreteRNumber(2.0)))
    @test LogExpFunctions.logaddexp(-10.0, -20.0) ≈
        @jit(LogExpFunctions.logaddexp(ConcreteRNumber(-10.0), ConcreteRNumber(-20.0)))
    @test LogExpFunctions.logaddexp(0.0, 0.0) ≈
        @jit(LogExpFunctions.logaddexp(ConcreteRNumber(0.0), ConcreteRNumber(0.0)))
    @test LogExpFunctions.logaddexp(100.0, 100.0) ≈
        @jit(LogExpFunctions.logaddexp(ConcreteRNumber(100.0), ConcreteRNumber(100.0)))
    @test LogExpFunctions.logaddexp(Float32(1.0), Float32(2.0)) ≈ @jit(
        LogExpFunctions.logaddexp(
            ConcreteRNumber(Float32(1.0)), ConcreteRNumber(Float32(2.0))
        )
    )
end

@testset "logsubexp" begin
    @test LogExpFunctions.logsubexp(2.0, 1.0) ≈
        @jit(LogExpFunctions.logsubexp(ConcreteRNumber(2.0), ConcreteRNumber(1.0)))
    @test LogExpFunctions.logsubexp(1.0, 2.0) ≈
        @jit(LogExpFunctions.logsubexp(ConcreteRNumber(1.0), ConcreteRNumber(2.0)))
    @test LogExpFunctions.logsubexp(-10.0, -20.0) ≈
        @jit(LogExpFunctions.logsubexp(ConcreteRNumber(-10.0), ConcreteRNumber(-20.0)))
end

@testset "cloglog" begin
    @test LogExpFunctions.cloglog(0.1) ≈ @jit(LogExpFunctions.cloglog(ConcreteRNumber(0.1)))
    @test LogExpFunctions.cloglog(0.5) ≈ @jit(LogExpFunctions.cloglog(ConcreteRNumber(0.5)))
    @test LogExpFunctions.cloglog(0.9) ≈ @jit(LogExpFunctions.cloglog(ConcreteRNumber(0.9)))
end

@testset "cexpexp" begin
    @test LogExpFunctions.cexpexp(-1.0) ≈
        @jit(LogExpFunctions.cexpexp(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.cexpexp(0.0) ≈ @jit(LogExpFunctions.cexpexp(ConcreteRNumber(0.0)))
    @test LogExpFunctions.cexpexp(1.0) ≈ @jit(LogExpFunctions.cexpexp(ConcreteRNumber(1.0)))
end

@testset "loglogistic" begin
    @test LogExpFunctions.loglogistic(0.0) ≈
        @jit(LogExpFunctions.loglogistic(ConcreteRNumber(0.0)))
    @test LogExpFunctions.loglogistic(1.0) ≈
        @jit(LogExpFunctions.loglogistic(ConcreteRNumber(1.0)))
    @test LogExpFunctions.loglogistic(-1.0) ≈
        @jit(LogExpFunctions.loglogistic(ConcreteRNumber(-1.0)))
end

@testset "logitexp" begin
    @test LogExpFunctions.logitexp(-1.0) ≈
        @jit(LogExpFunctions.logitexp(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.logitexp(-0.5) ≈
        @jit(LogExpFunctions.logitexp(ConcreteRNumber(-0.5)))
end

@testset "log1mlogistic" begin
    @test LogExpFunctions.log1mlogistic(0.0) ≈
        @jit(LogExpFunctions.log1mlogistic(ConcreteRNumber(0.0)))
    @test LogExpFunctions.log1mlogistic(1.0) ≈
        @jit(LogExpFunctions.log1mlogistic(ConcreteRNumber(1.0)))
    @test LogExpFunctions.log1mlogistic(-1.0) ≈
        @jit(LogExpFunctions.log1mlogistic(ConcreteRNumber(-1.0)))
end

@testset "logit1mexp" begin
    @test LogExpFunctions.logit1mexp(-1.0) ≈
        @jit(LogExpFunctions.logit1mexp(ConcreteRNumber(-1.0)))
    @test LogExpFunctions.logit1mexp(-0.5) ≈
        @jit(LogExpFunctions.logit1mexp(ConcreteRNumber(-0.5)))
end

@testset "log1pmx" begin
    @test LogExpFunctions.log1pmx(0.0) ≈ @jit(LogExpFunctions.log1pmx(ConcreteRNumber(0.0)))
    @test LogExpFunctions.log1pmx(0.1) ≈ @jit(LogExpFunctions.log1pmx(ConcreteRNumber(0.1)))
    @test LogExpFunctions.log1pmx(-0.1) ≈
        @jit(LogExpFunctions.log1pmx(ConcreteRNumber(-0.1)))
    @test LogExpFunctions.log1pmx(0.5) ≈ @jit(LogExpFunctions.log1pmx(ConcreteRNumber(0.5)))
    @test LogExpFunctions.log1pmx(-0.5) ≈
        @jit(LogExpFunctions.log1pmx(ConcreteRNumber(-0.5)))
end

@testset "logmxp1" begin
    @test LogExpFunctions.logmxp1(0.5) ≈ @jit(LogExpFunctions.logmxp1(ConcreteRNumber(0.5)))
    @test LogExpFunctions.logmxp1(1.0) ≈ @jit(LogExpFunctions.logmxp1(ConcreteRNumber(1.0)))
    @test LogExpFunctions.logmxp1(2.0) ≈ @jit(LogExpFunctions.logmxp1(ConcreteRNumber(2.0)))
    @test LogExpFunctions.logmxp1(0.1) ≈ @jit(LogExpFunctions.logmxp1(ConcreteRNumber(0.1)))
end

@testset "logsumexp" begin
    # Test with arrays
    x = [1.0, 2.0, 3.0]
    x_ra = Reactant.to_rarray(x)
    @test LogExpFunctions.logsumexp(x) ≈ @jit(LogExpFunctions.logsumexp(x_ra))

    # Test with 2D arrays
    x2 = [1.0 2.0; 3.0 4.0]
    x2_ra = Reactant.to_rarray(x2)
    @test LogExpFunctions.logsumexp(x2) ≈ @jit(LogExpFunctions.logsumexp(x2_ra))

    # Test with dims
    @test LogExpFunctions.logsumexp(x2; dims=1) ≈
        Array(@jit(LogExpFunctions.logsumexp(x2_ra; dims=1)))
    @test LogExpFunctions.logsumexp(x2; dims=2) ≈
        Array(@jit(LogExpFunctions.logsumexp(x2_ra; dims=2)))

    # Test Float32
    x32 = Float32[1.0, 2.0, 3.0]
    x32_ra = Reactant.to_rarray(x32)
    @test LogExpFunctions.logsumexp(x32) ≈ @jit(LogExpFunctions.logsumexp(x32_ra))
end

@testset "softmax" begin
    # Test 1D
    x = [1.0, 2.0, 3.0]
    x_ra = Reactant.to_rarray(x)
    @test LogExpFunctions.softmax(x) ≈ Array(@jit(LogExpFunctions.softmax(x_ra)))

    # Test 2D
    x2 = [1.0 2.0; 3.0 4.0]
    x2_ra = Reactant.to_rarray(x2)
    @test LogExpFunctions.softmax(x2) ≈ Array(@jit(LogExpFunctions.softmax(x2_ra)))

    # Test with dims
    @test LogExpFunctions.softmax(x2; dims=1) ≈
        Array(@jit(LogExpFunctions.softmax(x2_ra; dims=1)))
    @test LogExpFunctions.softmax(x2; dims=2) ≈
        Array(@jit(LogExpFunctions.softmax(x2_ra; dims=2)))

    # Test Float32
    x32 = Float32[1.0, 2.0, 3.0]
    x32_ra = Reactant.to_rarray(x32)
    @test LogExpFunctions.softmax(x32) ≈ Array(@jit(LogExpFunctions.softmax(x32_ra)))
end
