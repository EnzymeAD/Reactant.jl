using SpecialFunctions, Reactant

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")
const RunningOnMetal = contains(string(Reactant.devices()[1]), "MTL")

macro ≈(a, b)
    return quote
        isapprox($a, $b; atol=1e-14)
    end
end

@testset "gamma" begin
    @test SpecialFunctions.gamma(0.5) ≈ @jit(SpecialFunctions.gamma(ConcreteRNumber(0.5))) atol =
        1e-5 rtol = 1e-3 skip = RunningOnMetal
    @test SpecialFunctions.gamma(Int32(2)) ≈
        @jit(SpecialFunctions.gamma(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "loggamma" begin
    @test SpecialFunctions.loggamma(0.5) ≈
        @jit(SpecialFunctions.loggamma(ConcreteRNumber(0.5))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
    @test SpecialFunctions.loggamma(Int32(2)) ≈
        @jit(SpecialFunctions.loggamma(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "digamma" begin
    @test SpecialFunctions.digamma(0.5) ≈
        @jit(SpecialFunctions.digamma(ConcreteRNumber(0.5))) skip = RunningOnMetal
    @test SpecialFunctions.digamma(Int32(2)) ≈
        @jit(SpecialFunctions.digamma(ConcreteRNumber(Int32(2)))) skip = RunningOnMetal
end

@testset "trigamma" begin
    @test SpecialFunctions.trigamma(0.5) ≈
        @jit(SpecialFunctions.trigamma(ConcreteRNumber(0.5))) skip = RunningOnMetal
    @test SpecialFunctions.trigamma(Int32(2)) ≈
        @jit(SpecialFunctions.trigamma(ConcreteRNumber(Int32(2)))) skip = RunningOnMetal
end

@testset "beta" begin
    @test SpecialFunctions.beta(0.5, 0.6) ≈
        @jit(SpecialFunctions.beta(ConcreteRNumber(0.5), ConcreteRNumber(0.6))) skip =
        RunningOnMetal
    @test SpecialFunctions.beta(Int32(2), Int32(4)) ≈
        @jit(SpecialFunctions.beta(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))) skip =
        RunningOnMetal
end

@testset "logbeta" begin
    @test SpecialFunctions.logbeta(0.5, 0.6) ≈
        @jit(SpecialFunctions.logbeta(ConcreteRNumber(0.5), ConcreteRNumber(0.6))) skip =
        RunningOnMetal
    @test SpecialFunctions.logbeta(Int32(2), Int32(4)) ≈ @jit(
        SpecialFunctions.logbeta(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))
    ) skip = RunningOnMetal
end

@testset "erf" begin
    @test SpecialFunctions.erf(0.5) ≈ @jit(SpecialFunctions.erf(ConcreteRNumber(0.5))) skip =
        RunningOnMetal
    @test SpecialFunctions.erf(Int32(2)) ≈
        @jit(SpecialFunctions.erf(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "erf with 2 arguments" begin
    @test SpecialFunctions.erf(0.5, 0.6) ≈
        @jit(SpecialFunctions.erf(ConcreteRNumber(0.5), ConcreteRNumber(0.6))) skip =
        RunningOnMetal
    @test SpecialFunctions.erf(Int32(2), Int32(4)) ≈
        @jit(SpecialFunctions.erf(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))) atol =
        1e-5 rtol = 1e-3 skip = RunningOnMetal
end

@testset "erfinv" begin
    @test SpecialFunctions.erfinv(0.5) ≈ @jit(SpecialFunctions.erfinv(ConcreteRNumber(0.5))) skip =
        RunningOnMetal
    @test SpecialFunctions.erfinv(Int32(0)) ≈
        @jit(SpecialFunctions.erfinv(ConcreteRNumber(Int32(0)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "erfc" begin
    @test SpecialFunctions.erfc(0.5) ≈ @jit(SpecialFunctions.erfc(ConcreteRNumber(0.5))) skip =
        RunningOnMetal
    @test SpecialFunctions.erfc(Int32(2)) ≈
        @jit(SpecialFunctions.erfc(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "erfcinv" begin
    @test SpecialFunctions.erfcinv(0.5) ≈
        @jit(SpecialFunctions.erfcinv(ConcreteRNumber(0.5))) skip = RunningOnMetal
    @test SpecialFunctions.erfcinv(Int32(1)) ≈
        @jit(SpecialFunctions.erfcinv(ConcreteRNumber(Int32(1)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "logerf" begin
    @test SpecialFunctions.logerf(0.5, 0.6) ≈
        @jit(SpecialFunctions.logerf(ConcreteRNumber(0.5), ConcreteRNumber(0.6))) skip =
        RunningOnMetal
    @test SpecialFunctions.logerf(Int32(2), Int32(4)) ≈ @jit(
        SpecialFunctions.logerf(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))
    ) atol = 1e-5 rtol = 1e-3 skip = RunningOnMetal
end

@testset "erfcx" begin
    @test SpecialFunctions.erfcx(0.5) ≈ @jit(SpecialFunctions.erfcx(ConcreteRNumber(0.5))) skip =
        RunningOnMetal
    @test SpecialFunctions.erfcx(Int32(2)) ≈
        @jit(SpecialFunctions.erfcx(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "logerfc" begin
    @test SpecialFunctions.logerfc(0.5) ≈
        @jit(SpecialFunctions.logerfc(ConcreteRNumber(0.5))) skip = RunningOnMetal
    @test SpecialFunctions.logerfc(Int32(2)) ≈
        @jit(SpecialFunctions.logerfc(ConcreteRNumber(Int32(2)))) skip = RunningOnMetal
end

@testset "logerfcx" begin
    @test SpecialFunctions.logerfcx(0.5) ≈
        @jit(SpecialFunctions.logerfcx(ConcreteRNumber(0.5))) skip = RunningOnMetal
    @test SpecialFunctions.logerfcx(Int32(2)) ≈
        @jit(SpecialFunctions.logerfcx(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3 skip =
        RunningOnMetal
end

@testset "loggamma1p" begin
    @test SpecialFunctions.loggamma1p(0.5) ≈
        (@jit SpecialFunctions.loggamma1p(ConcreteRNumber(0.5))) skip = RunningOnMetal
end

@testset "loggammadiv" begin
    @test SpecialFunctions.loggammadiv(Int32(150), Int32(20)) ≈
        (@jit SpecialFunctions.loggammadiv(
        ConcreteRNumber(Int32(150)), ConcreteRNumber(Int32(20))
    )) skip = RunningOnMetal
end

@testset "zeta" begin
    s = Reactant.to_rarray([1.0, 2.0, 50.0])
    z = Reactant.to_rarray([1e-8, 0.001, 2.0])
    @test SpecialFunctions.zeta.(Array(s), Array(z)) ≈ (@jit SpecialFunctions.zeta.(s, z)) skip =
        RunningOnMetal
end
