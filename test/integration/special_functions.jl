using SpecialFunctions, Reactant

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

macro ≈(a, b)
    return quote
        isapprox($a, $b; atol=1e-14)
    end
end

@testset "gamma" begin
    @test SpecialFunctions.gamma(0.5) ≈ @jit(SpecialFunctions.gamma(ConcreteRNumber(0.5))) atol =
        1e-5 rtol = 1e-3
    @test SpecialFunctions.gamma(Int32(2)) ≈
        @jit(SpecialFunctions.gamma(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "loggamma" begin
    @test SpecialFunctions.loggamma(0.5) ≈
        @jit(SpecialFunctions.loggamma(ConcreteRNumber(0.5))) atol = 1e-5 rtol = 1e-3
    @test SpecialFunctions.loggamma(Int32(2)) ≈
        @jit(SpecialFunctions.loggamma(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "digamma" begin
    @test SpecialFunctions.digamma(0.5) ≈
        @jit(SpecialFunctions.digamma(ConcreteRNumber(0.5)))
    @test SpecialFunctions.digamma(Int32(2)) ≈
        @jit(SpecialFunctions.digamma(ConcreteRNumber(Int32(2))))
end

@testset "trigamma" begin
    @test SpecialFunctions.trigamma(0.5) ≈
        @jit(SpecialFunctions.trigamma(ConcreteRNumber(0.5)))
    @test SpecialFunctions.trigamma(Int32(2)) ≈
        @jit(SpecialFunctions.trigamma(ConcreteRNumber(Int32(2))))
end

@testset "beta" begin
    @test SpecialFunctions.beta(0.5, 0.6) ≈
        @jit(SpecialFunctions.beta(ConcreteRNumber(0.5), ConcreteRNumber(0.6)))
    @test SpecialFunctions.beta(Int32(2), Int32(4)) ≈
        @jit(SpecialFunctions.beta(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4))))
end

@testset "logbeta" begin
    @test SpecialFunctions.logbeta(0.5, 0.6) ≈
        @jit(SpecialFunctions.logbeta(ConcreteRNumber(0.5), ConcreteRNumber(0.6)))
    @test SpecialFunctions.logbeta(Int32(2), Int32(4)) ≈ @jit(
        SpecialFunctions.logbeta(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))
    )
end

@testset "erf" begin
    @test SpecialFunctions.erf(0.5) ≈ @jit(SpecialFunctions.erf(ConcreteRNumber(0.5)))
    @test SpecialFunctions.erf(Int32(2)) ≈
        @jit(SpecialFunctions.erf(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "erf with 2 arguments" begin
    @test SpecialFunctions.erf(0.5, 0.6) ≈
        @jit(SpecialFunctions.erf(ConcreteRNumber(0.5), ConcreteRNumber(0.6)))
    @test SpecialFunctions.erf(Int32(2), Int32(4)) ≈
        @jit(SpecialFunctions.erf(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))) atol =
        1e-5 rtol = 1e-3
end

@testset "erfc" begin
    @test SpecialFunctions.erfc(0.5) ≈ @jit(SpecialFunctions.erfc(ConcreteRNumber(0.5)))
    @test SpecialFunctions.erfc(Int32(2)) ≈
        @jit(SpecialFunctions.erfc(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "logerf" begin
    @test SpecialFunctions.logerf(0.5, 0.6) ≈
        @jit(SpecialFunctions.logerf(ConcreteRNumber(0.5), ConcreteRNumber(0.6)))
    @test SpecialFunctions.logerf(Int32(2), Int32(4)) ≈ @jit(
        SpecialFunctions.logerf(ConcreteRNumber(Int32(2)), ConcreteRNumber(Int32(4)))
    ) atol = 1e-5 rtol = 1e-3
end

@testset "erfcx" begin
    @test SpecialFunctions.erfcx(0.5) ≈ @jit(SpecialFunctions.erfcx(ConcreteRNumber(0.5)))
    @test SpecialFunctions.erfcx(Int32(2)) ≈
        @jit(SpecialFunctions.erfcx(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "logerfc" begin
    @test SpecialFunctions.logerfc(0.5) ≈
        @jit(SpecialFunctions.logerfc(ConcreteRNumber(0.5)))
    @test SpecialFunctions.logerfc(Int32(2)) ≈
        @jit(SpecialFunctions.logerfc(ConcreteRNumber(Int32(2))))
end

@testset "logerfcx" begin
    @test SpecialFunctions.logerfcx(0.5) ≈
        @jit(SpecialFunctions.logerfcx(ConcreteRNumber(0.5)))
    @test SpecialFunctions.logerfcx(Int32(2)) ≈
        @jit(SpecialFunctions.logerfcx(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "loggamma1p" begin
    @test SpecialFunctions.loggamma1p(0.5) ≈
        @jit SpecialFunctions.loggamma1p(ConcreteRNumber(0.5))
end

@testset "loggammadiv" begin
    @test SpecialFunctions.loggammadiv(Int32(150), Int32(20)) ≈
        @jit SpecialFunctions.loggammadiv(
        ConcreteRNumber(Int32(150)), ConcreteRNumber(Int32(20))
    )
end

@testset "zeta" begin
    s = Reactant.to_rarray([1.0, 2.0, 50.0])
    z = Reactant.to_rarray([1e-8, 0.001, 2.0])
    @test SpecialFunctions.zeta.(Array(s), Array(z)) ≈ @jit SpecialFunctions.zeta.(s, z)
end

@testset "bessel" begin
    x = ConcreteRNumber(0.5)

    @test contains(
        repr(@code_hlo(SpecialFunctions.besselj0(x))), "enzymexla.special.besselj"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselj1(x))), "enzymexla.special.besselj"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselj(2.0, x))), "enzymexla.special.besselj"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besseljx(2.0, x))), "enzymexla.special.besseljx"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.sphericalbesselj(2.0, x))),
        "enzymexla.special.sphericalbesselj",
    )

    @test contains(
        repr(@code_hlo(SpecialFunctions.bessely0(x))), "enzymexla.special.bessely"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.bessely1(x))), "enzymexla.special.bessely"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.bessely(2.0, x))), "enzymexla.special.bessely"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselyx(2.0, x))), "enzymexla.special.besselyx"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.sphericalbessely(2.0, x))),
        "enzymexla.special.sphericalbessely",
    )

    @test contains(
        repr(@code_hlo(SpecialFunctions.besselh(2.0, 1, x))), "enzymexla.special.besselh"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselh(2.0, 2, x))), "enzymexla.special.besselh"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselh(2.0, ConcreteRNumber(1.0), x))),
        "enzymexla.special.besselh",
    )

    @test contains(
        repr(@code_hlo(SpecialFunctions.hankelh1(2.0, x))), "enzymexla.special.besselh"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.hankelh1x(2.0, x))), "enzymexla.special.hankelh1x"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.hankelh2(2.0, x))), "enzymexla.special.besselh"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.hankelh2x(2.0, x))), "enzymexla.special.hankelh2x"
    )

    @test contains(
        repr(@code_hlo(SpecialFunctions.besseli(2.0, x))), "enzymexla.special.besseli"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselix(2.0, x))), "enzymexla.special.besselix"
    )

    @test contains(
        repr(@code_hlo(SpecialFunctions.besselk(2.0, x))), "enzymexla.special.besselk"
    )
    @test contains(
        repr(@code_hlo(SpecialFunctions.besselkx(2.0, x))), "enzymexla.special.besselkx"
    )

    @test contains(repr(@code_hlo(SpecialFunctions.jinc(x))), "enzymexla.special.jinc")
end
