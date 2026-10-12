using SpecialFunctions, Reactant, Enzyme

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

@testset "erfinv" begin
    @test SpecialFunctions.erfinv(0.5) ≈ @jit(SpecialFunctions.erfinv(ConcreteRNumber(0.5)))
    @test SpecialFunctions.erfinv(Int32(0)) ≈
        @jit(SpecialFunctions.erfinv(ConcreteRNumber(Int32(0)))) atol = 1e-5 rtol = 1e-3
end

@testset "erfc" begin
    @test SpecialFunctions.erfc(0.5) ≈ @jit(SpecialFunctions.erfc(ConcreteRNumber(0.5)))
    @test SpecialFunctions.erfc(Int32(2)) ≈
        @jit(SpecialFunctions.erfc(ConcreteRNumber(Int32(2)))) atol = 1e-5 rtol = 1e-3
end

@testset "erfcinv" begin
    @test SpecialFunctions.erfcinv(0.5) ≈
        @jit(SpecialFunctions.erfcinv(ConcreteRNumber(0.5)))
    @test SpecialFunctions.erfcinv(Int32(1)) ≈
        @jit(SpecialFunctions.erfcinv(ConcreteRNumber(Int32(1)))) atol = 1e-5 rtol = 1e-3
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

@testset "complex error functions" begin
    # https://github.com/EnzymeAD/Reactant.jl/issues/2874
    z = 1.0 + 0.5im
    @test SpecialFunctions.erf(z) ≈ @jit(SpecialFunctions.erf(ConcreteRNumber(z))) rtol =
        1e-13

    grid = [Complex{Float64}(x, y) for x in -6:0.5:6, y in -6:0.5:6]
    extra = ComplexF64[0, 1e-10, 1e-3 - 2e-3im, 0.999im, 2im, 5.0 + 1.0im, 12.0 + 3.0im]
    pts = vcat(vec(grid), extra, -extra, conj.(extra), im .* extra)

    # relative to max(|ref|, 1): erf gets huge along the imaginary axis. Points
    # where the reference overflows (e.g. erf(-3 + 12im) in Float32) are skipped.
    function maxerr(f, pts, got)
        return maximum(
            abs(ComplexF64(a) - ComplexF64(b)) / max(abs(ComplexF64(a)), 1) for
            (a, b) in zip(f.(pts), got) if isfinite(a)
        )
    end

    @testset "$f $T" for f in (
            SpecialFunctions.erf,
            SpecialFunctions.erfc,
            SpecialFunctions.erfcx,
            SpecialFunctions.erfi,
            SpecialFunctions.faddeeva,
        ),
        (T, tol) in ((ComplexF64, 1e-13), (ComplexF32, 2e-6))

        x = T.(pts)
        x_ra = Reactant.to_rarray(x)
        got = Array(@jit f.(x_ra))
        @test eltype(got) == T
        @test maxerr(f, x, got) < tol
        # scalar
        @test f(x[1]) ≈ @jit(f(ConcreteRNumber(x[1]))) atol = tol
    end

    # relative accuracy near the origin
    for z in (1e-10 + 1e-10im, 1e-3 - 2e-3im, 0.3 + 0.2im)
        @test SpecialFunctions.erf(z) ≈
            @jit(SpecialFunctions.erf(ConcreteRNumber(z))) rtol = 1e-13
        @test SpecialFunctions.erfi(z) ≈
            @jit(SpecialFunctions.erfi(ConcreteRNumber(z))) rtol = 1e-13
    end
    # erfc keeps relative accuracy for large positive real part
    for z in (5.0 + 1.0im, 10.0 + 3.0im, 20.0 - 2.0im)
        @test SpecialFunctions.erfc(z) ≈
            @jit(SpecialFunctions.erfc(ConcreteRNumber(z))) rtol = 1e-13
    end
    # large arguments
    @test @jit(SpecialFunctions.erf(ConcreteRNumber(30.0 + 0.0im))) == 1
    @test @jit(SpecialFunctions.erf(ConcreteRNumber(-30.0 + 0.0im))) == -1
    @test @jit(SpecialFunctions.erf(ConcreteRNumber(1e200 + 0.0im))) == 1
    @test @jit(SpecialFunctions.erfc(ConcreteRNumber(1e200 + 0.0im))) == 0

    # real inputs of the newly added functions
    @test SpecialFunctions.erfi(0.7) ≈ @jit(SpecialFunctions.erfi(ConcreteRNumber(0.7)))
    @test SpecialFunctions.erfi(2) ≈ @jit(SpecialFunctions.erfi(ConcreteRNumber(2)))
    @test SpecialFunctions.faddeeva(0.7) ≈
        @jit(SpecialFunctions.faddeeva(ConcreteRNumber(0.7)))

    # differentiable: d/dz erf(z) = 2/√π exp(-z²)
    f(x, y) = sum(real.(SpecialFunctions.erf.(Complex.(x, y))))
    xs = [1.0, 0.3, -2.0, 0.0, 0.0, 3.0, -0.7, 1e-12, 30.0]
    ys = [0.5, -0.2, 1.5, 0.0, 2.0, 0.0, -3.1, 0.0, 1.0]
    d = 2 / sqrt(pi) .* exp.(-Complex.(xs, ys) .^ 2)
    gx, gy = @jit Enzyme.gradient(
        Reverse, f, Reactant.to_rarray(xs), Reactant.to_rarray(ys)
    )
    @test Array(gx) ≈ real.(d) rtol = 1e-10 atol = 1e-12
    @test Array(gy) ≈ -imag.(d) rtol = 1e-10 atol = 1e-12
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
