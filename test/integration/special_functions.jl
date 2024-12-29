using SpecialFunctions, Reactant

macro ≈(a, b)
    return quote
        isapprox($a, $b; atol=1e-14)
    end
end

@testset "$op" for (op, n_args) in [
    (:gamma, 1),
    (:loggamma, 1),
    (:digamma, 1),
    (:trigamma, 1),
    (:beta, 2),
    (:logbeta, 2),
    (:erf, 1),
    (:erf, 2),
    (:erfc, 1),
    (:logerf, 2),
    (:erfcx, 1),
    (:logerfc, 1),
    (:logerfcx, 1),
]
    for data in ([0.5, 0.6], [2, 4])
        x = data[1:n_args]
        @eval @test @≈ @jit(SpecialFunctions.$op(ConcreteRNumber.($x)...)) SpecialFunctions.$op(
            $x...
        )
    end
end

@testset "loggamma1p" begin
    @test SpecialFunctions.loggamma1p(0.5) ≈
        @jit SpecialFunctions.loggamma1p(ConcreteRNumber(0.5))
end

@testset "loggammadiv" begin
    @test SpecialFunctions.loggammadiv(150, 20) ≈
        @jit SpecialFunctions.loggammadiv(ConcreteRNumber(150), ConcreteRNumber(20))
end

@testset "zeta" begin
    s = ConcreteRArray([1.0, 2.0, 50.0])
    z = ConcreteRArray([1e-8, 0.001, 2.0])
    @test SpecialFunctions.zeta.(Array(s), Array(z)) ≈ @jit SpecialFunctions.zeta.(s, z)
end
