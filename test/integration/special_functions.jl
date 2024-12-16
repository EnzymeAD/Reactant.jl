using SpecialFunctions, Reactant
@testset "Generic" begin
    values = [0.5, 0.6]
    for (op, n_args) in [
        (:gamma, 1),
        (:loggamma, 1),
        (:loggamma1p, 1),
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
        x = values[1:n_args]
        @eval @test @jit(SpecialFunctions.$op(ConcreteRNumber.($x)...)) ≈
            SpecialFunctions.$op($x...)
    end
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
