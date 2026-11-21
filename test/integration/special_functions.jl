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

# These tests require execution, rather than only generating Bessel op names.
_besselix_values(n, x) = SpecialFunctions.besselix.(n, x)
function _besselix_gradient(x, n)
    return only(
        Enzyme.gradient(Enzyme.Reverse, y -> sum(SpecialFunctions.besselix.(n, y)), x)
    )
end

@testset "integer-order besselix" begin
    for T in (Float32, Float64)
        orders = repeat([-50, -3, -1, 0, 1, 2, 3, 8, 20, 50, 100], 14)
        arguments = repeat(
            T[
                0,
                1e-12,
                0.1,
                1.5625,
                16,
                32,
                prevfloat(T(64)),
                64,
                nextfloat(T(64)),
                100,
                1000,
                1e5,
                -0.1,
                -100,
            ];
            inner=11,
        )
        rn, rx = Reactant.to_rarray(orders), Reactant.to_rarray(arguments)
        compiled = @compile _besselix_values(rn, rx)
        reference = SpecialFunctions.besselix.(orders, arguments)
        actual = Array(compiled(rn, rx))
        tolerance = T === Float64 ? 5e-12 : 8e-5
        @test all(isapprox.(actual, reference; rtol=tolerance, atol=100floatmin(T)))
        @test eltype(actual) === T
        # Reuse the executable after changing every order and argument.
        rn2 = Reactant.to_rarray(-orders)
        rx2 = Reactant.to_rarray(-arguments)
        @test all(
            isapprox.(
                Array(compiled(rn2, rx2)),
                SpecialFunctions.besselix.(-orders, -arguments);
                rtol=tolerance,
                atol=100floatmin(T),
            ),
        )
        @test T(@jit SpecialFunctions.besselix(3, ConcreteRNumber(T(1.5625)))) ≈
            T(SpecialFunctions.besselix(3, T(1.5625))) rtol = tolerance
    end
    @test Float64(@jit SpecialFunctions.besselix(typemin(Int64), ConcreteRNumber(1.0))) == 0
    @test Float64(@jit SpecialFunctions.besselix(typemax(UInt64), ConcreteRNumber(1.0))) ==
        0
end

@testset "integer-order besselix reverse mode" begin
    for T in (Float32, Float64)
        orders = [0, 1, 2, 3, 8, 20, 50, 100, 0, 1, -3, 4, 2, 1000]
        arguments = T[0, 0, 0, 1.5625, 64, 65, 100, 1000, 1e6, -0.1, -65, -1e6, 1e5, 1e5]
        rn, rx = Reactant.to_rarray(orders), Reactant.to_rarray(arguments)
        compiled = @compile _besselix_gradient(rx, rn)
        reference =
            (
                SpecialFunctions.besselix.(orders .- 1, arguments) .+
                SpecialFunctions.besselix.(orders .+ 1, arguments)
            ) ./ 2 .- sign.(arguments) .* SpecialFunctions.besselix.(orders, arguments)
        actual = Array(compiled(rx, rn))
        @test all(isfinite, actual)
        @test all(
            isapprox.(
                actual,
                reference;
                rtol=(T === Float64 ? 5e-8 : 2e-3),
                atol=(T === Float64 ? 1e-15 : 3e-12),
            ),
        )
        @test actual[1:3] == T[0, 0.5, 0]
    end
end

function _besselix_periodic_weights(rho, orders)
    a = inv(sum(rho)^2)
    return sqrt.(2 .* SpecialFunctions.besselix.(orders, a))
end
function _besselix_periodic_gradient(rho, orders)
    return only(
        Enzyme.gradient(
            Enzyme.Reverse, r -> sum(_besselix_periodic_weights(r, orders)), rho
        ),
    )
end

@testset "periodic Bessel weights and retained loops" begin
    structures = Dict{String,Int}[]
    for k in (3, 5)
        orders = vcat(1:k, 1:k)
        rn, rr = Reactant.to_rarray(orders), Reactant.to_rarray([0.8])
        hlo = repr(@code_hlo optimize = false _besselix_periodic_weights(rr, rn))
        structure = Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
            structure[m.match] = get(structure, m.match, 0) + 1
        end
        push!(structures, structure)
        @test get(structure, "stablehlo.while", 0) > 0
        @test get(structure, "stablehlo.if", 0) > 0
        primal = @compile _besselix_periodic_weights(rr, rn)
        reverse = @compile _besselix_periodic_gradient(rr, rn)
        for rho in (0.8, 0.1, 0.05, 10.0)
            arg = Reactant.to_rarray([rho])
            a = inv(rho^2)
            scaled = SpecialFunctions.besselix.(orders, a)
            weights = sqrt.(2 .* scaled)
            partials =
                (
                    SpecialFunctions.besselix.(orders .- 1, a) .+
                    SpecialFunctions.besselix.(orders .+ 1, a)
                ) ./ 2 .- scaled
            reference_gradient = -2 / rho^3 * sum(partials ./ weights)
            @test Array(primal(arg, rn)) ≈ weights rtol = 5e-12
            @test only(Array(reverse(arg, rn))) ≈ reference_gradient rtol = 2e-9 atol =
                1e-12
        end
    end
    @test structures[1] == structures[2]
end
