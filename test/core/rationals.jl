using Reactant, Test

@testset "rationals" begin
    # Test basic conversion
    r = 1//2
    r_ra = Reactant.to_rarray(r; track_numbers=true)
    @test r_ra isa Reactant.TracedRational
    @test numerator(r_ra) isa ConcreteRNumber
    @test denominator(r_ra) isa ConcreteRNumber

    # Test arithmetic operations
    r1 = Reactant.to_rarray(1//2; track_numbers=true)
    r2 = Reactant.to_rarray(1//3; track_numbers=true)

    # Addition
    r_sum = @jit(r1 + r2)
    @test Rational(r_sum) ≈ 1//2 + 1//3

    # Subtraction
    r_diff = @jit(r1 - r2)
    @test Rational(r_diff) ≈ 1//2 - 1//3

    # Multiplication
    r_prod = @jit(r1 * r2)
    @test Rational(r_prod) ≈ 1//2 * 1//3

    # Division
    r_quot = @jit(r1 / r2)
    @test Rational(r_quot) ≈ (1//2) / (1//3)

    # Negation
    r_neg = @jit(-r1)
    @test Rational(r_neg) ≈ -(1//2)
end

@testset "rational conversions" begin
    r = Reactant.to_rarray(3//4; track_numbers=true)

    # Convert to float
    r_float = @jit(float(r))
    @test r_float ≈ 0.75

    # Convert to Float64
    r_f64 = @jit(Float64(r))
    @test r_f64 ≈ 0.75

    # Convert to Float32
    r_f32 = @jit(Float32(r))
    @test r_f32 ≈ 0.75f0
end

@testset "rational construction with //" begin
    # Test // operator with TracedRNumber
    i1 = Reactant.to_rarray(3; track_numbers=true)
    i2 = Reactant.to_rarray(4; track_numbers=true)

    r = @jit(i1//i2)
    @test r isa Reactant.TracedRational
    @test Rational(r) == 3//4

    # Test // with mixed types
    i = Reactant.to_rarray(5; track_numbers=true)
    r2 = @jit(i//2)
    @test Rational(r2) == 5//2

    r3 = @jit(7//i2)
    @test Rational(r3) == 7//4
end
