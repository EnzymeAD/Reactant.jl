using Test
using Reactant

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

@testset "Complex runtime: $CT" for CT in (ComplexF32, ComplexF64)
    @test begin
        a = Reactant.to_rarray(ones(CT, 2))
        b = Reactant.to_rarray(ones(CT, 2))
        c = Reactant.compile(+, (a, b))(a, b)
        c == ones(CT, 2) + ones(CT, 2)
    end skip = CT == ComplexF64 && RunningOnTPU
end

const SCALAR_LIST = (1.0, 1.0 + 2.0im)

const ARRAY_LIST = (
    fill(1.0 + 2.0im),
    fill(1.0),
    [1.0 + 2.0im; 3.0 + 4.0im],
    [1.0; 3.0],
    [1.0 + 2.0im 3.0 + 4.0im],
    [1.0 2.0],
    [1.0+2.0im 3.0+4.0im; 5.0+6.0im 7.0+8.0im],
    [1.0 3.0; 5.0 7.0],
)

@testset "$(string(fn))" for fn in (conj, conj!, real, imag)
    if !endswith(string(fn), "!")
        @testset "$(typeof(x))" for x in SCALAR_LIST
            @test begin
                x_concrete = Reactant.to_rarray(x)
                only(@jit(fn(x_concrete))) == fn(x)
            end skip = RunningOnTPU && eltype(x) == ComplexF64
        end
    end

    @testset "$(typeof(x))" for x in ARRAY_LIST
        @test begin
            x_concrete = Reactant.to_rarray(x)
            @jit(fn(x_concrete)) == fn(x)
        end skip = RunningOnTPU && eltype(x) == ComplexF64
    end
end

@testset "abs: $T" for T in (Float32, ComplexF32)
    x = randn(T, 10)
    x_concrete = Reactant.to_rarray(x)
    @test @jit(abs.(x_concrete)) ≈ abs.(x)
end

@testset "promote_to Complex" begin
    x = ComplexF32(1.0 + 2.0im)
    y = ConcreteRNumber(x)

    f = Reactant.compile((y,)) do z
        z + Reactant.TracedUtils.promote_to(
            Reactant.TracedRNumber{ComplexF32}, ComplexF32(1.0 - 3.0im)
        )
    end

    @test isapprox(f(y), ComplexF32(2.0 - 1.0im))
end

@testset "complex reduction" begin
    x = randn(ComplexF32, 10, 10)
    x_ra = Reactant.to_rarray(x)
    @test @jit(sum(abs2, x_ra)) ≈ sum(abs2, x)
end

@testset "create complex numbers" begin
    x = randn(ComplexF32)
    x_ra = Reactant.to_rarray(x; track_numbers=true)
    @test @jit(Complex(x_ra)) == x_ra

    x = randn(Float32)
    y = randn(Float64)
    x_ra = Reactant.to_rarray(x; track_numbers=true)
    y_ra = Reactant.to_rarray(y; track_numbers=true)
    @test @jit(Complex(x_ra, y_ra)) == Complex(x, y) skip = RunningOnTPU
    @test @jit(Complex(x_ra, y)) == Complex(x, y) skip = RunningOnTPU
    @test @jit(Complex(x, y_ra)) == Complex(x, y) skip = RunningOnTPU
    @test @jit(Complex(x_ra)) == Complex(x) == @jit(Complex(x_ra, 0))
end
