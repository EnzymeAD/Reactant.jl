using Reactant, Test, FillArrays

fn(x, y) = (2 .* x .- 3) * y'

@testset "Fill" begin
    x = Fill(2.0f0, 4, 5)
    rx = Reactant.to_rarray(x)

    @test @jit(fn(rx, rx)) ≈ fn(x, x)

    @testset "Ones" begin
        y = Ones(Float32, 4, 5)
        ry = Reactant.to_rarray(y)
        @test @jit(fn(rx, ry)) ≈ fn(x, y)
    end

    @testset "Zeros" begin
        y = Zeros(Float32, 4, 5)
        ry = Reactant.to_rarray(y)
        @test @jit(fn(rx, ry)) ≈ fn(x, y)
    end
end

fn2(x, y) = ((2 .* x .- 3) * y')[:, 3]

@testset "OneElement" begin
    x = OneElement(3.4f0, (3, 4), (32, 32))
    rx = Reactant.to_rarray(x)

    @test @jit(fn2(rx, rx)) ≈ fn2(x, x)
end
