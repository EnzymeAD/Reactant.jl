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

@testset "indexing" begin
    @testset "scalar" begin
        fscalar(x) = x[3]

        N = 4
        for x in (Fill(2.0f0, N), Ones(Float32, N), Zeros(Float32, N))
            rx = Reactant.to_rarray(x)
            @test @jit(fscalar(rx)) ≈ fscalar(x)
        end

        N = 2
        for x in (Fill(2.0f0, N), Ones(Float32, N), Zeros(Float32, N))
            rx = Reactant.to_rarray(x)
            @test_throws BoundsError @jit(fscalar(rx))
        end
    end

    @testset "slice" begin
        fslice(x) = x[3:5, 2]

        N = (6, 3)
        for x in (Fill(2.0f0, N...), Ones(Float32, N...), Zeros(Float32, N...))
            rx = Reactant.to_rarray(x)
            @test @jit(fslice(rx)) ≈ fslice(x)
        end

        N = (4, 3)
        for x in (Fill(2.0f0, N...), Ones(Float32, N...), Zeros(Float32, N...))
            rx = Reactant.to_rarray(x)
            @test_throws BoundsError @jit(fslice(rx))
        end
    end
end
