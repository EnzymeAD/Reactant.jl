using Reactant, Test, FixedSizeArrays

fn(x, y) = (2 .* x .- 3) * y'

@testset "FixedSizeArrays" begin
    @testset "1D" begin
        x = FixedSizeArray(fill(3.0f0, 100))
        rx = Reactant.to_rarray(x)
        @test rx isa FixedSizeArray{Float32,1}

        res = @jit fn(rx, rx)
        @test res ≈ fn(x, x)
    end

    @testset "2D" begin
        x = FixedSizeArray(fill(3.0f0, (4, 5)))
        rx = Reactant.to_rarray(x)
        @test rx isa FixedSizeArray{Float32,2}

        res = @jit fn(rx, rx)
        @test res isa FixedSizeArray
        @test @allowscalar(Array(res)) ≈ fn(x, x)
    end
end
