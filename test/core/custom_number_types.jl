using Float8s, DLFP8Types, Reactant
using Reactant: TracedRNumber

@testset "Custom number types: $(T)" for T in [Float8_4, Float8_E4M3FN, Float8_E4M3FNUZ]
    x = T[
        -1.125 -0.21875 1.12
        1.875 0.4375 1.0
        0.5625 -1.0 0.937
        -0.375 -0.34375 -0.6875
        0.46875 0.75 -0.23437
        -0.6875 -0.203125 0.375
        0.875 -0.8125 2.5
        -0.6875 -0.1171875 -1.625
        0.75 0.9375 1.0
        0.5 0.203125 1.75
    ]
    x_64 = Float64.(x)
    x_ra = Reactant.to_rarray(x)

    @testset "Reductions" begin
        sumall(x) = TracedRNumber{Float64}(sum(x))

        @test @jit(sumall(x_ra)) ≈ sum(x_64) atol = 1e-1 rtol = 1e-1

        sum1(x) = TracedRNumber{Float64}.(sum(x; dims=1))
        sum2(x) = TracedRNumber{Float64}.(sum(x; dims=2))
        sum12(x) = TracedRNumber{Float64}.(sum(x; dims=(1, 2)))

        @test @jit(sum1(x_ra)) ≈ sum(x_64; dims=1) atol = 1e-1 rtol = 1e-1
        @test @jit(sum2(x_ra)) ≈ sum(x_64; dims=2) atol = 1e-1 rtol = 1e-1
        @test @jit(sum12(x_ra)) ≈ sum(x_64; dims=(1, 2)) atol = 1e-1 rtol = 1e-1
    end

    @testset "Broadcasting" begin
        fn(x) = TracedRNumber{Float64}.(x .+ 1)
        @test @jit(fn(x_ra)) ≈ (x_64 .+ 1) atol = 1e-1 rtol = 1e-1
    end
end
