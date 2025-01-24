using Float8s, Reactant
using Reactant: TracedRNumber

Reactant.reactant_primitive(::Type{Float8_4}) = Reactant.F8E4M3FN

x = Float8_4.(rand(Float32, 10, 3))
x_64 = Float64.(x)
x_ra = Reactant.to_rarray(x)

@testset "Reductions" begin
    sumall(x) = TracedRNumber{Float64}(sum(x))

    @test @jit(sumall(x_ra)) ≈ sum(x_64) atol=1e-1 rtol=1e-1

    sum1(x) = TracedRNumber{Float64}.(sum(x; dims=1))
    sum2(x) = TracedRNumber{Float64}.(sum(x; dims=2))
    sum12(x) = TracedRNumber{Float64}.(sum(x; dims=(1, 2)))

    @test @jit(sum1(x_ra)) ≈ sum(x_64; dims=1) atol=1e-1 rtol=1e-1
    @test @jit(sum2(x_ra)) ≈ sum(x_64; dims=2) atol=1e-1 rtol=1e-1
    @test @jit(sum12(x_ra)) ≈ sum(x_64; dims=(1, 2)) atol=1e-1 rtol=1e-1
end

@testset "Broadcasting" begin
    fn(x) = TracedRNumber{Float64}.(x .+ 1)
    @test @jit(fn(x_ra)) ≈ (x_64 .+ 1) atol=1e-1 rtol=1e-1
end
