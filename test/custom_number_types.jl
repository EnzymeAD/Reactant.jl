using Float8s, Reactant
using Reactant: TracedRNumber

Reactant.reactant_primitive(::Type{Float8_4}) = Reactant.F8E4M3FNUZ

x = Float8_4.(rand(Float32, 10, 3))
x_ra = Reactant.to_rarray(x)

@testset "Reductions" begin
    sumall(x) = TracedRNumber{Float64}(sum(x))

    @test @jit(sumall(x_ra)) ≈ sumall(x)

    sum1(x) = sum(x; dims=1)
    sum2(x) = sum(x; dims=2)
    sum12(x) = sum(x; dims=(1, 2))

    @test @jit(sum1(x_ra)) ≈ sum1(x)
    @test @jit(sum2(x_ra)) ≈ sum2(x)
    @test @jit(sum12(x_ra)) ≈ sum12(x)
end
