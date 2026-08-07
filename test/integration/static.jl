using Test
using Reactant
using Static

@testset "type promotion" begin
    for (sv, T) in ((static(true), Bool), (static(2), Int), (static(1.5), Float64))
        @test Reactant.unwrapped_eltype(typeof(sv)) == T
        @test promote_type(typeof(sv), Reactant.TracedRNumber{Float64}) ==
            Reactant.traced_number_type(promote_type(T, Float64))
        @test promote_type(Reactant.TracedRNumber{Float64}, typeof(sv)) ==
            Reactant.traced_number_type(promote_type(T, Float64))
    end
end

@testset "arithmetic with static numbers" begin
    x = Reactant.to_rarray(rand(4))

    f1(x) = sum(x) + static(1.5)
    @test Float64(@jit f1(x)) ≈ f1(Array(x))

    f2(x) = static(2) * sum(x)
    @test Float64(@jit f2(x)) ≈ f2(Array(x))

    f3(x) = x .+ static(1.0)
    @test Array(@jit f3(x)) ≈ f3(Array(x))
end
