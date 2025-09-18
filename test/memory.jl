using Reactant, Test

fn(x,y) = sin.(x) .+ cos.(y)

@testset "Memory test" begin
    x = Memory{Float32}(fill(2.0f0, 10))
    x_ra = Reactant.to_rarray(x)

    @test @jit(fn(x_ra,x_ra)) ≈ fn(x,x)
end

