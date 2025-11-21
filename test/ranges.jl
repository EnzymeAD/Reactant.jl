using Reactant, Test

@testset "ranges" begin
    i = Reactant.to_rarray(5; track_numbers=true)
    @test Array{Int64}(@jit(1:i)) == collect(1:5)
    @test Array{Int64}(@jit(i:10)) == collect(5:10)
    j = Reactant.to_rarray(10; track_numbers=true)
    @test Array{Int64}(@jit(i:j)) == collect(5:10)
end
