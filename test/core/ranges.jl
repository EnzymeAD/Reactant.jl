using Reactant, Test

@testset "ranges" begin
    i = Reactant.to_rarray(5; track_numbers=true)
    @test Array{Int64}(@jit(1:i)) == collect(1:5)
    @test Array{Int64}(@jit(i:10)) == collect(5:10)
    j = Reactant.to_rarray(10; track_numbers=true)
    @test Array{Int64}(@jit(i:j)) == collect(5:10)
end

broadcast_over_range(a, kx, ky) = a .* (kx .^ 2 .+ ky' .^ 2)

@testset "broadcast over ranges" begin
    a = Reactant.TestUtils.construct_test_array(Float32, 32, 32)
    a_ra = Reactant.to_rarray(a)
    kx = 1:32
    ky = 30:61
    kx_ra = Reactant.to_rarray(kx)
    ky_ra = Reactant.to_rarray(ky)

    @test kx_ra isa Reactant.TracedUnitRange
    @test ky_ra isa Reactant.TracedUnitRange
    @test @jit(broadcast_over_range(a_ra, kx_ra, ky_ra)) ≈ broadcast_over_range(a, kx, ky)
end
