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
    kx_ra = Reactant.to_rarray(kx; track_numbers=true)
    ky_ra = Reactant.to_rarray(ky; track_numbers=true)

    @test kx_ra isa Reactant.TracedUnitRange
    @test ky_ra isa Reactant.TracedUnitRange
    @test @jit(broadcast_over_range(a_ra, kx_ra, ky_ra)) ≈ broadcast_over_range(a, kx, ky)

    kx_2 = 1.0f0:32.0f0
    ky_2 = 30.0f0:61.0f0
    kx_2_ra = Reactant.to_rarray(kx_2; track_numbers=true)
    ky_2_ra = Reactant.to_rarray(ky_2; track_numbers=true)

    @test kx_2_ra isa Reactant.TracedStepRangeLen
    @test ky_2_ra isa Reactant.TracedStepRangeLen
    @test @jit(broadcast_over_range(a_ra, kx_2_ra, ky_2_ra)) ≈
        broadcast_over_range(a, kx_2, ky_2)
end
