using Reactant
using StaticArrays
using Test
using LinearAlgebra

@testset "StaticArrays" begin
    x = SMatrix{2, 2}(rand(2,2))
    x_ra = Reactant.to_rarray(x; track_numbers=true)

    @test typeof(x_ra) <: SMatrix{2, 2, <:Reactant.RNumber}
    @test Reactant.materialize_traced_array(x_ra) == x

    out = @jit map(abs, x_ra)
    @test out ≈ map(abs, x)
    @test out isa SMatrix{2, 2, <:Reactant.RNumber}

    out = @jit mapreduce(abs, +, x_ra)
    @test out ≈ mapreduce(abs, +, x)
    @test out isa Reactant.RNumber

    y = SMatrix{2, 3}(rand(2,3))
    y_ra = Reactant.to_rarray(y; track_numbers=true)
    out = @jit x_ra * y_ra
    @test out ≈ x * y
    @test out isa SMatrix{2, 3, <:Reactant.RNumber}

    v = SVector{3}(rand(3))
    u = SVector{2}(rand(2))
    v_ra = Reactant.to_rarray(v; track_numbers=true)
    u_ra = Reactant.to_rarray(u; track_numbers=true)
    out = @jit dot(u_ra, y_ra, v_ra)
    @test out ≈ dot(u, y, v)
    @test out isa Reactant.RNumber

    out = @jit dot(u_ra, u_ra)
    @test out ≈ dot(u, u)
    @test out isa Reactant.RNumber

    out = @jit dot(x_ra, x_ra)
    @test out ≈ dot(x, x)
    @test out isa Reactant.RNumber

end