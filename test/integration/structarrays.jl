using StructArrays, StaticArrays, Reactant, LinearAlgebra, Test

@testset "StructArray to_rarray and make_tracer" begin
    x = StructArray(;
        a=rand(10, 2), b=fill("some strings", (10, 2)), c=rand(Float32, 10, 2)
    )
    x_ra = Reactant.to_rarray(x)

    # Note that the element type (the NamedTuple) contains ConcreteRNumbers even though track_numbers is not enabled.
    # This is because when the backing arrays are converted to TracedRArrays, their elements will contain TracedRNumbers.
    # In order for the element type to match the backing arrays, we need to use ConcreteRNumbers here as well:
    @test typeof(x_ra) <: (StructArray{
        <:NamedTuple{
            (:a, :b, :c),
            <:Tuple{<:ConcreteRNumber{Float64},String,<:ConcreteRNumber{Float32}},
        },
        2,
        <:NamedTuple{
            (:a, :b, :c),
            <:Tuple{<:ConcreteRArray{Float64,2},Matrix{String},<:ConcreteRArray{Float32,2}},
        },
        CartesianIndex{2},
    })

    @test typeof(
        Reactant.make_tracer(Reactant.OrderedIdDict(), x_ra, (), Reactant.ConcreteToTraced)
    ) == StructArray{
        @NamedTuple{
            a::Reactant.TracedRNumber{Float64},
            b::String,
            c::Reactant.TracedRNumber{Float32},
        },
        2,
        @NamedTuple{
            a::Reactant.TracedRArray{Float64,2},
            b::Matrix{String},
            c::Reactant.TracedRArray{Float32,2},
        },
        Int64,
    }
end

@noinline function elwise(e::NamedTuple)
    return (; c=e.b, d=sin(e.a))
end

function broadcast_elwise(x)
    return elwise.(x)
end

@testset "structarray broadcasting" begin
    x = StructVector(; a=rand(10), b=rand(Float32, 10))

    x_ra = Reactant.to_rarray(x)

    result = @jit broadcast_elwise(x_ra)

    @test typeof(result) <: StructVector{
        <:NamedTuple{
            (:c, :d),<:Tuple{<:ConcreteRNumber{Float32},<:ConcreteRNumber{Float64}}
        },
        <:NamedTuple{
            (:c, :d),<:Tuple{<:ConcreteRArray{Float32,1},<:ConcreteRArray{Float64,1}}
        },
        CartesianIndex{1},
    }
    for (component_ra, component) in
        zip(components(result), components(broadcast_elwise(x)))
        @test component_ra ≈ component
    end
end

@testset "structarray with static array broadcasting" begin
    trel(x) = tr.(x)
    s1 = StructArray{SMatrix{2,2,Float64,4}}((
        fill(1.0, 4), fill(2.0, 4), fill(3.0, 4), fill(4.0, 4)
    ))
    sr = Reactant.to_rarray(s1)
    out = @jit(trel(sr))
    @test out ≈ trel(s1)
    @test out isa ConcreteRArray
    @test @jit(sum(sr)) ≈ sum(s1)

    s2 = StructArray{SMatrix{2,2,Float64,4}}((
        fill(1.0, 4, 4), fill(2.0, 4, 4), fill(3.0, 4, 4), fill(4.0, 4, 4)
    ))
    sr = Reactant.to_rarray(s2)
    out = @jit(trel(sr))
    @test out ≈ trel(s2)
    @test out isa ConcreteRArray
    @test @jit(sum(sr)) ≈ sum(s2)
end

@testset "structarray with complex numbers" begin
    s = randn(64)

    elcom(x) = complex(x, x)
    sr = Reactant.to_rarray(s)
    out = @jit(elcom.(sr))
    @test out ≈ elcom.(s)
    @test out isa ConcreteRArray
    @test @jit(sum(sr)) ≈ sum(s)
end

struct BcastFoo{T}
    a::T
    b::T
end
struct BcastBar{T}
    p::BcastFoo{T}
    c::T
end

bcast_h(s) = BcastFoo(s.a * s.b, s.a)
bcast_hconst(s) = BcastFoo(one(s.a), s.a)
bcast_hscalar(s) = s.a + s.b
bcast_hmix(s, x, y) = BcastFoo(s.a * x + y, s.b - x)
bcast_hnest(s) = BcastBar(BcastFoo(s.a + s.b, s.a - s.b), s.a * s.b)
bcast_hbar(b) = BcastFoo(b.p.a * b.c, b.p.b + b.c)

function bcast_foos(P)
    return StructArray{BcastFoo{eltype(P)}}((view(P, :, :, 1), view(P, :, :, 2)))
end
function bcast_via_structarray(P)
    Pn = similar(P)
    bcast_foos(Pn) .= bcast_h.(bcast_foos(P))
    return Pn
end
function bcast_inplace!(dest, sa)
    dest .= bcast_h.(sa)
    return dest
end
function bcast_closure(sa, x)
    c = @allowscalar x[1]
    return (s -> BcastFoo(s.a * c, s.b)).(sa)
end

function compare_components(r, c)
    for (rc, cc) in zip(Tuple(components(r)), Tuple(components(c)))
        @test Array(rc) ≈ cc
    end
end

# https://github.com/EnzymeAD/Reactant.jl/issues/3389
@testset "structarray broadcasting is batched (no while loop)" begin
    P = rand(8, 8, 2)
    P_ra = Reactant.to_rarray(P)

    hlo = repr(@code_hlo bcast_via_structarray(P_ra))
    @test !occursin("stablehlo.while", hlo)
    @test !occursin("stablehlo.dynamic_update_slice", hlo)
    @test Array(@jit(bcast_via_structarray(P_ra))) ≈ bcast_via_structarray(P)

    sa = StructArray{BcastFoo{Float64}}((rand(4, 3), rand(4, 3)))
    sa_ra = Reactant.to_rarray(sa)

    # struct -> struct
    @test !occursin("stablehlo.while", repr(@code_hlo bcast_h.(sa_ra)))
    res = @jit bcast_h.(sa_ra)
    @test res isa StructArray
    compare_components(res, bcast_h.(sa))

    # struct -> struct with a constant field
    @test !occursin("stablehlo.while", repr(@code_hlo bcast_hconst.(sa_ra)))
    compare_components(@jit(bcast_hconst.(sa_ra)), bcast_hconst.(sa))

    # struct -> scalar
    @test !occursin("stablehlo.while", repr(@code_hlo bcast_hscalar.(sa_ra)))
    res = @jit bcast_hscalar.(sa_ra)
    @test res isa ConcreteRArray
    @test Array(res) ≈ bcast_hscalar.(sa)

    # mixed StructArray / array / scalar arguments
    x = rand(4, 3)
    x_ra = Reactant.to_rarray(x)
    fmix(sa, x) = bcast_hmix.(sa, x, 2.0)
    @test !occursin("stablehlo.while", repr(@code_hlo fmix(sa_ra, x_ra)))
    compare_components(@jit(fmix(sa_ra, x_ra)), fmix(sa, x))

    # nested structs (nested StructArray result), chained broadcasts
    @test !occursin("stablehlo.while", repr(@code_hlo bcast_hnest.(sa_ra)))
    res = @jit bcast_hnest.(sa_ra)
    res_jl = bcast_hnest.(sa)
    @test Array(res.p.a) ≈ getfield.(res_jl.p, :a)
    @test Array(res.p.b) ≈ getfield.(res_jl.p, :b)
    @test Array(res.c) ≈ res_jl.c
    fnest(sa) = bcast_hbar.(bcast_hnest.(sa))
    @test !occursin("stablehlo.while", repr(@code_hlo fnest(sa_ra)))
    compare_components(@jit(fnest(sa_ra)), fnest(sa))

    # in-place into a StructArray destination
    dest = Reactant.to_rarray(StructArray{BcastFoo{Float64}}((zeros(4, 3), zeros(4, 3))))
    @test !occursin("stablehlo.while", repr(@code_hlo bcast_inplace!(dest, sa_ra)))
    @jit bcast_inplace!(dest, sa_ra)
    compare_components(dest, bcast_h.(sa))

    # closures capturing traced values still work (via the loop fallback)
    compare_components(@jit(bcast_closure(sa_ra, x_ra)), bcast_closure(sa, x))
end
