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

struct Pulse2869{T}
    α::T
    β::T
end

@inline pulse_vis_2869(m::Pulse2869, p) = cos(m.α * p.U + m.β * p.V)
@inline pulse_dot_2869(m::Pulse2869, p::Pulse2869) = sin(m.α * p.α + m.β * p.β)

function kernel_2869!(pvis, dp, m)
    pvis .*= pulse_vis_2869.(Ref(m), dp)
    return pvis
end

function kernel_struct_2869!(pvis, dp, m)
    pvis .*= pulse_dot_2869.(Ref(m), dp)
    return pvis
end

@testset "StructArray with host or ConcreteRArray columns in broadcast (#2869)" begin
    u = collect(range(-1.0, 1.0; length=16))
    v = collect(range(0.5, 2.5; length=16))
    m_cpu = Pulse2869(0.078, -0.042)
    m_ra = Pulse2869(ConcreteRNumber(0.078), ConcreteRNumber(-0.042))
    expected = ones(Float64, 16) .* pulse_vis_2869.(Ref(m_cpu), StructArray((U=u, V=v)))

    # Untraced host StructArray broadcasted against traced Ref(m)
    dp_host = StructArray((U=u, V=v))
    pvis1 = Reactant.to_rarray(ones(Float64, 16))
    @test Array(@jit kernel_2869!(pvis1, dp_host, m_ra)) ≈ expected

    # StructArray constructed directly from ConcreteRArray columns (without outer to_rarray)
    dp_rcols = StructArray((U=Reactant.to_rarray(u), V=Reactant.to_rarray(v)))
    pvis2 = Reactant.to_rarray(ones(Float64, 16))
    @test Array(@jit kernel_2869!(pvis2, dp_rcols, m_ra)) ≈ expected

    # Custom struct eltype StructArray{Pulse2869{Float64}} in both forms
    dp_pulse_host = StructArray{Pulse2869{Float64}}((u, v))
    dp_pulse_rcols = StructArray{Pulse2869{Float64}}((
        Reactant.to_rarray(u), Reactant.to_rarray(v)
    ))
    expected_pulse = ones(Float64, 16) .* pulse_dot_2869.(Ref(m_cpu), dp_pulse_host)

    pvis3 = Reactant.to_rarray(ones(Float64, 16))
    @test Array(@jit kernel_struct_2869!(pvis3, dp_pulse_host, m_ra)) ≈ expected_pulse

    pvis4 = Reactant.to_rarray(ones(Float64, 16))
    @test Array(@jit kernel_struct_2869!(pvis4, dp_pulse_rcols, m_ra)) ≈ expected_pulse
end
