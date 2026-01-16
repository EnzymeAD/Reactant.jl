using StructArrays, Reactant, Test

@testset "StructArray to_rarray and make_tracer" begin
    x = StructArray(;
        a=rand(10, 2), b=fill("some strings", (10, 2)), c=rand(Float32, 10, 2)
    )
    x_ra = Reactant.to_rarray(x)

    # Note that the element type (the NamedTuple) contains ConcreteRNumbers even though track_numbers is not enabled.
    # This is because when the backing arrays are converted to TracedRArrays, their elements will contain TracedRNumbers.
    # In order for the element type to match the backing arrays, we need to use ConcreteRNumbers here as well:
    @test typeof(x_ra) == StructArray{
        @NamedTuple{
            a::ConcretePJRTNumber{Float64,1}, b::String, c::ConcretePJRTNumber{Float32,1}
        },
        2,
        @NamedTuple{
            a::ConcretePJRTArray{Float64,2,1},
            b::Matrix{String},
            c::ConcretePJRTArray{Float32,2,1},
        },
        CartesianIndex{2},
    }

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

    @test typeof(result) == StructVector{
        @NamedTuple{c::ConcretePJRTNumber{Float32,1}, d::ConcretePJRTNumber{Float64,1}},
        @NamedTuple{c::ConcretePJRTArray{Float32,1,1}, d::ConcretePJRTArray{Float64,1,1}},
        CartesianIndex{1},
    }
    for (component_ra, component) in
        zip(components(result), components(broadcast_elwise(x)))
        @test component_ra ≈ component
    end
end
