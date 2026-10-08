using Reactant, Test, Adapt, OffsetArrays

struct AdaptTestParams{A,R,S}
    array::A
    range::R
    scale::S
end

function Adapt.adapt_structure(to, p::AdaptTestParams)
    return AdaptTestParams(
        Adapt.adapt(to, p.array), Adapt.adapt(to, p.range), Adapt.adapt(to, p.scale)
    )
end

@testset "ConcreteRArrayAdaptor" begin
    to = Reactant.ConcreteRArrayAdaptor()
    to_materialized = Reactant.ConcreteRArrayAdaptor(; materialize_ranges=true)

    @testset "arrays" begin
        x = collect(Float64, 1:4)
        rx = Adapt.adapt(to, x)
        @test rx isa ConcreteRArray{Float64,1}
        @test Array(rx) == x
        @test Adapt.adapt(to, rx) === rx

        # An AbstractArray without an `Adapt.adapt_structure` method
        rb = Adapt.adapt(to, trues(3))
        @test rb isa ConcreteRArray{Bool,1}
        @test Array(rb) == [true, true, true]

        # Arrays whose element type is not a ReactantPrimitive are left alone
        strings = ["a", "b"]
        @test Adapt.adapt(to, strings) === strings
        nested = [collect(1:2), collect(3:4)]
        @test Adapt.adapt(to, nested) === nested
        @test Adapt.adapt(to_materialized, big(1):big(3)) == big(1):big(3)
    end

    @testset "offset arrays" begin
        x = OffsetArray(collect(Float64, 1:6), -2)
        rx = Adapt.adapt(to, x)
        @test rx isa OffsetVector{Float64,<:ConcreteRArray}
        @test axes(rx) == axes(x)
        @test Array(parent(rx)) == parent(x)
    end

    @testset "ranges" begin
        for r in (1:5, Base.OneTo(5), 1:2:9, range(0, 1; length=5), LinRange(0, 1, 5))
            @test Adapt.adapt(to, r) == r
            @test typeof(Adapt.adapt(to, r)) == typeof(r)

            rr = Adapt.adapt(to_materialized, r)
            @test rr isa ConcreteRArray{eltype(r),1}
            @test Array(rr) == collect(r)
        end

        x = OffsetArray(range(0, 1; length=5), -2)
        rx = Adapt.adapt(to_materialized, x)
        @test rx isa OffsetVector{Float64,<:ConcreteRArray}
        @test axes(rx) == axes(x)
        @test Array(parent(rx)) == collect(parent(x))
    end

    @testset "structures" begin
        p = AdaptTestParams(collect(Float64, 1:3), range(0, 1; length=3), 2.0)

        rp = Adapt.adapt(to, p)
        @test rp.array isa ConcreteRArray{Float64,1}
        @test rp.range === p.range
        @test rp.scale === p.scale

        rp = Adapt.adapt(to_materialized, p)
        @test rp.array isa ConcreteRArray{Float64,1}
        @test rp.range isa ConcreteRArray{Float64,1}
        @test Array(rp.range) == collect(p.range)
        @test rp.scale === p.scale

        @test Adapt.adapt(to, nothing) === nothing
        @test Adapt.adapt(to, (collect(1:2), 1:2))[1] isa ConcreteRArray{Int,1}
    end
end
