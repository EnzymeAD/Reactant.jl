using Reactant, Test, Adapt

struct MyGrid{FT,AT} <: AbstractVector{FT}
    data::AT
    radius::FT
end

Adapt.parent(x::MyGrid) = x.data

Base.getindex(x::MyGrid, args...) = Base.getindex(x.data, args...)

Base.size(x::MyGrid) = Base.size(x.data)

function Base.show(io::IOty, X::MyGrid) where {IOty<:Union{IO,IOContext}}
    print(io, Core.Typeof(X), "(")
    if Adapt.parent(X) !== X
        Base.show(io, Adapt.parent(X))
    end
    return print(io, ")")
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{MyGrid{FT,AT}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT,AT}
    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)
    AT2 = Reactant.traced_type_inner(AT, seen, mode, track_numbers, sharding, runtime)

    for NF in (AT2,)
        FT2 = Reactant.promote_traced_type(FT2, eltype(NF))
    end

    res = MyGrid{FT2,AT2}
    return res
end

@inline Reactant.make_tracer(seen, @nospecialize(prev::MyGrid), args...; kwargs...) =
    Reactant.make_tracer_via_immutable_constructor(seen, prev, args...; kwargs...)

struct MyGrid2{FT,AT} <: AbstractVector{FT}
    data::AT
    radius::FT
    bar::FT
end

Adapt.parent(x::MyGrid2) = x.data

Base.getindex(x::MyGrid2, args...) = Base.getindex(x.data, args...)

Base.size(x::MyGrid2) = Base.size(x.data)

function Base.show(io::IOty, X::MyGrid2) where {IOty<:Union{IO,IOContext}}
    print(io, Core.Typeof(X), "(")
    if Adapt.parent(X) !== X
        Base.show(io, Adapt.parent(X))
    end
    return print(io, ")")
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{MyGrid2{FT,AT}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT,AT}
    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)
    AT2 = Reactant.traced_type_inner(AT, seen, mode, track_numbers, sharding, runtime)

    for NF in (AT2,)
        FT2 = Reactant.promote_traced_type(FT2, eltype(NF))
    end

    res = MyGrid2{FT2,AT2}
    return res
end

@inline Reactant.make_tracer(seen, @nospecialize(prev::MyGrid2), args...; kwargs...) =
    Reactant.make_tracer_via_immutable_constructor(seen, prev, args...; kwargs...)

function update!(g)
    @allowscalar g.data[1] = g.radius
    return nothing
end

function selfreturn(g)
    return g
end

function call_update!(g)
    @trace update!(g)
end

function call_selfreturn(g)
    @trace selfreturn(g)
end

@testset "Custom construction" begin
    g = MyGrid([3.14, 1.59], 2.7)
    rg = Reactant.to_rarray(g)

    @jit update!(rg)
    @test convert(Array, rg.data) ≈ [2.7, 1.59]

    rg = Reactant.to_rarray(g)
    res = @jit selfreturn(rg)
    @test convert(Array, res.data) ≈ [3.14, 1.59]
    @test res.radius ≈ 2.7
    @test typeof(res.radius) <: ConcreteRNumber

    rg = Reactant.to_rarray(g)

    @jit call_update!(rg)
    @test convert(Array, rg.data) ≈ [2.7, 1.59]

    rg = Reactant.to_rarray(g)
    res = @jit call_selfreturn(rg)
    @test convert(Array, res.data) ≈ [3.14, 1.59]
    @test res.radius ≈ 2.7
    @test typeof(res.radius) <: ConcreteRNumber
end

@testset "Custom construction2 " begin
    g = Ref(MyGrid([3.14, 1.59], 2.7))
    g = (g, g)

    rg = Reactant.to_rarray(g)
    res = @jit selfreturn(rg)
    @test convert(Array, res[1][].data) ≈ [3.14, 1.59]
    @test convert(Array, res[2][].data) ≈ [3.14, 1.59]
    @test res[1][].data == res[2][].data
end
