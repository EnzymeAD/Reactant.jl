using Reactant
using Reactant:
    traced_type,
    TracedRArray,
    TracedRNumber,
    ConcreteToTraced,
    ArrayToConcrete,
    NoFieldMatchError,
    TracedTypeError,
    ReactantPrimitive
using Test

struct Wrapper{A,B}
    a::A
    b::B
end

struct Descent{T}
    eta::T
end

struct RMSProp{Teta,Trho,Teps,C<:Bool}
    eta::Teta
    rho::Trho
    epsilon::Teps
    centred::C
end

@testset "Tracing" begin
    @testset "trace_type" begin
        @testset "mode = ConcreteToTraced" begin
            @testset "$origty" for (origty, targetty, targettynum) in [
                (Any, Any, Any),
                (Real, Real, Real),
                (Module, Module, Module),
                (DataType, DataType, DataType),
                # (Union{}, Union{}), # fails
                (Nothing, Nothing, Nothing),
                (Symbol, Symbol, Symbol),
                (Char, Char, Char),
                (AbstractString, AbstractString, AbstractString),
                (String, String, String),
                (VersionNumber, VersionNumber, VersionNumber),

                # Numeric types
                (AbstractFloat, AbstractFloat, AbstractFloat),
                (Float16, Float16, TracedRNumber{Float16}),
                (Float32, Float32, TracedRNumber{Float32}),
                (Float64, Float64, TracedRNumber{Float64}),
                (Integer, Integer, Integer),
                (Int8, Int8, TracedRNumber{Int8}),
                (Int16, Int16, TracedRNumber{Int16}),
                (Int32, Int32, TracedRNumber{Int32}),
                (Int64, Int64, TracedRNumber{Int64}),
                (UInt8, UInt8, TracedRNumber{UInt8}),
                (UInt16, UInt16, TracedRNumber{UInt16}),
                (UInt32, UInt32, TracedRNumber{UInt32}),
                (UInt64, UInt64, TracedRNumber{UInt64}),
                (Complex{Float32}, Complex{Float32}, TracedRNumber{Complex{Float32}}),
                (Complex{Float64}, Complex{Float64}, TracedRNumber{Complex{Float64}}),
                (Complex{Int8}, Complex{Int8}, TracedRNumber{Complex{Int8}}),
                (Complex{Int16}, Complex{Int16}, TracedRNumber{Complex{Int16}}),
                (Complex{Int32}, Complex{Int32}, TracedRNumber{Complex{Int32}}),
                (Complex{Int64}, Complex{Int64}, TracedRNumber{Complex{Int64}}),
                (Complex{UInt8}, Complex{UInt8}, TracedRNumber{Complex{UInt8}}),
                (Complex{UInt16}, Complex{UInt16}, TracedRNumber{Complex{UInt16}}),
                (Complex{UInt32}, Complex{UInt32}, TracedRNumber{Complex{UInt32}}),
                (Complex{UInt64}, Complex{UInt64}, TracedRNumber{Complex{UInt64}}),

                # RArray types
                (
                    ConcretePJRTArray{Float64,0,1,Sharding.NoShardInfo},
                    TracedRArray{Float64,0},
                    TracedRArray{Float64,0},
                ),
                (
                    ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo},
                    TracedRArray{Float64,1},
                    TracedRArray{Float64,1},
                ),
                (
                    ConcretePJRTArray{Float64,2,1,Sharding.NoShardInfo},
                    TracedRArray{Float64,2},
                    TracedRArray{Float64,2},
                ),
                (
                    ConcretePJRTArray{Float64,3,1,Sharding.NoShardInfo},
                    TracedRArray{Float64,3},
                    TracedRArray{Float64,3},
                ),

                # Array types
                (Array{Float64,1}, Array{Float64,1}, Array{TracedRNumber{Float64},1}),
                (
                    Array{ConcretePJRTArray{Float64,2,1,Sharding.NoShardInfo},1},
                    Array{TracedRArray{Float64,2},1},
                    Array{TracedRArray{Float64,2},1},
                ),

                # Union types
                (Union{Nothing,Int}, Union{Nothing,Int}, Union{Nothing,TracedRNumber{Int}}),
                (
                    Union{Nothing,ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo}},
                    Union{Nothing,TracedRArray{Float64,1}},
                    Union{Nothing,TracedRArray{Float64,1}},
                ),

                # Ptr types
                (Ptr{Float64}, Ptr{Float64}, Ptr{TracedRNumber{Float64}}),
                (
                    Ptr{ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo}},
                    Ptr{TracedRArray{Float64,1}},
                    Ptr{TracedRArray{Float64,1}},
                ),
                (
                    Core.LLVMPtr{Float64},
                    Core.LLVMPtr{Float64},
                    Core.LLVMPtr{TracedRNumber{Float64}},
                ),
                (
                    Core.LLVMPtr{ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                ),
                (
                    Base.RefValue{Float64},
                    Base.RefValue{Float64},
                    Base.RefValue{TracedRNumber{Float64}},
                ),
                (
                    Base.RefValue{ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo}},
                    Base.RefValue{TracedRArray{Float64,1}},
                    Base.RefValue{TracedRArray{Float64,1}},
                ),

                # Val types
                (Val{0}, Val{0}, Val{0}),
                (Val{0.5}, Val{0.5}, Val{0.5}),
                (Val{:x}, Val{:x}, Val{:x}),
                (
                    Dict{Int,ConcretePJRTArray{Float64,0,1,Sharding.NoShardInfo}},
                    Dict{Int,TracedRArray{Float64,0}},
                    Dict{Int,TracedRArray{Float64,0}},
                ),
                (Dict{Int}, Dict{Int}, Dict{Int}),
                (Dict, Dict, Dict),
                (
                    (Dict{A,ConcretePJRTArray{Float64,0,1,Sharding.NoShardInfo}} where {A}),
                    (Dict{A,TracedRArray{Float64,0}} where {A}),
                    (Dict{A,TracedRArray{Float64,0}} where {A}),
                ),
                (
                    (
                        Dict{
                            Symbol,NTuple{nsteps,SpectralVariable3D}
                        } where {nsteps} where {SpectralVariable3D}
                    ),
                    (
                        Dict{
                            Symbol,NTuple{nsteps,SpectralVariable3D}
                        } where {nsteps} where {SpectralVariable3D}
                    ),
                    (
                        Dict{
                            Symbol,NTuple{nsteps,SpectralVariable3D}
                        } where {nsteps} where {SpectralVariable3D}
                    ),
                ),
                (
                    Base.Pairs{Symbol,Union{}},
                    Base.Pairs{Symbol,Union{}},
                    Base.Pairs{Symbol,Union{}},
                ),
                (
                    NTuple{nsteps,SpectralVariable3D} where {nsteps,SpectralVariable3D},
                    NTuple{nsteps,SpectralVariable3D} where {nsteps,SpectralVariable3D},
                    NTuple{nsteps,SpectralVariable3D} where {nsteps,SpectralVariable3D},
                ),
                (
                    Base.RefValue{A} where {A},
                    Base.RefValue{A} where {A},
                    Base.RefValue{A} where {A},
                ),
                (Wrapper{Symbol,Symbol}, Wrapper{Symbol,Symbol}, Wrapper{Symbol,Symbol}),
                (
                    Wrapper{Float64,Vector{Float64}},
                    Wrapper{Float64,Vector{Float64}},
                    Wrapper{TracedRNumber{Float64},Vector{Float64}},
                ),
                (
                    Wrapper{Float64,ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo}},
                    Wrapper{Float64,TracedRArray{Float64,1}},
                    Wrapper{TracedRNumber{Float64},TracedRArray{Float64,1}},
                ),
                (Wrapper{Symbol}, Wrapper{Symbol}, Wrapper{Symbol}),
                (Wrapper{Float64}, Wrapper{Float64}, Wrapper{TracedRNumber{Float64}}),
                (
                    Wrapper{ConcretePJRTArray{Float64,1,1,Sharding.NoShardInfo}},
                    Wrapper{TracedRArray{Float64,1}},
                    Wrapper{TracedRArray{Float64,1}},
                ),
                (Wrapper, Wrapper, Wrapper),
            ]
                tracedty = traced_type(
                    origty, Val(ConcreteToTraced), Union{}, Sharding.NoSharding()
                )
                @test tracedty == targetty

                tracedty2 = traced_type(
                    origty, Val(ConcreteToTraced), ReactantPrimitive, Sharding.NoSharding()
                )
                @test tracedty2 == targetty
            end

            @testset "$type" for type in [
                TracedRArray{Float64,0},
                TracedRArray{Float64,1},
                TracedRArray{Float64,2},
                TracedRArray{Float64,3},
            ]
                @test_throws Union{ErrorException,String} traced_type(
                    type, Val(ConcreteToTraced), Union{}, Sharding.NoSharding()
                )
            end
        end
        @testset "traced_type exceptions" begin
            struct Node
                x::Vector{Float64}
                y::Union{Nothing,Node}
            end
            @test_throws NoFieldMatchError traced_type(
                Node, Val(ArrayToConcrete), Union{}, Sharding.NoSharding()
            )
        end
    end

    @testset "specialized dispatches" begin
        @test @inferred Union{Float64,ConcretePJRTArray{Float64}} Reactant.to_rarray(
            1.0; track_numbers=Number
        ) isa ConcretePJRTNumber
        @test @inferred Reactant.to_rarray(1.0) isa Float64
        @test @inferred Reactant.to_rarray(rand(3)) isa ConcretePJRTArray

        x_ra = Reactant.to_rarray(rand(3))
        @test @inferred Reactant.to_rarray(x_ra) isa ConcretePJRTArray

        x_ra = Reactant.to_rarray(1.0; track_numbers=Number)
        @test @inferred Reactant.to_rarray(x_ra) isa ConcretePJRTNumber
    end

    @testset "no trace Val" begin
        st = (; a=1, training=Val(true))
        st_traced = Reactant.to_rarray(st; track_numbers=Number)
        @test st_traced.training isa Val{true}
    end

    @testset "to_rarray(::AbstractRule)" begin
        opt = Descent(0.1)
        opt_traced = Reactant.to_rarray(opt; track_numbers=AbstractFloat)
        @test opt_traced.eta isa ConcreteRNumber{Float64}

        opt = RMSProp(0.1, 0.9, 1e-8, true)
        opt_traced = Reactant.to_rarray(opt; track_numbers=AbstractFloat)
        @test opt_traced.eta isa ConcreteRNumber{Float64}
        @test opt_traced.rho isa ConcreteRNumber{Float64}
        @test opt_traced.epsilon isa ConcreteRNumber{Float64}
        @test opt_traced.centred isa Bool
    end
end
