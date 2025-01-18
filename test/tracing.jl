using Reactant
using Reactant:
    traced_type,
    ConcreteRArray,
    TracedRArray,
    TracedRNumber,
    ConcreteToTraced,
    ArrayToConcrete,
    NoFieldMatchError,
    TracedTypeError,
    ReactantPrimitive
using Test

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
                (Int128, Int128, TracedRNumber{Int128}),
                (UInt8, UInt8, TracedRNumber{UInt8}),
                (UInt16, UInt16, TracedRNumber{UInt16}),
                (UInt32, UInt32, TracedRNumber{UInt32}),
                (UInt64, UInt64, TracedRNumber{UInt64}),
                (UInt128, UInt128, TracedRNumber{UInt128}),
                (Complex{Float16}, Complex{Float16}, TracedRNumber{Complex{Float16}}),
                (Complex{Float32}, Complex{Float32}, TracedRNumber{Complex{Float32}}),
                (Complex{Float64}, Complex{Float64}, TracedRNumber{Complex{Float64}}),
                (Complex{Int8}, Complex{Int8}, TracedRNumber{Complex{Int8}}),
                (Complex{Int16}, Complex{Int16}, TracedRNumber{Complex{Int16}}),
                (Complex{Int32}, Complex{Int32}, TracedRNumber{Complex{Int32}}),
                (Complex{Int64}, Complex{Int64}, TracedRNumber{Complex{Int64}}),
                (Complex{Int128}, Complex{Int128}, TracedRNumber{Complex{Int128}}),
                (Complex{UInt8}, Complex{UInt8}, TracedRNumber{Complex{UInt8}}),
                (Complex{UInt16}, Complex{UInt16}, TracedRNumber{Complex{UInt16}}),
                (Complex{UInt32}, Complex{UInt32}, TracedRNumber{Complex{UInt32}}),
                (Complex{UInt64}, Complex{UInt64}, TracedRNumber{Complex{UInt64}}),
                (Complex{UInt128}, Complex{UInt128}, TracedRNumber{Complex{UInt128}}),

                # RArray types
                (
                    ConcreteRArray{Float64,0},
                    TracedRArray{Float64,0},
                    TracedRArray{Float64,0},
                ),
                (
                    ConcreteRArray{Float64,1},
                    TracedRArray{Float64,1},
                    TracedRArray{Float64,1},
                ),
                (
                    ConcreteRArray{Float64,2},
                    TracedRArray{Float64,2},
                    TracedRArray{Float64,2},
                ),
                (
                    ConcreteRArray{Float64,3},
                    TracedRArray{Float64,3},
                    TracedRArray{Float64,3},
                ),

                # Array types
                (Array{Float64,1}, Array{Float64,1}, Array{TracedRNumber{Float64},1}),
                (
                    Array{ConcreteRArray{Float64,2},1},
                    Array{TracedRArray{Float64,2},1},
                    Array{TracedRArray{Float64,2},1},
                ),

                # Union types
                (Union{Nothing,Int}, Union{Nothing,Int}, Union{Nothing,TracedRNumber{Int}}),
                (
                    Union{Nothing,ConcreteRArray{Float64,1}},
                    Union{Nothing,TracedRArray{Float64,1}},
                    Union{Nothing,TracedRArray{Float64,1}},
                ),

                # Ptr types
                (Ptr{Float64}, Ptr{Float64}, Ptr{TracedRNumber{Float64}}),
                (
                    Ptr{ConcreteRArray{Float64,1}},
                    Ptr{TracedRArray{Float64,1}},
                    Ptr{TracedRArray{Float64,1}},
                ),
                (
                    Core.LLVMPtr{Float64},
                    Core.LLVMPtr{Float64},
                    Core.LLVMPtr{TracedRNumber{Float64}},
                ),
                (
                    Core.LLVMPtr{ConcreteRArray{Float64,1}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                ),
                (
                    Base.RefValue{Float64},
                    Base.RefValue{Float64},
                    Base.RefValue{TracedRNumber{Float64}},
                ),
                (
                    Base.RefValue{ConcreteRArray{Float64,1}},
                    Base.RefValue{TracedRArray{Float64,1}},
                    Base.RefValue{TracedRArray{Float64,1}},
                ),

                # Val types
                (Val{0}, Val{0}, Val{0}),
                (Val{0.5}, Val{0.5}, Val{0.5}),
                (Val{:x}, Val{:x}, Val{:x}),
                (
                    Dict{Int,ConcreteRArray{Float64,0}},
                    Dict{Int,TracedRArray{Float64,0}},
                    Dict{Int,TracedRArray{Float64,0}},
                ),
                (Dict{Int}, Dict{Int}, Dict{Int}),
                (Dict, Dict, Dict),
                (
                    (Dict{A,ConcreteRArray{Float64,0}} where {A}),
                    (Dict{A,TracedRArray{Float64,0}} where {A}),
                    (Dict{A,TracedRArray{Float64,0}} where {A}),
                ),
                (
                    Base.Pairs{Symbol,Union{}},
                    Base.Pairs{Symbol,Union{}},
                    Base.Pairs{Symbol,Union{}},
                ),
            ]
                tracedty = traced_type(origty, Val(ConcreteToTraced), Union{})
                @test tracedty == targetty

                tracedty2 = traced_type(origty, Val(ConcreteToTraced), ReactantPrimitive)
                @test tracedty2 == targetty
            end

            @testset "$type" for type in [
                TracedRArray{Float64,0},
                TracedRArray{Float64,1},
                TracedRArray{Float64,2},
                TracedRArray{Float64,3},
            ]
                @test_throws Union{ErrorException,String} traced_type(
                    type, Val(ConcreteToTraced), Union{}
                )
            end
        end
        @testset "traced_type exceptions" begin
            struct Node
                x::Vector{Float64}
                y::Union{Nothing,Node}
            end
            @test_throws NoFieldMatchError traced_type(Node, Val(ArrayToConcrete), Union{})
        end
    end

    @testset "specialized dispatches" begin
        @test @inferred Union{Float64,ConcreteRArray{Float64}} Reactant.to_rarray(
            1.0; track_numbers=Number
        ) isa ConcreteRNumber
        @test @inferred Reactant.to_rarray(1.0) isa Float64
        @test @inferred Reactant.to_rarray(rand(3)) isa ConcreteRArray

        x_ra = Reactant.to_rarray(rand(3))
        @test @inferred Reactant.to_rarray(x_ra) isa ConcreteRArray

        x_ra = Reactant.to_rarray(1.0; track_numbers=Number)
        @test @inferred Reactant.to_rarray(x_ra) isa ConcreteRNumber
    end
end
