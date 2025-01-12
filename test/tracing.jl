using Reactant
using Reactant:
    traced_type,
    ConcreteRArray,
    TracedRArray,
    ConcreteToTraced,
    ArrayToConcrete,
    NoFieldMatchError,
    TracedTypeError
using Test

@testset "Tracing" begin
    @testset "trace_type" begin
        @testset "mode = ConcreteToTraced" begin
            @testset "$origty" for (origty, targetty) in [
                (Any, Any),
                (Module, Module),
                (DataType, DataType),
                # (Union{}, Union{}), # fails
                (Nothing, Nothing),
                (Symbol, Symbol),
                (Char, Char),
                (AbstractString, AbstractString),
                (String, String),

                # Numeric types
                (AbstractFloat, AbstractFloat),
                (Float16, Float16),
                (Float32, Float32),
                (Float64, Float64),
                (Integer, Integer),
                (Int8, Int8),
                (Int16, Int16),
                (Int32, Int32),
                (Int64, Int64),
                (Int128, Int128),
                (UInt8, UInt8),
                (UInt16, UInt16),
                (UInt32, UInt32),
                (UInt64, UInt64),
                (UInt128, UInt128),
                (Complex{Float16}, Complex{Float16}),
                (Complex{Float32}, Complex{Float32}),
                (Complex{Float64}, Complex{Float64}),
                (Complex{Int8}, Complex{Int8}),
                (Complex{Int16}, Complex{Int16}),
                (Complex{Int32}, Complex{Int32}),
                (Complex{Int64}, Complex{Int64}),
                (Complex{Int128}, Complex{Int128}),
                (Complex{UInt8}, Complex{UInt8}),
                (Complex{UInt16}, Complex{UInt16}),
                (Complex{UInt32}, Complex{UInt32}),
                (Complex{UInt64}, Complex{UInt64}),
                (Complex{UInt128}, Complex{UInt128}),

                # RArray types
                (ConcreteRArray{Float64,0}, TracedRArray{Float64,0}),
                (ConcreteRArray{Float64,1}, TracedRArray{Float64,1}),
                (ConcreteRArray{Float64,2}, TracedRArray{Float64,2}),
                (ConcreteRArray{Float64,3}, TracedRArray{Float64,3}),

                # Array types
                (Array{Float64,1}, Array{Float64,1}),
                (Array{ConcreteRArray{Float64,2},1}, Array{TracedRArray{Float64,2},1}),

                # Union types
                (Union{Nothing,Int}, Union{Nothing,Int}),
                (
                    Union{Nothing,ConcreteRArray{Float64,1}},
                    Union{Nothing,TracedRArray{Float64,1}},
                ),

                # Ptr types
                (Ptr{Float64}, Ptr{Float64}),
                (Ptr{ConcreteRArray{Float64,1}}, Ptr{TracedRArray{Float64,1}}),
                (Core.LLVMPtr{Float64}, Core.LLVMPtr{Float64}),
                (
                    Core.LLVMPtr{ConcreteRArray{Float64,1}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                ),
                (Base.RefValue{Float64}, Base.RefValue{Float64}),
                (
                    Base.RefValue{ConcreteRArray{Float64,1}},
                    Base.RefValue{TracedRArray{Float64,1}},
                ),

                # Val types
                (Val{0}, Val{0}),
                (Val{0.5}, Val{0.5}),
                (Val{:x}, Val{:x}),
            ]
                tracedty = traced_type(
                    origty, Reactant.OrderedIdDict(), Val(ConcreteToTraced), ()
                )
                @test tracedty == targetty
            end

            @testset "$type" for type in [
                TracedRArray{Float64,0},
                TracedRArray{Float64,1},
                TracedRArray{Float64,2},
                TracedRArray{Float64,3},
            ]
                @test_throws Union{ErrorException,String} traced_type(
                    type, Reactant.OrderedIdDict(), Val(ConcreteToTraced), ()
                )
            end
        end
        @testset "traced_type exceptions" begin
            @test_throws TracedTypeError Reactant.traced_type(
                Real, Reactant.OrderedIdDict(), Val(Reactant.ArrayToConcrete), ()
            )

            struct Node
                x::Vector{Float64}
                y::Union{Nothing,Node}
            end
            @test_throws NoFieldMatchError traced_type(
                Node, Reactant.OrderedIdDict(), Val(ArrayToConcrete), ()
            )
        end
    end

    @testset "specialized dispatches" begin
        @test @inferred Union{Float64,ConcreteRArray{Float64}} Reactant.to_rarray(
            1.0; track_numbers=(Number,)
        ) isa ConcreteRNumber
        @test @inferred Reactant.to_rarray(1.0) isa Float64
        @test @inferred Reactant.to_rarray(rand(3)) isa ConcreteRArray

        x_ra = Reactant.to_rarray(rand(3))
        @test @inferred Reactant.to_rarray(x_ra) isa ConcreteRArray

        x_ra = Reactant.to_rarray(1.0; track_numbers=(Number,))
        @test @inferred Reactant.to_rarray(x_ra) isa ConcreteRNumber
    end
end
