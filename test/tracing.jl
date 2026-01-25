using Reactant, Test
using Reactant:
    traced_type,
    TracedRArray,
    TracedRNumber,
    ConcreteToTraced,
    ArrayToConcrete,
    NoFieldMatchError,
    TracedTypeError,
    ReactantPrimitive

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

@testset "Traced Type" begin
    # Note: With TracedRNumber as a Union type, Vector{Union{}} is technically a subtype
    # of AnyTracedRArray because Union{} <: AnyUnion. This is acceptable since Vector{Union{}}
    # is not a practically usable type. We test that actual traced arrays work correctly instead.
    @test TracedRArray{Float64,2} <: Reactant.AnyTracedRArray
    @test !(Vector{Float64} <: Reactant.AnyTracedRArray)
end

mul(a, b) = a .* b

struct MyFix{N,FT,XT} <: Base.Function
    f::FT
    x::XT
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
                (Float16, Float16, TracedRFloat{Float16}),
                (Float32, Float32, TracedRFloat{Float32}),
                (Float64, Float64, TracedRFloat{Float64}),
                (Integer, Integer, Integer),
                (Int8, Int8, TracedRInteger{Int8}),
                (Int16, Int16, TracedRInteger{Int16}),
                (Int32, Int32, TracedRInteger{Int32}),
                (Int64, Int64, TracedRInteger{Int64}),
                (UInt8, UInt8, TracedRInteger{UInt8}),
                (UInt16, UInt16, TracedRInteger{UInt16}),
                (UInt32, UInt32, TracedRInteger{UInt32}),
                (UInt64, UInt64, TracedRInteger{UInt64}),
                (Complex{Float32}, Complex{Float32}, TracedRComplex{Complex{Float32}}),
                (Complex{Float64}, Complex{Float64}, TracedRComplex{Complex{Float64}}),
                (Complex{Int8}, Complex{Int8}, TracedRComplex{Complex{Int8}}),
                (Complex{Int16}, Complex{Int16}, TracedRComplex{Complex{Int16}}),
                (Complex{Int32}, Complex{Int32}, TracedRComplex{Complex{Int32}}),
                (Complex{Int64}, Complex{Int64}, TracedRComplex{Complex{Int64}}),
                (Complex{UInt8}, Complex{UInt8}, TracedRComplex{Complex{UInt8}}),
                (Complex{UInt16}, Complex{UInt16}, TracedRComplex{Complex{UInt16}}),
                (Complex{UInt32}, Complex{UInt32}, TracedRComplex{Complex{UInt32}}),
                (Complex{UInt64}, Complex{UInt64}, TracedRComplex{Complex{UInt64}}),
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
                (Array{Float64,1}, Array{Float64,1}, Array{TracedRFloat{Float64},1}),
                (
                    Array{ConcreteRArray{Float64,2},1},
                    Array{TracedRArray{Float64,2},1},
                    Array{TracedRArray{Float64,2},1},
                ),

                # Union types
                (Union{Nothing,Int}, Union{Nothing,Int}, Union{Nothing,TracedRInteger{Int}}),
                (
                    Union{Nothing,ConcreteRArray{Float64,1}},
                    Union{Nothing,TracedRArray{Float64,1}},
                    Union{Nothing,TracedRArray{Float64,1}},
                ),

                # Ptr types
                (Ptr{Float64}, Ptr{Float64}, Ptr{TracedRFloat{Float64}}),
                (
                    Ptr{ConcreteRArray{Float64,1}},
                    Ptr{TracedRArray{Float64,1}},
                    Ptr{TracedRArray{Float64,1}},
                ),
                (
                    Core.LLVMPtr{Float64},
                    Core.LLVMPtr{Float64},
                    Core.LLVMPtr{TracedRFloat{Float64}},
                ),
                (
                    Core.LLVMPtr{ConcreteRArray{Float64,1}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                    Core.LLVMPtr{TracedRArray{Float64,1}},
                ),
                (
                    Base.RefValue{Float64},
                    Base.RefValue{Float64},
                    Base.RefValue{TracedRFloat{Float64}},
                ),
                (
                    Base.RefValue{ConcreteRArray{Float64,1}},
                    Base.RefValue{TracedRArray{Float64,1}},
                    Base.RefValue{TracedRArray{Float64,1}},
                ),

                # Function types
                (
                    MyFix{2,typeof(mul),ConcreteRArray{Float64,1}},
                    MyFix{2,typeof(mul),TracedRArray{Float64,1}},
                    MyFix{2,typeof(mul),TracedRArray{Float64,1}},
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
                    Wrapper{TracedRFloat{Float64},Vector{Float64}},
                ),
                (
                    Wrapper{Float64,ConcreteRArray{Float64,1}},
                    Wrapper{Float64,TracedRArray{Float64,1}},
                    Wrapper{TracedRFloat{Float64},TracedRArray{Float64,1}},
                ),
                (Wrapper{Symbol}, Wrapper{Symbol}, Wrapper{Symbol}),
                (Wrapper{Float64}, Wrapper{Float64}, Wrapper{TracedRFloat{Float64}}),
                (
                    Wrapper{ConcreteRArray{Float64,1}},
                    Wrapper{TracedRArray{Float64,1}},
                    Wrapper{TracedRArray{Float64,1}},
                ),
                (Wrapper, Wrapper, Wrapper),
            ]
                tracedty = traced_type(
                    origty,
                    Val(ConcreteToTraced),
                    Union{},
                    Sharding.NoSharding(),
                    Reactant.XLA.runtime(),
                )
                @test tracedty == targetty

                tracedty2 = traced_type(
                    origty,
                    Val(ConcreteToTraced),
                    ReactantPrimitive,
                    Sharding.NoSharding(),
                    Reactant.XLA.runtime(),
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
                    type,
                    Val(ConcreteToTraced),
                    Union{},
                    Sharding.NoSharding(),
                    Reactant.XLA.runtime(),
                )
            end
        end
        @testset "traced_type exceptions" begin
            struct Node
                x::Vector{Float64}
                y::Union{Nothing,Node}
            end
            @test_throws NoFieldMatchError traced_type(
                Node,
                Val(ArrayToConcrete),
                Union{},
                Sharding.NoSharding(),
                Reactant.XLA.runtime(),
            )
        end
        @testset "apply_type_with_promotion" begin
            struct Bar{T}
                b::T
            end
            struct Foo{T,B<:Bar{T},AT<:AbstractArray{T}}
                a::AT
                b::B
            end
            @test Reactant.apply_type_with_promotion(
                Foo, [Float64, Bar{Float64}, Reactant.TracedRArray{Float64,1}]
            ) == (
                Foo{
                    TracedRFloat{Float64},
                    Bar{TracedRFloat{Float64}},
                    Reactant.TracedRArray{Float64,1},
                },
                [true, true, false],
            )

            @test Reactant.apply_type_with_promotion(
                Foo,
                [
                    ConcreteRNumber{Float64},
                    Bar{ConcreteRNumber{Float64}},
                    ConcreteRArray{Float64,1},
                ],
            ) == (Foo{Float64,Bar{Float64},ConcreteRArray{Float64,1}}, [true, true, false])
        end
    end

    @testset "specialized dispatches" begin
        @test @inferred Union{Float64,ConcreteRArray{Float64}} Reactant.to_rarray(
            1.0; track_numbers=Number
        ) isa ConcreteRNumber
        @test @inferred Reactant.to_rarray(1.0) isa Float64
        @test @inferred Reactant.to_rarray(
            Reactant.TestUtils.construct_test_array(Float64, 3)
        ) isa ConcreteRArray

        x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3))
        @test @inferred Reactant.to_rarray(x_ra) isa ConcreteRArray

        x_ra = Reactant.to_rarray(1.0; track_numbers=Number)
        @test @inferred Reactant.to_rarray(x_ra) isa ConcreteRNumber
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

    @testset "@skip_rewrite_func" begin
        a = ConcreteRArray([1.0 2.0; 3.0 4.0])

        # TODO we should test it with a type-unstable method
        add_skip_rewrite(x) = x + x
        Reactant.@skip_rewrite_func add_skip_rewrite

        # wrapper because `@skip_rewrite_*` doesn't work with top-functions
        f(x) = add_skip_rewrite(x)

        # warmup
        @code_hlo optimize = false f(a)

        t = @timed @code_hlo optimize = false f(a)

        # `@timed` only measures compile time from v1.11.0 onward
        @static if VERSION >= v"1.11.0"
            @test iszero(t.compile_time)
        end
    end
end
