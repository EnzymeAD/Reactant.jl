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
    @test !(Vector{Union{}} <: Reactant.AnyTracedRArray)
end

mul(a, b) = a .* b

struct MyFix{N,FT,XT} <: Base.Function
    f::FT
    x::XT
end

@testset "trace_type (mode = ConcreteToTraced)" begin
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
        (ConcreteRArray{Float64,0}, TracedRArray{Float64,0}, TracedRArray{Float64,0}),
        (ConcreteRArray{Float64,1}, TracedRArray{Float64,1}, TracedRArray{Float64,1}),
        (ConcreteRArray{Float64,2}, TracedRArray{Float64,2}, TracedRArray{Float64,2}),
        (ConcreteRArray{Float64,3}, TracedRArray{Float64,3}, TracedRArray{Float64,3}),

        # Array types
        (Array{Float64,1}, Array{Float64,1}, Array{TracedRNumber{Float64},1}),
        (
            Array{ConcreteRArray{Float64,2},1},
            Array{TracedRArray{Float64,2},1},
            Array{TracedRArray{Float64,2},1},
        ),

        # AbstractArray types
        (
            AbstractArray{Float64,1},
            AbstractArray{Float64,1},
            AbstractArray{TracedRNumber{Float64},1},
        ),
        (AbstractArray, AbstractArray, AbstractArray),
        (
            AbstractArray{Float64},
            AbstractArray{Float64},
            AbstractArray{TracedRNumber{Float64}},
        ),
        (AbstractVector, AbstractVector, AbstractVector),
        (AbstractMatrix, AbstractMatrix, AbstractMatrix),
        (AbstractVector{<:Integer}, AbstractVector{<:Integer}, AbstractVector{<:Integer}),
        (
            AbstractVector{<:Int},
            AbstractVector{<:Int},
            AbstractVector{<:TracedRNumber{Int}},
        ),

        # BitArray types
        (BitArray, BitArray, BitArray),
        (BitArray{1}, BitArray{1}, BitArray{1}),

        # Union types
        (Union{Nothing,Int}, Union{Nothing,Int}, Union{Nothing,TracedRNumber{Int}}),
        (
            Union{Nothing,ConcreteRArray{Float64,1}},
            Union{Nothing,TracedRArray{Float64,1}},
            Union{Nothing,TracedRArray{Float64,1}},
        ),

        # UnionAll types
        (Array, Array, Array),
        (Array{Float64}, Array{Float64}, Array{TracedRNumber{Float64}}),
        (Matrix, Matrix, Matrix),
        (Array{<:AbstractFloat}, Array{<:AbstractFloat}, Array{<:AbstractFloat}),
        (Array{>:Float32}, Array{>:Float32}, Array{>:Float32}),
        (ConcreteRNumber, TracedRNumber, TracedRNumber),
        (ConcreteRArray, TracedRArray, TracedRArray),
        (ConcreteRArray{Float64}, TracedRArray{Float64}, TracedRArray{Float64}),
        (
            ConcreteRArray{T,2} where {T},
            TracedRArray{T,2} where {T},
            TracedRArray{T,2} where {T},
        ),
        (
            ConcreteRArray{<:AbstractFloat},
            TracedRArray{<:AbstractFloat},
            TracedRArray{<:AbstractFloat},
        ),
        (ConcreteRArray{>:Float32}, TracedRArray{>:Float32}, TracedRArray{>:Float32}),
        (Union{Nothing,Array}, Union{Nothing,Array}, Union{Nothing,Array}),
        (
            Union{Nothing,Array{Float64}},
            Union{Nothing,Array{Float64}},
            Union{Nothing,Array{TracedRNumber{Float64}}},
        ),
        (
            Union{Nothing,ConcreteRArray},
            Union{Nothing,TracedRArray},
            Union{Nothing,TracedRArray},
        ),
        (
            Union{Nothing,ConcreteRArray{Float64}},
            Union{Nothing,TracedRArray{Float64}},
            Union{Nothing,TracedRArray{Float64}},
        ),
        (
            Union{Nothing,ConcreteRArray{T,2} where {T}},
            Union{Nothing,TracedRArray{T,2} where {T}},
            Union{Nothing,TracedRArray{T,2} where {T}},
        ),
        (
            Union{Nothing,ConcreteRArray{T,2}} where {T},
            Union{Nothing,TracedRArray{T,2}} where {T},
            Union{Nothing,TracedRArray{T,2}} where {T},
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
        (Ref{Float64}, Ref{Float64}, Ref{TracedRNumber{Float64}}),
        (
            Ref{ConcreteRArray{Float64,1}},
            Ref{TracedRArray{Float64,1}},
            Ref{TracedRArray{Float64,1}},
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

        # others
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
        (IOStream, IOStream, IOStream),
        (Wrapper{Symbol,Symbol}, Wrapper{Symbol,Symbol}, Wrapper{Symbol,Symbol}),
        (
            Wrapper{Float64,Vector{Float64}},
            Wrapper{Float64,Vector{Float64}},
            Wrapper{TracedRNumber{Float64},Vector{Float64}},
        ),
        (
            Wrapper{Float64,ConcreteRArray{Float64,1}},
            Wrapper{Float64,TracedRArray{Float64,1}},
            Wrapper{TracedRNumber{Float64},TracedRArray{Float64,1}},
        ),
        (Wrapper{Symbol}, Wrapper{Symbol}, Wrapper{Symbol}),
        (Wrapper{Float64}, Wrapper{Float64}, Wrapper{TracedRNumber{Float64}}),
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
end

@testset "traced_type exceptions" begin
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

    struct Node
        x::Vector{Float64}
        y::Union{Nothing,Node}
    end
    @test_throws NoFieldMatchError traced_type(
        Node, Val(ArrayToConcrete), Union{}, Sharding.NoSharding(), Reactant.XLA.runtime()
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
            TracedRNumber{Float64},
            Bar{TracedRNumber{Float64}},
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

@testset "BitArray" begin
    mask = BitVector([true, true, false])
    mask_ra = Reactant.to_rarray(mask)
    @test mask_ra isa ConcreteRArray{Bool,1}
    @test Array(mask_ra) == Bool[true, true, false]

    mask2d = BitArray([true false; false true])
    mask2d_ra = Reactant.to_rarray(mask2d)
    @test mask2d_ra isa ConcreteRArray{Bool,2}
    @test Array(mask2d_ra) == Bool[true false; false true]

    struct BitVecStorage{B}
        v::B
    end

    mask_struct = BitVecStorage(mask)
    mask_struct_ra = Reactant.to_rarray(mask_struct)
    @test mask_struct_ra isa BitVecStorage{<:ConcreteRArray{Bool,1}}
    @test mask_struct_ra.v isa ConcreteRArray{Bool,1}
    @test Array(mask_struct_ra.v) == Bool[true, true, false]
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

function aliased_trace_for(m::T) where {T}
    a = one(T)
    pow2 = a

    Reactant.@trace for _ in 1:8
        pow2 += a
    end

    return a
end

function aliased_trace_for_dead_code(m::T) where {T}
    a = one(T)
    b = zero(T)
    pow2 = a

    Reactant.@trace for _ in 1:8
        c = sin(a)
        pow2 += b
    end

    return a
end

@testset "aliased TracedRNumber in @trace for" begin
    ms = Float64[0.01, 0.1, 0.5, 0.9]
    m_r = Reactant.to_rarray(ms)

    @test @jit(aliased_trace_for.(m_r)) ≈ aliased_trace_for.(ms)
    @test @jit(aliased_trace_for_dead_code.(m_r)) ≈ aliased_trace_for_dead_code.(ms)
end

function aliased_trace_for_array(ms::AbstractArray{T}) where {T}
    a = ms
    pow2 = ms

    Reactant.@trace for _ in 1:8
        @allowscalar a[1] = zero(eltype(a))
        @allowscalar pow2[2] = zero(eltype(a))
    end

    return a, pow2
end

@testset "aliased TracedRArray in @trace for" begin
    ms = Float64[0.01, 0.1, 0.5, 0.9]
    ms_r = Reactant.to_rarray(ms)

    res_ra = @jit aliased_trace_for_array(ms_r)
    res = aliased_trace_for_array(copy(ms))

    @test Array(res_ra[1]) ≈ res[1]
    @test Array(res_ra[2]) ≈ res[2]
end

# `a` and `b` alias the same array on entry. Inside the loop `b` is *reassigned*
# (rebound) while `a` is mutated *in place*. The reassignment forces
# `Ops._unalias_while_loop_args` to clone `b`'s slot via `Base.copy`, giving it a
# distinct identity from `a`'s slot. Because the clone is a fresh object, the
# outer `b` binding must be written back from its ref after the loop, otherwise
# the returned `b` would incorrectly alias `a`.
function aliased_trace_for_array_reassign_and_mutate(ms::AbstractArray{T}) where {T}
    a = ms
    b = ms   # alias on entry

    Reactant.@trace for _ in 1:8
        b = b .+ one(eltype(b))   # reassign b -> hits `_unalias_while_loop_args` copy
        a .*= 2                   # in-place mutation of the other aliased slot
    end

    return a, b
end

@testset "aliased + reassigned + mutated TracedRArray in @trace for" begin
    ms = Float64[0.01, 0.1, 0.5, 0.9]
    ms_r = Reactant.to_rarray(ms)

    res_ra = @jit aliased_trace_for_array_reassign_and_mutate(ms_r)
    res = aliased_trace_for_array_reassign_and_mutate(copy(ms))

    @test Array(res_ra[1]) ≈ res[1]   # mutated-in-place slot
    @test Array(res_ra[2]) ≈ res[2]   # reassigned/cloned slot must not alias `a`
    # The two slots must hold genuinely different values after the loop.
    @test !(Array(res_ra[1]) ≈ Array(res_ra[2]))
end

# Known limitation: when the shared array is mutated in place BEFORE the aliased
# variable is reassigned within the same iteration, plain Julia has `b` observe
# the mutation (since `a === b` at that point) before `b` is rebound. The pre-loop
# unaliasing in `_unalias_while_loop_args` splits `a` and `b` into distinct slots
# *before* the loop, so the traced `b` never sees that first in-place mutation.
# This within-iteration cross-variable aliasing cannot be preserved by pre-loop
# unaliasing. `a` is still computed correctly.
function aliased_trace_for_array_mutate_then_reassign(ms::AbstractArray{T}) where {T}
    a = ms
    b = ms   # alias on entry

    Reactant.@trace for _ in 1:8
        a .+= one(eltype(a))   # in-place mutation of shared array
        b = b .* 2             # reassign b AFTER the mutation
    end

    return a, b
end

@testset "aliased mutate-then-reassign in @trace for (known limitation)" begin
    ms = Float64[0.01, 0.1, 0.5, 0.9]
    ms_r = Reactant.to_rarray(ms)

    res_ra = @jit aliased_trace_for_array_mutate_then_reassign(ms_r)
    res = aliased_trace_for_array_mutate_then_reassign(copy(ms))

    @test Array(res_ra[1]) ≈ res[1]            # mutated-in-place slot is correct
    @test_broken Array(res_ra[2]) ≈ res[2]     # within-iteration aliasing not preserved
end

@testset "@skip_rewrite_func" begin
    a = ConcreteRArray([1.0 2.0; 3.0 4.0])

    # TODO(#2253) we should test it with a type-unstable method
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
