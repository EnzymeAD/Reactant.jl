module Reactant

using PackageExtensionCompat

include("mlir/MLIR.jl")
include("XLA.jl")
include("utils.jl")

abstract type RArray{ElType,Shape,N} <: AbstractArray{ElType,N} end

@inline Base.eltype(::RArray{ElType,Shape}) where {ElType,Shape} = ElType
@inline Base.size(::RArray{ElType,Shape}) where {ElType,Shape} = Shape
@inline Base.size(::Type{<:RArray{ElType,Shape}}) where {ElType,Shape} = Shape
@inline Base.ndims(::RArray{ElType,Shape,N}) where {ElType,Shape,N} = N
@inline Base.ndims(::Type{<:RArray{ElType,Shape,N}}) where {ElType,Shape,N} = N

@inline mlir_type(::RArray{ElType,Shape,N}) where {ElType,Shape,N} =
    MLIR.IR.TensorType(Shape, MLIR.IR.Type(ElType))

@inline mlir_type(::Type{<:RArray{ElType,Shape,N}}) where {ElType,Shape,N} =
    MLIR.IR.TensorType(Shape, MLIR.IR.Type(ElType))

struct XLAArray{ElType,Shape,N} <: RArray{ElType,Shape,N} end

mutable struct ConcreteRArray{ElType,Shape,N} <: RArray{ElType,Shape,N}
    data::XLA.AsyncBuffer
    #	data::XLAArray{ElType, Shape, N}
end

function Base.convert(
    ::Type{T}, X::ConcreteRArray{ElType,Shape,N}
) where {T<:Array,ElType,Shape,N}
    data = Array{ElType,N}(undef, Shape...)
    XLA.await(X.data)
    buf = X.data.buffer
    GC.@preserve data buf begin
        XLA.BufferToHost(buf, pointer(data))
    end
    return data
    # XLA.from_row_major(data)
end

function to_float(X::ConcreteRArray{ElType,(),0}) where {ElType}
    data = Ref{ElType}()
    XLA.await(X.data)
    buf = X.data.buffer
    GC.@preserve data buf begin
        XLA.BufferToHost(buf, data)
    end
    return data[]
end

function Base.isapprox(x::ConcreteRArray{ElType,(),0}, y; kwargs...) where {ElType}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.isapprox(x, y::ConcreteRArray{ElType,(),0}; kwargs...) where {ElType}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.isapprox(
    x::ConcreteRArray{ElType,(),0}, y::ConcreteRArray{ElType2,(),0}; kwargs...
) where {ElType,ElType2}
    return Base.isapprox(to_float(x), y; kwargs...)
end

function Base.print_array(io::IO, X::ConcreteRArray)
    if X.data == XLA.AsyncEmptyBuffer
        println(io, "<Empty buffer>")
        return nothing
    end
    return Base.print_array(io, convert(Array, X))
end

function Base.show(io::IO, X::ConcreteRArray)
    if X.data == XLA.AsyncEmptyBuffer
        println(io, "<Empty buffer>")
        return nothing
    end
    return Base.show(io, convert(Array, X))
end

@inline function Base.getindex(
    a::ConcreteRArray{ElType,Shape}, args::Vararg{Int,N}
) where {ElType,Shape,N}
    if a.data == XLA.AsyncEmptyBuffer
        throw("Cannot getindex from empty buffer")
    end
    # error("""Scalar indexing is disallowed.""")
    XLA.await(a.data)
    if XLA.BufferOnCPU(a.data.buffer)
        buf = a.data.buffer
        GC.@preserve buf begin
            ptr = Base.unsafe_convert(Ptr{ElType}, XLA.UnsafeBufferPointer(buf))
            start = 0
            for i in 1:N
                start *= Shape[N - i + 1]
                start += (args[N - i + 1] - 1)
                # start *= Shape[i]
                # start += (args[i]-1)
            end
            start += 1
            return unsafe_load(ptr, start)
        end
    end
    return convert(Array, a)[args...]
end

@inline function ConcreteRArray(
    data::Array{ElType,N}; client=XLA.default_backend[], idx=XLA.default_device_idx[]
) where {ElType,N}
    device = XLA.ClientGetDevice(client, idx)
    return ConcreteRArray{ElType,size(data),N}(
        XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, data, device), nothing)
    )
    # ConcreteRArray{ElType, size(data), N}(XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, XLA.to_row_major(data), device), nothing))
end

@inline ConcreteRArray(data::T) where {T<:Number} = ConcreteRArray{T,(),0}(data)

function Base.similar(x::ConcreteRArray{T,Shape,N}, ::Type{T2}) where {T,Shape,N,T2}
    return ConcreteRArray{T,Shape,N}(x.data)
end

mutable struct TracedRArray{ElType,Shape,N} <: RArray{ElType,Shape,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    function TracedRArray{ElType,Shape,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}
    ) where {ElType,Shape,N}
        if mlir_data !== nothing
            @assert size(MLIR.IR.type(mlir_data)) == Shape
        end
        return new{ElType,Shape,N}(paths, mlir_data)
    end
end

using Enzyme

@inline function Enzyme.Compiler.active_reg_inner(
    ::Type{TracedRArray{ElType,Shape,N}},
    seen::ST,
    world::Union{Nothing,UInt},
    ::Val{justActive}=Val(false),
    ::Val{UnionSret}=Val(false),
)::Enzyme.Compiler.ActivityState where {ST,ElType,Shape,N,justActive,UnionSret}
    if Enzyme.Compiler.active_reg_inner(
        ElType, seen, world, Val(justActive), Val(UnionSret)
    ) == Enzyme.Compiler.AnyState
        return Enzyme.Compiler.AnyState
    else
        return Enzyme.Compiler.DupState
    end
end

@inline function Enzyme.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
)::RT where {copy_if_inactive,RT<:RArray}
    if haskey(seen, prev)
        return seen[prev]
    end
    if Enzyme.Compiler.guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    if RT <: ConcreteRArray
        res = RT(zeros(eltype(RT), size(prev)))
        seen[prev] = res
        return res
    end

    if RT <: TracedRArray
        res = broadcast_to_size(eltype(RT)(0), size(prev))
        seen[prev] = res
        return res
    end

    attr = fill(MLIR.IR.Attribute(eltype(RT)(0)), mlir_type(prev))
    cst = MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
    res = RT((), cst)
    seen[prev] = res
    return res
end

function Base.promote_rule(
    A::Type{TracedRArray{T,Shape,N}}, B::Type{TracedRArray{S,Shape,N}}
) where {T,S,Shape,N}
    return TracedRArray{Base.promote_type(T, S),Shape,N}
end

function Base.promote_rule(A::Type{T}, B::Type{TracedRArray{S,Shape,N}}) where {T,S,Shape,N}
    return TracedRArray{Base.promote_type(T, S),Shape,N}
end

function Base.show(io::IO, X::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
    print(io, "TracedRArray{", ElType, ",", Shape, ",", N, "N}(", X.paths, ", ")
    return print(io, X.mlir_data, ")")
end

include("overloads.jl")

using Enzyme

@inline val_value(::Val{T}) where {T} = T
@inline val_value(::Type{Val{T}}) where {T} = T

@enum TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
end

@inline getmap(::Val{T}) where T = nothing
@inline getmap(::Val{T}, a, b, args...) where {T} = getmap(Val(T), args...)
@inline getmap(::Val{T}, ::Val{T}, ::Val{T2}, args...) where {T, T2} = T2

@inline is_concrete_tuple(x::T2) where {T2} =
    (x <: Tuple) && !(x === Tuple) && !(x isa UnionAll)
@inline function traced_type(val::Type{T}, seen::ST, ::Val{mode}) where {ST,T,mode}
    if T <: ConcreteRArray
        if mode == ConcreteToTraced
            @inline base_typet(TV::TT) where {TT<:UnionAll} =
                UnionAll(TV.var, base_typet(TV.body))
            @inline base_typet(TV::TT) where {TT<:DataType} = TracedRArray{TV.parameters...}
            return base_typet(T)
        elseif mode == TracedToConcrete
            return T
        else
            throw("Abstract RArray cannot be made concrete")
        end
    end
    if T <: TracedRArray
        if mode == ConcreteToTraced
            throw("TracedRArray $T cannot be traced")
        elseif mode == TracedToConcrete
            @inline base_typec(TV::TT) where {TT<:UnionAll} =
                UnionAll(TV.var, base_typec(TV.body))
            @inline base_typec(TV::TT) where {TT<:DataType} =
                ConcreteRArray{TV.parameters...}
            return base_typec(T)
        elseif mode == TracedTrack || mode == TracedSetPath
            return T
        else
            throw("Abstract RArray $T cannot be made concrete in mode $mode")
        end
    end

    if T <: XLAArray
        throw("XLA $T array cannot be traced")
    end
    if T <: RArray
        return T
    end

    if T === Any
        return T
    end

    if T === Symbol
        return T
    end

    if T <: Val
        val = val_value(T)
        if traced_type(typeof(val), seen, Val(mode)) == typeof(val)
            return T
        end
        throw("Val type $T cannot be traced")
    end

    if T === Union{}
        return T
    end

    if T == Nothing
        return T
    end

    if T == Char
        return T
    end

    if T <: Complex && !(T isa UnionAll)
        return Complex{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: AbstractFloat
        return T
    end

    if T <: Ptr
        return Ptr{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: Core.LLVMPtr
        return Core.LLVMPtr{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: Base.RefValue
        return Base.RefValue{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: Array
        if mode == ArrayToConcrete && eltype(T) <: AbstractFloat
            return (ConcreteRArray{eltype(T),Shape,ndims(T)} where {Shape})
        else
            return Array{
                traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode)),ndims(T)
            }
        end
    end

    if T <: Integer
        return T
    end

    if Enzyme.Compiler.isghostty(T) || Core.Compiler.isconstType(T)
        return T
    end

    if T <: Function
        # functions are directly returned
        if sizeof(T) == 0
            return T
        end

        # in closures, enclosured variables need to be traced
        N = fieldcount(T)
        traced_fieldtypes = ntuple(Val(N)) do i
            return traced_type(fieldtype(T, i), seen, Val(mode))
        end

        # closure are struct types with the types of enclosured vars as type parameters
        return Core.apply_type(T.name.wrapper, traced_fieldtypes...)
    end

    if T <: DataType
        return T
    end
    if T <: Module
        return T
    end
    if T <: AbstractString
        return T
    end

    # unknown number of fields
    if T isa UnionAll
        aT = Base.argument_datatype(T)
        if isnothing(aT)
            throw("Unhandled type $T")
        end
        if isnothing(Base.datatype_fieldcount(aT))
            throw("Unhandled type $T")
        end
    end

    if T isa Union
        return Union{traced_type(T.a, seen, Val(mode)),traced_type(T.b, seen, Val(mode))}
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        throw("Unhandled abstract type $T")
    end

    @assert !Base.isabstracttype(T)

    if !(Base.isconcretetype(T) || is_concrete_tuple(T) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    if is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
        Tuple{((T2 isa Core.TypeofVararg ? Any : T2) for T2 in T.parameters)...}
        throw(AssertionError("Type tuple of vararg $T is not supported"))
    end

    if is_concrete_tuple(T)
        return Tuple{(traced_type(T2, seen, Val(mode)) for T2 in T.parameters)...}
    end

    if T <: NamedTuple
        @inline tup_name(::Type{NamedTuple{A,B}}) where {A,B} = A
        @inline tup_val(::Type{NamedTuple{A,B}}) where {A,B} = B
        return NamedTuple{tup_name(T),traced_type(tup_val(T), seen, Val(mode))}
    end

    if T <: Dict
        @inline dict_name(::Type{Dict{A,B}}) where {A,B} = A
        @inline dict_val(::Type{Dict{A,B}}) where {A,B} = B
        return Dict{dict_name(T),traced_type(dict_val(T), seen, Val(mode))}
    end

    if T <: IdDict
        @inline iddict_name(::Type{IdDict{A,B}}) where {A,B} = A
        @inline iddict_val(::Type{IdDict{A,B}}) where {A,B} = B
        return IdDict{iddict_name(T),traced_type(iddict_val(T), seen, Val(mode))}
    end

    nextTy = getmap(Val(T), seen...)
    if nextTy != nothing
        return nextTy
    end

    seen2 = (Val(T), Val(T), seen...)

    changed = false
    subTys = Type[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type(subT, seen2, Val(mode))
        changed |= subT != subTT
        push!(subTys, subTT)
    end

    if !changed
        return T
    end

    subParms = []
    for SST in T.parameters
        if SST isa Type
            TrT = traced_type(SST, seen, Val(mode))
            push!(subParms, TrT)
        else
            push!(subParms, SST)
        end
    end

    TT2 = Core.apply_type(T.name.wrapper, subParms...)
    seen3 = (Val(T), Val(TT2), seen...)
    if fieldcount(T) == fieldcount(TT2)
        legal = true
        for f in 1:fieldcount(T)
            subT = fieldtype(T, f)
            subT2 = fieldtype(TT2, f)
            subTT = traced_type(subT, seen3, Val(mode))
            if subT2 != subTT
                legal = false
                break
            end
        end
        if legal
            return TT2
        end
    end

    name = Symbol[]
    throw(error("Cannot convert type $T, best attempt $TT2 failed"))
end

function append_path(path, i)
    return (path..., i)
end

@inline function make_tracer(seen::IdDict, prev::RT, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing) where {RT}
    if haskey(seen, prev)
        return seen[prev]
    end
    TT = traced_type(RT, (), Val(mode))
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    if TT <: NamedTuple
        changed = false
        subs = []
        for i in 1:nf
            xi = Base.getfield(prev, i)
            xi2 = make_tracer(seen, xi, append_path(path, i), mode; toscalar, tobatch)
            if xi !== xi2
                changed = true
            end
            push!(subs, xi2)
        end
        if !changed
            seen[prev] = prev
            return prev
        end
        tup = (subs...,)
        return NamedTuple{TT.parameters[1],typeof(tup)}(tup)
    end

    if ismutabletype(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        seen[prev] = y
        changed = false
        for i in 1:nf
            if isdefined(prev, i)
                xi = Base.getfield(prev, i)
                xi2 = make_tracer(seen, xi, append_path(path, i), mode; toscalar, tobatch)
                if xi !== xi2
                    changed = true
                end
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, xi2)
            end
        end
        if !changed
            seen[prev] = prev
            return prev
        end
        return y
    end

    if nf == 0
        return prev
    end

    flds = Vector{Any}(undef, nf)
    changed = false
    for i in 1:nf
        if isdefined(prev, i)
            xi = Base.getfield(prev, i)
            xi2 = make_tracer(seen, xi, append_path(path, i), mode; toscalar, tobatch)
            if xi !== xi2
                changed = true
            end
            flds[i] = xi2
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    if !changed
        seen[prev] = prev
        return prev
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), TT, flds, nf)
    seen[prev] = y
    return y
end

@inline function make_tracer(
    seen::IdDict, prev::ConcreteRArray{ElType,Shape,N}, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing
) where {ElType,Shape,N}
    if mode == ArrayToConcrete
        return prev
    end
    if mode != ConcreteToTraced
        throw("Cannot trace concrete")
    end
    if haskey(seen, prev)
        return seen[prev]::TracedRArray{ElType,Shape,N}
    end
    @assert N isa Int
    res = TracedRArray{ElType,Shape,N}((path,), nothing)
    seen[prev] = res
    return res
end

@inline function make_tracer(
    seen::IdDict, prev::TracedRArray{ElType,Shape,N}, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing
) where {ElType,Shape,N}
    if mode == ConcreteToTraced
        throw("Cannot trace existing trace type")
    end
    if mode == TracedTrack
        prev.paths = (prev.paths..., path)
        if !haskey(seen, prev)
            return seen[prev] = prev
        end
        return prev
    end
    if mode == TracedSetPath
        if haskey(seen, prev)
            return seen[prev]
        end
        res = if toscalar
            TracedRArray{ElType,(),0}((path,), nothing)
        elseif tobatch !== nothing
            TracedRArray{ElType,tobatch,length(tobatch)}((path,), prev.mlir_data)
        else
            TracedRArray{ElType,Shape,N}((path,), prev.mlir_data)
        end
        seen[prev] = res
        return res
    end

    if mode == TracedToConcrete
        if haskey(seen, prev)
            return seen[prev]::ConcreteRArray{ElType,Shape,N}
        end
        res = ConcreteRArray{ElType,Shape,N}(XLA.AsyncEmptyBuffer)
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

@inline function make_tracer(seen::IdDict, prev::RT, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing) where {RT<:AbstractFloat}
    return prev
end

@inline function make_tracer(seen::IdDict, prev::Complex{RT}, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing) where {RT}
    return Complex(
        make_tracer(seen, prev.re, append_path(path, :re), mode; toscalar, tobatch),
        make_tracer(seen, prev.im, append_path(path, :im), mode; toscalar, tobatch),
    )
end

@inline function make_tracer(seen::IdDict, prev::RT, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing) where {RT<:Array}
    if haskey(seen, prev)
        return seen[prev]
    end
    if mode == ArrayToConcrete && eltype(RT) <: AbstractFloat
        return seen[prev] = ConcreteRArray(prev)
    end
    TT = traced_type(eltype(RT), (), Val(mode))
    newa = Array{TT,ndims(RT)}(undef, size(prev))
    seen[prev] = newa
    same = true
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            nv = make_tracer(seen, pv, append_path(path, I), mode; toscalar, tobatch)
            if pv !== nv
                same = false
            end
            @inbounds newa[I] = nv
        end
    end
    if same
        seen[prev] = prev
        return prev
    end
    return newa
end

@inline function make_tracer(seen::IdDict, prev::RT, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing) where {RT<:Tuple}
    return (
        (make_tracer(seen, v, append_path(path, i), mode; toscalar, tobatch) for (i, v) in enumerate(prev))...,
    )
end

@inline function make_tracer(seen::IdDict, prev::NamedTuple{A,RT}, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing) where {A,RT}
    return NamedTuple{A,traced_type(RT, (), Val(mode))}((
        (
            make_tracer(seen, Base.getfield(prev, i), append_path(path, i), mode; toscalar, tobatch) for
            i in 1:length(A)
        )...,
    ))
end

@inline function make_tracer(seen::IdDict, prev::Core.Box, path::Tuple, mode::TraceMode; toscalar=false, tobatch=nothing)
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    tr = make_tracer(seen, prev2, append_path(path, :contents), mode; toscalar, tobatch)
    if tr == prev2
        seen[prev] = prev
        return prev
    end
    res = Core.Box(tr)
    seen[prev] = res
    return res
end

struct MakeConcreteRArray{T} end
struct MakeArray{AT,Vals} end
struct MakeString{AT,Val} end
struct MakeStruct{AT,Val} end
struct MakeVal{AT} end
struct MakeSymbol{AT} end

function make_valable(tocopy)
    if tocopy isa ConcreteRArray
        return MakeConcreteRArray{typeof(tocopy)}
    end
    if tocopy isa Array
        return MakeArray{Core.Typeof(tocopy),Tuple{map(make_valable, tocopy)...}}
    end
    if tocopy isa Symbol
        return tocopy
    end
    if tocopy isa Int || tocopy isa AbstractFloat || tocopy isa Nothing || tocopy isa Type
        return MakeVal{Val{tocopy}}
    end
    if tocopy isa AbstractString
        return MakeString{Core.Typeof(tocopy),Symbol(string)} || T <: Nothing
    end
    T = Core.Typeof(tocopy)
    if tocopy isa Tuple || tocopy isa NamedTuple || isstructtype(T)
        elems = []
        nf = fieldcount(T)
        for i in 1:nf
            push!(elems, make_valable(getfield(tocopy, i)))
        end
        return MakeStruct{Core.Typeof(tocopy),Tuple{elems...}}
    end

    return error("cannot copy $tocopy of type $(Core.Typeof(tocopy))")
end

function create_result(tocopy::Type{MakeConcreteRArray{T}}, path, result_stores) where {T}
    return :($T($(result_stores[path])))
end

function create_result(tocopy::Tuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, (path..., k), result_stores))
    end
    return quote
        ($(elems...),)
    end
end

function create_result(tocopy::NamedTuple, path, result_stores)
    elems = Union{Symbol,Expr}[]
    for (k, v) in pairs(tocopy)
        push!(elems, create_result(v, (path..., k), result_Stores))
    end
    return quote
        NamedTuple{$(keys(tocopy))}($elems)
    end
end

function create_result(::Type{MakeArray{AT,tocopy}}, path, result_stores) where {AT,tocopy}
    elems = Expr[]
    for (i, v) in enumerate(tocopy.parameters)
        push!(elems, create_result(v, (path..., i), result_stores))
    end
    return quote
        $(eltype(AT))[$(elems...)]
    end
end

function create_result(tocopy::Type{MakeVal{Val{nothing}}}, path, result_stores)
    return :(nothing)
end

function create_result(tocopy::Type{MakeVal{Val{elem}}}, path, result_stores) where {elem}
    return :($elem)
end

function create_result(tocopy::Symbol, path, result_stores)
    return Meta.quot(tocopy)
end

function create_result(tocopy::Type{MakeString{AT,Val}}, path, result_stores) where {AT,Val}
    return :($(AT(Val)))
end

function create_result(::Type{MakeStruct{AT,tocopy}}, path, result_stores) where {AT,tocopy}
    elems = Union{Symbol,Expr}[]
    for (i, v) in enumerate(tocopy.parameters)
        ev = create_result(v, (path..., i), result_stores)
        push!(elems, ev)
    end
    return Expr(:new, AT, elems...)
end

struct Thunk{
    linear_results_paths,
    linear_args_paths,
    preserved_args_paths,
    concrete_result_ty,
    closure_ty,
}
    exec::XLA.LoadedExecutable
    fnwrap::closure_ty
end

@generated function (
    thunk::Thunk{
        Val{linear_results_paths},
        Val{linear_args_paths},
        Val{preserved_args_paths},
        concrete_result_ty,
        closure_ty,
    }
)(
    args::Vararg{Any,N}
) where {
    linear_results_paths,
    linear_args_paths,
    preserved_args_paths,
    N,
    concrete_result_ty,
    closure_ty,
}
    arg_syncs = Expr[]
    topres = Symbol[]
    linearized_args = Union{Symbol,Expr}[]

    for (i, argpaths) in enumerate(linear_args_paths)
        paths = ((p for p in argpaths if p[1] == :args)...,)
        path = if length(paths) == 1
            paths[1]
        else
            throw("Invalid path duplication $(argpaths) into $(paths)")
        end
        res = :(args[$(path[2])])
        for p in path[3:end]
            res = :(Base.getfield($res, $(Meta.quot(p))))
        end
        sym = Symbol("sbuf_$i")
        sbuf = :($sym = XLA.synced_buffer($res.data))
        push!(arg_syncs, sbuf)

        push!(topres, sym)

        res = :($sym.buffer)
        push!(linearized_args, res)
    end

    concretize = Expr[]
    for idx in 1:length(linear_results_paths)
        push!(concretize, :($(Symbol("concrete_res_$(idx)")) = linearized_results[$idx]))
    end

    delinearized_results = Expr[]

    result_stores = Dict{Tuple,Symbol}()

    for (idx, result_paths) in enumerate(linear_results_paths)
        paths = ((p for p in result_paths if p[1] != :args)...,)
        for path in paths
            if path[1] == :result
                res = Symbol("result")
                path = path[2:end]
                result_stores[path] = Symbol("concrete_res_$(idx)")
                continue
            else
                if path[1] != :resargs
                    @show idx #, result
                    @show paths
                    @show path
                end
                @assert path[1] == :resargs
                res = :(args[$(path[2])])
                path = path[3:end]
            end
            for p in path
                res = :(Base.getfield($res, $(Meta.quot(p))))
            end
            res = :($res.data = $(Symbol("concrete_res_$(idx)")))
            push!(delinearized_results, res)
        end
    end

    for (result_paths, arg_idx) in preserved_args_paths
        for path in result_paths
            argpaths = linear_args_paths[arg_idx + 1]
            argpath = only((p for p in argpaths if p[1] == :args))

            if path[1] == :result
                res = Symbol("result")
                path = path[2:end]
            else
                @assert path[1] == :resargs || path[1] == :args
                # We can optimize cases where we set the arg to itself
                if path[2:end] == argpath[2:end]
                    continue
                end
                @show path, argpath
                res = :(args[path[2]])
                path = path[3:end]
            end
            for p in path
                res = :(Base.getfield($res, $(Meta.quot(p))))
            end

            argres = :(args[argpath[2]])
            for p in argpath[3:end]
                argres = :(Base.getfield($argres, $(Meta.quot(p))))
            end

            res = :($res.data = $argres.data)
            push!(delinearized_results, res)
        end
    end

    donated_args_set = zeros(UInt8, length(linearized_args))
    preserved_argnums = [i for (_, i) in preserved_args_paths]
    for i in 1:length(linear_args_paths)
        if !in(i, preserved_argnums)
            donated_args_set[i] = 1
        end
    end
    donated_args_set = (donated_args_set...,)

    exec_call = if length(linear_results_paths) == 0
        :()
    else
        quote
            $(arg_syncs...)
            GC.@preserve $(topres...) begin
                linearized_results = XLA.ExecutableCall(
                    thunk.exec,
                    ($(linearized_args...),),
                    $donated_args_set,
                    Val($(length(linear_results_paths))),
                )
            end
        end
    end

    resexpr = create_result(concrete_result_ty, (), result_stores)
    expr = quote
        Base.@_inline_meta
        $(
            # if `f` is a closure, then prepend the closure into `args`
            # the closure fields will be correctly extracted from it as the tracer has already passed through it
            if !(closure_ty <: Nothing)
                :(args = (thunk.fnwrap, args...))
            end
        )
        $exec_call
        $(concretize...)
        # Needs to store into result
        result = $resexpr
        $(delinearized_results...)
        return result
    end
    return expr
end

function generate_jlfunc(
    concrete_result,
    client,
    mod,
    linear_args,
    linear_results,
    preserved_args,
    fnwrap::closure_ty,
) where {closure_ty}
    linear_results_paths = (map(x -> x.paths, linear_results)...,)
    linear_args_paths = (map(x -> x.paths, linear_args)...,)
    preserved_args_paths = (map(x -> (x[1].paths, x[2]), preserved_args)...,)
    exec = XLA.Compile(client, mod)
    v = make_valable(concrete_result)
    return Thunk{
        Val{linear_results_paths},
        Val{linear_args_paths},
        Val{preserved_args_paths},
        v,
        closure_ty,
    }(
        exec, fnwrap
    )
end

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
    # PackageExtensionCompat: required for weakdeps to work in Julia <1.9
    @require_extensions

    registry[] = MLIR.IR.DialectRegistry()
    @ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(
        registry[]::MLIR.API.MlirDialectRegistry
    )::Cvoid
end

const opt_passes = """
            inline{default-pipeline=canonicalize max-iterations=4},
            canonicalize,cse,
            canonicalize,
            enzyme-hlo-generate-td{
            patterns=compare_op_canon<16>;
transpose_transpose<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;

cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;

concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
reshape_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;

bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
reshape_iota<16>;
slice_reshape_slice<1>;
dot_general_simplify<16>;
transpose_simplify<16>;
reshape_empty_broadcast<1>;
add_pad_pad_to_concat<1>;
broadcast_reshape<1>;

slice_reshape_concat<1>;
slice_reshape_elementwise<1>;
slice_reshape_transpose<1>;
slice_reshape_dot_general<1>;
concat_pad<1>;

reduce_pad<1>;
broadcast_pad<1>;

zero_product_reshape_pad<1>;
mul_zero_pad<1>;
div_zero_pad<1>;

binop_const_reshape_pad<1>;
binop_const_pad_add<1>;
binop_const_pad_subtract<1>;
binop_const_pad_mul<1>;
binop_const_pad_div<1>;

slice_reshape_pad<1>;
binop_binop_pad_pad_add<1>;
binop_binop_pad_pad_mul<1>;
binop_pad_pad_add<1>;
binop_pad_pad_subtract<1>;
binop_pad_pad_mul<1>;
binop_pad_pad_div<1>;
binop_pad_pad_min<1>;
binop_pad_pad_max<1>;

unary_pad_push_convert<1>;
unary_pad_push_tanh<1>;
unary_pad_push_exp<1>;

transpose_pad<1>;

transpose_dot_reorder<1>;
dot_transpose<1>;
convert_convert_float<1>;
concat_to_pad<1>;
concat_appending_reshape<1>;
reshape_iota<1>;

broadcast_reduce<1>;
slice_dot_general<1>;

dot_reshape_pad<1>;
pad_dot_general<1>(0);

dot_reshape_pad<1>;
pad_dot_general<1>(1);
            },
            transform-interpreter,
            enzyme-hlo-remove-transform
"""

function compile_to_module(mod, f, args; optimize=true)
    fnwrapped, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = 
    MLIR.IR.block!(MLIR.IR.body(mod)) do
        return make_mlir_fn(
            f, args, (), "main", true
        )
    end

    concrete_seen = IdDict()

    concrete_result = make_tracer(
        concrete_seen, traced_result, ("result",), TracedToConcrete
    )

    if optimize
        XLA.RunPassPipeline(
            opt_passes * ",enzyme-batch,"*
            opt_passes *
            ",enzyme,arith-raise{stablehlo=true},canonicalize, remove-unnecessary-enzyme-ops, enzyme-simplify-math," *
            opt_passes,
            mod,
        )
    end

    preserved_args = Tuple{TracedRArray,Int}[]
    results = [MLIR.IR.operand(ret, i) for i in 1:MLIR.IR.noperands(ret)]
    nresults = MLIR.IR.Value[]
    linear_results2 = TracedRArray[]
    for (i, op) in enumerate(results)
        if !MLIR.IR.is_block_arg(op)
            push!(nresults, op)
            push!(linear_results2, linear_results[i])
            continue
        end
        push!(preserved_args, (linear_results[i], MLIR.IR.block_arg_num(op)))
    end
    fnbody = MLIR.IR.block(ret)
    MLIR.API.mlirOperationDestroy(ret.operation)
    ret.operation = MLIR.API.MlirOperation(C_NULL)
    MLIR.IR.block!(fnbody) do
        return MLIR.Dialects.func.return_(nresults)
    end

    out_tys2 = [MLIR.IR.type(a) for a in nresults]

    func3 = MLIR.Dialects.func.func_(;
        sym_name="main",
        function_type=MLIR.IR.FunctionType(in_tys, out_tys2),
        body=MLIR.IR.Region(),
    )
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func3, 1), MLIR.IR.region(func2, 1))

    push!(MLIR.IR.body(mod), func3)

    MLIR.API.mlirOperationDestroy(func2.operation)
    func2.operation = MLIR.API.MlirOperation(C_NULL)

    return linear_args,
    linear_results2, preserved_args, seen_args, concrete_result,
    fnwrapped
end

function compile(
    f::FTy, args::VAT; pipeline_options="", client=nothing
) where {FTy,VAT<:Tuple}
    N = length(args)
    ctx = MLIR.IR.Context()
    Base.append!(registry[]; context=ctx)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())
        MLIR.IR.mmodule!(mod) do
            linear_args, linear_results2, preserved_args, seen_args, concrete_result, fnwrapped = compile_to_module(
                mod, f, args; optimize=true
            )

            if isnothing(client)
                if length(linear_args) > 0
                    for (k, v) in seen_args
                        if !(v isa TracedRArray)
                            continue
                        end
                        client = XLA.client(k.data)
                    end
                end
                if isnothing(client)
                    client = XLA.default_backend[]
                end
            end

            return generate_jlfunc(
                concrete_result,
                client,
                mod,
                linear_args,
                linear_results2,
                preserved_args,
                fnwrapped ? f : nothing,
            )
        end
    end
end

struct CompiledModule
    mod::MLIR.IR.Module
    ctx::MLIR.IR.Context
end

Base.show(io::IO, cm::CompiledModule) = show(io, cm.mod)

"""
    @code_hlo [optimize = ...] f(args...)
"""
macro code_hlo(options, maybe_call=nothing)
    call = something(maybe_call, options)
    options = isnothing(maybe_call) ? :(optimize = true) : options
    Meta.isexpr(call, :call) || error("@code_mlir: expected call, got $call")
    if !Meta.isexpr(options, :(=)) || options.args[1] != :optimize
        error("@code_mlir: expected options in format optimize=value, got $options")
    end

    options = Expr(:tuple, Expr(:parameters, Expr(:kw, options.args...)))

    quote
        options = $(esc(options))
        f = $(esc(call.args[1]))
        args = $(esc(Expr(:vect, call.args[2:end]...)))

        ctx = MLIR.IR.Context()
        Base.append!(registry[]; context=ctx)
        @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
        MLIR.IR.context!(ctx) do
            mod = MLIR.IR.Module(MLIR.IR.Location())
            compile_to_module(mod, f, args; optimize=options.optimize)
            CompiledModule(mod, ctx)
        end
    end
end

function set_default_backend(backend::XLA.Client)
    return XLA.default_backend[] = backend
end

function set_default_backend(backend::String)
    backend = XLA.backends[backend]
    return XLA.default_backend[] = backend
end

end # module
