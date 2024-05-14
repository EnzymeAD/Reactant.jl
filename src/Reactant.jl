module Reactant

using ArrayInterface: ArrayInterface

include("mlir/MLIR.jl")
include("XLA.jl")
include("utils.jl")

abstract type RArray{ElType,Shape,N} <: AbstractArray{ElType,N} end

@inline Base.eltype(::RArray{ElType,Shape}) where {ElType,Shape} = ElType
@inline Base.size(::RArray{ElType,Shape}) where {ElType,Shape} = Shape
@inline Base.size(::Type{<:RArray{ElType,Shape}}) where {ElType,Shape} = Shape
@inline Base.ndims(::RArray{ElType,Shape,N}) where {ElType,Shape,N} = N
@inline Base.ndims(::Type{<:RArray{ElType,Shape,N}}) where {ElType,Shape,N} = N

ArrayInterface.can_setindex(::Type{<:RArray}) = false
ArrayInterface.fast_matrix_colors(::Type{<:RArray}) = false

@inline mlir_type(::RArray{ElType,Shape,N}) where {ElType,Shape,N} =
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
        return T
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
        if aT === nothing
            throw("Unhandled type $T")
        end
        if Base.datatype_fieldcount(aT) === nothing
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

    if Val(T) ∈ seen
        return T
    end

    seen = (Val(T), seen...)

    changed = false
    subTys = Type[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type(subT, seen, Val(mode))
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
    if fieldcount(T) == fieldcount(TT2)
        legal = true
        for f in 1:fieldcount(T)
            subT = fieldtype(T, f)
            subT2 = fieldtype(TT2, f)
            subTT = traced_type(subT, seen, Val(mode))
            legal &= subT2 == subTT
        end
        if legal
            return TT2
        end
    end

    name = Symbol[]

    return NamedTuple{fieldnames(T),Tuple{subTys...}}
end

function append_path(path, i)
    return (path..., i)
end

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT}
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
            xi2 = make_tracer(seen, xi, append_path(path, i), mode, data)
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
        @show TT, subs, tup
        return NamedTuple{TT.parameters[1],typeof(tup)}(tup)
    end

    if ismutabletype(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        seen[prev] = y
        changed = false
        for i in 1:nf
            if isdefined(prev, i)
                xi = Base.getfield(prev, i)
                xi2 = make_tracer(seen, xi, append_path(path, i), mode, data)
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
            xi2 = make_tracer(seen, xi, append_path(path, i), mode, data)
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
    seen::IdDict, prev::ConcreteRArray{ElType,Shape,N}, path, mode, data
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
    seen::IdDict, prev::TracedRArray{ElType,Shape,N}, path, mode, data
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
        res = TracedRArray{ElType,Shape,N}((path,), prev.mlir_data)
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

@inline function make_tracer(
    seen::IdDict, prev::RT, path, mode, data
) where {RT<:AbstractFloat}
    return prev
end

@inline function make_tracer(seen::IdDict, prev::Complex{RT}, path, mode, data) where {RT}
    return Complex(
        make_tracer(seen, prev.re, append_path(path, :re), mode, data),
        make_tracer(seen, prev.im, append_path(path, :im), mode, data),
    )
end

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT<:Array}
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
            nv = make_tracer(seen, pv, append_path(path, I), mode, data)
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

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT<:Tuple}
    return (
        (
            make_tracer(seen, v, append_path(path, i), mode, data) for
            (i, v) in enumerate(prev)
        )...,
    )
end

@inline function make_tracer(
    seen::IdDict, prev::NamedTuple{A,RT}, path, mode, data
) where {A,RT}
    return NamedTuple{A,traced_type(RT, (), Val(mode))}((
        (
            make_tracer(
                seen, Base.getfield(prev, name), append_path(path, name), mode, data
            ) for name in A
        )...,
    ))
end

@inline function make_tracer(seen::IdDict, prev::Core.Box, path, mode, data)
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    tr = make_tracer(seen, prev2, append_path(path, :contents), mode, data)
    if tr == prev2
        seen[prev] = prev
        return prev
    end
    res = Core.Box(tr)
    seen[prev] = res
    return res
end

function generate_jlfunc(
    concrete_result, client, mod, Nargs, linear_args, linear_results, preserved_args
)
    args = ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        return Symbol("arg_$i")
    end

    arg_syncs = Expr[]
    topres = Symbol[]
    linearized_args = Union{Symbol,Expr}[]

    for (i, arg) in enumerate(linear_args)
        paths = ((p for p in arg.paths if p[1] == "args")...,)
        path = if length(paths) == 1
            paths[1]
        else
            throw("Invalid path duplication $(arg.paths) into $(paths)")
        end
        res = Symbol("arg_$(path[2])")
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
    for (idx, res) in enumerate(linear_results)
        push!(concretize, :($(Symbol("concrete_res_$(idx)")) = linearized_results[$idx]))
    end

    delinearized_results = Expr[]

    result_stores = Dict{Tuple,Symbol}()

    for (idx, result) in enumerate(linear_results)
        paths = ((p for p in result.paths if p[1] != "args")...,)
        for path in paths
            if path[1] == "result"
                res = Symbol("result")
                path = path[2:end]
                result_stores[path] = Symbol("concrete_res_$(idx)")
                continue
            else
                if path[1] != "resargs"
                    @show idx #, result
                    @show paths
                    @show path
                end
                @assert path[1] == "resargs"
                res = Symbol("arg_$(path[2])")
                path = path[3:end]
            end
            for p in path
                res = :(Base.getfield($res, $p))
            end
            res = :($res.data = $(Symbol("concrete_res_$(idx)")))
            push!(delinearized_results, res)
        end
    end

    for (result, arg_idx) in preserved_args
        for path in result.paths
            arg = linear_args[arg_idx + 1]
            argpath = only((p for p in arg.paths if p[1] == "args"))

            if path[1] == "result"
                res = Symbol("result")
                path = path[2:end]
            else
                @assert path[1] == "resargs" || path[1] == "args"
                # We can optimize cases where we set the arg to itself
                if path[2:end] == argpath[2:end]
                    continue
                end
                @show path, argpath
                res = Symbol("arg_$(path[2])")
                path = path[3:end]
            end
            for p in path
                res = :(Base.getfield($res, $p))
            end

            argres = Symbol("arg_$(argpath[2])")
            for p in argpath[3:end]
                argres = :(Base.getfield($argres, $p))
            end

            res = :($res.data = $argres.data)
            push!(delinearized_results, res)
        end
    end

    exec = XLA.Compile(client, mod)

    donated_args_set = zeros(UInt8, length(linearized_args))
    preserved_argnums = [i for (_, i) in preserved_args]
    for (i, val) in enumerate(linear_args)
        if !in(i, preserved_args)
            donated_args_set[i] = 1
        end
    end
    donated_args_set = (donated_args_set...,)

    exec_call = if length(linear_results) == 0
        :()
    else
        quote
            $(arg_syncs...)
            GC.@preserve $(topres...) begin
                linearized_results = XLA.ExecutableCall(
                    $exec,
                    ($(linearized_args...),),
                    $donated_args_set,
                    Val($(length(linear_results))),
                )
            end
        end
    end

    concrete_result_maker = Expr[]
    function create_result(tocopy::T, resname::Symbol, path) where {T}
        if T <: ConcreteRArray
            push!(concrete_result_maker, :($resname = $T($(result_stores[path]))))
            return nothing
        end
        if T <: Tuple
            elems = Symbol[]
            for (k, v) in pairs(tocopy)
                sym = Symbol(resname, :_, k)
                create_result(v, sym, (path..., k))
                push!(elems, sym)
            end
            push!(
                concrete_result_maker,
                quote
                    $resname = ($(elems...),)
                end,
            )
            return nothing
        end
        if T <: NamedTuple
            elems = Symbol[]
            for (k, v) in pairs(tocopy)
                sym = Symbol(resname, :_, k)
                create_result(v, sym, (path..., k))
                push!(elems, sym)
            end
            push!(
                concrete_result_maker,
                quote
                    $resname = NamedTuple{$(keys(tocopy))}($elems)
                end,
            )
            return nothing
        end
        if T <: Array
            elems = Symbol[]
            for (i, v) in enumerate(tocopy)
                sym = Symbol(string(resname) * "_" * string(i))
                create_result(v, sym, (path..., i))
                push!(elems, sym)
            end
            push!(
                concrete_result_maker,
                quote
                    $resname = $(eltype(T))[$(elems...)]
                end,
            )
            return nothing
        end
        if T <: Int || T <: AbstractFloat || T <: AbstractString || T <: Nothing
            push!(concrete_result_maker, :($resname = $tocopy))
            return nothing
        end
        if T <: Symbol
            push!(concrete_result_maker, :($resname = $(QuoteNode(tocopy))))
            return nothing
        end
        if isstructtype(T)
            elems = Symbol[]
            nf = fieldcount(T)
            for i in 1:nf
                sym = Symbol(resname, :_, i)
                create_result(getfield(tocopy, i), sym, (path..., i))
                push!(elems, sym)
            end
            push!(
                concrete_result_maker,
                quote
                    flds = Any[$(elems...)]
                    $resname = ccall(
                        :jl_new_structv, Any, (Any, Ptr{Cvoid}, UInt32), $T, flds, $nf
                    )
                end,
            )
            return nothing
        end
        # TODO: Add NamedTuple here?

        return error("cannot copy $T")
    end

    create_result(concrete_result, :result, ())
    func = quote
        ($(args...),) -> begin
            $exec_call
            $(concretize...)
            # Needs to store into result
            $(concrete_result_maker...)
            $(delinearized_results...)
            return result
        end
    end
    return eval(func)
end

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
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
            fnwrapped, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
                mod, f, args, (), "main", true
            )
            @assert !fnwrapped

            concrete_seen = IdDict()

            concrete_result = make_tracer(
                concrete_seen, traced_result, ("result",), TracedToConcrete, nothing
            ) #=data=#

            if client === nothing
                if length(linear_args) > 0
                    for (k, v) in seen_args
                        if !(v isa TracedRArray)
                            continue
                        end
                        client = XLA.client(k.data)
                    end
                end
                if client === nothing
                    client = XLA.default_backend[]
                end
            end

            XLA.RunPassPipeline(
                opt_passes *
                ",enzyme,arith-raise{stablehlo=true},canonicalize, remove-unnecessary-enzyme-ops, enzyme-simplify-math," *
                opt_passes,
                mod,
            )

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

            # println(string(mod))

            return generate_jlfunc(
                concrete_result,
                client,
                mod,
                N,
                linear_args,
                linear_results2,
                preserved_args,
            )
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
