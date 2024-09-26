@enum TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
end

for T in (
    DataType,
    Module,
    Nothing,
    Symbol,
    AbstractChar,
    AbstractFloat,
    Integer,
    AbstractString,
    RArray,
)
    @eval function traced_type(::Type{T}, seen, mode) where {T<:$T}
        return T
    end
end

function traced_type(::Type{C}, seen, mode) where {T,C<:Complex{T}}
    if !(C isa UnionAll)
        return Complex{traced_type(T, seen, mode)}
    else
        return @invoke traced_type(C::Type{Any}, seen, mode)
    end
end

function traced_type(::Type{T}, seen, mode) where {T<:Function}
    # functions are directly returned
    if sizeof(T) == 0
        return T
    end

    # in closures, enclosured variables need to be traced
    N = fieldcount(T)
    changed = false
    traced_fieldtypes = ntuple(Val(N)) do i
        next = traced_type(fieldtype(T, i), seen, mode)
        changed |= next != fieldtype(T, i)
        next
    end

    if !changed
        return T
    end

    # closure are struct types with the types of enclosured vars as type parameters
    return Core.apply_type(T.name.wrapper, traced_fieldtypes...)
end

@inline is_concrete_tuple(x::T2) where {T2} =
    (x <: Tuple) && !(x === Tuple) && !(x isa UnionAll)

function traced_type(::Type{T}, seen, mode) where {T<:Tuple}
    if !Base.isconcretetype(T) || !is_concrete_tuple(T) || T isa UnionAll
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    elseif is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
        # Tuple{((T2 isa Core.TypeofVararg ? Any : T2) for T2 in T.parameters)...}
        throw(AssertionError("Type tuple of vararg $T is not supported"))
    end
    TT = [traced_type(T.parameters[i], seen, mode) for i in 1:length(T.parameters)]
    return Tuple{TT...}
end

function traced_type(::Type{T}, seen, mode) where {N,V,T<:NamedTuple{N,V}}
    return NamedTuple{N,traced_type(V, seen, mode)}
end

function traced_type(::Type{T}, seen, mode) where {K,V,T<:AbstractDict{K,V}}
    dictty = T.name.wrapper
    return dictty{K,traced_type(V, seen, mode)}
end

@inline getmap(::Val{T}) where {T} = nothing
@inline getmap(::Val{T}, a, b, args...) where {T} = getmap(Val(T), args...)
@inline getmap(::Val{T}, ::Val{T}, ::Val{T2}, args...) where {T,T2} = T2

function traced_type(::Type{T}, seen, mode) where {T}
    if T === Any
        return T
    end

    if T === Union{}
        return T
    end

    if Enzyme.Compiler.isghostty(T) || Core.Compiler.isconstType(T)
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
        return Union{traced_type(T.a, seen, mode),traced_type(T.b, seen, mode)}
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        throw("Unhandled abstract type $T")
    end

    if !(Base.isconcretetype(T) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    nextTy = getmap(Val(T), seen...)
    if !isnothing(nextTy)
        return nextTy
    end

    seen2 = (Val(T), Val(T), seen...)

    changed = false
    subTys = Type[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type(subT, seen2, mode)
        changed |= subT != subTT
        push!(subTys, subTT)
    end

    if !changed
        return T
    end

    subParms = []
    for SST in T.parameters
        if SST isa Type
            TrT = traced_type(SST, seen, mode)
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
            subTT = traced_type(subT, seen3, mode)
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

function traced_type(::Type{T}, seen, ::Val{mode}) where {T<:ConcreteRArray,mode}
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

function traced_type(::Type{T}, seen::ST, ::Val{mode}) where {ST,T<:TracedRArray,mode}
    if mode == ConcreteToTraced
        throw("TracedRArray $T cannot be traced")
    elseif mode == TracedToConcrete
        @inline base_typec(TV::TT) where {TT<:UnionAll} =
            UnionAll(TV.var, base_typec(TV.body))
        @inline base_typec(TV::TT) where {TT<:DataType} = ConcreteRArray{TV.parameters...}
        return base_typec(T)
    elseif mode == TracedTrack || mode == TracedSetPath
        return T
    else
        throw("Abstract RArray $T cannot be made concrete in mode $mode")
    end
end

function traced_type(::Type{T}, seen, mode) where {T<:XLAArray}
    throw("XLA $T array cannot be traced")
end

function traced_type(::Type{A}, seen::ST, ::Val{mode}) where {T,N,A<:Array{T,N},ST,mode}
    if mode == ArrayToConcrete && T <: AbstractFloat
        return ConcreteRArray{T,N}
    else
        return Array{traced_type(T, seen, Val(mode)),N}
    end
end

for P in (Ptr, Core.LLVMPtr, Base.RefValue)
    @eval function traced_type(::Type{P}, seen, mode) where {T,P<:$P{T}}
        return $P{traced_type(T, seen, mode)}
    end
end

function traced_type(::Type{Val{T}}, seen, mode) where {T}
    if traced_type(typeof(T), seen, mode) == typeof(T)
        return Val{T}
    end
    throw("Val type $(Val{T}) cannot be traced")
end

append_path(path, i) = (path..., i)

function make_tracer(
    seen,
    @nospecialize(prev::RT),
    @nospecialize(path),
    mode;
    toscalar=false,
    tobatch=nothing,
) where {RT}
    if haskey(seen, prev)
        return seen[prev]
    end
    TT = traced_type(RT, (), Val(mode))
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

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

function make_tracer(
    seen, @nospecialize(prev::ConcreteRArray{T,N}), @nospecialize(path), mode; kwargs...
) where {T,N}
    if mode == ArrayToConcrete
        return prev
    end
    if mode != ConcreteToTraced
        throw("Cannot trace concrete")
    end
    if haskey(seen, prev)
        return seen[prev]::TracedRArray{T,N}
    end
    @assert N isa Int
    res = TracedRArray{T,N}((path,), nothing, size(prev))
    seen[prev] = res
    return res
end

function make_tracer(
    seen,
    @nospecialize(prev::TracedRArray{T,N}),
    @nospecialize(path),
    mode;
    toscalar=false,
    tobatch=nothing,
) where {T,N}
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
            TracedRArray{T,0}((path,), nothing, ())
        elseif !isnothing(tobatch)
            TracedRArray{T,length(tobatch)}((path,), prev.mlir_data, tobatch)
        else
            TracedRArray{T,N}((path,), prev.mlir_data, size(prev))
        end
        seen[prev] = res
        return res
    end

    if mode == TracedToConcrete
        if haskey(seen, prev)
            return seen[prev]::ConcreteRArray{T,N}
        end
        res = ConcreteRArray{T,N}(XLA.AsyncEmptyBuffer, size(prev))
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

function make_tracer(
    seen, @nospecialize(prev::RT), @nospecialize(path), mode; kwargs...
) where {RT<:AbstractFloat}
    return prev
end

make_tracer(seen, prev::Symbol, @nospecialize(path), mode; kwargs...) = prev

function make_tracer(
    seen,
    @nospecialize(prev::Complex{RT}),
    @nospecialize(path),
    mode;
    toscalar=false,
    tobatch=nothing,
) where {RT}
    return Complex(
        make_tracer(seen, prev.re, append_path(path, :re), mode; toscalar, tobatch),
        make_tracer(seen, prev.im, append_path(path, :im), mode; toscalar, tobatch),
    )
end

function make_tracer(
    seen, @nospecialize(prev::RT), @nospecialize(path), mode; kwargs...
) where {RT<:Array}
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
            nv = make_tracer(seen, pv, append_path(path, I), mode; kwargs...)
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

function make_tracer(
    seen, @nospecialize(prev::RT), @nospecialize(path), mode; kwargs...
) where {RT<:Tuple}
    return (
        (
            make_tracer(seen, v, append_path(path, i), mode; kwargs...) for
            (i, v) in enumerate(prev)
        )...,
    )
end

function make_tracer(
    seen, @nospecialize(prev::NamedTuple{A,RT}), @nospecialize(path), mode; kwargs...
) where {A,RT}
    return NamedTuple{A,traced_type(RT, (), Val(mode))}((
        (
            make_tracer(
                seen, Base.getfield(prev, i), append_path(path, i), mode; kwargs...
            ) for i in 1:length(A)
        )...,
    ))
end

function make_tracer(seen, prev::Core.Box, @nospecialize(path), mode; kwargs...)
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    tr = make_tracer(seen, prev2, append_path(path, :contents), mode; kwargs...)
    if tr === prev2
        seen[prev] = prev
        return prev
    end
    res = Core.Box(tr)
    seen[prev] = res
    return res
end

@inline to_rarray(@nospecialize(x)) = make_tracer(IdDict(), x, (), Reactant.ArrayToConcrete)
