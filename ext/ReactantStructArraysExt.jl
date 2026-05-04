module ReactantStructArraysExt

using Reactant: Reactant
using StructArrays: StructArrays

import StructArrays:
    StructArrayStyle,
    StructArray,
    index_type,
    components,
    createinstance,
    get_ith,
    maybe_convert_elt,
    foreachfield
import Reactant:
    TraceMode,
    TracedToTypes,
    traced_type_inner,
    append_path,
    make_tracer,
    traced_type,
    ReactantPrimitive,
    broadcast_to_size,
    TracedRNumber,
    TracedRArray,
    unwrapped_eltype
import Reactant.TracedRArrayOverrides: AbstractReactantArrayStyle, _copy
import Base.Broadcast: Broadcasted

function __init__()
    Reactant.@skip_rewrite_func StructArrays.index_type
end

StructArrays.always_struct_broadcast(::AbstractReactantArrayStyle) = true

function Base.copy(
    bc::Broadcasted{StructArrays.StructArrayStyle{S,N}}
) where {S<:AbstractReactantArrayStyle,N}
    return _copy(bc)
end

function Reactant.broadcast_to_size(arg::StructArray{T}, rsize) where {T}
    new = Tuple((broadcast_to_size(c, rsize) for c in components(arg)))
    return StructArray{T}(new)
end

function Base.copyto!(
    dest::StructArray, bc::Base.Broadcast.Broadcasted{<:AbstractReactantArrayStyle}
)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)
    res = Reactant.TracedUtils.elem_apply_via_while_loop(bc.f, args...; track_numbers=Union{})
    copyto!(dest, res)

    return dest
end

function Reactant.TracedRArrayOverrides._copyto!(
    dest::StructArray, bc::Base.Broadcast.Broadcasted{<:AbstractReactantArrayStyle}
)
    return copyto!(dest, bc)
end

function Base.copyto!(
    dest::Reactant.TracedRArray, bc::Broadcasted{StructArrayStyle{S,N}}
) where {S<:AbstractReactantArrayStyle,N}
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)
    args = (Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)
    res = Reactant.TracedUtils.elem_apply_via_while_loop(bc.f, args...; track_numbers=Union{})
    return copyto!(dest, res)
end

function alloc_sarr(bc, T)
    # Short circuit for Complex since in Reactant they are just a regular number
    T <: Complex && return similar(bc, T)
    asa = Base.Fix1(alloc_sarr, bc)
    if StructArrays.isnonemptystructtype(T)
        return StructArrays.buildfromschema(asa, T)
    else
        return similar(bc, T)
    end
end

function Base.similar(
    bc::Broadcasted{StructArrayStyle{S,N}}, ::Type{ElType}
) where {S<:AbstractReactantArrayStyle,N,ElType}
    bc′ = convert(Broadcasted{S}, bc)
    # It is possible that we have multiple broadcasted arguments
    return alloc_sarr(bc′, ElType)
end

Base.@propagate_inbounds function StructArrays._getindex(
    x::StructArray{T}, I::Vararg{TracedRNumber{<:Integer}}
) where {T}
    cols = components(x)
    @boundscheck checkbounds(x, I...)
    return createinstance(T, get_ith(cols, I...)...)
end

setstruct(col, val, I) = @inbounds Reactant.@allowscalar col[I] = val
struct SetStruct{T}
    I::T
end
(s::SetStruct)(col, val) = setstruct(col, val, s.I)
(s::SetStruct)(vals) = s(vals...)

Base.@propagate_inbounds function Base.setindex!(
    s::StructArray{T,<:Any,<:Any,Int}, vals, I::TracedRNumber{TI}
) where {T,TI<:Integer}
    valsT = maybe_convert_elt(T, vals)
    setter = SetStruct(I)
    foreachfield(setter, s, valsT)
    return s
end

const MRarr = Union{Reactant.AnyTracedRArray,Reactant.RArray}
getstruct(col, n, I...) = @inbounds Reactant.@allowscalar col[n][I...]
struct GetStruct{C,Idx}
    cols::C
    I::Idx
end
(g::GetStruct)(n) = getstruct(g.cols, n, g.I...)

function StructArrays.get_ith(cols::NamedTuple{N,<:NTuple{K,<:MRarr}}, I...) where {N,K}
    getter = GetStruct(cols, I)
    ith = ntuple(getter, Val(K))
    return ith
end

function StructArrays.get_ith(cols::NTuple{K,<:MRarr}, I...) where {K}
    getter = GetStruct(cols, I)
    ith = ntuple(getter, Val(K))
    return ith
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(prev::Type{StructArray{ET,N,C,I}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {ET,N,C,I}
    ET_traced = traced_type_inner(
        ET, seen, mode, Union{ReactantPrimitive,track_numbers}, sharding, runtime
    )
    C_traced = traced_type_inner(C, seen, mode, track_numbers, sharding, runtime)
    return StructArray{ET_traced,N,C_traced,index_type(fieldtypes(C_traced))}
end

@inline function Reactant.traced_getfield(@nospecialize(obj::StructArray), field)
    return Base.getfield(obj, field)
end

# This is to tell StructArrays to leave these array types alone.
StructArrays.staticschema(::Type{<:Reactant.AnyTracedRArray}) = NamedTuple{()}
StructArrays.staticschema(::Type{<:Reactant.RArray}) = NamedTuple{()}
StructArrays.staticschema(::Type{<:Reactant.RNumber}) = NamedTuple{()}
# # Even though RNumbers we have fields we want them to be threated as empty structs
StructArrays.isnonemptystructtype(::Type{<:Reactant.RNumber}) = false
StructArrays.isnonemptystructtype(::Type{<:Reactant.TracedRArray}) = false
end
