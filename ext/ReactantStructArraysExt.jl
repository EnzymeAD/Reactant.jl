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

# Broadcasting over StructArrays
#
# Broadcasting `f` over StructArrays is lowered like any other elementwise operation in
# Reactant: `f` is traced exactly once on scalar (`TracedRNumber`) inputs and the result is
# batched over the array shape with `enzyme.batch` (see `TracedUtils.elem_apply`). To reuse
# that machinery with StructArray arguments / results we
#   1. flatten every StructArray argument into its leaf component arrays,
#   2. wrap `f` in `StructBroadcastFn`, which rebuilds the struct elements from the scalar
#      leaves, calls `f`, and flattens the (struct) result into a tuple of scalar leaves
#      following the layout of the destination's components,
#   3. rebuild the destination layout from the batched leaf arrays (one traced array per
#      output field).
# This emits plain elementwise StableHLO ops instead of a per-element `stablehlo.while`
# loop (which also destroys sharding of the inputs). The while loop is kept as a fallback
# for functions that cannot be batched (mirroring `TracedUtils.elem_apply`).

# Describes how (scalar / array) leaves map to the components of a `StructArray{T}`.
# `components` is a `Tuple` / `NamedTuple` whose entries are either `nothing` (leaf) or
# a nested `StructTemplate`.
struct StructTemplate{T,C}
    components::C
end
StructTemplate{T}(components::C) where {T,C} = StructTemplate{T,C}(components)

function struct_template(x::StructArray{T}) where {T}
    return StructTemplate{T}(map(struct_template, components(x)))
end
struct_template(@nospecialize(x)) = nothing

function collect_leaves!(leaves, x::StructArray)
    foreach(c -> collect_leaves!(leaves, c), components(x))
    return leaves
end
function collect_leaves!(leaves, xs::Tuple)
    foreach(x -> collect_leaves!(leaves, x), xs)
    return leaves
end
collect_leaves!(leaves, @nospecialize(x)) = (push!(leaves, x); leaves)

# scalar leaves -> struct element
rebuild_elem(::Nothing, leaves, i) = (leaves[i], i + 1)
function rebuild_elem(t::StructTemplate{T}, leaves, i) where {T}
    vals, i = rebuild_elems(Tuple(t.components), leaves, i)
    return createinstance(T, vals...), i
end
rebuild_elems(::Tuple{}, leaves, i) = ((), i)
function rebuild_elems(ts::Tuple, leaves, i)
    v, i = rebuild_elem(first(ts), leaves, i)
    rest, i = rebuild_elems(Base.tail(ts), leaves, i)
    return (v, rest...), i
end

# struct element -> scalar leaves
flatten_elem(::Nothing, val) = (val,)
flatten_elem(t::StructTemplate, val) = flatten_elems(t.components, val)
function flatten_elems(ts::NamedTuple{names}, val) where {names}
    return _flatten_elems(Tuple(ts), names, val)
end
flatten_elems(ts::Tuple, val) = _flatten_elems(ts, ntuple(identity, length(ts)), val)
_flatten_elems(::Tuple{}, ::Tuple{}, val) = ()
function _flatten_elems(ts::Tuple, keys::Tuple, val)
    return (
        flatten_elem(first(ts), StructArrays.component(val, first(keys)))...,
        _flatten_elems(Base.tail(ts), Base.tail(keys), val)...,
    )
end

# leaf arrays -> StructArray
function leaf_array(x, sz)
    x isa Reactant.TracedRArray && return x
    # `f` returned a constant that does not depend on its arguments
    x isa Union{Number,TracedRNumber} && return Reactant.broadcast_to_size(x, sz)
    return fill(x, sz)
end
rebuild_array(::Nothing, leaves, i, sz) = (leaf_array(leaves[i], sz), i + 1)
function rebuild_array(
    t::StructTemplate{T,<:NamedTuple{names}}, leaves, i, sz
) where {T,names}
    comps, i = rebuild_arrays(Tuple(t.components), leaves, i, sz)
    return StructArray{T}(NamedTuple{names}(comps)), i
end
function rebuild_array(t::StructTemplate{T,<:Tuple}, leaves, i, sz) where {T}
    comps, i = rebuild_arrays(t.components, leaves, i, sz)
    return StructArray{T}(comps), i
end
rebuild_arrays(::Tuple{}, leaves, i, sz) = ((), i)
function rebuild_arrays(ts::Tuple, leaves, i, sz)
    v, i = rebuild_array(first(ts), leaves, i, sz)
    rest, i = rebuild_arrays(Base.tail(ts), leaves, i, sz)
    return (v, rest...), i
end

# Scalar function applied by the batched `elem_apply`: takes the scalar leaves of all
# arguments, rebuilds the struct elements, applies `f`, and returns the scalar leaves of
# the result (converted to the destination eltype `ET`, like `setindex!` does).
struct StructBroadcastFn{F,A,R,ET} <: Function
    f::F
    arg_templates::A
    res_template::R
end
function StructBroadcastFn{ET}(f::F, arg_templates::A, res_template::R) where {F,A,R,ET}
    return StructBroadcastFn{F,A,R,ET}(f, arg_templates, res_template)
end
Base.show(io::IO, w::StructBroadcastFn) = print(io, "StructBroadcastFn(", w.f, ")")

function (w::StructBroadcastFn{F,A,R,ET})(leaves::Vararg{Any,N}) where {F,A,R,ET,N}
    args, _ = rebuild_elems(w.arg_templates, leaves, 1)
    res = w.f(args...)
    return flatten_elem(w.res_template, maybe_convert_elt(ET, res))
end

function struct_broadcast_elem_apply(f, args::Tuple, dest)
    if Reactant.use_overlayed_version(f)
        # `f` captures traced values; `elem_apply` cannot batch it either, loop instead.
        return Reactant.TracedUtils.elem_apply_via_while_loop(
            f, args...; track_numbers=Union{}
        )
    end
    res_template = struct_template(dest)
    ET = dest isa StructArray ? eltype(dest) : Any
    wrapped = StructBroadcastFn{ET}(f, map(struct_template, args), res_template)
    res_leaves = Reactant.TracedUtils.elem_apply(wrapped, collect_leaves!(Any[], args)...)
    res, _ = rebuild_array(res_template, res_leaves, 1, size(dest))
    return res
end

function Base.copyto!(
    dest::StructArray, bc::Base.Broadcast.Broadcasted{<:AbstractReactantArrayStyle}
)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = Tuple(Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)
    res = struct_broadcast_elem_apply(bc.f, args, dest)
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
    args = Tuple(Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)
    res = struct_broadcast_elem_apply(bc.f, args, dest)
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
