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
    new = [broadcast_to_size(c, rsize) for c in components(arg)]
    return StructArray{T}(NamedTuple(Base.propertynames(arg) .=> new))
end

function Base.copyto!(
    dest::StructArray, bc::Base.Broadcast.Broadcasted{<:AbstractReactantArrayStyle}
)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = Reactant.TracedUtils.elem_apply_via_while_loop(bc.f, args...)

    return copyto!(dest, res)
end

Base.@propagate_inbounds function StructArrays._getindex(
    x::StructArray{T}, I::Vararg{TracedRNumber{<:Integer}}
) where {T}
    cols = components(x)
    @boundscheck checkbounds(x, I...)
    return createinstance(T, get_ith(cols, I...)...)
end

Base.@propagate_inbounds function Base.setindex!(
    s::StructArray{T,<:Any,<:Any,Int}, vals, I::TracedRNumber{TI}
) where {T,TI<:Integer}
    valsT = maybe_convert_elt(T, vals)
    foreachfield((col, val) -> (@inbounds col[I] = val), s, valsT)
    return s
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(prev::Type{StructArray{ET, N, C, I}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {ET, N, C, I}
    ET_traced = traced_type_inner(ET, seen, mode, Union{ReactantPrimitive,track_numbers}, sharding, runtime)
    C_traced  = traced_type_inner(C, seen, mode, track_numbers, sharding, runtime)
    return StructArray{ET_traced,N,C_traced,index_type(fieldtypes(C_traced))}
end

function Reactant.make_tracer(
    seen,
    @nospecialize(prev::StructArray{NT,N}),
    @nospecialize(path),
    mode;
    track_numbers=false,
    sharding=Reactant.Sharding.Sharding.NoSharding(),
    runtime=nothing,
    kwargs...,
) where {NT<:NamedTuple,N}
    track_numbers isa Bool && (track_numbers = track_numbers ? Number : Union{})
    components = getfield(prev, :components)
    if mode == TracedToTypes
        push!(path, typeof(prev))
        for c in components
            make_tracer(seen, c, path, mode; track_numbers, sharding, runtime, kwargs...)
        end
        return nothing
    end
    traced_components = make_tracer(
        seen,
        components,
        append_path(path, 1),
        mode;
        track_numbers,
        sharding,
        runtime,
        kwargs...,
    )
    T_traced = traced_type(typeof(prev), Val(mode), track_numbers, sharding, runtime)
    return StructArray{first(T_traced.parameters)}(traced_components)
end

@inline function Reactant.traced_getfield(@nospecialize(obj::StructArray), field)
    return Base.getfield(obj, field)
end

end
