module ReactantStructArraysExt

import Reactant
import StructArrays

import StructArrays: StructArrayStyle, StructArray, StructVector, index_type
import Reactant: TraceMode, TracedToTypes, traced_type_inner, append_path, make_tracer, traced_type, ReactantPrimitive
import Reactant.TracedRArrayOverrides: AbstractReactantArrayStyle, _copy
import Base.Broadcast: Broadcasted

StructArrays.always_struct_broadcast(::AbstractReactantArrayStyle) = true

function Base.copy(bc::Broadcasted{StructArrays.StructArrayStyle{S, N}}) where {S<:AbstractReactantArrayStyle, N}
    return _copy(bc)
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(prev::Type{<:StructArray{NT}}),
    seen,
    @nospecialize(mode::TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {NT <: NamedTuple}
    T, N, C, I = prev.parameters
    C_traced = traced_type_inner(
        C,
        seen,
        mode,
        track_numbers,
        sharding,
        runtime,
    )

    names = T.parameters[1]
    valuetypes = T.parameters[2].parameters
    traced_value_types = map(valuetypes) do VT
        # The elements in the NamedTuple are backed by vectors,
        # these vectors are converted to RArrays so we need to track numbers:
        track_numbers = VT <: ReactantPrimitive ? ReactantPrimitive : track_numbers
        traced_type_inner(
            VT,
            seen,
            mode,
            track_numbers,
            sharding,
            runtime,
        )
    end
    T_traced = NamedTuple{names, Tuple{traced_value_types...}}

    return StructVector{T_traced, C_traced, index_type(fieldtypes(C_traced))}
end

function Reactant.make_tracer(
    seen, @nospecialize(prev::StructArray{NT, N}), @nospecialize(path), mode; track_numbers=false, sharding=Reactant.Sharding.Sharding.NoSharding(), runtime=nothing, kwargs...
) where {NT <: NamedTuple, N}
    track_numbers isa Bool && (track_numbers = track_numbers ? Number : Union{})
    components = getfield(prev, :components)
    if mode == TracedToTypes
        push!(path, typeof(prev))
        for c in components
            make_tracer(seen, c, path, mode; track_numbers, sharding, runtime, kwargs...)
        end
        return nothing
    end
    traced_components = make_tracer(seen, components, append_path(path, :components), mode; track_numbers, sharding, runtime, kwargs...)
    T_traced = traced_type(
        typeof(prev),
        Val(mode),
        track_numbers,
        sharding,
        runtime,
    )
    return StructArray{first(T_traced.parameters)}(traced_components)
end

@inline function Reactant.traced_getfield(@nospecialize(obj::StructArray), field)
    return Base.getfield(obj, field)
end

end