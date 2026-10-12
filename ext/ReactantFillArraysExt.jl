module ReactantFillArraysExt

using Reactant: Reactant, TracedUtils, TracedRNumber, Ops, Sharding, unwrapped_eltype
using ReactantCore: ReactantCore
using FillArrays: FillArrays, AbstractFill, Fill, Ones, Zeros, OneElement
using GPUArraysCore: @allowscalar

# Tracing
Reactant._parent_type(T::Type{<:AbstractFill}) = T
Reactant._parent_type(T::Type{<:OneElement}) = T

# The scalar stored inside a `Fill`/`Ones`/`Zeros`/`OneElement` plays the role of the array
# elements, so when converting to concrete arrays (`Reactant.to_rarray`) it must always be
# promoted, exactly like the elements of an `Array{<:ReactantPrimitive}`. In every other
# mode we must respect the caller's `track_numbers`: a plain `Fill{Float64}` passed to a
# compiled function is a compile-time constant (just like a plain `Array{Float64}`), and
# unconditionally tracking the number would mint a fresh `TracedRNumber` for it in
# `TracedTrack`/`NoStopTracedTrack` mode. That value then becomes a spurious output of the
# compiled function which the thunk tries to `setfield!` back into the immutable struct.
@inline function _fill_track_numbers(mode::Reactant.TraceMode, @nospecialize(track_numbers))
    mode == Reactant.ArrayToConcrete && return Number
    return track_numbers
end

for AT in (Fill, Ones, Zeros)
    @eval Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(FA::Type{$(AT){T,N,Axes}}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    ) where {T,N,Axes}
        return $(AT){
            Reactant.traced_type_inner(
                T, seen, mode, _fill_track_numbers(mode, track_numbers), ndevices, runtime
            ),
            N,
            Axes,
        }
    end
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::Fill{T,N,Axes}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    kwargs...,
) where {T,N,Axes}
    return Fill(
        Reactant.make_tracer(
            seen,
            prev.value,
            (path..., 1),
            mode;
            kwargs...,
            track_numbers=_fill_track_numbers(mode, track_numbers),
        ),
        prev.axes,
    )
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::Ones{T,N,Axes}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    kwargs...,
) where {T,N,Axes}
    return Ones(
        Reactant.traced_type(
            T, Val(mode), _fill_track_numbers(mode, track_numbers), sharding, runtime
        ),
        prev.axes,
    )
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::Zeros{T,N,Axes}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    kwargs...,
) where {T,N,Axes}
    return Zeros(
        Reactant.traced_type(
            T, Val(mode), _fill_track_numbers(mode, track_numbers), sharding, runtime
        ),
        prev.axes,
    )
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(FA::Type{OneElement{T,N,I,A}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {T,N,I,A}
    return OneElement{
        Reactant.traced_type_inner(
            T, seen, mode, _fill_track_numbers(mode, track_numbers), ndevices, runtime
        ),
        N,
        I,
        A,
    }
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::OneElement{T,N,I,A}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    kwargs...,
) where {T,N,I,A}
    return OneElement(
        Reactant.make_tracer(
            seen,
            prev.val,
            (path..., 1),
            mode;
            kwargs...,
            track_numbers=_fill_track_numbers(mode, track_numbers),
        ),
        prev.ind,
        prev.axes,
    )
end

# Materialize into a dense array
function ReactantCore.materialize_traced_array(x::Fill{T}) where {T}
    return Reactant.broadcast_to_size(
        Reactant.promote_to(TracedRNumber{unwrapped_eltype(T)}, x.value), size(x)
    )
end

function ReactantCore.materialize_traced_array(x::Ones{T}) where {T}
    return Reactant.broadcast_to_size(unwrapped_eltype(T)(1), size(x))
end

function ReactantCore.materialize_traced_array(x::Zeros{T}) where {T}
    return Reactant.broadcast_to_size(unwrapped_eltype(T)(0), size(x))
end

function ReactantCore.materialize_traced_array(x::OneElement{T}) where {T}
    y = Reactant.broadcast_to_size(unwrapped_eltype(T)(0), size(x))
    @allowscalar setindex!(y, x.val, x.ind...)
    return y
end

for AT in (Fill, Ones, Zeros, OneElement)
    # overloads to prevent ambiguity
    @eval function Base.getindex(
        x::$AT{<:TracedRNumber}, idxs::Vararg{Union{Int64,Reactant.TracedRNumber{Int64}},N}
    ) where {N}
        @boundscheck checkbounds(x, idxs...)
        return FillArrays.getindex_value(x)
    end
    @eval function Base.getindex(x::$AT{<:TracedRNumber}, idxs...)
        @boundscheck checkbounds(x, idxs...)
        shape = Base.index_shape(idxs...)
        return FillArrays.fillsimilar(x, shape)
    end

    # for performance improvements
    @eval function Base.similar(x::$AT{<:TracedRNumber}, ::Type{T}, dims::Dims) where {T}
        return Reactant.broadcast_to_size(unwrapped_eltype(T)(0), dims)
    end
end

# for performance improvements
function Base.Broadcast.broadcasted(
    ::Base.Broadcast.DefaultArrayStyle,
    ::typeof(Reactant.to_number),
    r::FillArrays.AbstractZeros{<:Reactant.AbstractConcreteNumber},
)
    return Zeros{typeof(Reactant.to_number(FillArrays.getindex_value(r)))}(r.axes)
end

end
