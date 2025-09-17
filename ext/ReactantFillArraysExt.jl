module ReactantFillArraysExt

using Reactant: Reactant, TracedUtils, TracedRNumber, Ops, Sharding, unwrapped_eltype
using ReactantCore: ReactantCore
using FillArrays: FillArrays, AbstractFill, Fill, Ones, Zeros, OneElement
using GPUArraysCore: @allowscalar

# Tracing
Reactant._parent_type(T::Type{<:AbstractFill}) = T
Reactant._parent_type(T::Type{<:OneElement}) = T

for AT in (Fill, Ones, Zeros)
    @eval Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(FA::Type{$(AT){T,N,Axes}}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(sharding),
        @nospecialize(runtime)
    ) where {T,N,Axes}
        # T will be a number so we need to trace it
        return $(AT){
            Reactant.traced_type_inner(T, seen, mode, Number, sharding, runtime),N,Axes
        }
    end
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen, @nospecialize(prev::Fill{T,N,Axes}), @nospecialize(path), mode; kwargs...
) where {T,N,Axes}
    return Fill(
        Reactant.make_tracer(
            seen, prev.value, (path..., 1), mode; kwargs..., track_numbers=Number
        ),
        prev.axes,
    )
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::Ones{T,N,Axes}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    kwargs...,
) where {T,N,Axes}
    return Ones(
        Reactant.traced_type_inner(T, seen, mode, Number, sharding, runtime), prev.axes
    )
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::Zeros{T,N,Axes}),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    @nospecialize(runtime = nothing),
    kwargs...,
) where {T,N,Axes}
    return Zeros(
        Reactant.traced_type_inner(T, seen, mode, Number, sharding, runtime), prev.axes
    )
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(FA::Type{OneElement{T,N,I,A}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N,I,A}
    # T will be a number so we need to trace it
    return OneElement{
        Reactant.traced_type_inner(T, seen, mode, Number, sharding, runtime),N,I,A
    }
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen, @nospecialize(prev::OneElement{T,N,I,A}), @nospecialize(path), mode; kwargs...
) where {T,N,I,A}
    return OneElement(
        Reactant.make_tracer(
            seen, prev.val, (path..., 1), mode; kwargs..., track_numbers=Number
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

# some functions to avoid bad performance
for AT in (Fill, Ones, Zeros, OneElement)
    @eval function Base.similar(x::$AT{<:TracedRNumber}, ::Type{T}, dims::Dims) where {T}
        return Reactant.broadcast_to_size(unwrapped_eltype(T)(0), dims)
    end
end

end
