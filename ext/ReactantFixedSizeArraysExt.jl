module ReactantFixedSizeArraysExt

using FixedSizeArrays: FixedSizeArrays, FixedSizeArray, new_fixed_size_array
using Reactant: Reactant, AnyTracedRVector, TracedRArray, TracedRNumber, TracedUtils
using ReactantCore: ReactantCore

function Reactant.traced_type_inner(
    @nospecialize(_::Type{FixedSizeArray{T,N,M}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N,M}
    return FixedSizeArray{
        Reactant.TracedRNumber{T},
        N,
        Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime),
    }
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen, @nospecialize(prev::FixedSizeArray{T,N,M}), @nospecialize(path), mode; kwargs...
) where {T,N,M}
    return new_fixed_size_array(
        Reactant.make_tracer(
            seen, parent(prev), (path..., 1), mode; kwargs..., track_numbers=Number
        ),
        size(prev),
    )
end

function FixedSizeArrays.with_stripped_type_parameters_unchecked(
    ::FixedSizeArrays.TypeParametersElementType, ::Type{TracedRArray{T,1}}
) where {T}
    return Val{TracedRArray{E,1} where {E}}()
end

function Base.similar(
    x::T, ::Type{E}, size::NTuple{N,Int}
) where {T<:FixedSizeArray{<:Any,<:Any,<:TracedRArray},N,E}
    return new_fixed_size_array(
        similar(parent(x), Reactant.unwrapped_eltype(E), prod(size)), size
    )
end

function Base.similar(
    bc::Broadcast.Broadcasted{FixedSizeArrays.FixedSizeArrayBroadcastStyle{N,Mem}},
    ::Type{T},
) where {N,T,Mem<:TracedRArray{<:Any,1}}
    return similar(
        FixedSizeArray{T,N,TracedRArray{Reactant.unwrapped_eltype(T),1}}, axes(bc)
    )
end

# function Broadcast.BroadcastStyle(
#     ::Type{<:FixedSizeArray{T,N,<:AnyTracedRVector}}
# ) where {T,N}
#     return Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{N}()
# end
# function Broadcast.BroadcastStyle(
#     ::Type{<:SubArray{T,N,FixedSizeArray{T,N,<:AnyTracedRVector}}}
# ) where {T,N}
#     return Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{N}()
# end

function TracedUtils.materialize_traced_array(x::FixedSizeArray)
    return TracedUtils.materialize_traced_array(reshape(parent(x), size(x)))
end

function TracedUtils.set_mlir_data!(
    x::FixedSizeArray{T,N,<:AnyTracedRVector}, data
) where {T,N}
    TracedUtils.set_mlir_data!(
        parent(x), TracedUtils.get_mlir_data(vec(TracedRArray(data)))
    )
    return x
end

end
