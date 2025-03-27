module TracedROffsetArrayOverrides


using ReactantCore: ReactantCore

ReactantCore.is_traced(::TracedROffsetArray, seen) = true
ReactantCore.is_traced(::TracedOffsetRArray) = true


#------------------------------------------------------------------------------
# Vendored from: https://github.com/JuliaArrays/OffsetArrays.jl/tree/master
#------------------------------------------------------------------------------

# ensure that the indices are consistent in the constructor
_checkindices(A::AbstractArray, indices, label) = _checkindices(ndims(A), indices, label)
function _checkindices(N::Integer, indices, label)
    throw_argumenterror(N, indices, label) = throw(
        ArgumentError(label*" $indices are not compatible with a $(N)D array")
    )
    N == length(indices) || throw_argumenterror(N, indices, label)
end


struct TracedROffsetArray{T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::AA
    offsets::NTuple{N,Int}
    @inline function TracedROffsetArray{T, N, AA}(
            parent::AA, offsets::NTuple{N, Int};
            checkoverflow = true
        ) where {T, N, AA<:AbstractArray{T,N}}

        # allocation of `map` on tuple is optimized away
        checkoverflow && map(overflow_check, axes(parent), offsets)
        new{T, N, AA}(parent, offsets)
    end
end

const TracedROffsetVector{T,AA<:AbstractVector{T}} = TracedROffsetArray{T,1,AA}
const TracedROffsetMatrix{T,AA<:AbstractMatrix{T}} = TracedROffsetArray{T,2,AA}


function overflow_check(r::AbstractUnitRange, offset::Integer)
    Base.hastypemax(eltype(r)) || return nothing
    # This gives some performance boost https://github.com/JuliaLang/julia/issues/33273
    throw_upper_overflow_error(val) = throw(
        OverflowError(
            "offset should be <= $(typemax(Int) - val) corresponding to the axis $r, received an offset $offset"
        )
    )
    throw_lower_overflow_error(val) = throw(
        OverflowError(
            "offset should be >= $(typemin(Int) - val) corresponding to the axis $r, received an offset $offset"
        )
    )

    # With ranges in the picture, first(r) might not necessarily be < last(r)
    # we therefore use the min and max of first(r) and last(r) to check for overflow
    firstlast_min, firstlast_max = minmax(first(r), last(r))

    if offset > 0 && firstlast_max > typemax(Int) - offset
        throw_upper_overflow_error(firstlast_max)
    elseif offset < 0 && firstlast_min < typemin(Int) - offset
        throw_lower_overflow_error(firstlast_min)
    end

    return nothing
end


# Tuples of integers are treated as offsets. Empty Tuples are handled here
@inline function TracedROffsetArray(
        A::AbstractArray, offsets::Tuple{Vararg{Integer}};
        kw...
    )
    _checkindices(A, offsets, "offsets")
    TracedROffsetArray{eltype(A), ndims(A), typeof(A)}(A, offsets; kw...)
end

# These methods are necessary to disallow incompatible dimensions for the
# OffsetVector and the OffsetMatrix constructors
for (FT, ND) in ((:TracedROffsetVector, :1), (:TracedROffsetMatrix, :2))
    @eval @inline function $FT(
            A::AbstractArray{<:Any,$ND}, offsets::Tuple{Vararg{Integer}};
            kw...
        )
        _checkindices(A, offsets, "offsets")
        TracedROffsetArray{eltype(A), $ND, typeof(A)}(A, offsets; kw...)
    end
    FTstr = string(FT)
    @eval @inline function $FT(
            A::AbstractArray, offsets::Tuple{Vararg{Integer}};
            kw...
        )
        throw(ArgumentError($FTstr*" requires a "*string($ND)*"D array"))
    end
end


end
