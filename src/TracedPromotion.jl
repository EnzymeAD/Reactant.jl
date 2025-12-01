# convert julia types to traced types
## Promote to a Traced Type
promote_to(::TracedRArray{T,N}, rhs) where {T,N} = promote_to(TracedRArray{T,N}, rhs)
promote_to(::TracedRNumber{T}, rhs) where {T} = promote_to(TracedRNumber{T}, rhs)

## Array types
function promote_to(::Type{TracedRArray}, rhs)
    return promote_to(TracedRArray{unwrapped_eltype(rhs),ndims(rhs)}, rhs)
end
function promote_to(::Type{TracedRArray{T}}, rhs) where {T}
    return promote_to(TracedRArray{T,ndims(rhs)}, rhs)
end

promote_to(::Type{TracedRArray{T,N}}, rhs::TracedRArray{T,N}) where {T,N} = rhs
function promote_to(::Type{TracedRArray{T,N}}, rhs::TracedRArray{T2,N}) where {T,T2,N}
    return @opcall convert(TracedRArray{T,N}, rhs)
end

function promote_to(
    ::Type{TracedRArray{T,N}}, rhs::AbstractArray{<:TracedRNumber,N}
) where {T,N}
    return @opcall convert(TracedRArray{T,N}, aos_to_soa(materialize_traced_array(rhs)))
end

function promote_to(
    ::Type{TracedRArray{T,1}},
    rhs::Union{UnitRange,UnitRange{<:TracedRNumber},<:TracedUnitRange},
) where {T}
    return @opcall add(
        @opcall(iota(eltype(rhs), [length(rhs)]; iota_dimension=1)),
        @opcall(fill(first(rhs), [length(rhs)])),
    )
end

function promote_to(
    ::Type{TracedRArray{T,1}},
    rhs::Union{
        StepRange,
        StepRangeLen,
        StepRange{<:TracedRNumber},
        StepRangeLen{<:TracedRNumber},
        TracedStepRangeLen,
    },
) where {T}
    step_arr = broadcast_to_size(step(rhs), (length(rhs),))
    iota = @opcall iota(unwrapped_eltype(rhs), [length(rhs)]; iota_dimension=1)
    first_arr = broadcast_to_size(first(rhs), (length(rhs),))
    return @opcall add(@opcall(multiply(step_arr, iota)), first_arr)
end

function promote_to(::Type{TracedRArray{T,1}}, rhs::Base.OneTo) where {T}
    return promote_to(TracedRArray{T,1}, first(rhs):last(rhs))
end

function promote_to(::Type{TracedRArray{T,N}}, rhs::LinearAlgebra.Diagonal) where {T,N}
    return LinearAlgebra.diagm(promote_to(TracedRArray{T,1}, rhs.diag))
end

function promote_to(::Type{TracedRArray{T,N}}, rhs::AbstractArray{<:Any,N}) where {T,N}
    if ancestor(rhs) isa AnyTracedRArray
        return promote_to(TracedRArray{T,N}, materialize_traced_array(rhs))
    end
    ## fallback option, almost certainly will emit a really bad IR
    return promote_to(TracedRArray{T,N}, @opcall(constant(rhs)))
end

## Number types
function promote_to(::Type{TracedRNumber}, rhs)
    T = rhs isa AbstractIrrational ? Float64 : unwrapped_eltype(rhs)
    return promote_to(TracedRNumber{T}, rhs)
end

promote_to(::Type{TracedRNumber{T}}, rhs::TracedRNumber{T}) where {T} = rhs
function promote_to(::Type{TracedRNumber{T}}, rhs::TracedRNumber{T2}) where {T,T2}
    return @opcall convert(TracedRNumber{T}, rhs)
end

function promote_to(::Type{TracedRArray{T,0}}, rhs::TracedRNumber{T2}) where {T,T2}
    return TracedRArray{T,0}((), @opcall(convert(TracedRNumber{T}, rhs)).mlir_data, ())
end
function promote_to(::Type{TracedRNumber{T}}, rhs::TracedRArray{T2,0}) where {T,T2}
    return TracedRNumber{T}((), @opcall(convert(TracedRArray{T,0}, rhs)).mlir_data)
end

function promote_to(::Type{TracedRNumber{T}}, rhs::Number) where {T}
    res = @opcall(fill(rhs))
    return @opcall convert(
        TracedRNumber{T}, TracedRNumber{unwrapped_eltype(res)}((), res.mlir_data)
    )
end

function ReactantCore.promote_to_traced(x)
    return promote_to(TracedRNumber{unwrapped_eltype(typeof(x))}, x)
end

## Promote to a Traced Type and broadcast to a given size
function broadcast_to_size(arg::AbstractArray, rsize)
    return broadcast_to_size(promote_to(TracedRArray, arg), rsize)
end

function broadcast_to_size(arg::Number, rsize)
    return broadcast_to_size(promote_to(TracedRNumber, arg), rsize)
end

broadcast_to_size(arg::TracedRNumber, ::Tuple{}) = arg
broadcast_to_size(arg::TracedRNumber, rsize) = @opcall fill(arg, collect(Int64, rsize))

function broadcast_to_size(arg::TracedRArray, rsize)
    return @opcall broadcast_in_dim(
        arg, collect(Int64, 1:ndims(arg)), collect(Int64, rsize)
    )
end

function broadcast_to_size(arg::Broadcast.Extruded, rsize)
    rsize2 = (keep ? rsizev : 1 for (keep, rsizev) in zip(arg.keeps, rsize))
    return broadcast_to_size(broadcast_to_size(arg.x, rsize2), rsize)
end

broadcast_to_size(arg::Base.RefValue, rsize) = arg
