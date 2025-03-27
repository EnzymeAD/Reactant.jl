# Useful for testing indexing
struct ZeroBasedRange{T,A<:AbstractRange{T}} <: AbstractRange{T}
    a :: A
    function ZeroBasedRange(a::AbstractRange{T}) where {T}
        @assert !Base.has_offset_axes(a)
        new{T, typeof(a)}(a)
    end
end

struct ZeroBasedUnitRange{T,A<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
    a :: A
    function ZeroBasedUnitRange(a::AbstractUnitRange{T}) where {T}
        @assert !Base.has_offset_axes(a)
        new{T, typeof(a)}(a)
    end
end

for Z in [:ZeroBasedRange, :ZeroBasedUnitRange]
    @eval Base.parent(A::$Z) = A.a
    @eval Base.first(A::$Z) = first(A.a)
    @eval Base.length(A::$Z) = length(A.a)
    @eval Base.last(A::$Z) = last(A.a)
    @eval Base.size(A::$Z) = size(A.a)
    @eval Base.axes(A::$Z) = map(x -> IdentityUnitRange(0:x-1), size(A.a))
    @eval Base.getindex(A::$Z, i::Int) = A.a[i + 1]
    @eval Base.getindex(A::$Z, i::Integer) = A.a[i + 1]
    @eval Base.firstindex(A::$Z) = 0
    @eval Base.step(A::$Z) = step(A.a)
    @eval OffsetArrays.no_offset_view(A::$Z) = A.a
    @eval function Base.show(io::IO, A::$Z)
        show(io, A.a)
        print(io, " with indices $(axes(A,1))")
    end
end

for Z in [:ZeroBasedRange, :ZeroBasedUnitRange]
    for R in [:AbstractRange, :AbstractUnitRange, :StepRange]
        @eval @inline function Base.getindex(A::$Z, r::$R{<:Integer})
            @boundscheck checkbounds(A, r)
            OffsetArrays._indexedby(A.a[r .+ 1], axes(r))
        end
    end

    for R in [:ZeroBasedUnitRange, :ZeroBasedRange]
        @eval @inline function Base.getindex(A::$Z, r::$R{<:Integer})
            @boundscheck checkbounds(A, r)
            OffsetArrays._indexedby(A.a[r.a .+ 1], axes(r))
        end
    end

    for R in [:IIUR, :IdOffsetRange]
        @eval @inline function Base.getindex(A::$Z, r::$R)
            @boundscheck checkbounds(A, r)
            OffsetArrays._indexedby(A.a[r .+ 1], axes(r))
        end
    end

    for R in [:AbstractUnitRange, :IdOffsetRange, :IdentityUnitRange, :SliceIntUR, :StepRange, :StepRangeLen, :LinRange]
        @eval @inline function Base.getindex(A::$R, r::$Z)
            @boundscheck checkbounds(A, r)
            OffsetArrays._indexedby(A[r.a], axes(r))
        end
    end
    @eval @inline function Base.getindex(A::StepRangeLen{<:Any,<:Base.TwicePrecision,<:Base.TwicePrecision}, r::$Z)
        @boundscheck checkbounds(A, r)
        OffsetArrays._indexedby(A[r.a], axes(r))
    end

    @eval Base.reshape(z::$Z, inds::Tuple{}) = reshape(parent(z), inds)
    @eval Base.reshape(z::$Z, inds::Tuple{Int, Vararg{Int}}) = reshape(parent(z), inds)
    @eval Base.reshape(z::$Z, inds::Tuple{Union{Int, AbstractUnitRange{<:Integer}}, Vararg{Union{Int, AbstractUnitRange{<:Integer}}}}) = reshape(parent(z), inds)
end

# A basic range that does not have specialized vector indexing methods defined
# In this case the best that we may do is to return an OffsetArray
# Despite this, an indexing operation involving this type should preserve the axes of the indices
struct CustomRange{T,A<:AbstractRange{T}} <: AbstractRange{T}
    a :: A
end
Base.parent(r::CustomRange) = r.a
Base.size(r::CustomRange) = size(parent(r))
Base.length(r::CustomRange) = length(parent(r))
Base.axes(r::CustomRange) = axes(parent(r))
Base.first(r::CustomRange) = first(parent(r))
Base.last(r::CustomRange) = last(parent(r))
Base.step(r::CustomRange) = step(parent(r))
Base.getindex(r::CustomRange, i::Int) = getindex(parent(r), i)
