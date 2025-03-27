"""
    Origin(indices...)
    Origin(origin::Tuple)
    Origin(origin::CartesianIndex)

A helper type to construct OffsetArray with a given origin. This is not exported.

The `origin` of an array is defined as the tuple of the first index along each axis, i.e., `first.(axes(A))`.

# Example

```jldoctest origin; setup=:(using OffsetArrays)
julia> a = [1 2; 3 4];

julia> using OffsetArrays: Origin

julia> OffsetArray(a, Origin(0, 1))
2×2 OffsetArray(::$(Array{Int,2}), 0:1, 1:2) with eltype $Int with indices 0:1×1:2:
 1  2
 3  4

julia> OffsetArray(a, Origin(0)) # short notation for `Origin(0, 0)`
2×2 OffsetArray(::$(Array{Int, 2}), 0:1, 0:1) with eltype $Int with indices 0:1×0:1:
 1  2
 3  4
```

An `Origin` object is callable, and it may shift the origin of an array to the specified point.

```jldoctest origin
julia> b = Origin(0)(a) # shift the origin of the array to (0,0)
2×2 OffsetArray(::$(Array{Int, 2}), 0:1, 0:1) with eltype $Int with indices 0:1×0:1:
 1  2
 3  4
```

The type `Origin`, when called with an `AbstractArray` as the argument, will return an instance
corresponding ot the origin of the array.

```jldoctest origin
julia> origin_b = Origin(b) # retrieve the origin of the array as an Origin instance
Origin(0, 0)

julia> origin_b(ones(2,2)) # shift the origin of another array to that of b, in this case to (0,0)
2×2 OffsetArray(::$(Array{Float64, 2}), 0:1, 0:1) with eltype Float64 with indices 0:1×0:1:
 1.0  1.0
 1.0  1.0
```

!!! tip
    One may broadcast an `Origin` instance over multiple arrays to shift them all to the same origin.
    ```jldoctest
    julia> using OffsetArrays: Origin

    julia> a = [1 2; 3 4]; # origin at (1,1)

    julia> b = Origin(2,3)(a); # origin at (2,3)

    julia> c = Origin(4)(a); # origin at (4,4)

    julia> ao, bo, co = Origin(0).((a, b, c)); # shift all origins to (0,0)

    julia> first.(axes(ao)) == first.(axes(bo)) == first.(axes(co)) == (0,0)
    true

    julia> ao, bo, co = Origin(b).((a, b, c)); # shift all origins to that of b

    julia> first.(axes(ao)) == first.(axes(bo)) == first.(axes(co)) == (2,3)
    true

    julia> ao, bo, co = OffsetArray.((a, b, c), Origin(b)); # another way to do the same

    julia> first.(axes(ao)) == first.(axes(bo)) == first.(axes(co)) == (2,3)
    true
    ```
"""
struct Origin{T<:Union{Tuple{Vararg{Int}}, Int}}
    index::T
end
Origin(I::Tuple{Vararg{Int}}) = Origin{typeof(I)}(I)
Origin(I::Tuple{Vararg{Number}}) = Origin(map(Int, I))
Origin(I::CartesianIndex) = Origin(Tuple(I))
Origin(I::Number...) = Origin(I)
# Origin(0) != Origin((0, )) but they work the same with broadcasting
Origin(n::Number) = Origin{Int}(Int(n))

Base.Broadcast.broadcastable(o::Origin) = Ref(o)

_showidx(index::Integer) = "(" * string(index) * ")"
_showidx(index::Tuple) = string(index)
Base.show(io::IO, o::Origin) = print(io, "Origin", _showidx(o.index))
