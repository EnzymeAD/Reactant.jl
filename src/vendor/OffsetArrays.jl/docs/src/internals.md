# For developers

Writing code that supports OffsetArrays is generally fairly straightforward.
The majority of cases can be handled with these tips:

- replace many uses of `size` with `axes`
- replace `1:length(A)` with `eachindex(A)`, or if you need an integer index with `LinearIndices(A)`
- replace explicit allocations like `Array{Int}(undef, size(B))` with `similar(Array{Int}, axes(B))`

More information can be found in [Julia's developer documentation](https://docs.julialang.org/en/v1/devdocs/offset-arrays/).
The most subtle issues tend to arise around the axes, and further detail specific to
OffsetArrays.jl follows below.

## Internals

How does OffsetArrays work? The fundamental principle is very simple:
an `OffsetArray` is just a wrapper around a "parent" array, together
with an index offset:

```jldoctest oa; setup=:(using OffsetArrays)
julia> oa = OffsetArray([1 2; 3 4], 0:1, 5:6)
2×2 OffsetArray(::Matrix{Int64}, 0:1, 5:6) with eltype Int64 with indices 0:1×5:6:
 1  2
 3  4

julia> parent(oa)
2×2 Matrix{Int64}:
 1  2
 3  4

julia> oa.offsets
(-1, 4)
```

So `parent(oa)` is the original array we constructed it with, and `oa.offsets` is a tuple,
each entry encoding the index-shift to be applied along the corresponding axis.
When you index `oa[i,j]`, it "translates" the `i,j` indexes back to the parent array's
indexes and then returns the value in the parent.

## The axes of OffsetArrays

The internal of offset computing is achieved by [`IdOffsetRange`](@ref OffsetArrays.IdOffsetRange)
type:

```jldoctest oa
julia> ax = axes(oa, 2)
OffsetArrays.IdOffsetRange(values=5:6, indices=5:6)
```

This has a similar design to `Base.IdentityUnitRange` that `ax[x] == x` always holds.

```jldoctest oa
julia> ax[5]
5
julia> ax[1]
ERROR: BoundsError: attempt to access 2-element OffsetArrays.IdOffsetRange{Int64, Base.OneTo{Int64}} with indices 5:6 at index [1]
[...]
```

This property makes sure that they tend to be their own axes:

```jldoctest oa
julia> axes(ax)
(OffsetArrays.IdOffsetRange(values=5:6, indices=5:6),)

julia> axes(ax[ax])
(OffsetArrays.IdOffsetRange(values=5:6, indices=5:6),)
```

This example of indexing is [idempotent](https://en.wikipedia.org/wiki/Idempotence).
This is a useful characteristic for ensuring the "fundamental axiom" of generalized indexing,
that `a[ax][i] == a[ax[i]]`:

```jldoctest; setup=:(using OffsetArrays)
julia> oa2 = OffsetArray([5, 10, 15, 20], 0:3)
4-element OffsetArray(::Vector{Int64}, 0:3) with eltype Int64 with indices 0:3:
  5
 10
 15
 20

julia> ax2 = axes(oa2, 1)
OffsetArrays.IdOffsetRange(values=0:3, indices=0:3)

julia> oa2[2]
15

julia> oa2[ax2][2]
15

julia> oa2[ax2[2]]
15
```

`IdOffsetRange`s apply the offset both to the values and the indices of the range, and otherwise preserve the parent range.

!!! warning

    There are circumstances where constructing a specific type of `IdOffsetRange` cannot be supported without changing the axes of the range (see [`OffsetArrays.IdOffsetRange`](@ref).)
    In the future, this package will distinguish between *construction*  and *conversion*:

    - construction (aka, *coercion*) will always succeed, even if it has to change the axes of the result (Examples: `RangeType(rng)`, `typeof(rng1)(rng2)`)
    - conversion will succeed only if it can preserve both the values and the axes (Examples: `convert(RangeType, rng)`, `oftype(rng1, rng2)`)

    While these behave equivalently now (conversion currently performs coercion), developers are encouraged to "future-proof" their code by choosing the behavior appropriate for each usage.


## Wrapping other offset array types

An `OffsetArray` may wrap any subtype of `AbstractArray`, including ones that do not use `1`-based indexing. Such arrays however need to satisfy the fundamental axiom of idempotent indexing for things to work correctly. In other words, an axis of an offset array needs to have the same values as its own axis. This property is built into `OffsetArray`s if the parent uses 1-based indexing, but it's up to the user to ensure the correctness in case a type is to be wrapped that uses offset indices.

We demonstrate this through an example by creating a custom 0-based range type that we wrap in an `OffsetArray`:

```jldoctest zerobasedrange; setup=:(using OffsetArrays)
julia> struct ZeroBasedRange{T,A<:AbstractRange{T}} <: AbstractRange{T}
           a :: A
           function ZeroBasedRange(a::AbstractRange{T}) where {T}
               @assert !Base.has_offset_axes(a)
               new{T, typeof(a)}(a)
           end
       end;

julia> Base.parent(A::ZeroBasedRange) = A.a;

julia> Base.first(A::ZeroBasedRange) = first(A.a);

julia> Base.length(A::ZeroBasedRange) = length(A.a);

julia> Base.last(A::ZeroBasedRange) = last(A.a);

julia> Base.size(A::ZeroBasedRange) = size(A.a);

julia> Base.axes(A::ZeroBasedRange) = map(x -> 0:x-1, size(A.a));

julia> Base.getindex(A::ZeroBasedRange, i::Int) = A.a[i + 1];

julia> Base.step(A::ZeroBasedRange) = step(A.a);

julia> function Base.show(io::IO, A::ZeroBasedRange)
           show(io, A.a)
           print(io, " with indices $(axes(A,1))")
       end;
```

This definition of a `ZeroBasedRange` appears to have the correct indices, for example:

```jldoctest zerobasedrange
julia> z = ZeroBasedRange(1:4)
1:4 with indices 0:3

julia> z[0]
1

julia> z[3]
4
```

However this does not use idempotent indexing, as the axis of a `ZeroBasedRange` is not its own axis.

```jldoctest zerobasedrange
julia> axes(z, 1)
0:3

julia> axes(axes(z, 1), 1)
Base.OneTo(4)
```

This will lead to complications in certain functions --- for example `LinearIndices` --- that tends to implicitly assume idempotent indexing. In this case the `LinearIndices` of `z` will not match its axis.

```jldoctest zerobasedrange
julia> LinearIndices(z)
4-element LinearIndices{1, Tuple{UnitRange{Int64}}}:
 1
 2
 3
 4
```

Wrapping such a type in an `OffsetArray` might lead to unexpected bugs.

```jldoctest zerobasedrange
julia> zo = OffsetArray(z, 1);

julia> axes(zo, 1)
OffsetArrays.IdOffsetRange(values=1:4, indices=2:5)

julia> Array(zo)
ERROR: BoundsError: attempt to access 4-element UnitRange{Int64} at index [5]
[...]
```

The `Array` conversion errors despite `zo` having 1-based indices. The function `axes(zo, 1)` hints at the underlying problem --- the values and the indices of the axis are different. We may check that the axis of `zo` is not its own axis:

```jldoctest zerobasedrange
julia> axes(zo, 1)
OffsetArrays.IdOffsetRange(values=1:4, indices=2:5)

julia> axes(axes(zo, 1), 1)
OffsetArrays.IdOffsetRange(values=2:5, indices=2:5)
```

In this case the bug may be fixed by defining the `axes` of a `ZeroBasedRange` to be idempotent, for example using the `OffsetArrays.IdentityUnitRange` wrapper:

```jldoctest zerobasedrange
julia> Base.axes(A::ZeroBasedRange) = map(x -> OffsetArrays.IdentityUnitRange(0:x-1), size(A.a))

julia> axes(zo, 1)
OffsetArrays.IdOffsetRange(values=1:4, indices=1:4)
```

With this new definition, the values and indices of the axis are identical, which makes indexing idempotent. The conversion to an `Array` works as expected now:

```jldoctest zerobasedrange
julia> Array(zo)
4-element Vector{Int64}:
 1
 2
 3
 4
```

## Caveats

Because `IdOffsetRange` behaves quite differently to the normal `UnitRange` type, there are some
cases that you should be aware of, especially when you are working with multi-dimensional arrays.

One such cases is `getindex`:

```jldoctest getindex; setup = :(using OffsetArrays)
julia> Ao = zeros(-3:3, -3:3); Ao[:] .= 1:49;

julia> Ao[-3:0, :] |> axes # the first dimension does not preserve offsets
(OffsetArrays.IdOffsetRange(values=1:4, indices=1:4), OffsetArrays.IdOffsetRange(values=-3:3, indices=-3:3))

julia> Ao[-3:0, -3:3] |> axes # neither dimensions preserve offsets
(Base.OneTo(4), Base.OneTo(7))

julia> Ao[axes(Ao)...] |> axes # offsets are preserved
(OffsetArrays.IdOffsetRange(values=-3:3, indices=-3:3), OffsetArrays.IdOffsetRange(values=-3:3, indices=-3:3))

julia> Ao[:] |> axes # This is linear indexing
(Base.OneTo(49),)
```

Note that if you pass a `UnitRange`, the offsets in corresponding dimension will not be preserved.
This might look weird at first, but since it follows the `a[ax][i] == a[ax[i]]` rule, it is not a
bug.

```jldoctest getindex
julia> I = -3:0; # UnitRange always starts at index 1

julia> Ao[I, 0][1] == Ao[I[1], 0]
true

julia> ax = axes(Ao, 1) # ax starts at index -3
OffsetArrays.IdOffsetRange(values=-3:3, indices=-3:3)

julia> Ao[ax, 0][1] == Ao[ax[1], 0]
true
```

## Using custom axis types

While a wide variety of `AbstractUnitRange`s provided by `Base` may be used as indices to construct an `OffsetArray`, at times it might be convenient to define custom types. The `OffsetArray` constructor accepts any type that may be converted to an `AbstractUnitRange`. This proceeds through a two-step process. Let's assume that the constructor called is `OffsetArray(A, indstup)`, where `indstup` is a `Tuple` of indices.

1. At the first step, the constructor calls `to_indices(A, axes(A), indstup)` to lower `indstup` to a `Tuple` of `AbstractUnitRange`s. This step converts --- among other things --- `Colon`s to axis ranges. Custom types may extend `Base.to_indices(A, axes(A), indstup)` with the desired conversion of `indstup` to `Tuple{Vararg{AbstractUnitRange{Int}}}` if this is feasible.

2. At the second step, the result obtained from the previous step treated again to convert it to a `Tuple` of `AbstractUnitRange`s to handle cases where the first step doesn't achieve this. An additional customization option may be specified at this stage: a type may be converted either to a single `AbstractUnitRange{Int}`, or to a `Tuple` of them. A type might specify which of these two behaviours is desired by extending [`OffsetArrays.AxisConversionStyle`](@ref). An example of a type that is acted upon at this stage is `CartesianIndices`, which is converted to a `Tuple` of `AbstractUnitRange`s.

For example, here are a couple of custom type that facilitate zero-based indexing:

```jldoctest; setup = :(using OffsetArrays)
julia> struct ZeroBasedIndexing end

julia> Base.to_indices(A, inds, ::Tuple{ZeroBasedIndexing}) = map(x -> 0:length(x)-1, inds)

julia> a = zeros(3, 3);

julia> oa = OffsetArray(a, ZeroBasedIndexing());

julia> axes(oa)
(OffsetArrays.IdOffsetRange(values=0:2, indices=0:2), OffsetArrays.IdOffsetRange(values=0:2, indices=0:2))
```

In this example we had to define the action of `to_indices` as the type `ZeroBasedIndexing` did not have a familiar hierarchy. Things are even simpler if we subtype `AbstractUnitRange`, in which case we need to define `first` and `length` for the custom range to be able to use it as an axis:

```jldoctest; setup = :(using OffsetArrays)
julia> struct ZeroTo <: AbstractUnitRange{Int}
       n :: Int
       ZeroTo(n) = new(n < 0 ? -1 : n)
       end

julia> Base.first(::ZeroTo) = 0

julia> Base.length(r::ZeroTo) = r.n + 1

julia> oa = OffsetArray(zeros(2,2), ZeroTo(1), ZeroTo(1));

julia> axes(oa)
(OffsetArrays.IdOffsetRange(values=0:1, indices=0:1), OffsetArrays.IdOffsetRange(values=0:1, indices=0:1))
```

Note that zero-based indexing may also be achieved using the pre-defined type [`OffsetArrays.Origin`](@ref).
