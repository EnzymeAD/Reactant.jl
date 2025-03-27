# OffsetArrays.jl

OffsetArrays provides Julia users with arrays that have arbitrary
indices, similar to those found in some other programming languages
like Fortran. Below is the basic usage found in the README, followed
by a couple of short examples illustrating circumstances in which
OffsetArrays can be useful. For a lengthier discussion, see
[this blog post](https://julialang.org/blog/2017/04/offset-arrays/).

## Usage

You can construct such arrays as follows:

```julia
OA = OffsetArray(A, axis1, axis2, ...)
```

where you want `OA` to have axes `(axis1, axis2, ...)` and be indexed by values that
fall within these axis ranges.

```@repl index
using OffsetArrays

A = Float64.(reshape(1:15, 3, 5))

OA = OffsetArray(A, -1:1, 0:4) # OA will have axes (-1:1, 0:4)

OA = OffsetArray(A, CartesianIndex(-1, 0):CartesianIndex(1, 4))

OA[-1,0], OA[1,4]
```

You could also pass integers as offsets, where `0` means no offsets are applied:

```@repl index
OA = OffsetArray(A, -2, -1)
```

When you create a new `OffsetArray` on the top of another `OffsetArray`, the offsets are
accumulated:

```@repl index
OOA = OffsetArray(OA, 2, 1)
```

For the special cases that you want to compensate the offset back to the ordinary 1-based array, you
can use [`OffsetArrays.no_offset_view(A)`](@ref). Furthermore, you could use
`Base.require_one_based_indexing` if you want to ensure the array does not have offsets.

```@repl index
OffsetArrays.no_offset_view(OA)

Base.require_one_based_indexing(ans)

Base.require_one_based_indexing(OA)
```

[`OffsetArrays.Origin`](@ref) can be convenient if you want to directly specify the origin of the output
OffsetArray, it will automatically compute the corresponding offsets. For example:

```@repl index
OffsetArray(A, OffsetArrays.Origin(-1, -1))
OffsetArray(OA, OffsetArrays.Origin(-1, -1))
```

An equivalent — but possibly more convenient — way to specify the origin of an array is

```@repl index
OffsetArrays.Origin(-1, -1)(A)
```

Sometimes, it will be convenient to shift the center coordinate of the given array to `(0, 0, ...)`,
`OffsetArrays.centered` is a helper for this very purpose:

```@repl index
Ao = OffsetArrays.centered(A)
Ao[0, 0] == 8.0
```

and `OffsetArrays.center` tells you the center coordinate of given array:

```@repl index
c = OffsetArrays.center(A)
A[c...] == 8.0
```

## Example: Relativistic Notation

Suppose we have a position vector `r = [:x, :y, :z]` which is naturally one-based, ie. `r[1] == :x`, `r[2] == :y`,  `r[3] == :z` and we also want to construct a relativistic position vector which includes time as the 0th component. This can be done with OffsetArrays like

```jldoctest; setup = :(using OffsetArrays)
julia> r = [:x, :y, :z];

julia> x = OffsetVector([:t, r...], 0:3)
4-element OffsetArray(::Vector{Symbol}, 0:3) with eltype Symbol with indices 0:3:
 :t
 :x
 :y
 :z

julia> x[0]
:t

julia> x[1:3]
3-element Vector{Symbol}:
 :x
 :y
 :z
```

## Example: Polynomials

Suppose one wants to represent the Laurent polynomial

```math
6/x + 5 - 2*x + 3*x^2 + x^3
```

The coefficients of this polynomial are a naturally `-1` based list, since the `n`th element of the list
(counting from `-1`) `6, 5, -2, 3, 1` is the coefficient corresponding to the `n`th power of `x`. This Laurent polynomial can be evaluated at say `x = 2` as follows.

```jldoctest; setup = :(using OffsetArrays)
julia> coeffs = OffsetVector([6, 5, -2, 3, 1], -1:3)
5-element OffsetArray(::Vector{Int64}, -1:3) with eltype Int64 with indices -1:3:
  6
  5
 -2
  3
  1

julia> polynomial(x, coeffs) = sum(coeffs[n]*x^n for n in eachindex(coeffs))
polynomial (generic function with 1 method)

julia> polynomial(2.0, coeffs)
24.0
```

Notice our use of the `eachindex` function which does not assume that the given array starts at `1`.
