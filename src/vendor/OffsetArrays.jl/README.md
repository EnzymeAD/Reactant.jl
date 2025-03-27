# OffsetArrays.jl

| Documentation |  Build Status | Code coverage | Version |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] |  [![][action-img]][action-url] | [![][codecov-img]][codecov-url] | [![][ver-img]][ver-url]  |
| [![][docs-dev-img]][docs-dev-url]  |  [![][pkgeval-img]][pkgeval-url] | | [![][deps-img]][deps-url] |

## Introduction

OffsetArrays provides Julia users with arrays that have arbitrary
indices, similar to those found in some other programming languages
like Fortran.

An `OffsetArray` is a lightweight wrapper around an `AbstractArray` that shifts its indices.
Generally, indexing into an `OffsetArray` should be as performant as the parent array.

## Usage

There are two ways to construct `OffsetArray`s: by specifying the axes of the array, or
by specifying its origin.

The first way to construct an `OffsetArray` by specifying its axes is:

```julia
OA = OffsetArray(A, axis1, axis2, ...)
```

where you want `OA` to have axes `(axis1, axis2, ...)` and be indexed by values that
fall within these axis ranges. Example:

```julia
julia> using OffsetArrays

julia> A = Float64.(reshape(1:15, 3, 5))
3×5 Matrix{Float64}:
 1.0  4.0  7.0  10.0  13.0
 2.0  5.0  8.0  11.0  14.0
 3.0  6.0  9.0  12.0  15.0

julia> axes(A) # indices of a Matrix start from 1 along each axis
(Base.OneTo(3), Base.OneTo(5))

julia> OA = OffsetArray(A, -1:1, 0:4) # OA will have the axes (-1:1, 0:4)
3×5 OffsetArray(::Matrix{Float64}, -1:1, 0:4) with eltype Float64 with indices -1:1×0:4:
 1.0  4.0  7.0  10.0  13.0
 2.0  5.0  8.0  11.0  14.0
 3.0  6.0  9.0  12.0  15.0

julia> OA[-1, 0]
1.0

julia> OA[1, 4]
15.0
```

The second way to construct an `OffsetArray` is by specifying the origin, that is, the first index
along each axis. This is particularly useful if one wants, eg., arrays that are 0-indexed as opposed
to 1-indexed.

A convenient way to construct an `OffsetArray` this way is by using `OffsetArrays.Origin`:

```julia
julia> using OffsetArrays: Origin

julia> Origin(0)(A) # indices begin at 0 along all axes
3×5 OffsetArray(::Matrix{Float64}, 0:2, 0:4) with eltype Float64 with indices 0:2×0:4:
 1.0  4.0  7.0  10.0  13.0
 2.0  5.0  8.0  11.0  14.0
 3.0  6.0  9.0  12.0  15.0

julia> Origin(2, 3)(A) # indices begin at 2 along the first axis and 3 along the second
3×5 OffsetArray(::Matrix{Float64}, 2:4, 3:7) with eltype Float64 with indices 2:4×3:7:
 1.0  4.0  7.0  10.0  13.0
 2.0  5.0  8.0  11.0  14.0
 3.0  6.0  9.0  12.0  15.0
```

While the examples here refer to the common case where the parent arrays have indices starting at 1,
this is not necessary. An `OffsetArray` may wrap any array that has integer indices, irrespective of
where the indices begin.

## How to go back to 1-indexed arrays

Certain libraries, such as `LinearAlgebra`, require arrays to be indexed from 1. Passing an `OffsetArray`
with shifted indices would lead to an error here.

```julia
julia> A = Float64.(reshape(1:16, 4, 4));

julia> AO = Origin(0)(A);

julia> using LinearAlgebra

julia> Diagonal(AO)
ERROR: ArgumentError: offset arrays are not supported but got an array with index other than 1
```

The way to obtain a `1`-indexed array from an `OffsetArray` is by using `OffsetArrays.no_offset_view`.

An example of this is:

```julia
julia> OffsetArrays.no_offset_view(AO)
4×4 Matrix{Float64}:
 1.0  5.0   9.0  13.0
 2.0  6.0  10.0  14.0
 3.0  7.0  11.0  15.0
 4.0  8.0  12.0  16.0
```

This may now be passed to `LinearAlgebra`:

```julia
julia> D = Diagonal(OffsetArrays.no_offset_view(AO))
4×4 Diagonal{Float64, Vector{Float64}}:
 1.0   ⋅     ⋅     ⋅
  ⋅   6.0    ⋅     ⋅
  ⋅    ⋅   11.0    ⋅
  ⋅    ⋅     ⋅   16.0
```

If we want to restore the original indices of `AO`, we may wrap an `OffsetArray` around the `Diagonal` as:

```julia
julia> Origin(AO)(D)
4×4 OffsetArray(::Diagonal{Float64, Vector{Float64}}, 0:3, 0:3) with eltype Float64 with indices 0:3×0:3:
 1.0   ⋅     ⋅     ⋅
  ⋅   6.0    ⋅     ⋅
  ⋅    ⋅   11.0    ⋅
  ⋅    ⋅     ⋅   16.0
```

Here, `Origin(AO)` is able to automatically infer and use the indices of `AO`.

## Best practice on adopting OffsetArrays

For some applications, OffsetArrays give users an easy-to-understand interface. However, handling
the non-conventional axes of OffsetArrays requires extra care. Otherwise, the code might
error, crash, or return incorrect results. You can read [the Julialang documentation on
offset](https://docs.julialang.org/en/v1/devdocs/offset-arrays/) for more information. Here
we briefly summarize some of the best practices for users and package authors.

### There is no need to support OffsetArrays for every function

You don't need to support offset arrays for _internal functions_ that only consume standard 1-based
arrays -- it doesn't change or improve anything.

You don't need to support offset arrays for functions that _have no well-defined behavior on custom
axes_. For instance, many linear algebra functions such as matrix multiplication `A * B` does not
have an agreed behavior for offset arrays. In this case, it is a better practice to let users do the
conversion.

The helper function `Base.require_one_based_indexing` can be used to early check the axes and throw
a meaningful error. If your interface functions do not intend to support offset arrays, we recommend
you add this check before starting the real computation.

### use `axes` instead of `size`/`length`

Many implementations assume the array axes start at 1 by writing loops such as `for i in
1:length(x)` or `for i in 1:size(x, 1)`. A better practice is to use `for i in eachindex(x)` or `for
i in axes(x, 1)` -- `axes` provides more information than `size` with no performance overhead.

Also, if you know what indices type you want to use, [`LinearIndices`][doc_LinearIndices] and
[`CartesianIndices`][doc_CartesianIndices] allow you to loop multidimensional arrays efficiently
without worrying about the axes.

### test against OffsetArrays

For package authors that declare support for `AbstractArray`, we recommend having a few test cases
against `OffsetArray` to ensure the function works well for arrays with custom axes. This gives you
more confidence that users don't run into strange situations.

For package users that want to use offset arrays, many numerical correctness issues come from the
fact that `@inbounds` is used inappropriately with the 1-based indexing assumption. Thus for debug
purposes, it is not a bad idea to start Julia with `--check-bounds=yes`, which turns all `@inbounds`
into a no-op and uncover potential out-of-bound errors.

<!-- badges -->

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/O/OffsetArrays.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html

[action-img]: https://github.com/JuliaArrays/OffsetArrays.jl/workflows/Unit%20test/badge.svg
[action-url]: https://github.com/JuliaArrays/OffsetArrays.jl/actions

[codecov-img]: https://codecov.io/github/JuliaArrays/OffsetArrays.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/JuliaArrays/OffsetArrays.jl

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliaarrays.github.io/OffsetArrays.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliaarrays.github.io/OffsetArrays.jl/dev/

[ver-img]: https://juliahub.com/docs/OffsetArrays/version.svg
[ver-url]: https://juliahub.com/ui/Packages/OffsetArrays/UDEDl

[deps-img]: https://juliahub.com/docs/OffsetArrays/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/OffsetArrays/UDEDl

[doc_LinearIndices]: https://docs.julialang.org/en/v1/base/arrays/#Base.LinearIndices
[doc_CartesianIndices]: https://docs.julialang.org/en/v1/base/arrays/#Base.IteratorsMD.CartesianIndices
