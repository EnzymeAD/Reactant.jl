struct AffineMap
    map::API.MlirAffineMap

    function AffineMap(map::API.MlirAffineMap)
        @assert !mlirIsNull(map) "cannot create AffineMap with null MlirAffineMap"
        return new(map)
    end
end

"""
    AffineMap(; context=context())

Creates a zero result affine map with no dimensions or symbols in the context.
The affine map is owned by the context.
"""
AffineMap(; context::Context=context()) = AffineMap(API.mlirAffineMapEmptyGet(context))

Base.convert(::Core.Type{API.MlirAffineMap}, map::AffineMap) = map.map

"""
    ==(a, b)

Checks if two affine maps are equal.
"""
Base.:(==)(a::AffineMap, b::AffineMap) = API.mlirAffineMapEqual(a, b)

"""
    compose(affineExpr, affineMap)

Composes the given map with the given expression.
"""
compose(expr::AffineExpr, map::AffineMap) = AffineExpr(API.mlirAffineExprCompose(expr, map))

"""
    context(affineMap)

Gets the context that the given affine map was created with.
"""
context(map::AffineMap) = API.mlirAffineMapGetContext(map)

"""
    AffineMap(ndims, nsymbols; context=context())

Creates a zero result affine map of the given dimensions and symbols in the context.
The affine map is owned by the context.
"""
AffineMap(ndims, nsymbols; context::Context=context()) =
    AffineMap(API.mlirAffineMapZeroResultGet(context, ndims, nsymbols))

"""
    AffineMap(ndims, nsymbols, affineExprs; context=context())

Creates an affine map with results defined by the given list of affine expressions.
The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.
"""
AffineMap(ndims, nsymbols, exprs::Vector{AffineExpr}; context::Context=context()) =
    AffineMap(API.mlirAffineMapGet(context, ndims, nsymbols, length(exprs), exprs))

"""
    ConstantAffineMap(val; context=context())

Creates a single constant result affine map in the context. The affine map is owned by the context.
"""
ConstantAffineMap(val; context::Context=context()) =
    AffineMap(API.mlirAffineMapConstantGet(context, val))

"""
    IdentityAffineMap(ndims; context=context())

Creates an affine map with 'ndims' identity in the context. The affine map is owned by the context.
"""
IdentityAffineMap(ndims; context::Context=context()) =
    AffineMap(API.mlirAffineMapMultiDimIdentityGet(context, ndims))

"""
    MinorIdentityAffineMap(ndims, nresults; context=context())

Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context.
The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function MinorIdentityAffineMap(ndims, nresults; context::Context=context())
    @assert ndims >= nresults "number of dimensions must be greater or equal to the number of results"
    return AffineMap(API.mlirAffineMapMinorIdentityGet(context, ndims, nresults))
end

"""
    PermutationAffineMap(permutation; context=context())

Creates an affine map with a permutation expression and its size in the context.
The permutation expression is a non-empty vector of integers.
The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid invalid permutation).
The affine map is owned by the context.
"""
function PermutationAffineMap(permutation; context::Context=context())
    @assert Base.isperm(permutation) "$permutation must be a valid permutation"
    zero_perm = permutation .- 1
    return AffineMap(API.mlirAffineMapPermutationGet(context, length(zero_perm), zero_perm))
end

"""
    isidentity(affineMap)

Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
isidentity(map::AffineMap) = API.mlirAffineMapIsIdentity(map)

"""
    isminoridentity(affineMap)

Checks whether the given affine map is a minor identity affine map.
"""
isminoridentity(map::AffineMap) = API.mlirAffineMapIsMinorIdentity(map)

"""
    isempty(affineMap)

Checks whether the given affine map is an empty affine map.
"""
Base.isempty(map::AffineMap) = API.mlirAffineMapIsEmpty(map)

"""
    issingleconstant(affineMap)

Checks whether the given affine map is a single result constant affine map.
"""
issingleconstant(map::AffineMap) = API.mlirAffineMapIsSingleConstant(map)

"""
    result(affineMap)

Returns the constant result of the given affine map. The function asserts that the map has a single constant result.
"""
function result(map::AffineMap)
    @assert issingleconstant(map) "affine map must have a single constant result"
    return API.mlirAffineMapGetSingleConstantResult(map)
end

"""
    ndims(affineMap)

Returns the number of dimensions of the given affine map.
"""
Base.ndims(map::AffineMap) = API.mlirAffineMapGetNumDims(map)

"""
    nsymbols(affineMap)

Returns the number of symbols of the given affine map.
"""
nsymbols(map::AffineMap) = API.mlirAffineMapGetNumSymbols(map)

"""
    nresults(affineMap)

Returns the number of results of the given affine map.
"""
nresults(map::AffineMap) = API.mlirAffineMapGetNumResults(map)

"""
    result(affineMap, pos)

Returns the result at the given position.
"""
result(map::AffineMap, pos) = AffineExpr(API.mlirAffineMapGetResult(map, pos))

"""
    ninputs(affineMap)

Returns the number of inputs (dimensions + symbols) of the given affine map.
"""
ninputs(map::AffineMap) = API.mlirAffineMapGetNumInputs(map)

"""
    isprojperm(affineMap)

Checks whether the given affine map represents a subset of a symbol-less permutation map.
"""
isprojperm(map::AffineMap) = API.mlirAffineMapIsProjectedPermutation(map)

"""
    isperm(affineMap)

Checks whether the given affine map represents a symbol-less permutation map.
"""
Base.isperm(map::AffineMap) = API.mlirAffineMapIsPermutation(map)

"""
    submap(affineMap, positions)

Returns the affine map consisting of the `positions` subset.
"""
submap(map::AffineMap, pos::Vector{Int}) =
    AffineMap(API.mlirAffineMapGetSubMap(map, length(pos), pos))

"""
    majorsubmap(affineMap, nresults)

Returns the affine map consisting of the most major `nresults` results.
Returns the null AffineMap if the `nresults` is equal to zero.
Returns the `affineMap` if `nresults` is greater or equals to number of results of the given affine map.
"""
majorsubmap(map::AffineMap, nresults) =
    AffineMap(API.mlirAffineMapGetMajorSubMap(map, nresults))

"""
    minorsubmap(affineMap, nresults)

Returns the affine map consisting of the most minor `nresults` results. Returns the null AffineMap if the `nresults` is equal to zero.
Returns the `affineMap` if `nresults` is greater or equals to number of results of the given affine map.
"""
minorsubmap(map::AffineMap, nresults) =
    AffineMap(API.mlirAffineMapGetMinorSubMap(map, nresults))

"""
    mlirAffineMapReplace(affineMap, expression => replacement, numResultDims, numResultSyms)

Apply `AffineExpr::replace(map)` to each of the results and return a new new AffineMap with the new results and the specified number of dims and symbols.
"""
Base.replace(
    map::AffineMap, old_new::Pair{AffineExpr,AffineExpr}, nresultdims, nresultsyms
) = AffineMap(
    API.mlirAffineMapReplace(map, old_new.first, old_new.second, nresultdims, nresultsyms),
)

"""
    simplify(affineMaps, size, result, populateResult)

Returns the simplified affine map resulting from dropping the symbols that do not appear in any of the individual maps in `affineMaps`.
Asserts that all maps in `affineMaps` are normalized to the same number of dims and symbols.
Takes a callback `populateResult` to fill the `res` container with value `m` at entry `idx`.
This allows returning without worrying about ownership considerations.
"""
# TODO simplify(map::AffineMap, ...) = AffineMap(API.mlirAffineMapCompressUnusedSymbols(map, ...))

function Base.show(io::IO, map::AffineMap)
    print(io, "AffineMap(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirAffineMapPrint(map, c_print_callback, ref)
    return print(io, " =#)")
end

walk(f, other) = f(other)
function walk(f, expr::Expr)
    expr = f(expr)
    return Expr(expr.head, map(arg -> walk(f, arg), expr.args)...)
end

"""
    @affinemap (d1, d2, d3, ...)[s1, s2, ...] -> (d0 + d1, ...)

Returns an affine map from the provided Julia expression.
On the right hand side are allowed the following function calls:

 - +, *, ÷, %, fld, cld

The rhs can only contains dimensions and symbols present on the left hand side or integer literals.

```julia
julia> using Reactant.MLIR: IR

julia> IR.context!(IR.Context()) do
           IR.@affinemap (d1, d2)[s0] -> (d1 + s0, d2 % 10)
       end
MLIR.IR.AffineMap(#= (d0, d1)[s0] -> (d0 + s0, d1 mod 10) =#)
```
"""
macro affinemap(expr)
    @assert Meta.isexpr(expr, :(->), 2) "invalid affine expression $expr"
    Base.remove_linenums!(expr)

    lhs, rhs = expr.args
    rhs = Meta.isexpr(rhs, :block) ? only(rhs.args) : rhs

    @assert Meta.isexpr(rhs, :tuple) "invalid expression rhs $(rhs) (expected tuple)"

    dims, syms = if Meta.isexpr(lhs, :ref)
        collection, key... = lhs.args
        collection.args, key
    else
        lhs.args, Symbol[]
    end

    @assert all(x -> x isa Symbol, dims) "invalid dimensions $dims"
    @assert all(x -> x isa Symbol, syms) "invalid symbols $syms"

    dimexprs = map(enumerate(dims)) do (i, dim)
        return :($dim = AffineDimensionExpr($(i - 1)))
    end

    symexprs = map(enumerate(syms)) do (i, sym)
        return :($sym = SymbolExpr($(i - 1)))
    end

    known_binops = [:+, :-, :*, :÷, :%, :fld, :cld]

    affine_exprs = Expr(
        :vect, map(rhs.args) do ex
            walk(ex) do v
                if v isa Integer
                    Expr(:call, ConstantExpr, Int64(v))
                elseif Meta.isexpr(v, :call)
                    v
                elseif v isa Symbol
                    if v in dims || v in syms || v in known_binops
                        v
                    else
                        error("unknown item $v")
                    end
                else
                    v
                end
            end
        end...
    )

    quote
        $(dimexprs...)
        $(symexprs...)

        AffineMap($(length(dims)), $(length(syms)), $(affine_exprs))
    end
end
