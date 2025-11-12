struct IntegerSet
    set::API.MlirIntegerSet

    function IntegerSet(set)
        @assert !mlirIsNull(set) "cannot create IntegerSet with null MlirIntegerSet"
        return new(set)
    end
end

"""
    Integerset(ndims, nsymbols; context=context())

Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.
"""
IntegerSet(ndims, nsymbols; context::Context=context()) =
    IntegerSet(API.mlirIntegerSetEmptyGet(context, ndims, nsymbols))

"""
    IntegerSet(ndims, nsymbols, constraints, eqflags; context=context())

Gets or creates a new integer set in the given context.
The set is defined by a list of affine constraints, with the given number of input dimensions and symbols, which are treated as either equalities (eqflags is 1) or inequalities (eqflags is 0).
Both `constraints` and `eqflags` need to be arrays of the same length.
"""
IntegerSet(ndims, nsymbols, constraints, eqflags; context::Context=context()) = IntegerSet(
    API.mlirIntegerSetGet(
        context, ndims, nsymbols, length(constraints), constraints, eqflags
    ),
)

"""
    mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)

Gets or creates a new integer set in which the values and dimensions of the given set are replaced with the given affine expressions.
`dimReplacements` and `symbolReplacements` are expected to point to at least as many consecutive expressions as the given set has dimensions and symbols, respectively.
The new set will have `numResultDims` and `numResultSymbols` dimensions and symbols, respectively.
"""
Base.replace(set::IntegerSet, dim_replacements, symbol_replacements) = IntegerSet(
    API.mlirIntegerSetReplaceGet(
        set,
        dim_replacements,
        symbol_replacements,
        length(dim_replacements),
        length(symbol_replacements),
    ),
)

Base.convert(::Core.Type{API.MlirIntegerSet}, set::IntegerSet) = set.set

"""
    ==(s1, s2)

Checks if two integer set objects are equal. This is a "shallow" comparison of two objects.
Only the sets with some small number of constraints are uniqued and compare equal here.
Set objects that represent the same integer set with different constraints may be considered non-equal by this check.
Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.
"""
Base.:(==)(a::IntegerSet, b::IntegerSet) = API.mlirIntegerSetEqual(a, b)

"""
    context(set)

Gets the context in which the given integer set lives.
"""
context(set::IntegerSet) = Context(API.mlirIntegerSetGetContext(set.set))

"""
    isempty(set)

Checks whether the given set is a canonical empty set, e.g., the set returned by [`Reactant.MLIR.API.mlirIntegerSetEmptyGet`](@ref).
"""
Base.isempty(set::IntegerSet) = API.mlirIntegerSetIsCanonicalEmpty(set)

"""
    ndims(set)

Returns the number of dimensions in the given set.
"""
Base.ndims(set::IntegerSet) = API.mlirIntegerSetGetNumDims(set)

"""
    nsymbols(set)

Returns the number of symbols in the given set.
"""
nsymbols(set::IntegerSet) = API.mlirIntegerSetGetNumSymbols(set)

"""
    ninputs(set)

Returns the number of inputs (dimensions + symbols) in the given set.
"""
ninputs(set::IntegerSet) = API.mlirIntegerSetGetNumInputs(set)

"""
    nconstraints(set)

Returns the number of constraints (equalities + inequalities) in the given set.
"""
nconstraints(set::IntegerSet) = API.mlirIntegerSetGetNumConstraints(set)

"""
    nequalities(set)

Returns the number of equalities in the given set.
"""
nequalities(set::IntegerSet) = API.mlirIntegerSetGetNumEqualities(set)

"""
    ninequalities(set)

Returns the number of inequalities in the given set.
"""
ninequalities(set::IntegerSet) = API.mlirIntegerSetGetNumInequalities(set)

"""
    mlirIntegerSetGetConstraint(set, i)

Returns `i`-th constraint of the set.
"""
constraint(set::IntegerSet, i) = API.mlirIntegerSetGetConstraint(set, i)

"""
    mlirIntegerSetIsConstraintEq(set, i)

Returns `true` of the `i`-th constraint of the set is an equality constraint, `false` otherwise.
"""
isconstrainteq(set::IntegerSet, i) = API.mlirIntegerSetIsConstraintEq(set, i)

function Base.show(io::IO, set::IntegerSet)
    print(io, "IntegerSet(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirIntegerSetPrint(set, c_print_callback, ref)
    return print(io, " =#)")
end
