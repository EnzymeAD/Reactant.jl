struct AffineExpr
    expr::API.MlirAffineExpr

    function AffineExpr(expr)
        @assert !mlirIsNull(expr) "cannot create AffineExpr with null MlirAffineExpr"
        return new(expr)
    end
end

Base.convert(::Core.Type{API.MlirAffineExpr}, expr::AffineExpr) = expr.expr

"""
    ==(a, b)

Returns `true` if the two affine expressions are equal.
"""
Base.:(==)(a::AffineExpr, b::AffineExpr) = API.mlirAffineExprEqual(a, b)

"""
    context(affineExpr)

Gets the context that owns the affine expression.
"""
context(expr::AffineExpr) = Context(API.mlirAffineExprGetContext(expr))

"""
    is_symbolic_or_constant(affineExpr)

Checks whether the given affine expression is made out of only symbols and constants.
"""
is_symbolic_or_constant(expr::AffineExpr) = API.mlirAffineExprIsSymbolicOrConstant(expr)

"""
    is_pure_affine(affineExpr)

Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
"""
is_pure_affine(expr::AffineExpr) = API.mlirAffineExprIsPureAffine(expr)

"""
    gcd(affineExpr)

Returns the greatest known integral divisor of this affine expression. The result is always positive.
"""
Base.gcd(expr::AffineExpr) = API.mlirAffineExprGetLargestKnownDivisor(expr)

"""
    ismultipleof(affineExpr, factor)

Checks whether the given affine expression is a multiple of 'factor'.
"""
ismultipleof(expr::AffineExpr, factor) = API.mlirAffineExprIsMultipleOf(expr, factor)

"""
    isfunctionofdimexpr(affineExpr, position)

Checks whether the given affine expression involves AffineDimExpr 'position'.
"""
isfunctionofdimexpr(expr::AffineExpr, position) =
    API.mlirAffineExprIsFunctionOfDim(expr, position)

"""
    isdimexpr(affineExpr)

Checks whether the given affine expression is a dimension expression.
"""
isdimexpr(expr::AffineExpr) = API.mlirAffineExprIsADim(expr)

"""
    AffineDimensionExpr(position; context=context)

Creates an affine dimension expression with 'position' in the context.
"""
AffineDimensionExpr(position; context::Context=context()) =
    AffineExpr(API.mlirAffineDimExprGet(context, position))

"""
    issymbolexpr(affineExpr)

Checks whether the given affine expression is a symbol expression.
"""
issymbolexpr(expr::AffineExpr) = API.mlirAffineExprIsASymbol(expr)

"""
    SymbolExpr(position; context=context())

Creates an affine symbol expression with 'position' in the context.
"""
SymbolExpr(position; context::Context=context()) =
    AffineExpr(API.mlirAffineSymbolExprGet(context, position))

"""
    position(affineExpr)

Returns the position of the given affine dimension expression, affine symbol expression or ...
"""
function position(expr::AffineExpr)
    if isdimexpr(expr)
        API.mlirAffineDimExprGetPosition(expr)
    elseif issymbolexpr(expr)
        API.mlirAffineSymbolExprGetPosition(expr)
    else
        throw(
            ArgumentError(
                "The given affine expression is not a affine dimension expression or affine symbol expression",
            ),
        )
    end
end

"""
    isconstantexpr(affineExpr)

Checks whether the given affine expression is a constant expression.
"""
isconstantexpr(expr::AffineExpr) = API.mlirAffineExprIsAConstant(expr)

"""
    ConstantExpr(constant::Int; context=context())

Creates an affine constant expression with 'constant' in the context.
"""
ConstantExpr(constant; context::Context=context()) =
    AffineExpr(API.mlirAffineConstantExprGet(context, constant))

"""
    value(affineExpr)

Returns the value of the given affine constant expression.
"""
function value(expr::AffineExpr)
    @assert isconstantexpr(expr) "The given affine expression is not a constant expression"
    return API.mlirAffineConstantExprGetValue(expr)
end

"""
    isadd(affineExpr)

Checks whether the given affine expression is an add expression.
"""
isadd(expr::AffineExpr) = API.mlirAffineExprIsAAdd(expr)

"""
    +(lhs, rhs)

Creates an affine add expression with 'lhs' and 'rhs'.
"""
Base.:(+)(lhs::AffineExpr, rhs::AffineExpr) = AffineExpr(API.mlirAffineAddExprGet(lhs, rhs))

"""
    ismul(affineExpr)

Checks whether the given affine expression is an mul expression.
"""
ismul(expr::AffineExpr) = API.mlirAffineExprIsAMul(expr)

"""
    *(lhs, rhs)

Creates an affine mul expression with 'lhs' and 'rhs'.
"""
Base.:(*)(lhs::AffineExpr, rhs::AffineExpr) = AffineExpr(API.mlirAffineMulExprGet(lhs, rhs))

"""
    ismod(affineExpr)

Checks whether the given affine expression is an mod expression.
"""
ismod(expr::AffineExpr) = API.mlirAffineExprIsAMod(expr)

"""
    mod(lhs, rhs)

Creates an affine mod expression with 'lhs' and 'rhs'.
"""
Base.mod(lhs::AffineExpr, rhs::AffineExpr) = AffineExpr(API.mlirAffineModExprGet(lhs, rhs))

"""
    isfloordiv(affineExpr)

Checks whether the given affine expression is an floordiv expression.
"""
isfloordiv(expr::AffineExpr) = API.mlirAffineExprIsAFloorDiv(expr)

"""
    div(lhs, rhs)
    ÷(lhs, rhs)
    fld(lhs, rhs)

Creates an affine floordiv expression with 'lhs' and 'rhs'.
"""
Base.div(lhs::AffineExpr, rhs::AffineExpr) =
    AffineExpr(API.mlirAffineFloorDivExprGet(lhs, rhs))
Base.fld(lhs::AffineExpr, rhs::AffineExpr) = div(lhs, rhs)

"""
    isceildiv(affineExpr)

Checks whether the given affine expression is an ceildiv expression.
"""
isceildiv(expr::AffineExpr) = API.mlirAffineExprIsACeilDiv(expr)

"""
    cld(lhs, rhs)

Creates an affine ceildiv expression with 'lhs' and 'rhs'.
"""
Base.cld(lhs::AffineExpr, rhs::AffineExpr) =
    AffineExpr(API.mlirAffineCeilDivExprGet(lhs, rhs))

"""
    isbinary(affineExpr)

Checks whether the given affine expression is binary.
"""
isbinary(expr::AffineExpr) = API.mlirAffineExprIsABinary(expr)

"""
    lhs(affineExpr)

Returns the left hand side affine expression of the given affine binary operation expression.
"""
lhs(expr::AffineExpr) = AffineExpr(API.mlirAffineBinaryOpExprGetLHS(expr))

"""
    rhs(affineExpr)

Returns the right hand side affine expression of the given affine binary operation expression.
"""
rhs(expr::AffineExpr) = AffineExpr(API.mlirAffineBinaryOpExprGetRHS(expr))

function Base.show(io::IO, affineExpr::AffineExpr)
    print(io, "AffineExpr(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirAffineExprPrint(affineExpr, c_print_callback, ref)
    return print(io, " =#)")
end
