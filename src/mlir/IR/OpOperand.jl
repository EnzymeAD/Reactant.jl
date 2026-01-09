@checked struct OpOperand
    ref::API.MlirOpOperand
end

Base.cconvert(::Core.Type{API.MlirOpOperand}, op::OpOperand) = op.ref

"""
    first_use(value)

Returns an `OpOperand` representing the first use of the value, or a `nothing` if there are no uses.
"""
function first_use(value::Value)
    operand = API.mlirValueGetFirstUse(value)
    mlirIsNull(operand) && return nothing
    return OpOperand(operand)
end

"""
    owner(opOperand)

Returns the owner operation of an op operand.
"""
owner(op::OpOperand) = Operation(API.mlirOpOperandGetOwner(op))

"""
    operandindex(opOperand)

Returns the operand number of an op operand.
"""
operandindex(op::OpOperand) = API.mlirOpOperandGetOperandNumber(op)

"""
    next(opOperand)

Returns an op operand representing the next use of the value, or `nothing` if there is no next use.
"""
function next(op::OpOperand)
    op = API.mlirOpOperandGetNextUse(op)
    mlirIsNull(op) && return nothing
    return OpOperand(op)
end
