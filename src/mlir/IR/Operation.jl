mutable struct Operation
    operation::API.MlirOperation
    @atomic owned::Bool

    function Operation(operation, owned=true)
        @assert !mlirIsNull(operation) "cannot create Operation with null MlirOperation"
        finalizer(new(operation, owned)) do op
            if op.owned
                API.mlirOperationDestroy(op.operation)
            end
        end
    end
end

Base.cconvert(::Core.Type{API.MlirOperation}, operation::Operation) = operation
function Base.unsafe_convert(::Core.Type{API.MlirOperation}, operation::Operation)
    return operation.operation
end
Base.:(==)(op::Operation, other::Operation) = API.mlirOperationEqual(op, other)

"""
    copy(op)

Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.
"""
Base.copy(operation::Operation) = Operation(API.mlirOperationClone(operation))

"""
    context(op)

Gets the context this operation is associated with.
"""
context(operation::Operation) = Context(API.mlirOperationGetContext(operation))

"""
    location(op)

Gets the location of the operation.
"""
location(operation::Operation) = Location(API.mlirOperationGetLocation(operation))

"""
    typeid(op)

Gets the type id of the operation. Returns null if the operation does not have a registered operation description.
"""
typeid(op::Operation) = TypeID(API.mlirOperationGetTypeID(op))

"""
    name(op)

Gets the name of the operation as an identifier.
"""
name(operation::Operation) = String(API.mlirOperationGetName(operation))

"""
    block(op)

Gets the block that owns this operation, returning null if the operation is not owned.
"""
block(operation::Operation) = Block(API.mlirOperationGetBlock(operation), false)

"""
    parent_op(op)

Gets the operation that owns this operation, returning null if the operation is not owned.
"""
parent_op(operation::Operation) =
    Operation(API.mlirOperationGetParentOperation(operation), false)

"""
    rmfromparent!(op)

Removes the given operation from its parent block. The operation is not destroyed.
The ownership of the operation is transferred to the caller.
"""
function rmfromparent!(operation::Operation)
    API.mlirOperationRemoveFromParent(operation)
    @atomic operation.owned = true
    return operation
end

dialect(operation::Operation) = Symbol(first(split(name(operation), '.')))

"""
    nregions(op)

Returns the number of regions attached to the given operation.
"""
nregions(operation::Operation) = API.mlirOperationGetNumRegions(operation)

"""
    region(op, i)

Returns `i`-th region attached to the operation.
"""
function region(operation::Operation, i)
    i ∉ 1:nregions(operation) && throw(BoundsError(operation, i))
    return Region(API.mlirOperationGetRegion(operation, i - 1), false)
end

"""
    nresults(op)

Returns the number of results of the operation.
"""
nresults(operation::Operation) = API.mlirOperationGetNumResults(operation)

"""
    result(op, i)

Returns `i`-th result of the operation.
"""
function result(operation::Operation, i=1)
    i ∉ 1:nresults(operation) && throw(BoundsError(operation, i))
    return Value(API.mlirOperationGetResult(operation, i - 1))
end
results(operation) = [result(operation, i) for i in 1:nresults(operation)]

"""
    noperands(op)

Returns the number of operands of the operation.
"""
noperands(operation::Operation) = API.mlirOperationGetNumOperands(operation)

"""
    operand(op, i)

Returns `i`-th operand of the operation.
"""
function operand(operation::Operation, i=1)
    i ∉ 1:noperands(operation) && throw(BoundsError(operation, i))
    return Value(API.mlirOperationGetOperand(operation, i - 1))
end

"""
    operands(op)

Return an array of all operands of the operation.
"""
operands(op) = Value[operand(op, i) for i in 1:noperands(op)]

"""
    operand!(op, i, value)

Sets the `i`-th operand of the operation.
"""
function operand!(operation::Operation, i, value)
    i ∉ 1:noperands(operation) && throw(BoundsError(operation, i))
    API.mlirOperationSetOperand(operation, i - 1, value)
    return value
end

"""
    nsuccessors(op)

Returns the number of successor blocks of the operation.
"""
nsuccessors(operation::Operation) = API.mlirOperationGetNumSuccessors(operation)

"""
    successor(op, i)

Returns `i`-th successor of the operation.
"""
function successor(operation::Operation, i)
    i ∉ 1:nsuccessors(operation) && throw(BoundsError(operation, i))
    return Block(API.mlirOperationGetSuccessor(operation, i - 1), false)
end

"""
    nattrs(op)

Returns the number of attributes attached to the operation.
"""
nattrs(operation::Operation) = API.mlirOperationGetNumAttributes(operation)

"""
    attr(op, i)

Return `i`-th attribute of the operation.
"""
function attr(operation::Operation, i)
    i ∉ 1:nattrs(operation) && throw(BoundsError(operation, i))
    return NamedAttribute(API.mlirOperationGetAttribute(operation, i - 1))
end

"""
    attr(op, name)

Returns an attribute attached to the operation given its name.
"""
function attr(operation::Operation, name::AbstractString)
    raw_attr = API.mlirOperationGetAttributeByName(operation, name)
    if mlirIsNull(raw_attr)
        return nothing
    end
    return Attribute(raw_attr)
end

"""
    attr!(op, name, attr)

Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise.
"""
function attr!(operation::Operation, name, attribute)
    API.mlirOperationSetAttributeByName(operation, name, attribute)
    return operation
end

"""
    rmattr!(op, name)

Removes an attribute by name. Returns false if the attribute was not found and true if removed.
"""
rmattr!(operation::Operation, name) =
    API.mlirOperationRemoveAttributeByName(operation, name)

function lose_ownership!(operation::Operation)
    @assert operation.owned
    @atomic operation.owned = false
    return operation
end

function Base.show(io::IO, operation::Operation)
    if mlirIsNull(operation.operation)
        return write(io, "Operation(NULL)")
    end

    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))

    buffer = IOBuffer()
    ref = Ref(buffer)

    flags = API.mlirOpPrintingFlagsCreate()

    API.mlirOpPrintingFlagsEnableDebugInfo(flags, get(io, :debug, false), false)
    API.mlirOperationPrintWithFlags(operation, flags, c_print_callback, ref)
    API.mlirOpPrintingFlagsDestroy(flags)

    return write(io, rstrip(String(take!(buffer))))
end

"""
    parse(::Type{Operation}, code; context=context())

Parses an operation from the string and transfers ownership to the caller.
"""
function Base.parse(
    ::Core.Type{Operation},
    code;
    verify::Bool=false,
    context::Context=context(),
    block=Block(),
    location::Location=Location(),
)
    return Operation(
        @ccall API.mlir_c.mlirOperationParse(
            context::API.MlirContext,
            block::API.MlirBlock,
            code::API.MlirStringRef,
            location::API.MlirLocation,
            verify::Bool,
        )::API.MlirOperation
    )
end

"""
    verify(op)

Verify the operation and return true if it passes, false if it fails.
"""
verify(operation::Operation) = API.mlirOperationVerify(operation)

"""
    move_after!(op, other)

Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function move_after!(operation::Operation, other::Operation)
    lose_ownership!(operation)
    return API.mlirOperationMoveAfter(operation, other)
end

"""
    move_before!(op, other)

Moves the given operation immediately before the other operation in its parent block.
The given operation may be owner by the caller or by its current block.
The other operation must belong to a block.
In any case, the ownership is transferred to the block of the other operation.
"""
function move_before!(operation::Operation, other::Operation)
    lose_ownership!(operation)
    return API.mlirOperationMoveBefore(operation, other)
end

"""
    is_registered(name; context=context())

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context.
This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
is_registered(opname; context::Context=context()) =
    API.mlirContextIsRegisteredOperation(context, opname)

function create_operation_common(
    name,
    loc;
    results=nothing,
    operands=nothing,
    owned_regions=nothing,
    successors=nothing,
    attributes=nothing,
    result_inference=isnothing(results),
)
    GC.@preserve name loc begin
        state = Ref(API.mlirOperationStateGet(name, loc))
        if !isnothing(results)
            if result_inference
                error("Result inference and provided results conflict")
            end
            API.mlirOperationStateAddResults(state, length(results), results)
        end
        if !isnothing(operands)
            API.mlirOperationStateAddOperands(state, length(operands), operands)
        end
        if !isnothing(owned_regions)
            lose_ownership!.(owned_regions)
            GC.@preserve owned_regions begin
                mlir_regions = Base.unsafe_convert.(API.MlirRegion, owned_regions)
                API.mlirOperationStateAddOwnedRegions(
                    state, length(mlir_regions), mlir_regions
                )
            end
        end
        if !isnothing(successors)
            GC.@preserve successors begin
                mlir_blocks = Base.unsafe_convert.(API.MlirBlock, successors)
                API.mlirOperationStateAddSuccessors(state, length(mlir_blocks), mlir_blocks)
            end
        end
        if !isnothing(attributes)
            API.mlirOperationStateAddAttributes(state, length(attributes), attributes)
        end
        if result_inference
            API.mlirOperationStateEnableResultTypeInference(state)
        end
        op = API.mlirOperationCreate(state)
        if mlirIsNull(op)
            error("Create Operation '$name' failed")
        end
        return Operation(op, true)
    end
end

function create_operation(args...; kwargs...)
    res = create_operation_common(args...; kwargs...)
    if _has_block()
        push!(block(), res)
    end
    return res
end

function create_operation_at_front(args...; kwargs...)
    res = create_operation_common(args...; kwargs...)
    Base.pushfirst!(block(), res)
    return res
end

function FunctionType(op::Operation)
    is_function_op = @ccall API.mlir_c.mlirIsFunctionOpInterface(
        op::API.MlirOperation
    )::Bool
    if is_function_op
        return Type(
            @ccall API.mlir_c.mlirGetFunctionTypeFromOperation(
                op::API.MlirOperation
            )::API.MlirType
        )
    else
        throw("operation is not a function operation")
    end
end
