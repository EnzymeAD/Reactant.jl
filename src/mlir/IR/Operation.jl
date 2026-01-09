@checked struct Operation
    ref::API.MlirOperation
end

dispose!(op::Operation) = mark_dispose(API.mlirOperationDestroy, op)

Base.cconvert(::Core.Type{API.MlirOperation}, op::Operation) = mark_use(op).ref

Base.:(==)(op::Operation, other::Operation) = API.mlirOperationEqual(op, other)

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
    return Operation(mark_alloc(
        @ccall API.mlir_c.mlirOperationParse(
            context::API.MlirContext,
            block::API.MlirBlock,
            code::API.MlirStringRef,
            location::API.MlirLocation,
            verify::Bool,
        )::API.MlirOperation
    ))
end

function Base.show(io::IO, op::Operation)
    if mlirIsNull(op.ref)
        return write(io, "Operation(NULL)")
    end

    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))

    buffer = IOBuffer()
    ref = Ref(buffer)

    flags = API.mlirOpPrintingFlagsCreate()

    API.mlirOpPrintingFlagsEnableDebugInfo(flags, get(io, :debug, false), false)
    API.mlirOperationPrintWithFlags(op, flags, c_print_callback, ref)
    API.mlirOpPrintingFlagsDestroy(flags)

    return write(io, rstrip(String(take!(buffer))))
end

"""
    copy(op)

Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.
"""
Base.copy(op::Operation) = Operation(mark_alloc(API.mlirOperationClone(op)))

"""
    context(op)

Gets the context this operation is associated with.
"""
context(op::Operation) = Context(API.mlirOperationGetContext(op))

"""
    location(op)

Gets the location of the operation.
"""
location(op::Operation) = Location(API.mlirOperationGetLocation(op))

"""
    typeid(op)

Gets the type id of the operation. Returns null if the operation does not have a registered operation description.
"""
typeid(op::Operation) = TypeID(API.mlirOperationGetTypeID(op))

"""
    name(op)

Gets the name of the operation as an identifier.
"""
name(op::Operation) = String(API.mlirOperationGetName(op))

"""
    block(op)

Gets the block that owns this operation, returning null if the operation is not owned.
"""
block(op::Operation) = Block(API.mlirOperationGetBlock(op))

"""
    parent_op(op)

Gets the operation that owns this operation, returning null if the operation is not owned.
"""
parent_op(op::Operation) = Operation(API.mlirOperationGetParentOperation(op))

"""
    rmfromparent!(op)

Removes the given operation from its parent block. The operation is not destroyed.
The ownership of the operation is transferred to the caller.
"""
function rmfromparent!(operation::Operation)
    API.mlirOperationRemoveFromParent(operation)
    return operation
end

dialect(op::Operation) = Symbol(first(split(name(op), '.')))

"""
    nregions(op)

Returns the number of regions attached to the given operation.
"""
nregions(op::Operation) = API.mlirOperationGetNumRegions(op)

"""
    region(op, i)

Returns `i`-th region attached to the operation.
"""
function region(op::Operation, i)
    if i ∉ 1:nregions(op)
        throw(BoundsError(op, i))
    end
    return Region(API.mlirOperationGetRegion(op, i - 1))
end

regions(op::Operation) = collect(op)

"""
    nresults(op)

Returns the number of results of the operation.
"""
nresults(op::Operation) = API.mlirOperationGetNumResults(op)

"""
    result(op, i)

Returns `i`-th result of the operation.
"""
function result(op::Operation, i=1)
    if i ∉ 1:nresults(op)
        throw(BoundsError(op, i))
    end
    return Value(API.mlirOperationGetResult(op, i - 1))
end

results(op::Operation) = [result(op, i) for i in 1:nresults(op)]

"""
    noperands(op)

Returns the number of operands of the operation.
"""
noperands(op::Operation) = API.mlirOperationGetNumOperands(op)

"""
    operand(op, i)

Returns `i`-th operand of the operation.
"""
function operand(op::Operation, i=1)
    if i ∉ 1:noperands(op)
        throw(BoundsError(op, i))
    end
    return Value(API.mlirOperationGetOperand(op, i - 1))
end

"""
    operands(op)

Return an array of all operands of the operation.
"""
operands(op) = Value[operand(op, i) for i in 1:noperands(op)]

"""
    setoperand!(op, i, value)

Sets the `i`-th operand of the operation.
"""
function setoperand!(op::Operation, i, value)
    i ∉ 1:noperands(op) && throw(BoundsError(op, i))
    API.mlirOperationSetOperand(op, i - 1, value)
    return value
end

"""
    nsuccessors(op)

Returns the number of successor blocks of the operation.
"""
nsuccessors(op::Operation) = API.mlirOperationGetNumSuccessors(op)

"""
    successor(op, i)

Returns `i`-th successor of the operation.
"""
function successor(op::Operation, i)
    if i ∉ 1:nsuccessors(op)
        throw(BoundsError(op, i))
    end
    return Block(API.mlirOperationGetSuccessor(op, i - 1))
end

"""
    nattrs(op)

Returns the number of attributes attached to the operation.
"""
nattrs(op::Operation) = API.mlirOperationGetNumAttributes(op)

"""
    getattr(op, i)

Return `i`-th attribute of the operation.
"""
function getattr(op::Operation, i)
    if i ∉ 1:nattrs(op)
        throw(BoundsError(op, i))
    end
    return NamedAttribute(API.mlirOperationGetAttribute(op, i - 1))
end

"""
    getattr(op, name)

Returns an attribute attached to the operation given its name.
"""
function getattr(op::Operation, name::AbstractString)
    raw_attr = API.mlirOperationGetAttributeByName(op, name)
    if mlirIsNull(raw_attr)
        return nothing
    end
    return Attribute(raw_attr)
end

"""
    setattr!(op, name, attr)

Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise.
"""
function setattr!(op::Operation, name, attribute)
    API.mlirOperationSetAttributeByName(op, name, attribute)
    return op
end

"""
    rmattr!(op, name)

Removes an attribute by name. Returns false if the attribute was not found and true if removed.
"""
rmattr!(op::Operation, name) = API.mlirOperationRemoveAttributeByName(op, name)

"""
    verify(op)

Verify the operation and return true if it passes, false if it fails.
"""
verify(op::Operation) = API.mlirOperationVerify(op)

"""
    move_after!(op, other)

Moves the given operation immediately after the other operation in its parent block.
The given operation may be owned by the caller or by its current block.
The other operation must belong to a block.
In any case, the ownership is transferred to the block of the other operation.
"""
move_after!(op::Operation, other::Operation) = API.mlirOperationMoveAfter(op, other)

"""
    move_before!(op, other)

Moves the given operation immediately before the other operation in its parent block.
The given operation may be owner by the caller or by its current block.
The other operation must belong to a block.
In any case, the ownership is transferred to the block of the other operation.
"""
move_before!(op::Operation, other::Operation) = API.mlirOperationMoveBefore(op, other)

"""
    is_registered(name; context=context())

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context.
This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
function is_registered(opname; context::Context=context())
    return API.mlirContextIsRegisteredOperation(context, opname)
end

Base.eltype(::Operation) = Region
Base.length(it::Operation) = nregions(it.op)

function Base.iterate(it::Operation)
    raw_region = API.mlirOperationGetFirstRegion(it.op)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region)
        (it, region)
    end
end

function Base.iterate(it::Operation, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region)
        (it, region)
    end
end

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
            GC.@preserve owned_regions begin
                mlir_regions = Base.cconvert.(API.MlirRegion, owned_regions)
                API.mlirOperationStateAddOwnedRegions(
                    state, length(mlir_regions), mlir_regions
                )
            end
        end
        if !isnothing(successors)
            GC.@preserve successors begin
                mlir_blocks = Base.cconvert.(API.MlirBlock, successors)
                API.mlirOperationStateAddSuccessors(state, length(mlir_blocks), mlir_blocks)
            end
        end
        if !isnothing(attributes)
            API.mlirOperationStateAddAttributes(state, length(attributes), attributes)
        end
        if result_inference
            API.mlirOperationStateEnableResultTypeInference(state)
        end
        op = mark_alloc(API.mlirOperationCreate(state))
        if mlirIsNull(op)
            error("Create Operation '$name' failed")
        end
        return Operation(op)
    end
end

function create_operation(args...; kwargs...)
    res = create_operation_common(args...; kwargs...)
    if has_current_block()
        push!(current_block(), res)
    end
    return res
end

function create_operation_at_front(args...; kwargs...)
    res = create_operation_common(args...; kwargs...)
    Base.pushfirst!(current_block(), res)
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

"""
    verifyall(operation; debug=false)

Prints the operations which could not be verified.
"""
function verifyall(operation::Operation; debug=false)
    io = IOBuffer()
    visit(operation) do op
        ok = verifyall(op; debug)
        if !ok || !verify(op)
            if ok
                show(IOContext(io, :debug => debug), op)
                error(String(take!(io)))
            end
            false
        else
            true
        end
    end
end
