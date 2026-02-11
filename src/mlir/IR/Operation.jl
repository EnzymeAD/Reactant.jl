@checked struct Operation
    ref::API.MlirOperation
end

dispose(op::Operation) = mark_dispose(API.mlirOperationDestroy, op)

Base.cconvert(::Core.Type{API.MlirOperation}, op::Operation) = op
Base.unsafe_convert(::Core.Type{API.MlirOperation}, op::Operation) = mark_use(op).ref

Base.:(==)(op::Operation, other::Operation) = API.mlirOperationEqual(op, other)

"""
    parse(::Type{Operation}, code; context=current_context())

Parses an operation from the string and transfers ownership to the caller.
"""
function Base.parse(
    ::Core.Type{Operation},
    code;
    verify::Bool=false,
    context::Context=current_context(),
    block=Block(),
    location::Location=Location(),
)
    return Operation(
        mark_alloc(
            @ccall API.mlir_c.mlirOperationParse(
                context::API.MlirContext,
                block::API.MlirBlock,
                code::API.MlirStringRef,
                location::API.MlirLocation,
                verify::Bool,
            )::API.MlirOperation
        ),
    )
end

function Base.show(io::IO, operation::Operation)
    if mlirIsNull(operation.ref)
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

Base.IteratorSize(::Core.Type{Operation}) = Base.HasLength()
Base.IteratorEltype(::Core.Type{Operation}) = Base.HasEltype()
Base.eltype(::Operation) = Region
Base.length(it::Operation) = nregions(it)

"""
    Base.iterate(op::Operation)

Iterates over all sub-regions for the given operation.
"""
function Base.iterate(it::Operation)
    raw_region = API.mlirOperationGetFirstRegion(it)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region)
        (region, region)
    end
end

function Base.iterate(::Operation, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region)
        (region, region)
    end
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
function parent_op(op::Operation)
    return Operation(API.mlirOperationGetParentOperation(op))
end

"""
    rmfromparent!(op)

Removes the given operation from its parent block. The operation is not destroyed.
The ownership of the operation is transferred to the caller.
"""
function rmfromparent!(op::Operation)
    API.mlirOperationRemoveFromParent(op)
    # TODO mark ownership moved to the caller
    return op
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
    i ∉ 1:nregions(op) && throw(BoundsError(op, i))
    return Region(API.mlirOperationGetRegion(op, i - 1))
end

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
    i ∉ 1:nresults(op) && throw(BoundsError(op, i))
    return Value(API.mlirOperationGetResult(op, i - 1))
end
results(op) = [result(op, i) for i in 1:nresults(op)]

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
    i ∉ 1:noperands(op) && throw(BoundsError(op, i))
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
    i ∉ 1:nsuccessors(op) && throw(BoundsError(op, i))
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
    i ∉ 1:nattrs(op) && throw(BoundsError(op, i))
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
verify(operation::Operation) = API.mlirOperationVerify(operation)

"""
    move_after!(op, other)

Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function move_after!(operation::Operation, other::Operation)
    mark_donate(operation)
    return API.mlirOperationMoveAfter(operation, other)
end

"""
    move_before!(op, other)

Moves the given operation immediately before the other operation in its parent block.
The given operation may be owner by the caller or by its current block.
The other operation must belong to a block.
In any case, the ownership is transferred to the block of the other operation.
"""
function move_before!(op::Operation, other::Operation)
    mark_donate(op)
    return API.mlirOperationMoveBefore(op, other)
end

"""
    is_registered(name; context=current_context())

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context.
This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
function is_registered(opname; context::Context=current_context())
    return API.mlirContextIsRegisteredOperation(context, opname)
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
            mark_donate.(owned_regions)
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
        return Operation(op)
    end
end

function create_operation(args...; kwargs...)
    res = create_operation_common(args...; kwargs...)
    if has_block()
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
