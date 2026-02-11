"""
    Block

A `Block` is a sequence of [`Operation`](@ref)s with a list of arguments.
"""
@checked struct Block
    ref::API.MlirBlock
end

Block() = Block(mark_alloc(API.mlirBlockCreate(0, C_NULL, C_NULL)))

"""
    Block(args, locs)

Creates a new empty block with the given argument types and transfers ownership to the caller.
"""
function Block(args::Vector{Type}, locs::Vector{Location})
    @assert length(args) == length(locs) "there should be one args for each locs (got $(length(args)) & $(length(locs)))"
    return Block(mark_alloc(API.mlirBlockCreate(length(args), args, locs)))
end

"""
    dispose(blk::Block)

Disposes the given block and releases its resources.
After calling this function, the block must not be used anymore.
"""
dispose(blk::Block) = mark_dispose(API.mlirBlockDestroy, blk)

"""
    ==(block, other)

Checks whether two blocks handles point to the same block. This does not perform deep comparison.
"""
Base.:(==)(a::Block, b::Block) = API.mlirBlockEqual(a, b)

Base.cconvert(::Core.Type{API.MlirBlock}, block::Block) = block
Base.unsafe_convert(::Core.Type{API.MlirBlock}, block::Block) = mark_use(block).ref

Base.IteratorSize(::Core.Type{Block}) = Base.SizeUnknown()
Base.IteratorEltype(::Core.Type{Block}) = Base.HasEltype()
Base.eltype(::Block) = Operation

"""
    Base.iterate(block::Block)

Iterates over all operations for the given block.
"""
function Base.iterate(it::Block)
    raw_op = API.mlirBlockGetFirstOperation(it)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op)
        (op, op)
    end
end

function Base.iterate(::Block, op)
    raw_op = API.mlirOperationGetNextInBlock(op)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op)
        (op, op)
    end
end

function Base.show(io::IO, block::Block)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    return API.mlirBlockPrint(block, c_print_callback, ref)
end

"""
    parent_op(block)

Returns the closest surrounding operation that contains this block.
"""
parent_op(block::Block) = Operation(API.mlirBlockGetParentOperation(block))

"""
    parent_region(block)

Returns the region that contains this block.
"""
parent_region(block::Block) = Region(API.mlirBlockGetParentRegion(block))

Base.parent(block::Block) = parent_region(block)

"""
    next(block)

Returns the block immediately following the given block in its parent region or `nothing` if last.
"""
function next(block::Block)
    block = API.mlirBlockGetNextInRegion(block)
    mlirIsNull(block) && return nothing
    return Block(block)
end

"""
    nargs(block)

Returns the number of arguments of the block.
"""
nargs(block::Block) = API.mlirBlockGetNumArguments(block)

"""
    argument(block, i)

Returns `i`-th argument of the block.
"""
function argument(block::Block, i)
    i ∉ 1:nargs(block) && throw(BoundsError(block, i))
    return Value(API.mlirBlockGetArgument(block, i - 1))
end

"""
    push_argument!(block, type; location=Location())

Appends an argument of the specified type to the block. Returns the newly added argument.
"""
function push_argument!(block::Block, type; location::Location=Location())
    return Value(API.mlirBlockAddArgument(block, type, location))
end

"""
    erase_argument!(block, i)

Erase argument `i` of the block. Returns the block.
"""
function erase_argument!(block, i)
    if i ∉ 1:nargs(block)
        throw(BoundsError(block, i))
    end
    API.mlirBlockEraseArgument(block, i - 1)
    return block
end

"""
    first_op(block)

Returns the first operation in the block or `nothing` if empty.
"""
function first_op(block::Block)
    op = API.mlirBlockGetFirstOperation(block)
    mlirIsNull(op) && return nothing
    return Operation(op)
end
Base.first(block::Block) = first_op(block)

"""
    terminator(block)

Returns the terminator operation in the block or `nothing` if no terminator.
"""
function terminator(block::Block)
    op = API.mlirBlockGetTerminator(block)
    mlirIsNull(op) && return nothing
    return Operation(op)
end

"""
    push!(block, operation)

Takes an operation owned by the caller and appends it to the block.
"""
function Base.push!(block::Block, op::Operation)
    API.mlirBlockAppendOwnedOperation(block, mark_donate(op))
    return op
end

"""
    insert!(block, index, operation)

Takes an operation owned by the caller and inserts it as `index` to the block.
This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.
"""
function Base.insert!(block::Block, index, op::Operation)
    API.mlirBlockInsertOwnedOperation(block, index - 1, mark_donate(op))
    return op
end

function Base.pushfirst!(block::Block, op::Operation)
    insert!(block, 1, op)
    return op
end

"""
    insert_after!(block, reference, operation)

Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.
"""
function insert_after!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationAfter(block, reference, mark_donate(op))
    return op
end

"""
    insert_before!(block, reference, operation)

Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.
"""
function insert_before!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationBefore(block, reference, mark_donate(op))
    return op
end

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate(blk::Block)
    stack = get!(task_local_storage(), :mlir_block) do
        return Block[]
    end::Vector{Block}
    Base.push!(stack, blk)
    return nothing
end

function deactivate(blk::Block)
    current_block() == blk || error("Deactivating wrong block")
    return Base.pop!(task_local_storage(:mlir_block))
end

function has_block()
    return haskey(task_local_storage(), :mlir_block) &&
           !Base.isempty(task_local_storage(:mlir_block)::Vector{Block})
end

function current_block(; throw_error::Core.Bool=true)
    if !has_block()
        throw_error && error("No MLIR block is active")
        return nothing
    end
    return last(task_local_storage(:mlir_block)::Vector{Block})
end
