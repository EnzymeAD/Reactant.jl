"""
    BlockIterator(region::Region)

Iterates over all blocks in the given region.
"""
struct BlockIterator
    region::Region
end

Base.IteratorSize(::Core.Type{BlockIterator}) = Base.SizeUnknown()
Base.eltype(::BlockIterator) = Block

function Base.iterate(it::BlockIterator)
    reg = it.region
    raw_block = API.mlirRegionGetFirstBlock(reg)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(::BlockIterator, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

"""
    RegionIterator(::Operation)

Iterates over all sub-regions for the given operation.
"""
struct RegionIterator
    op::Operation
end

Base.eltype(::RegionIterator) = Region
Base.length(it::RegionIterator) = nregions(it.op)

function Base.iterate(it::RegionIterator)
    raw_region = API.mlirOperationGetFirstRegion(it.op)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

function Base.iterate(it::RegionIterator, region)
    raw_region = API.mlirRegionGetNextInOperation(region)
    if mlirIsNull(raw_region)
        nothing
    else
        region = Region(raw_region, false)
        (region, region)
    end
end

"""
    OperationIterator(block::Block)

Iterates over all operations for the given block.
"""
struct OperationIterator
    block::Block
end

Base.IteratorSize(::Core.Type{OperationIterator}) = Base.SizeUnknown()
Base.eltype(::OperationIterator) = Operation

function Base.iterate(it::OperationIterator)
    raw_op = API.mlirBlockGetFirstOperation(it.block)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end

function Base.iterate(::OperationIterator, op)
    raw_op = API.mlirOperationGetNextInBlock(op)
    if mlirIsNull(raw_op)
        nothing
    else
        op = Operation(raw_op, false)
        (op, op)
    end
end
