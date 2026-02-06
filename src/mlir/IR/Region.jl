@checked struct Region
    ref::API.MlirRegion
end

"""
    Region()

Creates a new empty region and transfers ownership to the caller.
"""
Region() = Region(mark_alloc(API.mlirRegionCreate()))

"""
    dispose(region::Region)

Disposes the given region and releases its resources.
After calling this function, the region must not be used anymore.
"""
dispose(region::Region) = mark_dispose(API.mlirRegionDestroy, region)

Base.cconvert(::Core.Type{API.MlirRegion}, region::Region) = region
Base.unsafe_convert(::Core.Type{API.MlirRegion}, region::Region) = mark_use(region).ref

"""
    ==(region, other)

Checks whether two region handles point to the same region. This does not perform deep comparison.
"""
Base.:(==)(a::Region, b::Region) = API.mlirRegionEqual(a, b)

Base.IteratorSize(::Core.Type{Region}) = Base.SizeUnknown()
Base.IteratorEltype(::Core.Type{Region}) = Base.HasEltype()
Base.eltype(::Region) = Block

"""
    Base.iterate(region::Region)

Iterates over all [`Block`](@ref) in the given region.
"""
function Base.iterate(it::Region)
    raw_block = API.mlirRegionGetFirstBlock(it)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block)
        (b, b)
    end
end

function Base.iterate(::Region, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block)
        (b, b)
    end
end

"""
    push!(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function Base.push!(region::Region, block::Block)
    API.mlirRegionAppendOwnedBlock(region, mark_donate(block))
    return block
end

"""
    insert!(region, index, block)

Takes a block owned by the caller and inserts it at `index` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function Base.insert!(region::Region, index, block::Block)
    API.mlirRegionInsertOwnedBlock(region, index - 1, mark_donate(block))
    return block
end

function Base.pushfirst!(region::Region, block)
    insert!(region, 1, block)
    return block
end

"""
    insert_after!(region, reference, block)

Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.
"""
function insert_after!(region::Region, reference::Block, block::Block)
    return API.mlirRegionInsertOwnedBlockAfter(region, reference, mark_donate(block))
end

"""
    insert_before!(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
function insert_before!(region::Region, reference::Block, block::Block)
    return API.mlirRegionInsertOwnedBlockBefore(region, reference, mark_donate(block))
end

"""
    first_block(region)

Gets the first block in the region.
"""
function first_block(region::Region)
    block = API.mlirRegionGetFirstBlock(region)
    mlirIsNull(block) && return nothing
    return Block(block, false)
end
Base.first(region::Region) = first_block(region)
