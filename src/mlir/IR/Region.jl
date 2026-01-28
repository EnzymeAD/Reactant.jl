mutable struct Region
    ref::API.MlirRegion
    @atomic owned::Bool

    function Region(region, owned=true)
        @assert !mlirIsNull(region)
        finalizer(new(region, owned)) do region
            if region.owned
                API.mlirRegionDestroy(region.ref)
            end
        end
    end
end

"""
    Region()

Creates a new empty region and transfers ownership to the caller.
"""
Region() = Region(API.mlirRegionCreate())

Base.cconvert(::Core.Type{API.MlirRegion}, region::Region) = region
Base.unsafe_convert(::Core.Type{API.MlirRegion}, region::Region) = region.ref

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
        b = Block(raw_block, false)
        (b, b)
    end
end

function Base.iterate(::Region, block)
    raw_block = API.mlirBlockGetNextInRegion(block)
    if mlirIsNull(raw_block)
        nothing
    else
        b = Block(raw_block, false)
        (b, b)
    end
end

"""
    push!(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function Base.push!(region::Region, block::Block)
    API.mlirRegionAppendOwnedBlock(region, lose_ownership!(block))
    return block
end

"""
    insert!(region, index, block)

Takes a block owned by the caller and inserts it at `index` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function Base.insert!(region::Region, index, block::Block)
    API.mlirRegionInsertOwnedBlock(region, index - 1, lose_ownership!(block))
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
    return API.mlirRegionInsertOwnedBlockAfter(region, reference, lose_ownership!(block))
end

"""
    insert_before!(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
function insert_before!(region::Region, reference::Block, block::Block)
    return API.mlirRegionInsertOwnedBlockBefore(region, reference, lose_ownership!(block))
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

function lose_ownership!(region::Region)
    @assert region.owned
    @atomic region.owned = false
    return region
end
