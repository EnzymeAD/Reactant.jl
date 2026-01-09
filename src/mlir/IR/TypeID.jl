@checked struct TypeID
    ref::API.MlirTypeID
end

TypeID(type::Type) = TypeID(API.mlirTypeGetTypeID(type))

# TODO mlirTypeIDCreate

Base.cconvert(::Core.Type{API.MlirTypeID}, typeid::TypeID) = typeid.ref

"""
    ==(typeID1, typeID2)

Checks if two type ids are equal.
"""
Base.:(==)(a::TypeID, b::TypeID) = API.mlirTypeIDEqual(a, b)

"""
    hash(typeID)

Returns the hash value of the type id.
"""
Base.hash(typeid::TypeID) = API.mlirTypeIDHashValue(typeid)

@checked struct TypeIDAllocator
    ref::API.MlirTypeIDAllocator
end

TypeIDAllocator() = TypeIDAllocator(API.mlirTypeIDAllocatorCreate())

dispose!(alloc::TypeIDAllocator) = API.mlirTypeIDAllocatorDestroy(alloc)

function Base.cconvert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator)
    return alloc.ref
end

function TypeID(allocator::TypeIDAllocator)
    return TypeID(API.mlirTypeIDAllocatorAllocateTypeID(allocator))
end
