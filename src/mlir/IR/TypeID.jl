@checked struct TypeID
    ref::API.MlirTypeID
end

TypeID(type::Type) = TypeID(API.mlirTypeGetTypeID(type))

# mlirTypeIDCreate

Base.cconvert(::Core.Type{API.MlirTypeID}, typeid::TypeID) = typeid
Base.unsafe_convert(::Core.Type{API.MlirTypeID}, typeid::TypeID) = typeid.ref

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

TypeIDAllocator() = mark_alloc(TypeIDAllocator(API.mlirTypeIDAllocatorCreate()))

dispose(alloc::TypeIDAllocator) = mark_dispose(API.mlirTypeIDAllocatorDestroy, alloc)

Base.cconvert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator) = alloc
function Base.unsafe_convert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator)
    return mark_use(alloc).ref
end

function TypeID(alloc::TypeIDAllocator)
    # the type ID lives as long as its allocator
    return mark_alloc(TypeID(API.mlirTypeIDAllocatorAllocateTypeID(alloc)); owner=alloc)
end
