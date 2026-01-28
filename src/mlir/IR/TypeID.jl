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

TypeIDAllocator() = TypeIDAllocator(mark_alloc(API.mlirTypeIDAllocatorCreate()))

dispose(alloc::TypeIDAllocator) = mark_dispose(API.mlirTypeIDAllocatorDestroy(alloc))

Base.cconvert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator) = alloc
function Base.unsafe_convert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator)
    return alloc.ref
end

function TypeID(alloc::TypeIDAllocator)
    return TypeID(mark_alloc(API.mlirTypeIDAllocatorAllocateTypeID(alloc)))
end
