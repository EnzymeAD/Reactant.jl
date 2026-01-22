struct TypeID
    ref::API.MlirTypeID

    function TypeID(typeid)
        @assert !mlirIsNull(typeid) "cannot create TypeID with null MlirTypeID"
        return new(typeid)
    end
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
Base.hash(typeid::TypeID) = API.mlirTypeIDHashValue(typeid.ref)

mutable struct TypeIDAllocator
    ref::API.MlirTypeIDAllocator

    function TypeIDAllocator()
        ptr = API.mlirTypeIDAllocatorCreate()
        @assert ptr != C_NULL "cannot create TypeIDAllocator"
        return finalizer(API.mlirTypeIDAllocatorDestroy, new(ptr))
    end
end

Base.cconvert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator) = alloc
Base.unsafe_convert(::Core.Type{API.MlirTypeIDAllocator}, alloc::TypeIDAllocator) = alloc.ref

TypeID(alloc::TypeIDAllocator) = TypeID(API.mlirTypeIDAllocatorAllocateTypeID(alloc))
