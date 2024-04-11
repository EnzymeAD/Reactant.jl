struct TypeID
    typeid::API.MlirTypeID

    function TypeID(typeid)
        @assert !mlirIsNull(typeid) "cannot create TypeID with null MlirTypeID"
        new(typeid)
    end
end

TypeID(type::Type) = TypeID(API.mlirTypeGetTypeID(type))

# mlirTypeIDCreate

"""
    hash(typeID)

Returns the hash value of the type id.
"""
Base.hash(typeid::TypeID) = API.mlirTypeIDHashValue(typeid.typeid)

Base.convert(::Core.Type{API.MlirTypeID}, typeid::TypeID) = typeid.typeid

"""
    ==(typeID1, typeID2)

Checks if two type ids are equal.
"""
Base.:(==)(a::TypeID, b::TypeID) = API.mlirTypeIDEqual(a, b)

@static if isdefined(API, :MlirTypeIDAllocator)

    mutable struct TypeIDAllocator
        allocator::API.MlirTypeIDAllocator

        function TypeIDAllocator()
            ptr = API.mlirTypeIDAllocatorCreate()
            @assert ptr != C_NULL "cannot create TypeIDAllocator"
            finalizer(API.mlirTypeIDAllocatorDestroy, new(ptr))
        end
    end

    Base.cconvert(::Core.Type{API.MlirTypeIDAllocator}, allocator::TypeIDAllocator) = allocator
    Base.unsafe_convert(::Core.Type{API.MlirTypeIDAllocator}, allocator) = allocator.allocator

    TypeID(allocator::TypeIDAllocator) = TypeID(API.mlirTypeIDAllocatorAllocateTypeID(allocator))

else

    struct TypeIDAllocator end

end
