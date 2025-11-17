struct TypeID
    typeid::API.MlirTypeID

    function TypeID(typeid)
        @assert !mlirIsNull(typeid) "cannot create TypeID with null MlirTypeID"
        return new(typeid)
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
            return finalizer(API.mlirTypeIDAllocatorDestroy, new(ptr))
        end
    end

    function Base.cconvert(::Core.Type{API.MlirTypeIDAllocator}, allocator::TypeIDAllocator)
        return allocator
    end
    function Base.unsafe_convert(
        ::Core.Type{API.MlirTypeIDAllocator}, allocator::TypeIDAllocator
    )
        return allocator.allocator
    end

    function TypeID(allocator::TypeIDAllocator)
        return TypeID(API.mlirTypeIDAllocatorAllocateTypeID(allocator))
    end

else
    struct TypeIDAllocator end
end
