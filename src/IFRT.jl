module IFRT

using CEnum
using Reactant_jll
const libxla = Reactant_jll.libReactantExtra

const Span = @NamedTuple{len::Csize_t, ptr::Ptr{Cvoid}}

abstract type AbstractSerializable end

# Base virtual classes
abstract type AbstractValue end
struct Value <: AbstractValue
    ptr::Ptr{Cvoid}
    function Value(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractTuple <: AbstractValue end
struct Tuple <: AbstractTuple
    ptr::Ptr{Cvoid}
    function Tuple(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractMemory end
struct Memory <: AbstractMemory
    ptr::Ptr{Cvoid}
    function Memory(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractDevice end
struct Device <: AbstractDevice
    ptr::Ptr{Cvoid}
    function Device(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractSharding <: AbstractSerializable end
struct Sharding <: AbstractSharding
    ptr::Ptr{Cvoid}
    function Sharding(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractArray end
struct Array <: AbstractArray
    ptr::Ptr{Cvoid}
    function Array(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractTopology end
struct Topology <: AbstractTopology
    ptr::Ptr{Cvoid}
    function Topology(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractClient end
struct Client <: AbstractClient
    ptr::Ptr{Cvoid}
    function Client(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractHostCallback end
struct HostCallback <: AbstractHostCallback
    ptr::Ptr{Cvoid}
    function HostCallback(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractLoadedHostCallback end
struct LoadedHostCallback <: AbstractLoadedHostCallback
    ptr::Ptr{Cvoid}
    function LoadedHostCallback(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractExecutable end
struct Executable <: AbstractExecutable
    ptr::Ptr{Cvoid}
    function Executable(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractLoadedExecutable end
struct LoadedExecutable <: AbstractLoadedExecutable
    ptr::Ptr{Cvoid}
    function LoadedExecutable(x)
        @assert x != C_NULL
        return new(x)
    end
end

abstract type AbstractCompiler end
struct Compiler <: AbstractCompiler
    ptr::Ptr{Cvoid}
    function Compiler(x)
        @assert x != C_NULL
        return new(x)
    end
end

# Concrete classes
mutable struct DType
    ptr::Ptr{Cvoid}
    function DType(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_dtype_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct Shape
    ptr::Ptr{Cvoid}
    function Shape(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_shape_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct DynamicShape
    ptr::Ptr{Cvoid}
    function DynamicShape(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_dynamic_shape_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct Index
    ptr::Ptr{Cvoid}
    function Index(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_index_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct IndexDomain
    ptr::Ptr{Cvoid}
    function IndexDomain(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_indexdomain_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct MemoryKind
    ptr::Ptr{Cvoid}
    function MemoryKind(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_memorykind_free(z::Ptr{Cvoid})
        end
        return y
    end
end

# Value
function client(x::AbstractValue)
    return Client(@ccall libxla.ifrt_value_client(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO `get_ready_future`

# TODO it returns a `Future` object
# function Base.empty!(x::AbstractValue)
#     @ccall libxla.ifrt_value_delete(x::Ptr{Cvoid})::Cvoid
#     return x
# end

Base.isempty(x::AbstractValue) = @ccall libxla.ifrt_value_is_deleted(x::Ptr{Cvoid})::Bool

function debug_string(x::AbstractValue)
    return Base.unsafe_string(@ccall libxla.ifrt_value_debug_string(x::Ptr{Cvoid})::Cstring)
end

# Tuple
Base.length(x::AbstractTuple) = @ccall libxla.ifrt_tuple_arity(x::Ptr{Cvoid})::Int

# TODO `Unpack`?

# DTypeKind
@cenum DTypeKind::Int32 begin
    # Invalid data type.
    DTypeKindInvalid = 0

    # Predicates are two-state booleans.
    DTypeKindPred = 1

    # Signed integral values of fixed width.
    DTypeKindS2 = 26
    DTypeKindS4 = 21
    DTypeKindS8 = 2
    DTypeKindS16 = 3
    DTypeKindS32 = 4
    DTypeKindS64 = 5

    # Unsigned integral values of fixed width.
    DTypeKindU2 = 27
    DTypeKindU4 = 22
    DTypeKindU8 = 6
    DTypeKindU16 = 7
    DTypeKindU32 = 8
    DTypeKindU64 = 9

    # Floating-point values of fixed width.
    DTypeKindF16 = 10
    DTypeKindF32 = 11
    DTypeKindF64 = 12

    # Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
    # floating-point format, but uses 1 bit for the sign, 8 bits for the
    # exponent and 7 bits for the mantissa.
    DTypeKindBF16 = 16

    # Complex values of fixed width.
    DTypeKindC64 = 15 # Paired F32 (real, imag), as in std::complex<float>.
    DTypeKindC128 = 18 # Paired F64 (real, imag), as in std::complex<double>.

    # A token type threaded between side-effecting operations. Shapes of this
    # dtype will have empty dimensions.
    DTypeKindToken = 17

    # Opaque objects.
    DTypeKindOpaque = 14

    DTypeKindF8E3M4 = 29
    DTypeKindF8E4M3 = 28
    DTypeKindF8E4M3FN = 20
    DTypeKindF8E4M3B11FNUZ = 23
    DTypeKindF8E4M3FNUZ = 25
    DTypeKindF8E5M2 = 19
    DTypeKindF8E5M2FNUZ = 24

    # Variable-length string represented as raw bytes, as in `bytes` in Python,
    # i.e., no encoding enforcement. String is not support in XLA. DType.Kind
    # needs to match xla.PrimitiveType enum, so choose a large enum to avoid
    # collision.
    DTypeKindString = 99
end

Base.convert(::Type{DTypeKind}, ::Type) = DTypeKindInvalid
Base.convert(::Type{DTypeKind}, ::Type{Bool}) = DTypeKindPred
Base.convert(::Type{DTypeKind}, ::Type{Int8}) = DTypeKindS8
Base.convert(::Type{DTypeKind}, ::Type{Int16}) = DTypeKindS16
Base.convert(::Type{DTypeKind}, ::Type{Int32}) = DTypeKindS32
Base.convert(::Type{DTypeKind}, ::Type{Int64}) = DTypeKindS64
Base.convert(::Type{DTypeKind}, ::Type{UInt8}) = DTypeKindU8
Base.convert(::Type{DTypeKind}, ::Type{UInt16}) = DTypeKindU16
Base.convert(::Type{DTypeKind}, ::Type{UInt32}) = DTypeKindU32
Base.convert(::Type{DTypeKind}, ::Type{UInt64}) = DTypeKindU64
Base.convert(::Type{DTypeKind}, ::Type{Float16}) = DTypeKindF16
Base.convert(::Type{DTypeKind}, ::Type{Float32}) = DTypeKindF32
Base.convert(::Type{DTypeKind}, ::Type{Float64}) = DTypeKindF64
Base.convert(::Type{DTypeKind}, ::Type{Complex{Float32}}) = DTypeKindC64
Base.convert(::Type{DTypeKind}, ::Type{Complex{Float64}}) = DTypeKindC128
Base.convert(::Type{DTypeKind}, ::Type{String}) = DTypeKindString

function Base.convert(::Type{Type}, x::DTypeKind)
    if x == DTypeKindPred
        Bool
    elseif x == DTypeKindS8
        Int8
    elseif x == DTypeKindS16
        Int16
    elseif x == DTypeKindS32
        Int32
    elseif x == DTypeKindS64
        Int64
    elseif x == DTypeKindU8
        UInt8
    elseif x == DTypeKindU16
        UInt16
    elseif x == DTypeKindU32
        UInt32
    elseif x == DTypeKindU64
        UInt64
    elseif x == DTypeKindF16
        Float16
    elseif x == DTypeKindF32
        Float32
    elseif x == DTypeKindF64
        Float64
    elseif x == DTypeKindC64
        Complex{Float32}
    elseif x == DTypeKindC128
        Complex{Float64}
    elseif x == DTypeKindString
        String
    else
        error("Unknown mapping of DType to Julia type")
    end
end

# DType
DType(x::DTypeKind) = DType(@ccall libxla.ifrt_dtype_ctor(x::Int32)::Ptr{Cvoid})
DType(x::Type) = DType(convert(DTypeKind, x))

function Base.:(==)(x::DType, y::DType)
    return x.ptr == y.ptr || @ccall libxla.ifrt_dtype_eq(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.:(!=)(x::DType, y::DType)
    return x.ptr != y.ptr && @ccall libxla.ifrt_dtype_ne(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

kind(x::DType) = DTypeKind(@ccall libxla.ifrt_dtype_kind(x::Ptr{Cvoid})::Int32)

function byte_size(x::DType)
    len = @ccall libxla.ifrt_dtype_byte_size(x::Ptr{Cvoid})::Int
    return len < 0 ? nothing : len
end

function bit_size(x::DType)
    len = @ccall libxla.ifrt_dtype_bit_size(x::Ptr{Cvoid})::Int
    return len < 0 ? nothing : len
end

function debug_string(x::DType)
    return Base.unsafe_string(@ccall libxla.ifrt_dtype_debug_string(x::Ptr{Cvoid})::Cstring)
end

# Shape
function Shape(x::Vector{Int64})
    return Shape(@ccall libxla.ifrt_shape_ctor(x::Ptr{Int}, length(x)::Csize_t)::Ptr{Cvoid})
end

# function Base.ndims(x::Shape)
#     (; len, ptr) = @ccall libxla.ifrt_shape_dims(x::Ptr{Cvoid})::Span
#     @show len ptr
#     return Base.unsafe_wrap(Base.Array, reinterpret(Ptr{Int64}, ptr), len; own=false)
# end

Base.length(x::Shape) = @ccall libxla.ifrt_shape_num_elements(x::Ptr{Cvoid})::Int

function debug_string(x::Shape)
    return Base.unsafe_string(@ccall libxla.ifrt_shape_debug_string(x::Ptr{Cvoid})::Cstring)
end

# DynamicShape
function DynamicShape(shape::Shape, mask::Vector{Bool})
    @assert length(mask) == length(ndims(shape))

    return DynamicShape(
        @ccall libxla.ifrt_dynamic_shape_ctor(
            shape::Ptr{Cvoid}, mask::Ptr{Bool}
        )::Ptr{Cvoid}
    )
end

function Base.:(==)(x::DynamicShape, y::DynamicShape)
    return x.ptr == y.ptr ||
           @ccall libxla.ifrt_dynamicshape_eq(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.:(!=)(x::DynamicShape, y::DynamicShape)
    return x.ptr != y.ptr &&
           @ccall libxla.ifrt_dynamicshape_ne(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function pad(x::DynamicShape)
    return Shape(
        @ccall libxla.ifrt_dynamicshape_get_padded_shape(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function isdynamicdim(x::DynamicShape, d::Int)
    return @ccall libxla.ifrt_dynamicshape_is_dynamic_dim(x::Ptr{Cvoid}, d::Int)::Bool
end

function debug_string(x::DynamicShape)
    return Base.unsafe_string(
        @ccall libxla.ifrt_dynamicshape_debug_string(x::Ptr{Cvoid})::Cstring
    )
end

# Index
function Index(dims::Vector{Int64})
    return Index(
        @ccall libxla.ifrt_index_ctor(dims::Ptr{Int}, length(dims)::Csize_t)::Ptr{Cvoid}
    )
end

function Base.zeros(::Type{Index}, dims::Int)
    return Index(@ccall libxla.ifrt_index_zeros(dims::Int)::Ptr{Cvoid})
end

function Base.:(==)(x::Index, y::Index)
    return x.ptr == y.ptr || @ccall libxla.ifrt_index_eq(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.:(!=)(x::Index, y::Index)
    return x.ptr != y.ptr && @ccall libxla.ifrt_index_ne(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.:(+)(x::Index, y::Index)
    return Index(@ccall libxla.ifrt_index_add(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid})
end

function Base.:(-)(x::Index, y::Index)
    return Index(@ccall libxla.ifrt_index_sub(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO
# function Base.:(*)(x::Index, y::Index)
#     return Index(@ccall libxla.ifrt_index_mul(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid})
# end

# TODO add!, sub!, mul!

function debug_string(x::Index)
    return Base.unsafe_string(@ccall libxla.ifrt_index_debug_string(x::Ptr{Cvoid})::Cstring)
end

# IndexDomain
function IndexDomain(shape::Shape)
    return IndexDomain(@ccall libxla.ifrt_indexdomain_ctor(shape::Ptr{Cvoid})::Ptr{Cvoid})
end

function IndexDomain(origin::Index, shape::Shape)
    return IndexDomain(
        @ccall libxla.ifrt_indexdomain_ctor_with_orign(
            index::Ptr{Cvoid}, shape::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function Base.:(==)(x::IndexDomain, y::IndexDomain)
    return x.ptr == y.ptr ||
           @ccall libxla.ifrt_indexdomain_eq(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.:(!=)(x::IndexDomain, y::IndexDomain)
    return x.ptr != y.ptr &&
           @ccall libxla.ifrt_indexdomain_ne(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

# TODO are we taking ownership of this?
function origin(x::IndexDomain)
    return Index(@ccall libxla.ifrt_indexdomain_origin(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function shape(x::IndexDomain)
    return Shape(@ccall libxla.ifrt_indexdomain_shape(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function Base.:(+)(x::IndexDomain, y::IndexDomain)
    return IndexDomain(
        @ccall libxla.ifrt_indexdomain_add(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function Base.:(-)(x::IndexDomain, y::IndexDomain)
    return IndexDomain(
        @ccall libxla.ifrt_indexdomain_sub(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

# TODO add!, sub!

function debug_string(x::IndexDomain)
    return Base.unsafe_string(
        @ccall libxla.ifrt_indexdomain_debug_string(x::Ptr{Cvoid})::Cstring
    )
end

# MemoryKind
function MemoryKind(x::AbstractString)
    return MemoryKind(@ccall libxla.ifrt_memorykind_ctor(x::Cstring)::Ptr{Cvoid})
end

function Base.:(==)(x::MemoryKind, y::MemoryKind)
    return x.ptr == y.ptr ||
           @ccall libxla.ifrt_memorykind_eq(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.:(!=)(x::MemoryKind, y::MemoryKind)
    return x.ptr != y.ptr &&
           @ccall libxla.ifrt_memorykind_ne(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Bool
end

function Base.string(x::MemoryKind)
    return Base.unsafe_string(@ccall libxla.ifrt_memorykind_string(x::Ptr{Cvoid})::Cstring)
end

function canonicalize(x::MemoryKind, dev::Device)
    return MemoryKind(
        @ccall libxla.ifrt_memorykind_canonicalize(
            x::Ptr{Cvoid}, dev::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

# Memory
function id(x::Memory)
    return @ccall libxla.ifrt_memory_id(x::Ptr{Cvoid})::Int32
end

function kind(x::Memory)
    return MemoryKind(@ccall libxla.ifrt_memory_kind(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function Base.string(x::Memory)
    return Base.unsafe_string(@ccall libxla.ifrt_memory_string(x::Ptr{Cvoid})::Cstring)
end

function debug_string(x::Memory)
    return Base.unsafe_string(
        @ccall libxla.ifrt_memory_debug_string(x::Ptr{Cvoid})::Cstring
    )
end

# LoadedHostCallback
function client(x::AbstractLoadedHostCallback)
    return Client(@ccall libxla.ifrt_loadedhostcallback_client(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function serialize(x::AbstractLoadedHostCallback)
    return Base.unsafe_string(
        @ccall libxla.ifrt_loadedhostcallback_serialize(x::Ptr{Cvoid})::Cstring
    )
end

# IFRT-PjRt backend

mutable struct PjRtTuple <: AbstractTuple
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_tuple_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtMemory <: AbstractMemory
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_memory_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtDevice <: AbstractDevice
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_device_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtArray <: AbstractArray
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_array_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtTopology <: AbstractTopology
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_topology_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtClient <: AbstractClient
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_client_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtHostSendAndRecvLoadedHostCallback <: AbstractLoadedHostCallback
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_hostsendandrecv_loadhostcallback_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtExecutable <: AbstractExecutable
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_executable_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtLoadedExecutable <: AbstractLoadedExecutable
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_loadedexecutable_free(z::Ptr{Cvoid})
        end
        return y
    end
end

mutable struct PjRtCompiler <: AbstractCompiler
    ptr::Ptr{Cvoid}
    function _(x::Ptr{Cvoid})
        @assert x != C_NULL
        y = new(x)
        finalizer(y) do z
            @ccall libxla.ifrt_pjrt_compiler_free(z::Ptr{Cvoid})
        end
        return y
    end
end

# function PjRtTuple(client::AbstractPjRtCompatibleClient, values::Vector{AbstractValue})
#     return PjRtTuple(
#         @ccall libxla.ifrt_pjrt_tuple_ctor(
#             client::Ptr{Cvoid}, values::Vector{Ptr{Cvoid}}, length(values)::Int
#         )::Ptr{Cvoid}
#     )
# end

# TODO for PjRt-IFRT backend, implement `ifrt_to_primitive_type` and `ifrt_to_dtype`

end
