module IFRT

using CEnum
using Reactant_jll
const libxla = Reactant_jll.libReactantExtra

import ..Reactant: @cbinding
import ..PjRt
import ..XLA

const Cspan = @NamedTuple{len::Csize_t, ptr::Ptr{Cvoid}}

# Backend trait
abstract type Backend end

"""
    backend(x)

Returns the `Backend` trait of the given IFRT object.
"""
function backend end

# TODO remove this? `Serializable` doesn't have any method
abstract type AbstractSerializable end
abstract type AbstractValue end
abstract type AbstractTuple <: AbstractValue end
abstract type AbstractMemory end
abstract type AbstractDevice end
abstract type AbstractSharding <: AbstractSerializable end
abstract type AbstractArray end
abstract type AbstractTopology end
abstract type AbstractClient end
abstract type AbstractHostCallback end
abstract type AbstractLoadedHostCallback end
abstract type AbstractExecutable end
abstract type AbstractLoadedExecutable end
abstract type AbstractCompiler end
abstract type AbstractProgram <: AbstractSerializable end

# Base virtual classes
for (abstyp, contype) in [
    (:AbstractValue, :Value),
    (:AbstractSharding, :Sharding),
    # (:AbstractHostCallback, :HostCallback),
    # (:AbstractLoadedHostCallback, :LoadedHostCallback),
    # (:AbstractProgram, :Program),
]
    @eval begin
        struct $contype <: $abstyp
            ptr::Ptr{Cvoid}
            function $contype(x)
                @assert x != C_NULL
                return new(x)
            end
        end

        Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::$contype) = x.ptr
    end
end

# Base virtual classes with IFRT backend implementations
# WARN we redefine `Tuple` so you need to use `Core.Tuple` to access the original `Tuple`
for (abstyp, contype) in [
    (:AbstractTuple, :Tuple),
    (:AbstractMemory, :Memory),
    (:AbstractDevice, :Device),
    (:AbstractArray, :Array),
    (:AbstractTopology, :Topology),
    (:AbstractClient, :Client),
    # (:AbstractHostCallback, :HostCallback),
    # (:AbstractLoadedHostCallback, :LoadedHostCallback),
    (:AbstractExecutable, :Executable),
    (:AbstractLoadedExecutable, :LoadedExecutable),
    (:AbstractCompiler, :Compiler),
    # (:AbstractProgram, :Program),
]
    @eval begin
        struct $contype <: $abstyp
            backend::Backend
            ptr::Ptr{Cvoid}
            function $contype(backend, x)
                @assert x != C_NULL
                return new(backend, x)
            end
        end

        Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::$contype) = x.ptr
        backend(x::$contype) = x.backend
        free(x::$contype) = free(backend(x), x)
        function free(b::Backend, x::$contype)
            @warn("No free method for $contype with backend $b")
            return nothing
        end
    end
end

# Concrete classes
for (typ, dtor) in [
    (:DType, :(libxla.ifrt_dtype_free)),
    (:Shape, :(libxla.ifrt_shape_free)),
    (:DynamicShape, :(libxla.ifrt_dynamic_shape_free)),
    (:Index, :(libxla.ifrt_index_free)),
    (:IndexDomain, :(libxla.ifrt_indexdomain_free)),
    (:MemoryKind, :(libxla.ifrt_memorykind_free)),
]
    @eval begin
        mutable struct $typ
            ptr::Ptr{Cvoid}
            function $typ(x::Ptr{Cvoid})
                @assert x != C_NULL
                y = new(x)
                finalizer(y) do z
                    @ccall $dtor(z::Ptr{Cvoid})::Cvoid
                end
                return y
            end
        end

        Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::$typ) = x.ptr
    end
end

# Enums
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

@cenum ArrayCopySemantics::Int32 begin
    ArrayCopySemanticsAlwaysCopy = 0
    ArrayCopySemanticsReuseInput = 1
    ArrayCopySemanticsDonateInput = 2
end

# Value
function client(x::AbstractValue)
    return Client(@ccall libxla.ifrt_value_client(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function ready_future(x::AbstractValue)
    return XLA.Future(@ccall libxla.ifrt_value_ready(x::Ptr{Cvoid})::Cvoid)
end

# TODO use `Base.delete!` or `Base.empty!`?
# TODO use `PjRt.Future` when moved there
function delete!(x::AbstractValue)
    return XLA.Future(@ccall libxla.ifrt_value_delete(x::Ptr{Cvoid})::Cvoid)
end

isdeleted(x::AbstractValue) = @ccall libxla.ifrt_value_is_deleted(x::Ptr{Cvoid})::Bool

function debug_string(x::AbstractValue)
    return Base.unsafe_string(@ccall libxla.ifrt_value_debug_string(x::Ptr{Cvoid})::Cstring)
end

# Tuple
Base.length(x::AbstractTuple) = @ccall libxla.ifrt_tuple_arity(x::Ptr{Cvoid})::Int

# TODO `Unpack`?

# Memory
function id(x::Memory)
    return @ccall libxla.ifrt_memory_id(x::Ptr{Cvoid})::Int32
end

function kind(x::Memory)
    return MemoryKind(@ccall libxla.ifrt_memory_kind(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO check ownership of passing vector
function devices(x::Memory)
    (; len, ptr) = @ccall libxla.ifrt_memory_devices(x::Ptr{Cvoid})::Cspan
    return Base.unsafe_wrap(Base.Vector, reinterpret(Ptr{Device}, ptr), len; own=false)
end

function Base.string(x::Memory)
    return Base.unsafe_string(@ccall libxla.ifrt_memory_string(x::Ptr{Cvoid})::Cstring)
end

function debug_string(x::Memory)
    return Base.unsafe_string(
        @ccall libxla.ifrt_memory_debug_string(x::Ptr{Cvoid})::Cstring
    )
end

# Device
function client(x::AbstractDevice)
    return Client(@ccall libxla.ifrt_device_client(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function id(x::AbstractDevice)
    return @ccall libxla.ifrt_device_id(x::Ptr{Cvoid})::Int32
end

function kind(x::AbstractDevice)
    return Base.unsafe_string(@ccall libxla.ifrt_device_kind(x::Ptr{Cvoid})::Cstring)
end

function debug_string(x::AbstractDevice)
    return Base.unsafe_string(
        @ccall libxla.ifrt_device_debug_string(x::Ptr{Cvoid})::Cstring
    )
end

function default_memory(x::AbstractDevice)
    return Memory(@ccall libxla.ifrt_device_default_memory(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function isaddressable(x::AbstractDevice)
    return @ccall libxla.ifrt_device_is_addressable(x::Ptr{Cvoid})::Bool
end

function pid(x::AbstractDevice)
    return @ccall libxla.ifrt_device_process_index(x::Ptr{Cvoid})::Int
end

# TODO Sharding

# Array
function dtype(x::AbstractArray)
    return DType(@ccall libxla.ifrt_array_dtype(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# convert DType of Array to Julia type
Base.eltype(x::AbstractArray) = convert(Type, dtype(x))

function shape(x::AbstractArray)
    return Shape(@ccall libxla.ifrt_array_shape(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO `shape` to `size`

function sharding(x::AbstractArray)
    return Sharding(@ccall libxla.ifrt_array_sharding(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO layout, copy_to_host_buffer

# Topology
function platformname(x::AbstractTopology)
    return Base.unsafe_string(
        @ccall libxla.ifrt_topology_platform_name(x::Ptr{Cvoid})::Cstring
    )
end

function platformversion(x::AbstractTopology)
    return Base.unsafe_string(
        @ccall libxla.ifrt_topology_platform_version(x::Ptr{Cvoid})::Cstring
    )
end

function platformid(x::AbstractTopology)
    return @ccall libxla.ifrt_topology_platform_id(x::Ptr{Cvoid})::UInt64
end

# TODO `descriptions` from `ifrt_topology_device_descriptions`
# TODO `default_layout` from `xla::ifrt::Topology::GetDefaultLayout`

function serialize(x::AbstractTopology)
    return Base.unsafe_string(@ccall libxla.ifrt_topology_serialize(x::Ptr{Cvoid})::Cstring)
end

# Client
# TODO a lot of methods...

function runtime_type(x::AbstractClient)
    return Base.unsafe_string(
        @ccall libxla.ifrt_client_runtime_type(x::Ptr{Cvoid})::Cstring
    )
end

function platformname(x::AbstractClient)
    return Base.unsafe_string(
        @ccall libxla.ifrt_client_platform_name(x::Ptr{Cvoid})::Cstring
    )
end

function platformversion(x::AbstractClient)
    return Base.unsafe_string(
        @ccall libxla.ifrt_client_platform_version(x::Ptr{Cvoid})::Cstring
    )
end

function platformid(x::AbstractClient)
    return @ccall libxla.ifrt_client_platform_id(x::Ptr{Cvoid})::UInt64
end

function devices(x::AbstractClient; addressable::Bool=false)
    (; len, ptr) = if addressable
        @ccall libxla.ifrt_client_addressable_devices(x::Ptr{Cvoid})::Cspan
    else
        @ccall libxla.ifrt_client_devices(x::Ptr{Cvoid})::Cspan
    end
    return Base.unsafe_wrap(Base.Array, reinterpret(Ptr{Device}, ptr), len; own=true)
end

function pid(x::AbstractClient)
    return @ccall libxla.ifrt_client_process_index(x::Ptr{Cvoid})::Int
end

# NOTE potentially deprecated
function lookup_device(x::AbstractClient, id::Int; addressable::Bool=false)
    return Device(
        if addressable
            @ccall libxla.ifrt_client_lookup_addressable_device(
                x::Ptr{Cvoid}, id::Int
            )::Ptr{Cvoid}
        else
            @ccall libxla.ifrt_client_lookup_device(x::Ptr{Cvoid}, id::Int)::Ptr{Cvoid}
        end,
    )
end

# NOTE potentially deprecated
function default_compiler(x::AbstractClient)
    return Compiler(@ccall libxla.ifrt_client_default_compiler(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO ifrt_client_topology_for_devices, ifrt_client_default_layout_for_device

# HostCallback
function serialize(x::AbstractHostCallback)
    return Base.unsafe_string(
        @ccall libxla.ifrt_hostcallback_serialize(x::Ptr{Cvoid})::Cstring
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

# Executable
function name(x::AbstractExecutable)
    return Base.unsafe_string(@ccall libxla.ifrt_executable_name(x::Ptr{Cvoid})::Cstring)
end

function fingerprint(x::AbstractExecutable)
    return Base.unsafe_string(
        @ccall libxla.ifrt_executable_fingerprint(x::Ptr{Cvoid})::Cstring
    )
end

function serialize(x::AbstractExecutable)
    return Base.unsafe_string(
        @ccall libxla.ifrt_executable_serialize(x::Ptr{Cvoid})::Cstring
    )
end

function ndevices(x::AbstractExecutable)
    return @ccall libxla.ifrt_executable_num_devices(x::Ptr{Cvoid})::Int
end

function byte_size(x::AbstractExecutable)
    return @ccall libxla.ifrt_executable_byte_size(x::Ptr{Cvoid})::Int
end

# TODO missing `Executable` methods in the C-API

# LoadedExecutable
function client(x::AbstractLoadedExecutable)
    return Client(@ccall libxla.ifrt_loadedexecutable_client(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function name(x::AbstractLoadedExecutable)
    return Base.unsafe_string(
        @ccall libxla.ifrt_loadedexecutable_name(x::Ptr{Cvoid})::Cstring
    )
end

function fingerprint(x::AbstractLoadedExecutable)
    return Base.unsafe_string(
        @ccall libxla.ifrt_loadedexecutable_fingerprint(x::Ptr{Cvoid})::Cstring
    )
end

function serialize(x::AbstractLoadedExecutable)
    return Base.unsafe_string(
        @ccall libxla.ifrt_loadedexecutable_serialize(x::Ptr{Cvoid})::Cstring
    )
end

function ndevices(x::AbstractLoadedExecutable)
    return @ccall libxla.ifrt_loadedexecutable_num_devices(x::Ptr{Cvoid})::Int
end

function byte_size(x::AbstractLoadedExecutable)
    return @ccall libxla.ifrt_loadedexecutable_byte_size(x::Ptr{Cvoid})::Int
end

# TODO maybe use `Base.delete!` or `Base.empty!`?
# TODO use `PjRt.Future` when moved there
function delete!(x::AbstractLoadedExecutable)
    return XLA.Future(@ccall libxla.ifrt_loadedexecutable_delete(x::Ptr{Cvoid})::Cvoid)
end

function isdeleted(x::AbstractLoadedExecutable)
    @ccall libxla.ifrt_loadedexecutable_is_deleted(x::Ptr{Cvoid})::Bool
end

# TODO missing `LoadedExecutable` methods in the C-API

# Compiler
function compile(compiler::AbstractCompiler, program::AbstractProgram)
    return LoadedExecutable(
        @ccall libxla.ifrt_compiler_compile(
            compiler::Ptr{Cvoid}, program::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function compile(
    compiler::AbstractCompiler, program::AbstractProgram, topology::AbstractTopology
)
    return Executable(
        @ccall libxla.ifrt_compiler_compile_with_topology(
            compiler::Ptr{Cvoid}, program::Ptr{Cvoid}, topology::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function deserialize(compiler::AbstractCompiler, serialized::String)
    return LoadedExecutable(
        @ccall libxla.ifrt_compiler_deserialize_loadedexecutable(
            compiler::Ptr{Cvoid}, serialized::Cstring
        )::Ptr{Cvoid}
    )
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

Base.length(x::Shape) = @ccall libxla.ifrt_shape_num_elements(x::Ptr{Cvoid})::Int

function Base.size(x::Shape)
    (; len, ptr) = @ccall libxla.ifrt_shape_dims(x::Ptr{Cvoid})::Cspan
    return Base.unsafe_wrap(Base.Array, reinterpret(Ptr{Int64}, ptr), len; own=true)
end

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

function Base.collect(x::Index)
    (; len, ptr) = @ccall libxla.ifrt_index_elements(x::Ptr{Cvoid})::Cspan
    return Base.unsafe_wrap(Base.Array, reinterpret(Ptr{Int64}, ptr), len; own=true)
end

function Base.length(x::Index)
    return @ccall libxla.ifrt_index_count(x::Ptr{Cvoid})::Int
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

function Base.:(*)(x::Index, y::Vector{Int64})
    @boundscheck length(x) == length(y)
    return Index(@ccall libxla.ifrt_index_mul(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid})
end

function add!(x::Index, y::Index)
    @ccall libxla.ifrt_index_add_inplace(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Cvoid
    return x
end

function sub!(x::Index, y::Index)
    @ccall libxla.ifrt_index_sub_inplace(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Cvoid
    return x
end

function mul!(x::Index, y::Vector{Int64})
    @boundscheck length(x) == length(y)
    @ccall libxla.ifrt_index_mul_inplace(x::Ptr{Cvoid}, y::Ptr{Int})::Cvoid
    return x
end

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
            origin::Ptr{Cvoid}, shape::Ptr{Cvoid}
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

function Base.:(+)(x::IndexDomain, y::Index)
    return IndexDomain(
        @ccall libxla.ifrt_indexdomain_add(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function Base.:(-)(x::IndexDomain, y::Index)
    return IndexDomain(
        @ccall libxla.ifrt_indexdomain_sub(x::Ptr{Cvoid}, y::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function add!(x::IndexDomain, offset::Index)
    @ccall libxla.ifrt_indexdomain_add_inplace(x::Ptr{Cvoid}, offset::Ptr{Cvoid})::Cvoid
    return x
end

function sub!(x::IndexDomain, offset::Index)
    @ccall libxla.ifrt_indexdomain_sub_inplace(x::Ptr{Cvoid}, offset::Ptr{Cvoid})::Cvoid
    return x
end

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

# DTypeKind
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

include("BackendPjRt.jl")

end
