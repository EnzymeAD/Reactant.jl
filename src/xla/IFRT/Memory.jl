mutable struct Memory <: XLA.AbstractMemory
    ptr::Ptr{Cvoid}
end

function Base.show(io::IO, memory::Memory)
    str = GC.@preserve memory MLIR.API.ifrt_MemoryToString(memory.ptr)
    print(io, "XLA.IFRT.Memory(\"", XLA.unsafe_string_and_free(str), "\")")
    return nothing
end

mutable struct MemoryKind <: XLA.AbstractMemoryKind
    ptr::Ptr{Cvoid}
end

function MemoryKind(::Nothing)
    return MemoryKind(MLIR.API.ifrt_memory_kind_with_optional_memory_space())
end

function MemoryKind(str::AbstractString)
    return MemoryKind(MLIR.API.ifrt_memory_kind_from_string(string(str)))
end

function Base.isempty(memory_kind::MemoryKind)
    res = GC.@preserve memory_kind MLIR.API.ifrt_memory_kind_has_value(memory_kind.ptr)
    return !res
end

function Base.convert(::Type{MemoryKind}, memory::Memory)
    return MemoryKind(GC.@preserve memory MLIR.API.ifrt_MemoryGetMemoryKind(memory.ptr))
end

function Base.:(==)(a::MemoryKind, b::MemoryKind)
    return GC.@preserve a b MLIR.API.ifrt_MemoryKindsAreEqual(a.ptr, b.ptr)
end

function Base.string(memory_kind::MemoryKind)
    isempty(memory_kind) && return "<null>"
    str = GC.@preserve memory_kind MLIR.API.ifrt_MemoryKindToString(memory_kind.ptr)
    return XLA.unsafe_string_and_free(str)
end

function Base.show(io::IO, memory_kind::MemoryKind)
    print(io, "XLA.IFRT.MemoryKind(\"", string(memory_kind), "\")")
    return nothing
end
