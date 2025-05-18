mutable struct Memory <: XLA.AbstractMemory
    ptr::Ptr{Cvoid}
end

function Base.show(io::IO, memory::Memory)
    GC.@preserve memory begin
        str = @ccall MLIR.API.mlir_c.ifrt_MemoryToString(memory.ptr::Ptr{Cvoid})::Cstring
    end
    print(io, "XLA.IFRT.Memory(\"", XLA.unsafe_string_and_free(str), "\")")
    return nothing
end

mutable struct MemoryKind <: XLA.AbstractMemoryKind
    ptr::Ptr{Cvoid}
end

function MemoryKind(::Nothing)
    return MemoryKind(
        @ccall MLIR.API.mlir_c.ifrt_memory_kind_with_optional_memory_space()::Ptr{Cvoid}
    )
end

function MemoryKind(str::AbstractString)
    str = string(str)
    GC.@preserve str begin
        return MemoryKind(
            @ccall MLIR.API.mlir_c.ifrt_memory_kind_from_string(str::Cstring)::Ptr{Cvoid}
        )
    end
end

function Base.isempty(memory_kind::MemoryKind)
    GC.@preserve memory_kind begin
        has_value = @ccall MLIR.API.mlir_c.ifrt_memory_kind_has_value(
            memory_kind.ptr::Ptr{Cvoid}
        )::Bool
    end
    return !has_value
end

function Base.convert(::Type{MemoryKind}, memory::Memory)
    GC.@preserve memory begin
        return MemoryKind(
            @ccall MLIR.API.mlir_c.ifrt_MemoryGetMemoryKind(
                memory.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function Base.:(==)(a::MemoryKind, b::MemoryKind)
    GC.@preserve a b begin
        return @ccall MLIR.API.mlir_c.ifrt_MemoryKindsAreEqual(
            a.ptr::Ptr{Cvoid}, b.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function Base.string(memory_kind::MemoryKind)
    isempty(memory_kind) && return "<null>"
    GC.@preserve memory_kind begin
        str = @ccall MLIR.API.mlir_c.ifrt_MemoryKindToString(
            memory_kind.ptr::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function Base.show(io::IO, memory_kind::MemoryKind)
    print(io, "XLA.IFRT.MemoryKind(\"", string(memory_kind), "\")")
    return nothing
end
