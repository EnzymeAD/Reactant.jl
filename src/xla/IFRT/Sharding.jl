# xla::ifrt::HloSharding (distinct from xla::HloSharding)
mutable struct HloSharding
    ptr::Ptr{Cvoid}

    function HloSharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_hlo_sharding, new(ptr))
    end
end

function free_hlo_sharding(hlo_sharding::HloSharding)
    @ccall MLIR.API.mlir_c.free_ifrt_hlo_sharding(hlo_sharding.ptr::Ptr{Cvoid})::Cvoid
end

function HloSharding(device_list::BasicDeviceList, xla_hlo_sharding::XLA.HloSharding)
    default_memory_kind = convert(MemoryKind, XLA.default_memory(device_list))
    GC.@preserve device_list default_memory_kind xla_hlo_sharding begin
        return HloSharding(
            @ccall MLIR.API.mlir_c.ifrt_hlo_sharding_from_xla_hlo_sharding(
                device_list.ptr::Ptr{Cvoid},
                default_memory_kind.ptr::Ptr{Cvoid},
                xla_hlo_sharding.ptr::Ptr{Cvoid},
            )::Ptr{Cvoid}
        )
    end
end

function Base.string(hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        str = @ccall MLIR.API.mlir_c.ifrt_hlo_sharding_to_string(
            hlo_sharding.ptr::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function Base.show(io::IO, ::MIME"text/plain", hlo_sharding::HloSharding)
    print(io, "XLA.IFRT.HloSharding(\"", string(hlo_sharding), "\")")
    return nothing
end

# HloSharding is more specific than Sharding. But Sharding is a neater way to deal with
# most of the IFRT APIs.
mutable struct Sharding
    ptr::Ptr{Cvoid}

    function Sharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_sharding, new(ptr))
    end
end

function Sharding(device_list::BasicDeviceList, xla_hlo_sharding::XLA.HloSharding)
    return convert(Sharding, HloSharding(device_list, xla_hlo_sharding))
end

function free_sharding(sharding::Sharding)
    @ccall MLIR.API.mlir_c.free_ifrt_sharding(sharding.ptr::Ptr{Cvoid})::Cvoid
end

function Base.convert(::Type{Sharding}, hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        return Sharding(
            @ccall MLIR.API.mlir_c.ifrt_sharding_from_ifrt_hlo_sharding(
                hlo_sharding.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function Base.string(sharding::Sharding)
    GC.@preserve sharding begin
        str = @ccall MLIR.API.mlir_c.ifrt_sharding_to_string(
            sharding.ptr::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function Base.show(io::IO, ::MIME"text/plain", sharding::Sharding)
    print(io, "XLA.IFRT.Sharding(\"", string(sharding), "\")")
    return nothing
end
