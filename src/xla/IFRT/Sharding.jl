# xla::ifrt::HloSharding (distinct from xla::HloSharding)
mutable struct HloSharding
    ptr::Ptr{Cvoid}

    function HloSharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_hlo_module, new(ptr))
    end
end

function free_hlo_module(hlo_module::HloSharding)
    @ccall MLIR.API.mlir_c.free_ifrt_hlo_sharding(hlo_module.ptr::Ptr{Cvoid})::Cvoid
end

function HloSharding(device_list::BasicDeviceList, xla_hlo_sharding::XLA.HloSharding)
    default_memory_kind = convert(MemoryKind, XLA.default_memory(device_list))
    GC.@preserve device_list default_memory_kind xla_hlo_sharding begin
        return HloSharding(
            @ccall MLIR.API.mlir_c.ifrt_hlo_sharding_from_xla_hlo_sharding(
                device_list.ptr::Ptr{Cvoid},
                default_memory_kind.ptr::Ptr{Cvoid},
                xla_hlo_sharding.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", hlo_sharding::HloSharding)
    GC.@preserve hlo_sharding begin
        str = @ccall MLIR.API.mlir_c.ifrt_hlo_sharding_to_string(
            hlo_sharding.ptr::Ptr{Cvoid}
        )::Cstring
    end
    print(io, "XLA.IFRT.HloSharding(\"", XLA.unsafe_string_and_free(str), "\")")
    return nothing
end
