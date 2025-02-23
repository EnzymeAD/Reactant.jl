# xla::ifrt::HloSharding (distinct from xla::HloSharding)
mutable struct HloSharding
    ptr::Ptr{Cvoid}

    function HloSharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        # return finalizer(free_hlo_sharding, new(ptr))
        return new(ptr)
    end
end

function free_hlo_sharding(hlo_sharding::HloSharding)
    @ccall MLIR.API.mlir_c.free_ifrt_hlo_sharding(hlo_sharding.ptr::Ptr{Cvoid})::Cvoid
end

function Base.convert(::Type{XLA.HloSharding}, sharding::HloSharding)
    GC.@preserve sharding begin
        return XLA.HloSharding(
            @ccall MLIR.API.mlir_c.ifrt_hlo_sharding_to_xla_hlo_sharding(
                sharding.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function HloSharding(
    device_list::AbstractVector{<:Device}, xla_hlo_sharding::XLA.HloSharding
)
    addressable_devices = filter(XLA.is_addressable, device_list)
    default_memory_kind = convert(MemoryKind, XLA.default_memory(addressable_devices))
    return HloSharding(device_list, xla_hlo_sharding, default_memory_kind)
end

function HloSharding(
    device_list::AbstractVector{<:Device},
    xla_hlo_sharding::XLA.HloSharding,
    memoy_kind::AbstractString,
)
    return HloSharding(device_list, xla_hlo_sharding, MemoryKind(memoy_kind))
end

function HloSharding(
    device_list::AbstractVector{<:Device},
    xla_hlo_sharding::XLA.HloSharding,
    memory_kind::MemoryKind,
)
    client = XLA.client(device_list)
    GC.@preserve device_list memory_kind xla_hlo_sharding client begin
        return HloSharding(
            @ccall MLIR.API.mlir_c.ifrt_hlo_sharding_from_xla_hlo_sharding(
                client.client::Ptr{Cvoid},
                [d.device for d in device_list]::Ptr{Ptr{Cvoid}},
                length(device_list)::Int32,
                memory_kind.ptr::Ptr{Cvoid},
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
        # return finalizer(free_sharding, new(ptr))
        return new(ptr)
    end
end

function Sharding(device_list::AbstractVector{<:Device}, xla_hlo_sharding::XLA.HloSharding)
    return convert(Sharding, HloSharding(device_list, xla_hlo_sharding))
end

function Sharding(
    device_list::AbstractVector{<:Device},
    xla_hlo_sharding::XLA.HloSharding,
    memoy_kind::Union{AbstractString,MemoryKind},
)
    return convert(Sharding, HloSharding(device_list, xla_hlo_sharding, memoy_kind))
end

function free_sharding(sharding::Sharding)
    @ccall MLIR.API.mlir_c.free_ifrt_sharding(sharding.ptr::Ptr{Cvoid})::Cvoid
end

function XLA.devices(sharding::Sharding)
    GC.@preserve sharding begin
        ndevices = @ccall MLIR.API.mlir_c.ifrt_sharding_devices_size(
            sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
    devices = Ref{NTuple{Int64(ndevices),Ptr{Cvoid}}}()
    GC.@preserve sharding devices begin
        @ccall MLIR.API.mlir_c.ifrt_sharding_to_device_list(
            sharding.ptr::Ptr{Cvoid}, devices::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    return [Device(device) for device in devices[]]
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

function Base.convert(::Type{HloSharding}, sharding::Sharding)
    GC.@preserve sharding begin
        return HloSharding(
            @ccall MLIR.API.mlir_c.ifrt_sharding_to_ifrt_hlo_sharding(
                sharding.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function Base.convert(::Type{XLA.HloSharding}, sharding::Sharding)
    return convert(XLA.HloSharding, convert(HloSharding, sharding))
end

function Base.string(sharding::Sharding)
    GC.@preserve sharding begin
        str = @ccall MLIR.API.mlir_c.ifrt_sharding_to_string(
            sharding.ptr::Ptr{Cvoid}
        )::Cstring
    end
    return XLA.unsafe_string_and_free(str)
end

function is_fully_replicated(sharding::Sharding)
    GC.@preserve sharding begin
        return @ccall MLIR.API.mlir_c.ifrt_sharding_is_fully_replicated(
            sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function is_single_device_sharding(sharding::Sharding)
    GC.@preserve sharding begin
        return @ccall MLIR.API.mlir_c.ifrt_sharding_is_single_device_sharding(
            sharding.ptr::Ptr{Cvoid}
        )::Bool
    end
end

function Base.show(io::IO, ::MIME"text/plain", sharding::Sharding)
    print(io, "XLA.IFRT.Sharding(\"", string(sharding), "\")")
    return nothing
end
