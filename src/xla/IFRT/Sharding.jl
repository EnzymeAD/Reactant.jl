mutable struct Sharding
    ptr::Ptr{Cvoid}

    function Sharding(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_sharding, new(ptr))
    end
end

function Sharding(
    device_list::AbstractVector{<:Device},
    xla_hlo_sharding::XLA.HloSharding,
    memory_kind::Union{AbstractString,MemoryKind,Nothing}=nothing,
)
    !(memory_kind isa MemoryKind) && (memory_kind = MemoryKind(memory_kind))
    client = XLA.client(device_list)
    GC.@preserve device_list memory_kind xla_hlo_sharding client begin
        return Sharding(
            @ccall MLIR.API.mlir_c.ifrt_sharding_from_xla_hlo_sharding(
                client.client::Ptr{Cvoid},
                [d.device for d in device_list]::Ptr{Ptr{Cvoid}},
                length(device_list)::Int32,
                memory_kind.ptr::Ptr{Cvoid},
                xla_hlo_sharding.ptr::Ptr{Cvoid},
            )::Ptr{Cvoid}
        )
    end
end

function free_sharding(sharding::Sharding)
    @ccall MLIR.API.mlir_c.free_ifrt_sharding(sharding.ptr::Ptr{Cvoid})::Cvoid
end

function XLA.num_devices(sharding::Sharding)
    GC.@preserve sharding begin
        return @ccall MLIR.API.mlir_c.ifrt_sharding_devices_size(
            sharding.ptr::Ptr{Cvoid}
        )::Int32
    end
end

function XLA.devices(sharding::Sharding)
    ndevices = XLA.num_devices(sharding)
    devices = Ref{NTuple{Int64(ndevices),Ptr{Cvoid}}}()
    GC.@preserve sharding devices begin
        @ccall MLIR.API.mlir_c.ifrt_sharding_to_device_list(
            sharding.ptr::Ptr{Cvoid}, devices::Ptr{Ptr{Cvoid}}
        )::Cvoid
    end
    return map(Device, devices[])
end

function Base.convert(::Type{XLA.HloSharding}, sharding::Sharding)
    GC.@preserve sharding begin
        return XLA.HloSharding(
            @ccall MLIR.API.mlir_c.ifrt_sharding_to_xla_hlo_sharding(
                sharding.ptr::Ptr{Cvoid}
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

function XLA.sharding_to_concrete_array_indices(
    sharding::Sharding, shape, logical_device_ids
)
    shape = collect(Int64, shape)
    reverse!(shape)

    index_domain_origins = Vector{Int64}(undef, length(logical_device_ids) * length(shape))
    index_domain_shapes = Vector{Int64}(undef, length(logical_device_ids) * length(shape))

    GC.@preserve sharding index_domain_origins index_domain_shapes begin
        @ccall MLIR.API.mlir_c.ifrt_sharding_to_index_domains(
            sharding.ptr::Ptr{Cvoid},
            shape::Ptr{Int64},
            Int32(length(shape))::Int32,
            index_domain_origins::Ptr{Int64},
            index_domain_shapes::Ptr{Int64},
        )::Cvoid
    end

    needs_padding = false
    array_indices = Vector{NTuple{length(shape),UnitRange{Int64}}}(
        undef, length(logical_device_ids)
    )
    for i in logical_device_ids
        array_indices[i - 1] = ntuple(length(shape)) do j
            idx = i * length(shape) + j
            start_idx = index_domain_origins[idx] + 1
            stop_idx = start_idx + index_domain_shapes[idx] - 1
            !needs_padding && stop_idx > shape[j] && (needs_padding = true)
            return reverse(start_idx:stop_idx)
        end
    end

    return array_indices, needs_padding
end
