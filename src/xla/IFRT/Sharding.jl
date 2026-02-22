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
    return Sharding(
        MLIR.API.ifrt_sharding_from_xla_hlo_sharding(
            client.client,
            [d.device for d in device_list],
            length(device_list),
            memory_kind.ptr,
            xla_hlo_sharding.ptr,
        ),
    )
end

function free_sharding(sharding::Sharding)
    return MLIR.API.free_ifrt_sharding(sharding.ptr)
end

function XLA.num_devices(sharding::Sharding)
    return MLIR.API.ifrt_sharding_devices_size(sharding.ptr)
end

function XLA.devices(sharding::Sharding)
    ndevices = XLA.num_devices(sharding)
    devices = Ref{NTuple{Int64(ndevices),Ptr{Cvoid}}}()
    MLIR.API.ifrt_sharding_to_device_list(sharding.ptr, devices)
    return map(Device, devices[])
end

function Base.convert(::Type{XLA.HloSharding}, sharding::Sharding)
    return XLA.HloSharding(MLIR.API.ifrt_sharding_to_xla_hlo_sharding(sharding.ptr))
end

function Base.string(sharding::Sharding)
    str = MLIR.API.ifrt_sharding_to_string(sharding.ptr)
    return XLA.unsafe_string_and_free(str)
end

function is_fully_replicated(sharding::Sharding)
    return MLIR.API.ifrt_sharding_is_fully_replicated(sharding.ptr)
end

function is_single_device_sharding(sharding::Sharding)
    return MLIR.API.ifrt_sharding_is_single_device_sharding(sharding.ptr)
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

    MLIR.API.ifrt_sharding_to_index_domains(
        sharding.ptr, shape, length(shape), index_domain_origins, index_domain_shapes
    )

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
