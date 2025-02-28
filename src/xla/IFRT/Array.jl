mutable struct Array <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Array(buffer::Ptr{Cvoid})
        # return finalizer(free_ifrt_array, new(buffer))
        return new(buffer)
    end
end

function Array(
    client::Client,
    array::Base.Array{T,N},
    device::Device=XLA.default_device(client),
    memory_kind::AbstractString=string(convert(MemoryKind, XLA.default_memory(device))),
) where {T,N}
    sizear = collect(Int64, reverse(size(array)))
    buffer = GC.@preserve array sizear begin
        @ccall MLIR.API.mlir_c.ifrt_client_make_single_shard_array_from_host_buffer(
            client.client::Ptr{Cvoid},
            array::Ptr{T},
            XLA.primitive_type(T)::UInt64,
            N::Csize_t,
            sizear::Ptr{Int64},
            0::Cint, # kAlwaysCopy
            device.device::Ptr{Cvoid},
            string(memory_kind)::Cstring,
        )::Ptr{Cvoid}
    end
    return Array(buffer)
end

function Array(client::Client, array::Base.Array{T,N}, sharding::Sharding) where {T,N}
    sizear = collect(Int64, reverse(size(array)))

    if is_single_device_sharding(sharding) || is_fully_replicated(sharding)
        buffer = GC.@preserve array sizear begin
            @ccall MLIR.API.mlir_c.ifrt_client_make_array_from_host_buffer(
                client.client::Ptr{Cvoid},
                array::Ptr{T},
                XLA.primitive_type(T)::Cint,
                N::Csize_t,
                sizear::Ptr{Int64},
                sharding.ptr::Ptr{Cvoid},
                0::Cint, # kAlwaysCopy
            )::Ptr{Cvoid}
        end
        return Array(buffer)
    end

    all_devices = XLA.devices(sharding)
    array_slices, _ = XLA.sharding_to_concrete_array_indices(
        convert(XLA.HloSharding, sharding),
        size(array),
        collect(Int64, 0:(length(all_devices) - 1)),
    )
    array_shape = collect(Int64, reverse(size(array)))
    arrays_list = [
        Array(client, array[slice...], device).buffer for
        (device, slice) in zip(all_devices, array_slices) if XLA.is_addressable(device)
    ]

    buffer = GC.@preserve client arrays_list array_shape sharding begin
        @ccall MLIR.API.mlir_c.ifrt_client_assemble_array_from_single_shards(
            client.client::Ptr{Cvoid},
            Int32(length(array_shape))::Int32,
            array_shape::Ptr{Int64},
            sharding.ptr::Ptr{Cvoid},
            Int32(length(arrays_list))::Int32,
            arrays_list::Ptr{Ptr{Cvoid}},
            2::Cint, # kDonateInput
        )::Ptr{Cvoid}
    end

    return Array(buffer)
end

function Array(client::Client, array::Base.Array{T,N}, sharding) where {T,N}
    @assert sharding isa Reactant.Sharding.AbstractSharding
    if !(sharding isa Reactant.Sharding.HloSharding)
        sharding = convert(Reactant.Sharding.HloSharding, sharding)
    end

    (; hlo_sharding, mesh) = sharding
    devices = XLA.get_device.((client,), mesh.device_ids)
    ifrt_sharding = Sharding([devices...], hlo_sharding)

    return Array(client, array, ifrt_sharding)
end

@inline function free_ifrt_array(buffer::Array)
    sbuffer = buffer.buffer
    if sbuffer != C_NULL
        @ccall MLIR.API.mlir_c.ifrt_free_array(sbuffer::Ptr{Cvoid})::Cvoid
    end
end

function Base.ndims(buffer::Array)
    GC.@preserve buffer begin
        return @ccall MLIR.API.mlir_c.ifrt_array_ndims(buffer.buffer::Ptr{Cvoid})::Int64
    end
end

function Base.size(buffer::Array)
    GC.@preserve buffer begin
        sz = @ccall MLIR.API.mlir_c.ifrt_array_shape(buffer.buffer::Ptr{Cvoid})::Ptr{Int64}
    end
    return Tuple(unsafe_wrap(Base.Array, sz, ndims(buffer)))
end

function Base.eltype(buffer::Array)
    GC.@preserve buffer begin
        return XLA.julia_type(
            @ccall MLIR.API.mlir_c.ifrt_array_eltype(buffer.buffer::Ptr{Cvoid})::Cint
        )
    end
end

function XLA.device(::Array)
    return error("IFRT.Array can be sharded/replicated across multiple devices. Hence, \
                  `XLA.device` is not defined.")
end

function XLA.client(buffer::Array)
    GC.@preserve buffer begin
        return Client(
            @ccall MLIR.API.mlir_c.ifrt_array_to_client(
                buffer.buffer::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

XLA.synced_buffer(buffer::Array) = buffer

function XLA.buffer_on_cpu(::Array)
    return error("IFRT.Array does not support `XLA.buffer_on_cpu`")
end

function XLA.to_host(buffer::Array, data)
    sharding = XLA.sharding(buffer)
    all_devices = XLA.devices(sharding)

    if length(all_devices) == 1
        GC.@preserve buffer data begin
            @ccall MLIR.API.mlir_c.ifrt_array_copy_to_host_buffer(
                buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
            )::Cvoid
        end
        return nothing
    end

    if any(!is_addressable, all_devices)
        @warn "Not all devices are addressable. Currently we only fill in the data for \
               addressable devices. Remaining slices of data in `data` are left \
               untouched."
    end

    # While some client implementations might support directly copying to host, but we 
    # avoid the complexity of supporting that for now.
    single_device_arrays = disassemble_into_single_device_arrays(buffer, true)

    array_slices, _ = XLA.sharding_to_concrete_array_indices(
        convert(XLA.HloSharding, sharding),
        size(data),
        collect(Int64, 0:(length(all_devices) - 1)),
    )
    array_slices = [
        slice for
        (device, slice) in zip(all_devices, array_slices) if XLA.is_addressable(device)
    ]

    @assert length(array_slices) == length(single_device_arrays)

    for (slice, arr) in zip(array_slices, single_device_arrays)
        data_slice = data[slice...]
        XLA.to_host(arr, data_slice)
        data[slice...] .= data_slice
    end
    return nothing
end

function disassemble_into_single_device_arrays(array::Array, only_addressable_devices::Bool)
    c_single_device_shard_semantics = Int32(!only_addressable_devices)
    narrays = Ref{Int32}(0)
    arrays = GC.@preserve array begin
        @ccall MLIR.API.mlir_c.ifrt_array_disassemble_into_single_device_arrays(
            array.buffer::Ptr{Cvoid},
            Int32(0)::Int32,
            c_single_device_shard_semantics::Int32,
            narrays::Ptr{Int32},
        )::Ptr{Ptr{Cvoid}}
    end
    return [Array(unsafe_load(arrays, i)) for i in 1:narrays[]]
end

function XLA.unsafe_buffer_pointer(::Array)
    return error("IFRT.Array does not support `XLA.unsafe_buffer_pointer`")
end

function XLA.copy_buffer_to_device(::Array, ::Device)
    return error("IFRT.Array does not support `XLA.copy_buffer_to_device`")
end

function XLA.sharding(buffer::Array)
    GC.@preserve buffer begin
        return Sharding(
            @ccall MLIR.API.mlir_c.ifrt_array_to_sharding(
                buffer.buffer::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end
