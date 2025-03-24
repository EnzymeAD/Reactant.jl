mutable struct Array <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Array(buffer::Ptr{Cvoid})
        # return finalizer(free_ifrt_array, new(buffer))
        return new(buffer)
    end
end

function Array(
    client::Client,
    array::Reactant.ReactantPrimitive,
    device::Device=XLA.default_device(client),
    memory_kind::AbstractString=string(convert(MemoryKind, XLA.default_memory(device))),
)
    return Array(client, fill(array), device, memory_kind)
end

function Array(
    client::Client,
    array::Base.Array{T,N},
    device::Device=XLA.default_device(client),
    memory_kind::AbstractString=string(convert(MemoryKind, XLA.default_memory(device))),
) where {T<:Reactant.ReactantPrimitive,N}
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

function Array(
    client::Client, array::Base.Array{T,N}, sharding::Sharding
) where {T<:Reactant.ReactantPrimitive,N}
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

function Array(
    client::Client, array::Base.Array{T,N}, sharding
) where {T<:Reactant.ReactantPrimitive,N}
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

function XLA.device(buffer::Array)
    devices = XLA.devices(XLA.sharding(buffer))
    length(devices) == 1 && return only(devices)
    return nothing
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

function XLA.to_host(buffer::Array, data, reactant_sharding)
    reactant_sharding = Reactant.Sharding.unwrap_shardinfo(reactant_sharding)

    # While some client implementations might support directly copying to host, but we
    # avoid the complexity of supporting that for now.
    single_device_arrays = disassemble_into_single_device_arrays(buffer, true)

    if reactant_sharding isa Reactant.Sharding.NoSharding
        data_buffer = first(single_device_arrays)
        data_buffer_shape = reverse(size(data_buffer))
        @assert size(data) == data_buffer_shape "Expected data to be of size \
                                                 $(size(data)), got $(data_buffer_shape)"
        GC.@preserve data_buffer data begin
            @ccall MLIR.API.mlir_c.ifrt_array_copy_to_host_buffer(
                data_buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
            )::Cvoid
        end
        return data
    end

    if reactant_sharding isa Reactant.Sharding.HloSharding
        hlo_sharding = reactant_sharding.hlo_sharding
    else
        hlo_sharding =
            convert(Reactant.Sharding.HloSharding, reactant_sharding).hlo_sharding
    end

    client = XLA.client(buffer)
    all_devices = XLA.get_device.((client,), reactant_sharding.mesh.device_ids)

    if any(XLA.is_addressable, all_devices)
        # Take a fast path if all devices are addressable
        array_slices, _ = XLA.sharding_to_concrete_array_indices(
            convert(XLA.CondensedOpSharding, hlo_sharding),
            size(data),
            reactant_sharding.mesh.logical_device_ids,
        )
        array_slices = [
            slice for
            (device, slice) in zip(all_devices, array_slices) if XLA.is_addressable(device)
        ]

        @assert length(array_slices) == length(single_device_arrays)

        for (slice, arr) in zip(array_slices, single_device_arrays)
            data_slice = data isa Base.RefValue ? data : data[slice...]
            XLA.to_host(arr, data_slice, Reactant.Sharding.NoSharding())
            data isa Base.RefValue || (data[slice...] .= data_slice)
        end
    end

    # Here we need to copy data from all the processes to the host
    arr = replicate_array_to_all_devices(
        buffer, reactant_sharding, reactant_sharding.mesh, size(data)
    )
    XLA.to_host(arr, data, Reactant.Sharding.NoSharding())

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

function replicate_array_to_all_devices(array::Array, sharding, mesh, size_arr)
    is_fully_replicated(XLA.sharding(array)) && return array

    if sharding isa Reactant.Sharding.AbstractSharding
        hlo_sharding = convert(Reactant.Sharding.HloSharding, sharding).hlo_sharding
        reactant_sharding = sharding
    else
        hlo_sharding = convert(XLA.HloSharding, sharding)
        reactant_sharding = Reactant.Sharding.HloSharding(
            hlo_sharding,
            mesh,
            ntuple(Returns(1), length(size_arr)),
            ntuple(Returns(-1), length(size_arr)),
        )
    end

    shard_info = Reactant.Sharding.ShardInfo(
        reactant_sharding,
        Reactant.Sharding.sharding_to_array_slices(reactant_sharding, size_arr),
    )
    sharding_constraint = Reactant.Sharding.NamedSharding(
        mesh, ntuple(Returns(nothing), length(size_arr))
    )

    data = Reactant.ConcreteIFRTArray{eltype(array),length(size_arr),typeof(shard_info)}(
        AsyncArray(array, nothing), size_arr, shard_info
    )

    # TODO: Directly write the MLIR for this part??
    fn_compiled = Reactant.compile(
        identity,
        (data,);
        shardy_passes=:to_mhlo_shardings,
        optimize=false,
        output_shardings=Dict(1 => sharding_constraint),
    )

    return fn_compiled(data).data.buffer
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
