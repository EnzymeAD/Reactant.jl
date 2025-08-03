mutable struct Array <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Array(buffer::Ptr{Cvoid}, owned::Bool=true)
        !owned && return new(buffer)
        return finalizer(XLA.free_buffer, new(buffer))
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
    all_logical_device_ids = collect(Int64, 0:(length(all_devices) - 1))
    hlo_sharding = convert(XLA.HloSharding, sharding)

    slices, _ = XLA.sharding_to_concrete_array_indices(
        hlo_sharding, size(array), all_logical_device_ids
    )

    seen_slice = Dict{NTuple{N,UnitRange{Int64}},Int}()
    host_buffers = Base.Array{T,N}[]
    addressable_shard_indices = Vector{Int64}[]

    cur_shard = 0
    for (slice, device) in zip(slices, all_devices)
        XLA.is_addressable(device) || continue

        if haskey(seen_slice, slice)
            idx = seen_slice[slice]
            push!(addressable_shard_indices[idx], cur_shard)
        else
            host_buffer = let slice = array[slice...]
                slice isa Number ? collect(slice) : slice
            end
            push!(host_buffers, host_buffer)
            push!(addressable_shard_indices, Int64[cur_shard])
            seen_slice[slice] = length(host_buffers)
        end

        cur_shard += 1
    end

    return Array(client, host_buffers, addressable_shard_indices, size(array), sharding)
end

function Array(
    client::Client,
    host_buffers::Vector{Base.Array{T,N}},
    addressable_shard_indices::Vector{Vector{Int64}},
    array_shape,
    sharding::Sharding,
) where {T<:Reactant.ReactantPrimitive,N}
    # Construct using the slower path, the faster path is only implemented for IFRT-Proxy
    # and seems to cause issues with IFRT-PJRT
    all_addressable_devices = filter(XLA.is_addressable, XLA.devices(sharding))

    @assert !isempty(all_addressable_devices) "`IFRT.Array` requires atleast one \
                                               addressable device per process."

    single_device_arrays = Vector{Ptr{Nothing}}(
        undef, sum(length, addressable_shard_indices)
    )
    for (i, addr_shard_idxs) in enumerate(addressable_shard_indices)
        for addr_shard_idx in addr_shard_idxs
            idx = addr_shard_idx + 1
            device = all_addressable_devices[idx]
            single_device_arrays[idx] = Array(client, host_buffers[i], device).buffer
        end
    end

    array_shape = collect(Int64, reverse(array_shape))

    buffer = GC.@preserve client single_device_arrays array_shape sharding begin
        @ccall MLIR.API.mlir_c.ifrt_client_assemble_array_from_single_shards(
            client.client::Ptr{Cvoid},
            length(array_shape)::Int32,
            array_shape::Ptr{Int64},
            sharding.ptr::Ptr{Cvoid},
            length(single_device_arrays)::Int32,
            single_device_arrays::Ptr{Ptr{Cvoid}},
            2::Int32, # kDonateInput
        )::Ptr{Cvoid}
    end

    # host_buffer_shapes = Vector{Vector{Int64}}(undef, length(host_buffers))
    # addressable_shard_indices_sizes = Vector{Int64}(undef, length(host_buffers))

    # for (i, host_buffer) in enumerate(host_buffers)
    #     host_buffer_shapes[i] = collect(Int64, reverse(size(host_buffer)))
    #     addressable_shard_indices_sizes[i] = length(addressable_shard_indices[i])
    # end

    # array_shape = collect(Int64, reverse(array_shape))

    # buffer = GC.@preserve client host_buffers host_buffer_shapes addressable_shard_indices addressable_shard_indices_sizes array_shape sharding begin
    #     @ccall MLIR.API.mlir_c.ifrt_make_array_from_host_buffer_shards(
    #         client.client::Ptr{Cvoid},
    #         host_buffers::Ptr{Ptr{Cvoid}},
    #         length(host_buffers)::Cint,
    #         host_buffer_shapes::Ptr{Ptr{Int64}},
    #         addressable_shard_indices::Ptr{Ptr{Int64}},
    #         addressable_shard_indices_sizes::Ptr{Int64},
    #         XLA.primitive_type(T)::Cint,
    #         N::Cint,
    #         array_shape::Ptr{Int64},
    #         sharding.ptr::Ptr{Cvoid},
    #         0::Cint,
    #     )::Ptr{Cvoid}
    # end

    return Array(buffer)
end

function Array(
    client::Client, array::Base.Array{T,N}, sharding
) where {T<:Reactant.ReactantPrimitive,N}
    @assert sharding isa Reactant.Sharding.AbstractSharding
    if !(sharding isa Reactant.Sharding.HloSharding)
        sharding = Reactant.Sharding.HloSharding(sharding, size(array))
    end

    (; hlo_sharding, mesh) = sharding
    devices = XLA.get_device.((client,), mesh.device_ids)
    ifrt_sharding = Sharding([devices...], hlo_sharding)

    return Array(client, array, ifrt_sharding)
end

@inline function XLA.free_buffer(buffer::Array)
    if buffer.buffer != C_NULL
        @ccall MLIR.API.mlir_c.ifrt_free_array(buffer.buffer::Ptr{Cvoid})::Cvoid
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
        (; hlo_sharding) = reactant_sharding
    else
        (; hlo_sharding) = Reactant.Sharding.HloSharding(reactant_sharding, size(data))
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
            0::Int32,
            c_single_device_shard_semantics::Int32,
            narrays::Ptr{Int32},
        )::Ptr{Ptr{Cvoid}}
    end
    return [Array(unsafe_load(arrays, i)) for i in 1:narrays[]]
end

function replicate_array_to_all_devices(array::Array, sharding, mesh, size_arr)
    is_fully_replicated(XLA.sharding(array)) && return array

    if sharding isa Reactant.Sharding.AbstractSharding
        (; hlo_sharding) = Reactant.Sharding.HloSharding(sharding, size(array))
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

    XLA.is_replicated(hlo_sharding) && return array

    output_sharding = Reactant.Sharding.NamedSharding(
        mesh, ntuple(Returns(nothing), length(size_arr))
    )

    # Manually write the MLIR for resharding resharding
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    Reactant.Compiler.context_gc_vector[ctx] = Vector{
        Union{Reactant.TracedRArray,Reactant.TracedRNumber}
    }(
        undef, 0
    )
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
    MLIR.IR.activate!(ctx)

    sdycache = Reactant.Compiler.default_sdycache()
    Reactant.Compiler.activate_sdycache!(sdycache)
    output_buffer = try
        data_mlir_type = [
            MLIR.IR.TensorType(
                collect(Int, reverse(size_arr)), MLIR.IR.Type(eltype(array))
            ),
        ]
        mod = MLIR.IR.Module(MLIR.IR.Location(; context=ctx))

        (; sym_name, mesh_attr) = Reactant.Ops.mesh(mesh; mod)
        common_args = (ctx, sym_name, mesh_attr, size_arr)
        common_kwargs = (; dialect=:sdy, do_transpose=true)
        input_tensor_sharding_attr, _ = Reactant.Sharding.get_tensor_sharding_attribute(
            reactant_sharding, common_args...; common_kwargs...
        )
        output_tensor_sharding_attr, _ = Reactant.Sharding.get_tensor_sharding_attribute(
            output_sharding, common_args...; common_kwargs...
        )

        func = MLIR.Dialects.func.func_(;
            sym_name="main",
            function_type=MLIR.IR.FunctionType(data_mlir_type, data_mlir_type),
            no_inline=true,
            body=MLIR.IR.Region(),
        )
        fnbody = MLIR.IR.Block(data_mlir_type, [MLIR.IR.Location()])
        push!(MLIR.IR.region(func, 1), fnbody)
        MLIR.IR.activate!(fnbody)
        try
            MLIR.Dialects.func.return_([MLIR.IR.argument(fnbody, 1)])
        finally
            MLIR.IR.deactivate!(fnbody)
        end
        push!(MLIR.IR.body(mod), func)

        MLIR.API.mlirFuncSetArgAttr(func, 0, "sdy.sharding", input_tensor_sharding_attr)
        MLIR.API.mlirFuncSetResultAttr(func, 0, "sdy.sharding", output_tensor_sharding_attr)

        Reactant.Compiler.run_pass_pipeline!(
            mod,
            join(
                ["sdy-propagation-pipeline", "sdy-close-shardings", "canonicalize", "cse"],
                ",",
            ),
        )

        exec = XLA.compile(
            XLA.client(array),
            nothing,
            mod;
            is_sharded=true,
            global_device_ids=vec(mesh.device_ids),
            num_replicas=1,
            num_partitions=length(mesh.device_ids),
            num_outputs=1,                # unused
            num_parameters=1,             # unused
            use_shardy_partitioner=true,  # unused
        )

        only(XLA.execute(exec, (array.buffer,), (UInt8(0),), Val(1)))
    finally
        Reactant.Compiler.deactivate_sdycache!(sdycache)
        MLIR.IR.deactivate!(ctx)
    end
    delete!(Reactant.Compiler.context_gc_vector, ctx)

    return output_buffer
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

function copy_arrays_to_device_with_sharding(buffers::Vector{Array}, sharding::Sharding)
    ifrt_client = XLA.client(first(buffers)) # TODO: check all clients are the same?
    src_buffers = [buffer.buffer for buffer in buffers]
    GC.@preserve buffers ifrt_client begin
        dst_buffers = @ccall MLIR.API.mlir_c.ifrt_copy_arrays_to_device_with_sharding(
            ifrt_client.client::Ptr{Cvoid},
            src_buffers::Ptr{Ptr{Cvoid}},
            length(buffers)::Int32,
            sharding.ptr::Ptr{Cvoid},
            0::Cint, # kAlwaysCopy
        )::Ptr{Ptr{Cvoid}}
    end
    dst_arrays = Vector{Array}(undef, length(buffers))
    for i in 1:length(buffers)
        dst_arrays[i] = Array(unsafe_load(dst_buffers, i))
    end
    return dst_arrays
end

function Base.copy(b::Array)
    GC.@preserve b begin
        return Array(
            @ccall MLIR.API.mlir_c.ifrt_copy_array(b.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end
