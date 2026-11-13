"""
    Client

## Interface

- `destroy!`
- `platform_name`
- `platform_version`
- `process_index`
- `devices`
- `addressable_devices`
- `lookup_device`
- `lookup_addressable_device`
- `addressable_memories`
- `buffer_from_host_buffer`
- `compile!`
- `deserialize_and_load`
- `default_device_assignment`
- `load`
- `create_error_buffer`
- `create_alias_buffer`
- `update_global_process_info!`
- `dma_map!`
- `dma_unmap!`
- `create_buffers_for_async_host_to_device`
- `topology_description`
"""
abstract type Client end

@pjrt_catch_return function pjrt_client_destroy(
    args::Ptr{CAPI.PJRT_Client_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    unroot!(client)
    return ok()
end

"""
    platform_name(client::Client) -> String

Return a string that identifies the platform of `client` (e.g. `"cpu"`, `"gpu"`, `"tpu"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `client`.

Backs `PJRT_Client_PlatformName`.
"""
platform_name(::Client)

@pjrt_catch_return function pjrt_client_platform_name(
    args::Ptr{CAPI.PJRT_Client_PlatformName_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    str = platform_name(client)::String
    unsafe_store!(args.platform_name, Cstring(pointer(str)))
    unsafe_store!(args.platform_name_size, sizeof(str))
    return ok()
end

"""
    platform_version(client::Client) -> String

Return a string containing the platform version of `client`.

!!! warning
The string is handed to C by pointer, so it must be owned by `client`.

Backs `PJRT_Client_PlatformVersion`.
"""
platform_version(::Client)

@pjrt_catch_return function pjrt_client_platform_version(
    args::Ptr{CAPI.PJRT_Client_PlatformVersion_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    str = platform_version(client)::String
    unsafe_store!(args.platform_version, Cstring(pointer(str)))
    unsafe_store!(args.platform_version_size, sizeof(str))
    return ok()
end

"""
    process_index(client::Client) -> Integer

Return the index of the process of `client`. Always `0` in single-process settings.

Backs `PJRT_Client_ProcessIndex`.
"""
process_index(::Client)

@pjrt_catch_return function pjrt_client_process_index(
    args::Ptr{CAPI.PJRT_Client_ProcessIndex_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    pid = process_index(client)
    unsafe_store!(args.process_index, pid)
    return ok()
end

"""
    devices(client::Client) -> Vector{Device}

Return all the devices visible to `client`, including those that it cannot address.

!!! warning
The vector is handed to C by pointer, so it must be owned by `client` and never be resized.

Backs `PJRT_Client_Devices`.
"""
devices(::Client)

@pjrt_catch_return function pjrt_client_devices(
    args::Ptr{CAPI.PJRT_Client_Devices_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    devs = devices(client)::Vector{Device}
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end

"""
    addressable_devices(client::Client) -> Vector{Device}

Return the subset of `devices(client)` that `client` can address.

!!! warning
The vector is handed to C by pointer, so it must be owned by `client` and never be resized.

Backs `PJRT_Client_AddressableDevices`.
"""
addressable_devices(::Client)

@pjrt_catch_return function pjrt_client_addressable_devices(
    args::Ptr{CAPI.PJRT_Client_AddressableDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    devs = addressable_devices(client)::Vector{Device}
    unsafe_store!(args.addressable_devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_addressable_devices, length(devs))
    return ok()
end

"""
    lookup_device(client::Client, id::Integer) -> Device

Return the device of `devices(client)` whose description has id `id` (see `id(::DeviceDescription)`), or throw
[`invalid_argument`](@ref) if there is none.

Defaults to a linear search over `devices(client)`.

Backs `PJRT_Client_LookupDevice`.
"""
function lookup_device(client::Client, device_id::Integer)
    devs = devices(client)::Vector{Device}
    i = findfirst(dev -> id(description(dev)) == device_id, devs)
    isnothing(i) && throw(invalid_argument("no device with id $device_id"))
    return devs[i]
end

@pjrt_catch_return function pjrt_client_lookup_device(
    args::Ptr{CAPI.PJRT_Client_LookupDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    dev = lookup_device(client, unsafe_load(args.id))::Device
    unsafe_store!(args.device, Ptr{CAPI.PJRT_Device}(pointer_from_objref(dev)))
    return ok()
end

"""
    lookup_addressable_device(client::Client, local_hardware_id::Integer) -> Device

Return the device of `addressable_devices(client)` whose `local_hardware_id` is `local_hardware_id`, or throw
[`invalid_argument`](@ref) if there is none.

Defaults to a linear search over `addressable_devices(client)`. Since `-1` means that the local hardware id is
undefined, looking it up always throws.

Backs `PJRT_Client_LookupAddressableDevice`.
"""
function lookup_addressable_device(client::Client, hardware_id::Integer)
    hardware_id == -1 && throw(invalid_argument("local hardware id -1 is undefined"))
    devs = addressable_devices(client)::Vector{Device}
    i = findfirst(dev -> local_hardware_id(dev) == hardware_id, devs)
    isnothing(i) &&
        throw(invalid_argument("no addressable device with local hardware id $hardware_id"))
    return devs[i]
end

@pjrt_catch_return function pjrt_client_lookup_addressable_device(
    args::Ptr{CAPI.PJRT_Client_LookupAddressableDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    dev = lookup_addressable_device(client, unsafe_load(args.local_hardware_id))::Device
    unsafe_store!(args.addressable_device, Ptr{CAPI.PJRT_Device}(pointer_from_objref(dev)))
    return ok()
end

"""
    addressable_memories(client::Client) -> Vector{Memory}

Return the memories that `client` can address.

!!! warning
The vector is handed to C by pointer, so it must be owned by `client` and never be resized.

Backs `PJRT_Client_AddressableMemories`.
"""
addressable_memories(::Client)

@pjrt_catch_return function pjrt_client_addressable_memories(
    args::Ptr{CAPI.PJRT_Client_AddressableMemories_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    mems = addressable_memories(client)::Vector{Memory}
    unsafe_store!(args.addressable_memories, Ptr{Ptr{CAPI.PJRT_Memory}}(pointer(mems)))
    unsafe_store!(args.num_addressable_memories, length(mems))
    return ok()
end

"""
    buffer_from_host_buffer(
        client::Client, dev::Device, mem::Memory, data::Ptr, type, dims, host_layout, device_layout,
        host_buffer_semantics,
    ) -> Buffer

Return a new buffer placed on memory `mem` of device `dev`, with a copy of the host `data`. The new buffer has element
type `type::CAPI.PJRT_Buffer_Type` and dimensions `dims::Vector{Int64}`.

## Arguments
- `mem` is always one of `addressable_memories(dev)`. PJRT may pass only one of the two, so the caller fills in the other:

  - only `dev` given: `mem` is `default_memory(dev)`.
  - only `mem` given: `dev` is one of `addressable_devices(mem)`.

- `host_layout::Union{Nothing,StridedMemoryLayout}` is the layout of `data`: `nothing` for dense data in major-to-minor
  order, or the byte strides of each dimension (which can be negative, if `data` points inside the array).
- `device_layout::Union{Nothing,MemoryLayout}` is `nothing` for the default layout.
- `host_buffer_semantics::CAPI.PJRT_HostBufferSemantics` says how long `data` stays valid.
- `done_with_host_buffer` is signaled as soon as this returns, so `data` must not be accessed afterwards.

The caller GC-roots the returned buffer.

Backs `PJRT_Client_BufferFromHostBuffer`.
"""
buffer_from_host_buffer(::Client, ::Device, ::Memory, args...)

function buffer_from_host_buffer(client::Client, mem::Memory, args...)
    # we select the first addressable device from the memory because there is no default
    return buffer_from_host_buffer(client, first(addressable_devices(mem)), mem, args...)
end

function buffer_from_host_buffer(client::Client, dev::Device, args...)
    return buffer_from_host_buffer(client, dev, default_memory(dev), args...)
end

@pjrt_catch_return function pjrt_client_buffer_from_host_buffer(
    args::Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client

    data = unsafe_load(args.data)
    _eltype = unsafe_load(args.type)

    _ndims = unsafe_load(args.num_dims)
    _dims_ptr = unsafe_load(args.dims)
    _size = copy(unsafe_wrap(Vector{Int64}, _dims_ptr, _ndims))

    # the byte strides of `data` are a strided layout; if there are none, `data` is dense in major-to-minor order
    byte_strides_ptr = unsafe_load(args.byte_strides)
    num_byte_strides = unsafe_load(args.num_byte_strides)
    host_layout = if byte_strides_ptr == C_NULL || num_byte_strides == 0
        nothing
    else
        num_byte_strides == _ndims || throw(
            invalid_argument("got $num_byte_strides byte strides for $_ndims dimensions"),
        )
        StridedMemoryLayout(
            copy(unsafe_wrap(Vector{Int64}, byte_strides_ptr, num_byte_strides))
        )
    end

    c_device_layout = unsafe_load(args.device_layout)
    _device_layout = c_device_layout == C_NULL ? nothing : MemoryLayout(c_device_layout)
    _host_buffer_semantics = unsafe_load(args.host_buffer_semantics)

    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    dev_ptr = Ptr{Device}(unsafe_load(args.device))
    if mem_ptr == C_NULL && dev_ptr == C_NULL
        throw(invalid_argument("No device nor memory has been assigned"))
    end

    buffer = if mem_ptr != C_NULL
        mem = unsafe_pointer_to_objref(mem_ptr)::Memory
        buffer_from_host_buffer(
            client,
            mem,
            data,
            _eltype,
            _size,
            host_layout,
            _device_layout,
            _host_buffer_semantics,
        )
    else # dev_ptr != C_NULL
        dev = unsafe_pointer_to_objref(dev_ptr)::Device
        buffer_from_host_buffer(
            client,
            dev,
            data,
            _eltype,
            _size,
            host_layout,
            _device_layout,
            _host_buffer_semantics,
        )
    end::Buffer

    event = root!(Event())
    notify(event)
    unsafe_store!(
        args.done_with_host_buffer, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event))
    )

    root!(buffer)
    unsafe_store!(args.buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(buffer)))
    return ok()
end

"""
    create_uninitialized_buffer(client::Client, dev::Device, mem::Memory, type, dims, layout) -> Buffer

Return a new buffer placed on memory `mem` of device `dev`, with element type `type::CAPI.PJRT_Buffer_Type` and
dimensions `dims::Vector{Int64}`, whose contents are left uninitialized.

## Arguments
- `mem` is always one of `addressable_memories(dev)`, filled in like in [`buffer_from_host_buffer`](@ref).
- `layout::Union{Nothing,MemoryLayout}` is `nothing` for the default layout.

The caller GC-roots the returned buffer.

Backs `PJRT_Client_CreateUninitializedBuffer`.
"""
create_uninitialized_buffer(::Client, ::Device, ::Memory, args...)

function create_uninitialized_buffer(client::Client, mem::Memory, args...)
    return create_uninitialized_buffer(
        client, first(addressable_devices(mem)), mem, args...
    )
end

function create_uninitialized_buffer(client::Client, dev::Device, args...)
    return create_uninitialized_buffer(client, dev, default_memory(dev), args...)
end

@pjrt_catch_return function pjrt_client_create_uninitialized_buffer(
    args::Ptr{CAPI.PJRT_Client_CreateUninitializedBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    _eltype = unsafe_load(args.shape_element_type)
    _size = copy(
        unsafe_wrap(
            Vector{Int64}, unsafe_load(args.shape_dims), unsafe_load(args.shape_num_dims)
        ),
    )
    c_layout = unsafe_load(args.shape_layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    dev_ptr = Ptr{Device}(unsafe_load(args.device))
    if mem_ptr == C_NULL && dev_ptr == C_NULL
        throw(invalid_argument("No device nor memory has been assigned"))
    end

    buffer = if mem_ptr != C_NULL
        mem = unsafe_pointer_to_objref(mem_ptr)::Memory
        create_uninitialized_buffer(client, mem, _eltype, _size, layout)
    else
        dev = unsafe_pointer_to_objref(dev_ptr)::Device
        create_uninitialized_buffer(client, dev, _eltype, _size, layout)
    end::Buffer

    root!(buffer)
    unsafe_store!(args.buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(buffer)))
    return ok()
end

"""
    compile!(client::Client, code::MLIR.IR.Module, options::Vector{UInt8})

Compile `code` for the devices of `client` and return a loaded executable.

## Arguments

- `code`: MLIR module. Reactant always sends this format: a StableHLO portable artifact (VHLO
    bytecode) serialized to the version in the plugin's `stablehlo_current_version` attribute, or to XLA's default
    version if the plugin doesn't set it.
- `options` is a serialized `CompileOptionsProto` (replica/partition counts, device assignment, ...).

!!! warning
`code` and `options` are owned by C and only valid during the call, so they must be copied before being stored.

The caller GC-roots the returned executable.

!!! todo
Not supported yet: no executable type exists and `PJRT_Client_Compile` always returns `UNIMPLEMENTED`.

Backs `PJRT_Client_Compile`.
"""
compile!(::Client, args...)

# parse a `PJRT_Program`, which must be a StableHLO portable artifact (VHLO bytecode) in `"mlir"` format
function unsafe_load_program(program::Ptr{CAPI.PJRT_Program})
    format = unsafe_string(
        Ptr{Cchar}(unsafe_load(program.format)), unsafe_load(program.format_size)
    )
    format == "mlir" || throw(unimplemented("code in $format format is not supported"))
    code_str_ref = API.MlirStringRef(
        unsafe_load(program.code), unsafe_load(program.code_size)
    )
    ctx = Reactant.ReactantContext()
    return IR.Module(API.stablehloDeserializePortableArtifactNoError(code_str_ref, ctx))
end

# parse a serialized `CompileOptionsProto`
function unsafe_load_compile_options(ptr::Cstring, size::Integer)
    bytes = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(ptr), size)
    return Reactant.ProtoUtils.proto_from_bytes(
        Reactant.Proto.xla.CompileOptionsProto, bytes
    )
end

@pjrt_catch_return function pjrt_client_compile(
    args::Ptr{CAPI.PJRT_Client_Compile_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    mlir_mod = unsafe_load_program(Ptr{CAPI.PJRT_Program}(unsafe_load(args.program)))
    options = unsafe_load_compile_options(
        unsafe_load(args.compile_options), unsafe_load(args.compile_options_size)
    )
    loaded_exec = compile!(client, mlir_mod, options)::LoadedExecutable
    root!(loaded_exec)
    unsafe_store!(
        args.executable, Ptr{CAPI.PJRT_LoadedExecutable}(pointer_from_objref(loaded_exec))
    )
    return ok()
end

"""
    deserialize_and_load(
        client::Client, bytes::Vector{UInt8}, options::Union{Nothing,Reactant.Proto.xla.CompileOptionsProto}
    ) -> LoadedExecutable

Load on the devices of `client` an executable serialized with `serialize(::Executable)`. If `options` is not
`nothing`, it overrides the compile options stored in `bytes`.

The caller GC-roots the returned executable.

!!! todo
`PJRT_LoadOptions` are ignored.

Backs `PJRT_Executable_DeserializeAndLoad`.
"""
deserialize_and_load(::Client, args...)

@pjrt_catch_return function pjrt_executable_deserialize_and_load(
    args::Ptr{CAPI.PJRT_Executable_DeserializeAndLoad_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client

    # the serialized executable is owned by C, so copy it in case the implementation stores it
    bytes_ptr = Ptr{UInt8}(unsafe_load(args.serialized_executable))
    bytes_size = unsafe_load(args.serialized_executable_size)
    bytes = copy(unsafe_wrap(Vector{UInt8}, bytes_ptr, bytes_size))

    opts_ptr = Ptr{UInt8}(unsafe_load(args.overridden_serialized_compile_options))
    options = if opts_ptr == C_NULL
        nothing
    else
        opts_size = unsafe_load(args.overridden_serialized_compile_options_size)
        opts_bytes = unsafe_wrap(Vector{UInt8}, opts_ptr, opts_size)
        Reactant.ProtoUtils.proto_from_bytes(
            Reactant.Proto.xla.CompileOptionsProto, opts_bytes
        )
    end

    loaded_exec = deserialize_and_load(client, bytes, options)::LoadedExecutable
    root!(loaded_exec)
    unsafe_store!(
        args.loaded_executable,
        Ptr{CAPI.PJRT_LoadedExecutable}(pointer_from_objref(loaded_exec)),
    )
    return ok()
end

"""
    default_device_assignment(client::Client, num_replicas::Integer, num_partitions::Integer) -> AbstractMatrix{<:Integer}

Return the default device assignment of `client` as a `num_replicas × num_partitions` matrix, whose entry `[r, p]` is
the id of the device that runs replica `r - 1` and partition `p - 1`.

Backs `PJRT_Client_DefaultDeviceAssignment`.
"""
default_device_assignment(::Client, ::Integer, ::Integer)

@pjrt_catch_return function pjrt_client_default_device_assignment(
    args::Ptr{CAPI.PJRT_Client_DefaultDeviceAssignment_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    num_replicas = unsafe_load(args.num_replicas)
    num_partitions = unsafe_load(args.num_partitions)
    if num_replicas <= 0 || num_partitions <= 0
        throw(
            invalid_argument(
                "`num_replicas` and `num_partitions` must be positive, got $num_replicas and $num_partitions",
            ),
        )
    end

    capacity = unsafe_load(args.default_assignment_size)
    if capacity < num_replicas * num_partitions
        msg = "`default_assignment_size` $capacity < `num_replicas * num_partitions` = $(num_replicas * num_partitions)"
        throw(Error(CAPI.PJRT_Error_Code_FAILED_PRECONDITION, msg))
    end

    assignment = default_device_assignment(
        client, num_replicas, num_partitions
    )::AbstractMatrix{<:Integer}
    if size(assignment) != (num_replicas, num_partitions)
        msg = "`default_device_assignment` returned a $(size(assignment)) matrix, but ($num_replicas, $num_partitions) was expected"
        throw(Error(CAPI.PJRT_Error_Code_INTERNAL, msg))
    end

    # C expects a row-major (replica, partition) array
    dst = unsafe_load(args.default_assignment)
    for r in 1:num_replicas, p in 1:num_partitions
        unsafe_store!(dst, assignment[r, p], (r - 1) * num_partitions + p)
    end
    return ok()
end

"""
    load(client::Client, exe::Executable, options::Reactant.Proto.xla.CompileOptionsProto) -> LoadedExecutable

Load `exe` on the devices of `client` with the compile options `options`, e.g. an executable returned by `compile`.

The caller GC-roots the returned executable.

Backs `PJRT_Client_Load`.
"""
load(::Client, ::Executable, ::Any)

@pjrt_catch_return function pjrt_client_load(
    args::Ptr{CAPI.PJRT_Client_Load_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    options = unsafe_load_compile_options(
        unsafe_load(args.compile_options), unsafe_load(args.compile_options_size)
    )
    loaded_exec = load(client, exe, options)::LoadedExecutable
    root!(loaded_exec)
    unsafe_store!(
        args.loaded_executable,
        Ptr{CAPI.PJRT_LoadedExecutable}(pointer_from_objref(loaded_exec)),
    )
    return ok()
end

"""
    create_error_buffer(client::Client, mem::Memory, error::Error, type, dims, layout) -> Buffer

Return a new buffer placed on memory `mem`, with element type `type::CAPI.PJRT_Buffer_Type` and dimensions
`dims::Vector{Int64}`, that carries `error` without allocating any data: executing a program with it as argument must
fail with `error`.

`layout::Union{Nothing,MemoryLayout}` is `nothing` for the default layout.

The caller GC-roots the returned buffer.

!!! todo
Error payloads are not supported yet: if C passes any, `PJRT_Client_CreateErrorBuffer` returns `UNIMPLEMENTED`.

Backs `PJRT_Client_CreateErrorBuffer`.
"""
create_error_buffer(::Client, ::Memory, ::Error, args...)

@pjrt_catch_return function pjrt_client_create_error_buffer(
    args::Ptr{CAPI.PJRT_Client_CreateErrorBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    mem_ptr == C_NULL && throw(invalid_argument("No memory has been assigned"))
    mem = unsafe_pointer_to_objref(mem_ptr)::Memory

    unsafe_load(args.num_payload) == 0 ||
        throw(unimplemented("error payloads are not supported yet"))
    error = Error(
        unsafe_load(args.error_code),
        unsafe_load(args.error_message),
        unsafe_load(args.error_message_size),
    )

    _eltype = unsafe_load(args.shape_element_type)
    _size = copy(
        unsafe_wrap(
            Vector{Int64}, unsafe_load(args.shape_dims), unsafe_load(args.shape_num_dims)
        ),
    )
    # `device_layout` was added at the end of the struct, so older callers don't pass it
    has_device_layout =
        unsafe_load(args.struct_size) >=
        UInt(args.device_layout) - UInt(args) + sizeof(Ptr{Cvoid})
    c_layout = has_device_layout ? unsafe_load(args.device_layout) : C_NULL
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    buffer = create_error_buffer(client, mem, error, _eltype, _size, layout)::Buffer
    root!(buffer)
    unsafe_store!(args.buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(buffer)))
    return ok()
end

# GC-rooted holder of the callable returned by `create_alias_buffer`, since it may not be mutable
mutable struct FulfillAliasBufferCallback
    const fulfill::Any
end

"""
    create_alias_buffer(client::Client, mem::Memory, type, dims, layout) -> Tuple{Buffer,Any}

Return `(alias, fulfill)`: a new buffer `alias` placed on memory `mem`, with element type `type::CAPI.PJRT_Buffer_Type`
and dimensions `dims::Vector{Int64}`, which aliases a buffer that is not known yet; and a callable `fulfill`.

`alias` is not ready until `fulfill(result)` is called, where `result` is either the aliased `Buffer` or an `Error` that
`alias` must fail with. `fulfill` is called at most once, by `PJRT_Client_FulfillAliasBuffer`, and its errors are
returned to C.

`layout::Union{Nothing,MemoryLayout}` is `nothing` for the default layout.

The caller GC-roots `alias`, and keeps `fulfill` alive until it is called.

Backs `PJRT_Client_CreateAliasBuffer`.
"""
create_alias_buffer(::Client, ::Memory, args...)

@pjrt_catch_return function pjrt_client_create_alias_buffer(
    args::Ptr{CAPI.PJRT_Client_CreateAliasBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    mem_ptr == C_NULL && throw(invalid_argument("No memory has been assigned"))
    mem = unsafe_pointer_to_objref(mem_ptr)::Memory

    _eltype = unsafe_load(args.shape_element_type)
    _size = copy(
        unsafe_wrap(
            Vector{Int64}, unsafe_load(args.shape_dims), unsafe_load(args.shape_num_dims)
        ),
    )
    c_layout = unsafe_load(args.shape_layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    alias, fulfill =
        create_alias_buffer(client, mem, _eltype, _size, layout)::Tuple{Buffer,Any}
    root!(alias)
    cb = root!(FulfillAliasBufferCallback(fulfill))
    unsafe_store!(args.alias_buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(alias)))
    unsafe_store!(
        args.fulfill_alias_buffer_cb,
        Ptr{CAPI.PJRT_FulfillAliasBufferCallback}(pointer_from_objref(cb)),
    )
    return ok()
end

@pjrt_catch_return function pjrt_client_fulfill_alias_buffer(
    args::Ptr{CAPI.PJRT_Client_FulfillAliasBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    cb_ptr = unsafe_load(args.fulfill_alias_buffer_cb)
    cb_ptr == C_NULL && throw(invalid_argument("`fulfill_alias_buffer_cb` is null"))
    # the callback is consumed even if fulfilling fails, like in XLA
    cb = unsafe_pointer_to_objref(cb_ptr)::FulfillAliasBufferCallback
    unroot!(cb)

    code = unsafe_load(args.status_code)
    result = if code == CAPI.PJRT_Error_Code_OK
        buffer_ptr = unsafe_load(args.buffer)
        buffer_ptr == C_NULL &&
            throw(invalid_argument("buffer to fulfill the alias with is null"))
        unsafe_pointer_to_objref(buffer_ptr)::Buffer
    else
        Error(code, unsafe_load(args.error_message), unsafe_load(args.error_message_size))
    end
    cb.fulfill(result)
    return ok()
end

"""
    update_global_process_info!(client::Client, infos::Vector)

Update `client` with the state of every process of a distributed program. Each element of `infos` is a `NamedTuple`
with fields:

- `task_id::Int`
- `incarnation_id::UInt64`
- `state::CAPI.PJRT_ProcessState`
- `error::Union{Nothing,Error}`: the error of the process, if any

Backs `PJRT_Client_UpdateGlobalProcessInfo`.
"""
update_global_process_info!(::Client, ::Vector)

@pjrt_catch_return function pjrt_client_update_global_process_info(
    args::Ptr{CAPI.PJRT_Client_UpdateGlobalProcessInfo_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    c_infos = unsafe_load(args.process_infos)
    infos = map(1:unsafe_load(args.num_process_infos)) do i
        info = unsafe_load(c_infos, i)
        error = if info.error_code == 0
            nothing
        else
            # unlike in other structs, the error code is a plain `int` here
            code = CAPI.PJRT_Error_Code(info.error_code)
            Error(code, info.error_message, info.error_message_size)
        end
        (;
            task_id=Int(info.task_id),
            incarnation_id=info.incarnation_id,
            state=info.state,
            error,
        )
    end
    update_global_process_info!(client, infos)
    return ok()
end

"""
    dma_map!(client::Client, data::Ptr{Cvoid}, size::Integer)

Map the `size` bytes of host memory at `data` for fast transfers to and from the devices of `client`.

Backs `PJRT_Client_DmaMap`.
"""
dma_map!(::Client, ::Ptr{Cvoid}, ::Integer)

@pjrt_catch_return function pjrt_client_dma_map(
    args::Ptr{CAPI.PJRT_Client_DmaMap_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    dma_map!(client, unsafe_load(args.data), unsafe_load(args.size))
    return ok()
end

"""
    dma_unmap!(client::Client, data::Ptr{Cvoid})

Unmap the host memory at `data`, previously mapped with `dma_map!`.

Backs `PJRT_Client_DmaUnmap`.
"""
dma_unmap!(::Client, ::Ptr{Cvoid})

@pjrt_catch_return function pjrt_client_dma_unmap(
    args::Ptr{CAPI.PJRT_Client_DmaUnmap_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    dma_unmap!(client, unsafe_load(args.data))
    return ok()
end
