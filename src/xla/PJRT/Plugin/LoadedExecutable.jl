"""
    LoadedExecutable

A compiled program loaded on the devices of a `Client`, as returned by `compile!`.

### Interface

- `destroy`
- `executable`
- `delete!`
- `isdeleted`
- `addressable_devices`
- `addressable_device_logical_ids`
- `device_assignment`
- `execute`
"""
abstract type LoadedExecutable end

@pjrt_catch_return function pjrt_loaded_executable_destroy(
    args::Ptr{CAPI.PJRT_LoadedExecutable_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.executable)
    ptr == C_NULL && return ok()
    exec = unsafe_pointer_to_objref(ptr)::LoadedExecutable
    delete!(exec)
    unroot!(exec)
    return ok()
end

"""
    executable(exec::LoadedExecutable) -> Executable

Return the `Executable` of `exec`, i.e. its compiled program without the devices it is loaded on.

The returned executable is GC rooted until C calls `PJRT_Executable_Destroy` on it. If the same object is returned
on every call, `exec` must keep a reference to it, since the first `PJRT_Executable_Destroy` unroots it.

!!! warning
`PjRtCApiLoadedExecutable` calls this when wrapping `exec` and treats any error as fatal.

Backs `PJRT_LoadedExecutable_GetExecutable`.
"""
executable(::LoadedExecutable)

@pjrt_catch_return function pjrt_loaded_executable_get_executable(
    args::Ptr{CAPI.PJRT_LoadedExecutable_GetExecutable_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.loaded_executable))::LoadedExecutable
    exe = root!(executable(exec)::Executable)
    unsafe_store!(args.executable, Ptr{CAPI.PJRT_Executable}(pointer_from_objref(exe)))
    return ok()
end

"""
    delete!(exec::LoadedExecutable)

Free the runtime resources of `exec` (e.g. its compiled code) while keeping the object alive. Must be idempotent.
After this call, `isdeleted(exec)` must return `true`.

Also called by `PJRT_LoadedExecutable_Destroy` before `exec` is unrooted.

Backs `PJRT_LoadedExecutable_Delete`.
"""
Base.delete!(::LoadedExecutable)

@pjrt_catch_return function pjrt_loaded_executable_delete(
    args::Ptr{CAPI.PJRT_LoadedExecutable_Delete_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    delete!(exec)
    return ok()
end

"""
    isdeleted(exec::LoadedExecutable) -> Bool

Return whether the runtime resources of `exec` have been freed with `delete!`.

Backs `PJRT_LoadedExecutable_IsDeleted`.
"""
isdeleted(::LoadedExecutable)

@pjrt_catch_return function pjrt_loaded_executable_is_deleted(
    args::Ptr{CAPI.PJRT_LoadedExecutable_IsDeleted_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    pred = isdeleted(exec)::Bool
    unsafe_store!(args.is_deleted, pred)
    return ok()
end

"""
    addressable_devices(exec::LoadedExecutable) -> Vector{Device}

Return the devices that `exec` runs on, in the order used by the argument and output lists of `execute`.

!!! warning
The vector is handed to C by pointer, so it must be owned by `exec` and never be resized.
`PjRtCApiLoadedExecutable` calls this when wrapping `exec` and treats any error as fatal.

Backs `PJRT_LoadedExecutable_AddressableDevices`.
"""
addressable_devices(::LoadedExecutable)

@pjrt_catch_return function pjrt_loaded_executable_addressable_devices(
    args::Ptr{CAPI.PJRT_LoadedExecutable_AddressableDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    devs = addressable_devices(exec)::Vector{Device}
    unsafe_store!(args.addressable_devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_addressable_devices, length(devs))
    return ok()
end

"""
    addressable_device_logical_ids(exec::LoadedExecutable) -> Vector{CAPI.PJRT_LogicalDeviceIds}

Return the logical (replica, partition) ids, 0-based, of each device in `addressable_devices(exec)`, in the same order.

!!! warning
The vector is handed to C by pointer, so it must be owned by `exec` and never be resized.
Some versions of `PjRtCApiLoadedExecutable` call this when wrapping `exec` and treat any error as fatal.

Backs `PJRT_LoadedExecutable_AddressableDeviceLogicalIds`.
"""
addressable_device_logical_ids(::LoadedExecutable)

@pjrt_catch_return function pjrt_loaded_executable_addressable_device_logical_ids(
    args::Ptr{CAPI.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    ids = addressable_device_logical_ids(exec)::Vector{CAPI.PJRT_LogicalDeviceIds}
    unsafe_store!(args.addressable_device_logical_ids, pointer(ids))
    unsafe_store!(args.num_addressable_device_logical_ids, length(ids))
    return ok()
end

"""
    device_assignment(exec::LoadedExecutable) -> Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}

Return the device assignment that `exec` was compiled for, which maps every (replica, partition) pair to a device
id, or `nothing` if `exec` is portable (i.e. it runs on whichever device is given to `execute`).

It usually is `executable_build_options.device_assignment` of the `CompileOptionsProto` given to `compile!`.

!!! warning
`PjRtCApiLoadedExecutable` calls this when wrapping `exec` and treats any error as fatal.

Backs `PJRT_LoadedExecutable_GetDeviceAssignment`.
"""
device_assignment(::LoadedExecutable)

@pjrt_catch_return function pjrt_loaded_executable_get_device_assignment(
    args::Ptr{CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    proto = device_assignment(exec)::Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}

    # XLA calls the deleter even if there is no device assignment
    deleter = @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    unsafe_store!(args.serialized_device_assignment_deleter, deleter)

    if isnothing(proto)
        unsafe_store!(args.serialized_bytes, Cstring(C_NULL))
        unsafe_store!(args.serialized_bytes_size, 0)
        unsafe_store!(args.serialized_device_assignment, C_NULL)
        return ok()
    end

    bytes = root!(Reactant.ProtoUtils.proto_to_bytes(proto))
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(args.serialized_device_assignment, pointer_from_objref(bytes))
    return ok()
end

"""
    execute(
        exec::LoadedExecutable, arguments::Vector{Vector{Buffer}}, dev::Union{Nothing,Device},
        context::Union{Nothing,ExecuteContext},
    ) -> Vector{Vector{Buffer}}

Run `exec` and return its outputs.

`arguments[i]` holds the input buffers for the `i`-th device of `addressable_devices(exec)`. The result must follow the
same layout: one vector per device, each with as many new buffers as `exec` has outputs.

If `dev` is not `nothing`, run only on `dev` (one of `addressable_devices(exec)`). Then `arguments` and the result
have a single entry, for `dev`.

`context` is the [`ExecuteContext`](@ref) of this execution (e.g. with the user data for its FFI handlers), or `nothing`
if C passes none.

Execution is synchronous: the device-complete events are signaled as soon as this returns. The caller GC-roots the
returned buffers and destroys each of them, so they must all be distinct.

!!! todo
`PJRT_ExecuteOptions` (e.g. non-donatable inputs, send/recv callbacks) are ignored.

Backs `PJRT_LoadedExecutable_Execute`.
"""
execute(::LoadedExecutable, args...)

@pjrt_catch_return function pjrt_loaded_executable_execute(
    args::Ptr{CAPI.PJRT_LoadedExecutable_Execute_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    isdeleted(exec) && throw(invalid_argument("executable is deleted"))

    num_devices = unsafe_load(args.num_devices)
    num_args = unsafe_load(args.num_args)
    argument_lists = unsafe_load(args.argument_lists)
    arguments = map(1:num_devices) do i
        argument_list = unsafe_load(argument_lists, i)
        Buffer[
            unsafe_pointer_to_objref(unsafe_load(argument_list, j))::Buffer for
            j in 1:num_args
        ]
    end

    dev_ptr = unsafe_load(args.execute_device)
    dev = dev_ptr == C_NULL ? nothing : unsafe_pointer_to_objref(dev_ptr)::Device

    # `context` was added to `PJRT_ExecuteOptions` later, so older callers don't pass it
    options = unsafe_load(args.options)
    offset = fieldoffset(
        CAPI.PJRT_ExecuteOptions, Base.fieldindex(CAPI.PJRT_ExecuteOptions, :context)
    )
    has_context =
        options != C_NULL &&
        unsafe_load(Ptr{Csize_t}(options)) >= offset + sizeof(Ptr{Cvoid})
    context_ptr = if has_context
        unsafe_load(Ptr{Ptr{CAPI.PJRT_ExecuteContext}}(options + offset))
    else
        C_NULL
    end
    context = if context_ptr == C_NULL
        nothing
    else
        unsafe_pointer_to_objref(context_ptr)::ExecuteContext
    end

    outputs = execute(exec, arguments, dev, context)::Vector{Vector{Buffer}}
    if length(outputs) != num_devices
        msg = "`execute` returned outputs for $(length(outputs)) devices, but $num_devices were expected"
        throw(Error(CAPI.PJRT_Error_Code_INTERNAL, msg))
    end

    output_lists = unsafe_load(args.output_lists)
    for (i, output_list) in enumerate(outputs)
        c_output_list = unsafe_load(output_lists, i)
        for (j, output) in enumerate(output_list)
            root!(output)
            unsafe_store!(
                c_output_list, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(output)), j
            )
        end
    end

    events = unsafe_load(args.device_complete_events)
    if events != C_NULL
        for i in 1:num_devices
            event = root!(Event())
            notify(event)
            unsafe_store!(events, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)), i)
        end
    end
    return ok()
end
