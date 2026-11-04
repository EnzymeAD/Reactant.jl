module Julia

# DESIGN NOTES
# when dealing with memory at the Julia - C FFI boundary, there are two directions we can follow:
# - rely on Julia's GC, which requires to...
#   - globally mark objects
#   - refer to them via `pointer_from_objref`, `unsafe_pointer_to_objref`
#   - use `mutable` objects only (immutable objects do not have a stable memory address)
# - manually allocate/free memory, which requires to...
#   - use `Libc`
#   - only work with `isbits` objects (i.e. not mutable nor immutable with mutable fields)
#
# after trying both designs, i ultimately opted for the "globally GC rooted" design because
# - has a simpler design
# - it's easier to extend
# - safer against memory bugs
# - almost no overhead compared to manual allocations
#
# notes:
# - all objects passed through the XLA/PJRT boundary need to be GC rooted (i.e. use `(un)root!`), which requires objects to be `mutable` to have a stable pointer address
# - some objects require as a first field a pointer to their vtable
# - if for some PJRT object, the API requires it to return a string, store the string inside the object so that it's correctly freed (owner is the object)
# - if for some PJRT object, the API requires it to return a list, store the list as a `Vector{X}` where `X` is an abstract type
#   - the Julia array ABI layout states that an array with abstract eltype is represented as a pointer of pointers (no inline storage. check: https://docs.julialang.org/en/v1/manual/embedding/#Working-with-Arrays)
#   - the stored pointers are of type `jl_value_t*` which points to the object data (the type tag is stored before that pointer address. check https://docs.julialang.org/en/v1/devdocs/object/#Object-layout-(jl_value_t))
#   - once the object is created, DO NOT resize the array
# - DO NOT use moving GC (e.g. MMTk) as it would break `pointer_from_objref`

using Base: unsafe_convert
using ..Reactant: Reactant
using Reactant.XLA.CAPI: CAPI, csizeof
using Reactant.MLIR: IR, API
using LinearAlgebra: mul!
import Serialization: serialize

macro pjrt_catch_return(fdef)
    Meta.isexpr(fdef, :function) ||
        error("@pjrt_catch_return must be applied to a `function` definition")

    err = gensym(:err)
    fdef.args[2] = quote
        try
            $(fdef.args[2])
        catch $err
            if $err isa $Error
                return $pointer_from_objref($root!($err))
            else
                # we could return an "internal" error, but throwing makes it easier for Julia REPL sessions
                # return $pointer_from_objref($root!($Error($(CAPI.PJRT_Error_Code_INTERNAL), $sprint($showerror, $err))))
                $rethrow()
            end
        end
        return $ok()
    end
    return esc(fdef)
end

# interface functions
function addressable_devices end
function addressable_memories end
function attributes end
function buffer_from_host_buffer end
function compile_options end
function compile! end
function copy_to_device end
function copy_to_memory end
function debug_string end
function default_memory end
function description end
function device end
function device_assignment end
function addressable_device_logical_ids end
function devices end
function dimensions end
function element_type end
function executable end
function execute end
function fingerprint end
function id end
function is_addressable end
function is_on_cpu end
function isdeleted end
function kind end
function kind_id end
function local_hardware_id end
function memory end
function name end
function num_outputs end
function num_partitions end
function num_replicas end
function on_device_size_in_bytes end
function optimized_program end
function output_element_types end
function platform_name end
function platform_version end
function process_index end
function ready_event end
function size_of_generated_code_in_bytes end
function to_host_buffer end
function to_string end
function unsafe_pointer end

# GC roots of objects whose pointers are handed to C; objects stay alive until they are unmarked
const gc_roots = IdSet{Any}()
const gc_roots_lock = ReentrantLock()

function root!(obj::T) where {T}
    ismutabletype(T) || error("Only mutable types can be GC rooted")
    @lock gc_roots_lock push!(gc_roots, obj)
    # return Ptr{T}(pointer_from_objref(obj))
    return obj
end

unroot!(ptr::Ptr) = unroot!(unsafe_pointer_to_objref(ptr))
function unroot!(obj)
    @lock gc_roots_lock delete!(gc_roots, obj)
    return nothing
end

function pjrt_elsize(T::CAPI.PJRT_Buffer_Type)
    if T in (
        CAPI.PJRT_Buffer_Type_PRED,
        CAPI.PJRT_Buffer_Type_S8,
        CAPI.PJRT_Buffer_Type_U8,
        CAPI.PJRT_Buffer_Type_F8E5M2,
        CAPI.PJRT_Buffer_Type_F8E4M3FN,
        CAPI.PJRT_Buffer_Type_F8E4M3B11FNUZ,
        CAPI.PJRT_Buffer_Type_F8E5M2FNUZ,
        CAPI.PJRT_Buffer_Type_F8E4M3FNUZ,
        CAPI.PJRT_Buffer_Type_F8E4M3,
        CAPI.PJRT_Buffer_Type_F8E3M4,
        CAPI.PJRT_Buffer_Type_F8E8M0FNU,
    )
        1
    elseif T in (
        CAPI.PJRT_Buffer_Type_S16,
        CAPI.PJRT_Buffer_Type_U16,
        CAPI.PJRT_Buffer_Type_F16,
        CAPI.PJRT_Buffer_Type_BF16,
    )
        2
    elseif T in
        (CAPI.PJRT_Buffer_Type_S32, CAPI.PJRT_Buffer_Type_U32, CAPI.PJRT_Buffer_Type_F32)
        4
    elseif T in (
        CAPI.PJRT_Buffer_Type_S64,
        CAPI.PJRT_Buffer_Type_U64,
        CAPI.PJRT_Buffer_Type_F64,
        CAPI.PJRT_Buffer_Type_C64,
    )
        8
    elseif T == CAPI.PJRT_Buffer_Type_C128
        16
    else
        return nothing
    end
end

# TODO move `pjrt_eltype` to specialized pkg exts
pjrt_eltype(T::Type) = error("PJRT does not have a type mapping for $T")
pjrt_eltype(::Type{Bool}) = CAPI.PJRT_Buffer_Type_PRED
pjrt_eltype(::Type{Int8}) = CAPI.PJRT_Buffer_Type_S8
pjrt_eltype(::Type{UInt8}) = CAPI.PJRT_Buffer_Type_U8
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E5M2
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3FN
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3B11FNUZ
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E5M2FNUZ
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3FNUZ
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E3M4
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E8M0FNU
pjrt_eltype(::Type{Int16}) = CAPI.PJRT_Buffer_Type_S16
pjrt_eltype(::Type{UInt16}) = CAPI.PJRT_Buffer_Type_U16
pjrt_eltype(::Type{Float16}) = CAPI.PJRT_Buffer_Type_F16
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_BF16
pjrt_eltype(::Type{Int32}) = CAPI.PJRT_Buffer_Type_S32
pjrt_eltype(::Type{UInt32}) = CAPI.PJRT_Buffer_Type_U32
pjrt_eltype(::Type{Float32}) = CAPI.PJRT_Buffer_Type_F32
pjrt_eltype(::Type{Int64}) = CAPI.PJRT_Buffer_Type_S64
pjrt_eltype(::Type{UInt64}) = CAPI.PJRT_Buffer_Type_U64
pjrt_eltype(::Type{Float64}) = CAPI.PJRT_Buffer_Type_F64
pjrt_eltype(::Type{ComplexF32}) = CAPI.PJRT_Buffer_Type_C64
pjrt_eltype(::Type{ComplexF64}) = CAPI.PJRT_Buffer_Type_C128

## vtables
const vtable_api = Ref{CAPI.PJRT_Api}()
const vtable_error = Ref{CAPI.PJRT_Error_FunctionTable}()
const vtable_memory = Ref{CAPI.PJRT_Memory_FunctionTable}()

## Error
"""
    Error(code::CAPI.PJRT_Error_Code, message)

A `PJRT_Error` carrying an error `code` and a human-readable `message`.

Throw it from a `@pjrt_catch_return` callback to return it to C.

!!! info
Success is a null `PJRT_Error*` (`ok()`), not an `Error` with code `OK`.

!!! warning
It is GC rooted when handed to C and unrooted by `PJRT_Error_Destroy`.
Since every receiver destroys its own error, the same `Error` must never be handed out twice.
"""
mutable struct Error
    const vtable::Ptr{CAPI.PJRT_Error_FunctionTable}
    const code::CAPI.PJRT_Error_Code
    const message::String

    function Error(code::CAPI.PJRT_Error_Code, message)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Error_FunctionTable}, vtable_error), code, message
        )
    end
end

# XLA considers `nullptr` to be OK as it avoids allocations
ok() = C_NULL

"""
    unimplemented(reason="this feature is not implemented yet!") -> Error

Return an `UNIMPLEMENTED` [`Error`](@ref) with message `reason`, for features that are not supported yet.

!!! info
The error is not GC rooted.
`throw` it inside the scope of a `@pjrt_catch_return` function, which roots it and returns it to C.
"""
function unimplemented(reason="this feature is not implemented yet!")
    return Error(CAPI.PJRT_Error_Code_UNIMPLEMENTED, reason)
end

"""
    invalid_argument(reason="invalid argument") -> Error

Return an `INVALID_ARGUMENT` [`Error`](@ref) with message `reason`, for arguments that are malformed or that refer to
an object in the wrong state (e.g. a deleted buffer).

!!! info
The error is not GC rooted.
`throw` it inside the scope of a `@pjrt_catch_return` function, which roots it and returns it to C.
"""
function invalid_argument(reason="invalid argument")
    return Error(CAPI.PJRT_Error_Code_INVALID_ARGUMENT, reason)
end

"""
    resource_exhausted(reason="out of memory") -> Error

Return a `RESOURCE_EXHAUSTED` [`Error`](@ref) with message `reason`, for failed allocations.

!!! info
The error is not GC rooted.
`throw` it inside the scope of a `@pjrt_catch_return` function, which roots it and returns it to C.
"""
function resource_exhausted(reason="out of memory")
    return Error(CAPI.PJRT_Error_Code_RESOURCE_EXHAUSTED, reason)
end

# these are required by CAPI.PJRT_Api
function error_destroy(args::Ptr{CAPI.PJRT_Error_Destroy_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    error_destroy_obj(error)
    return nothing
end

function error_message(args::Ptr{CAPI.PJRT_Error_Message_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    error_message_obj(error, args.message, args.message_size)
    return nothing
end

function error_getcode(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    code = error_getcode_obj(error)
    unsafe_store!(args.code, code)
    return ok()
end

function error_for_each_payload(
    args::Ptr{CAPI.PJRT_Error_ForEachPayload_Args}
)::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    visitor = unsafe_load(args.visitor)
    user_arg = unsafe_load(args.user_arg)
    error_for_each_payload_obj(error, visitor, user_arg)
    return ok()
end

# this are required by CAPI.PJRT_Error_FunctionTable
function error_destroy_obj(error::Ptr{Error})::Cvoid
    error != C_NULL && unroot!(error)
    return nothing
end

function error_message_obj(
    error::Ptr{Error}, message::Ptr{Cstring}, message_size::Ptr{Csize_t}
)::Cvoid
    error == C_NULL && return nothing
    err = unsafe_pointer_to_objref(error)::Error
    msg = err.message
    unsafe_store!(message, Cstring(pointer(msg)))
    unsafe_store!(message_size, sizeof(msg))
    return nothing
end

function error_getcode_obj(error::Ptr{Error})::CAPI.PJRT_Error_Code
    error == C_NULL && return CAPI.PJRT_Error_Code_OK
    err = unsafe_pointer_to_objref(error)::Error
    return err.code
end

function error_for_each_payload_obj(
    error::Ptr{Error}, visitor::CAPI.PJRT_Error_PayloadVisitor, user_arg::Ptr{Cvoid}
)::Cvoid
    # do nothing
    return nothing
end

## Event
"""
    Event()

A `PJRT_Event`: a one-shot, thread-safe completion signal, which may carry an [`Error`](@ref) if the operation failed.

It starts unset and is set exactly once with [`notify`](@ref Base.notify(::Event)). Use
[`wait`](@ref Base.wait(::Event)) to block until it is set and [`isready`](@ref Base.isready(::Event)) to check without
blocking. Callbacks registered with `PJRT_Event_OnReady` run when it is set, or immediately if it already is.

Every `PJRT_Error` it hands to C (from `PJRT_Event_Error`, `PJRT_Event_Await` or an on-ready callback) is a new copy of
`event.error`, since each receiver destroys its own.

!!! warning
It is GC rooted when handed to C and unrooted by `PJRT_Event_Destroy`.
"""
mutable struct Event
    const cond::Threads.Condition
    ready::Bool
    error::Union{Nothing,Error}
    const callbacks::Vector{Tuple{CAPI.PJRT_Event_OnReadyCallback,Ptr{Cvoid}}}

    function Event()
        return new(
            Threads.Condition(),
            false,
            nothing,
            Tuple{CAPI.PJRT_Event_OnReadyCallback,Ptr{Cvoid}}[],
        )
    end
end

"""
    isready(event::Event) -> Bool

Return whether `event` has been set, without blocking.

Backs `PJRT_Event_IsReady`.

See also [`wait`](@ref Base.wait(::Event)), [`notify`](@ref Base.notify(::Event)).
"""
Base.isready(event::Event) = @lock event.cond event.ready

"""
    wait(event::Event)

Block the current task until `event` is set. Returns immediately if it already is.

It doesn't throw if `event` was set with an error: check `event.error` afterwards.

Backs `PJRT_Event_Await`.

See also [`isready`](@ref Base.isready(::Event)), [`notify`](@ref Base.notify(::Event)).
"""
function Base.wait(event::Event)
    @lock event.cond while !event.ready
        wait(event.cond)
    end
    return nothing
end

"""
    notify(event::Event, err::Union{Nothing,Error}=nothing)

Set `event`, as failed with `err` or as successful if `err` is `nothing`. This wakes every task blocked in
[`wait`](@ref Base.wait(::Event)) and then runs the on-ready callbacks, each with its own copy of `err`.

Callbacks run outside the lock, in the calling task, so they may call back into `event`.

Throws [`invalid_argument`](@ref) if `event` is already set.

Backs `PJRT_Event_Set`.

See also [`isready`](@ref Base.isready(::Event)).
"""
function Base.notify(event::Event, err::Union{Nothing,Error}=nothing)
    callbacks = @lock event.cond begin
        event.ready && throw(invalid_argument("event has already been set"))
        event.ready = true
        event.error = err
        notify(event.cond)
        callbacks = copy(event.callbacks)
        empty!(event.callbacks)
        callbacks
    end

    # callbacks run outside the lock so that they can call back into the event
    for (callback, user_arg) in callbacks
        ccall(
            callback,
            Cvoid,
            (Ptr{CAPI.PJRT_Error}, Ptr{Cvoid}),
            event_error_ptr(event),
            user_arg,
        )
    end
    return nothing
end

# every `PJRT_Error` handed out by an event is freed independently by its receiver, so each one must be a new object
@pjrt_catch_return function event_error_ptr(event::Event)::Ptr{CAPI.PJRT_Error}
    err = event.error
    isnothing(err) && return ok()
    return pointer_from_objref(root!(Error(err.code, err.message)))
end

@pjrt_catch_return function event_destroy(
    args::Ptr{CAPI.PJRT_Event_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.event)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

@pjrt_catch_return function event_is_ready(
    args::Ptr{CAPI.PJRT_Event_IsReady_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    unsafe_store!(args.is_ready, isready(event))
    return ok()
end

@pjrt_catch_return function event_error(
    args::Ptr{CAPI.PJRT_Event_Error_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    return event_error_ptr(event)
end

@pjrt_catch_return function event_await(
    args::Ptr{CAPI.PJRT_Event_Await_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    wait(event)
    return event_error_ptr(event)
end

@pjrt_catch_return function event_on_ready(
    args::Ptr{CAPI.PJRT_Event_OnReady_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    callback = unsafe_load(args.callback)
    user_arg = unsafe_load(args.user_arg)

    ready = @lock event.cond begin
        event.ready || push!(event.callbacks, (callback, user_arg))
        event.ready
    end

    # if the event is already set, the callback is run immediately
    if ready
        ccall(
            callback,
            Cvoid,
            (Ptr{CAPI.PJRT_Error}, Ptr{Cvoid}),
            event_error_ptr(event),
            user_arg,
        )
    end
    return ok()
end

@pjrt_catch_return function event_create(
    args::Ptr{CAPI.PJRT_Event_Create_Args}
)::Ptr{CAPI.PJRT_Error}
    event = root!(Event())
    unsafe_store!(args.event, pointer_from_objref(event))
    return ok()
end

@pjrt_catch_return function event_set(
    args::Ptr{CAPI.PJRT_Event_Set_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    code = unsafe_load(args.error_code)

    err = nothing
    if code != CAPI.PJRT_Error_Code_OK
        msg_ptr = Ptr{UInt8}(unsafe_load(args.error_message))
        msg_size = unsafe_load(args.error_message_size)
        msg = msg_ptr == C_NULL ? "" : unsafe_string(msg_ptr, msg_size)
        err = Error(code, msg)
    end

    return notify(event, err)
end

## Plugin
@pjrt_catch_return function plugin_initialize(
    args::Ptr{CAPI.PJRT_Plugin_Initialize_Args}
)::Ptr{CAPI.PJRT_Error}
    # do nothing
end

@pjrt_catch_return function plugin_attributes(
    args::Ptr{CAPI.PJRT_Plugin_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    # TODO return key-value pairs of `xla_version`, `stablehlo_current_version` and `stablehlo_minimum_version`
    unsafe_store!(args.attributes, C_NULL)
    unsafe_store!(args.num_attributes, 0)
    return ok()
end

## interfaces
"""
    Memory

### Inteface

- `id`
- `kind`
- `kind_id`
- `debug_string`
- `to_string`
- `addressable_devices`
"""
abstract type Memory end

function checkinterface(T::Type{<:Memory})
    if !ismutabletype(T)
        error("$T <: Memory must be a mutable type to be GC rootable")
    end

    if fieldtype(T, 1) != Ptr{CAPI.PJRT_Memory_FunctionTable}
        error("First field of $T must be a $(CAPI.PJRT_Memory_FunctionTable)")
    end

    return nothing
end

"""
    id(mem::Memory) -> Integer

Return the id of `mem`, unique among the memories of its client.

Backs `PJRT_Memory_Id`.
"""
id(::Memory)

@pjrt_catch_return function memory_id(
    args::Ptr{CAPI.PJRT_Memory_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    unsafe_store!(args.id, id(memory))
    return ok()
end

"""
    kind(mem::Memory) -> String

Return the kind of `mem` (e.g. `"device"`, `"pinned_host"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `mem`.

Backs `PJRT_Memory_Kind`.
"""
kind(::Memory)

@pjrt_catch_return function memory_kind(
    args::Ptr{CAPI.PJRT_Memory_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = kind(memory)
    unsafe_store!(args.kind, Cstring(pointer(str)))
    unsafe_store!(args.kind_size, sizeof(str))
    return ok()
end

"""
    kind_id(mem::Memory) -> Integer

Return a platform-dependent id that uniquely identifies the kind of `mem`.
Memories with the same `kind` must return the same `kind_id`.

Backs `PJRT_Memory_Kind_Id`.
"""
kind_id(::Memory)

@pjrt_catch_return function memory_kind_id(
    args::Ptr{CAPI.PJRT_Memory_Kind_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    kid = kind_id(memory)::Integer
    unsafe_store!(args.kind_id, kid)
    return ok()
end

"""
    debug_string(mem::Memory) -> String

Return a verbose description of `mem` for debugging.
    
!!! warning
The string is handed to C by pointer, so it must be owned by `mem`.

Backs `PJRT_Memory_DebugString`.
"""
debug_string(::Memory)

@pjrt_catch_return function memory_debug_string(
    args::Ptr{CAPI.PJRT_Memory_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = debug_string(memory)::String
    unsafe_store!(args.debug_string, Cstring(pointer(str)))
    unsafe_store!(args.debug_string_size, sizeof(str))
    return ok()
end

"""
    to_string(mem::Memory) -> String

Return a short, user-facing description of `mem`.

!!! warning
The string is handed to C by pointer, so it must be owned by `mem`.

Backs `PJRT_Memory_ToString`.
"""
to_string(::Memory)

@pjrt_catch_return function memory_to_string(
    args::Ptr{CAPI.PJRT_Memory_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    str = to_string(memory)::String
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, sizeof(str))
    return ok()
end

"""
    addressable_devices(mem::Memory) -> Vector{Device}

Return the devices that can address `mem`.

!!! warning
The vector is handed to C by pointer, so it must be owned by `mem` and never be resized.

Backs `PJRT_Memory_AddressableByDevices`.
"""
addressable_devices(::Memory)

@pjrt_catch_return function memory_addressable_by_devices(
    args::Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    memory = unsafe_pointer_to_objref(Ptr{Memory}(unsafe_load(args.memory)))::Memory
    devs = addressable_devices(memory)
    unsafe_store!(args.devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_devices, length(devs))
    return ok()
end

"""
    DeviceDescription

### Interface

- `id`
- `pid`
- `kind`
- `debug_string`
- `to_string`
- `attributes`
"""
abstract type DeviceDescription end

"""
    id(desc::DeviceDescription) -> Integer

Return the id of the device described by `desc`, unique among the devices of its platform across all processes.

Backs `PJRT_DeviceDescription_Id`.
"""
id(::DeviceDescription)

@pjrt_catch_return function device_description_id(
    args::Ptr{CAPI.PJRT_DeviceDescription_Id_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    unsafe_store!(args.id, id(desc))
    return ok()
end

"""
    process_index(desc::DeviceDescription) -> Integer

Return the index of the process that owns the device described by `desc`. Always `0` in single-process settings.

Backs `PJRT_DeviceDescription_ProcessIndex`.
"""
process_index(::DeviceDescription)

@pjrt_catch_return function device_description_process_index(
    args::Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    unsafe_store!(args.process_index, process_index(desc))
    return ok()
end

"""
    kind(desc::DeviceDescription) -> String

Return a vendor-dependent string that uniquely identifies the kind of device (e.g. `"Tesla V100-SXM2-16GB"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `desc`.

Backs `PJRT_DeviceDescription_Kind`.
"""
kind(::DeviceDescription)

@pjrt_catch_return function device_description_kind(
    args::Ptr{CAPI.PJRT_DeviceDescription_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = kind(desc)::String
    unsafe_store!(args.device_kind, Cstring(pointer(str)))
    unsafe_store!(args.device_kind_size, sizeof(str))
    return ok()
end

"""
    attributes(desc::DeviceDescription)

Return the device-specific attributes of the device described by `desc` as name-value pairs.
PJRT supports values of type `String`, `Int64`, `Vector{Int64}`, `Float32` and `Bool`.

!!! warning
`PjRtCApiClient` copies the attributes once, when it wraps the device description, and treats any error as fatal.

!!! todo
Not supported yet: this method is never called and `PJRT_DeviceDescription_Attributes` always returns an empty list.

Backs `PJRT_DeviceDescription_Attributes`.
"""
attributes(::DeviceDescription)

# TODO find the interface of `attributes`
@pjrt_catch_return function device_description_attributes(
    args::Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    desc = unsafe_pointer_to_objref(unsafe_load(args.device_description))::DeviceDescription
    # TODO attributes not yet supported
    # attrs = attributes(desc)
    unsafe_store!(args.num_attributes, 0)
    unsafe_store!(args.attributes, C_NULL)
    return ok()
end

"""
    debug_string(desc::DeviceDescription) -> String

Return a verbose description of the device for debugging.
    
!!! warning
The string is handed to C by pointer, so it must be owned by `desc`.

Backs `PJRT_DeviceDescription_DebugString`.
"""
debug_string(::DeviceDescription)

@pjrt_catch_return function device_description_debug_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = debug_string(desc)::String
    unsafe_store!(args.debug_string, Cstring(pointer(str)))
    unsafe_store!(args.debug_string_size, sizeof(str))
    return ok()
end

"""
    to_string(desc::DeviceDescription) -> String

Return a short, user-facing description of the device.

!!! warning
The string is handed to C by pointer, so it must be owned by `desc`.

Backs `PJRT_DeviceDescription_ToString`.
"""
to_string(::DeviceDescription)

@pjrt_catch_return function device_description_to_string(
    args::Ptr{CAPI.PJRT_DeviceDescription_ToString_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = to_string(desc)
    unsafe_store!(args.to_string, Cstring(pointer(str)))
    unsafe_store!(args.to_string_size, sizeof(str))
    return ok()
end

"""
    Device

### Interface

- `description`
- `is_addressable`
- `local_hardware_id`
- `addressable_memories`
- `default_memory`
- `memory_stats`
"""
abstract type Device end
function checkinterface(T::Type{<:Device})
    if !ismutabletype(T)
        error("$T <: Device must be a mutable type to be GC rootable")
    end
end

"""
    description(dev::Device) -> DeviceDescription

Return the description of `dev`. The description must be kept alive by `dev`.

Backs `PJRT_Device_GetDescription`.
"""
description(::Device)

@pjrt_catch_return function device_get_description(
    args::Ptr{CAPI.PJRT_Device_GetDescription_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    desc = description(device)::DeviceDescription
    unsafe_store!(
        args.device_description, Ptr{CAPI.PJRT_DeviceDescription}(pointer_from_objref(desc))
    )
    return ok()
end

"""
    is_addressable(dev::Device) -> Bool

Return whether `dev` can be addressed by the client of this process.

Backs `PJRT_Device_IsAddressable`.
"""
is_addressable(::Device)

@pjrt_catch_return function device_is_addressable(
    args::Ptr{CAPI.PJRT_Device_IsAddressable_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.is_addressable, is_addressable(device)::Bool)
    return ok()
end

"""
    local_hardware_id(dev::Device) -> Integer

Return an opaque hardware id of `dev` (e.g. the CUDA device number).
Use the vendor's handle for the physical device.
Not guaranteed to be dense (i.e. `∈(0:n-1)`), and `-1` if undefined.

Backs `PJRT_Device_LocalHardwareId`.
"""
local_hardware_id(::Device)

@pjrt_catch_return function device_local_hardware_id(
    args::Ptr{CAPI.PJRT_Device_LocalHardwareId_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_store!(args.local_hardware_id, local_hardware_id(device))
    return ok()
end

"""
    addressable_memories(dev::Device) -> Vector{Memory}

Return the memories that `dev` can address.

!!! warning
The vector is handed to C by pointer, so it must be owned by `dev` and never be resized.

Backs `PJRT_Device_AddressableMemories`.
"""
addressable_memories(::Device)

@pjrt_catch_return function device_addressable_memories(
    args::Ptr{CAPI.PJRT_Device_AddressableMemories_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mems = addressable_memories(device)::Vector{Memory}
    unsafe_store!(args.memories, Ptr{Ptr{CAPI.PJRT_Memory}}(pointer(mems)))
    unsafe_store!(args.num_memories, length(mems))
    return ok()
end

"""
    default_memory(dev::Device) -> Memory

Return the memory where buffers are placed on `dev` when no memory is given. Must be one of `addressable_memories(dev)`.

Backs `PJRT_Device_DefaultMemory`.
"""
default_memory(::Device)

@pjrt_catch_return function device_default_memory(
    args::Ptr{CAPI.PJRT_Device_DefaultMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mem = default_memory(device)::Memory
    unsafe_store!(args.memory, Ptr{CAPI.PJRT_Memory}(pointer_from_objref(mem)))
    return ok()
end

# TODO document and add interface function
@pjrt_catch_return function device_memory_stats(
    args::Ptr{CAPI.PJRT_Device_MemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    throw(unimplemented())
end

"""
    Buffer

### Interface

- `destroy`
- `delete!`
- `isdeleted`
- `element_type` -> eltype
- `dimensions` -> size
- `on_device_size_in_bytes` -> sizeof?
- `device`
- `memory`
- `copy_to_device` -> copyto!?
- `copy_to_memory`
- `to_host_buffer` -> copyto!?
- `is_on_cpu`
- `ready_event`
- `unsafe_pointer`
"""
abstract type Buffer end

@pjrt_catch_return function buffer_destroy(
    args::Ptr{CAPI.PJRT_Buffer_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    delete!(buffer)
    unroot!(buffer)
    return ok()
end

"""
    delete!(buffer::Buffer)

Free the data of `buffer` while keeping its metadata alive. Must be idempotent.
After this call, `isdeleted(buffer)` must return `true`.

Backs `PJRT_Buffer_Delete`.
"""
Base.delete!(::Buffer)

@pjrt_catch_return function buffer_delete(
    args::Ptr{CAPI.PJRT_Buffer_Delete_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    delete!(buffer)
    return ok()
end

"""
    isdeleted(buffer::Buffer) -> Bool

Return whether the data of `buffer` has been freed with `delete!`.

Backs `PJRT_Buffer_IsDeleted`.
"""
isdeleted(::Buffer)

@pjrt_catch_return function buffer_is_deleted(
    args::Ptr{CAPI.PJRT_Buffer_IsDeleted_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    pred = isdeleted(buffer)::Bool
    unsafe_store!(args.is_deleted, pred)
    return ok()
end

"""
    element_type(buffer::Buffer) -> CAPI.PJRT_Buffer_Type

Return the PJRT element type of `buffer`.

Backs `PJRT_Buffer_ElementType`.
"""
element_type(::Buffer)

@pjrt_catch_return function buffer_element_type(
    args::Ptr{CAPI.PJRT_Buffer_ElementType_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    # _eltype = eltype(buffer)::Type
    # _pjrt_eltype = pjrt_eltype(_eltype)::CAPI.PJRT_Buffer_Type
    _pjrt_eltype = element_type(buffer)::CAPI.PJRT_Buffer_Type
    unsafe_store!(args.type, _pjrt_eltype)
    return ok()
end

"""
    dimensions(buffer::Buffer) -> Vector{Int64}

Return the dimensions of `buffer`.

!!! warning
The returned vector is handed to C by pointer, so it must be owned by `buffer` and never be resized.

Backs `PJRT_Buffer_Dimensions`.
"""
dimensions(::Buffer)

@pjrt_catch_return function buffer_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dims = dimensions(buffer)::Vector{Int}
    unsafe_store!(args.dims, pointer(dims))
    unsafe_store!(args.num_dims, length(dims))
    return ok()
end

"""
    device(buffer::Buffer) -> Device

Return the device where `buffer` resides. The device must be kept alive by its client.

Backs `PJRT_Buffer_Device`.
"""
device(::Buffer)

@pjrt_catch_return function buffer_device(
    args::Ptr{CAPI.PJRT_Buffer_Device_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dev = device(buffer)::Device
    ptr = Ptr{CAPI.PJRT_Device}(pointer_from_objref(dev))
    unsafe_store!(args.device, ptr)
    return ok()
end

"""
    memory(buffer::Buffer) -> Memory

Return the memory space where `buffer` resides. The memory must be kept alive by its client.

Backs `PJRT_Buffer_Memory`.
"""
memory(::Buffer)

@pjrt_catch_return function buffer_memory(
    args::Ptr{CAPI.PJRT_Buffer_Memory_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    mem = memory(buffer)::Memory
    ptr = Ptr{CAPI.PJRT_Memory}(pointer_from_objref(mem))
    unsafe_store!(args.memory, ptr)
    return ok()
end

"""
    on_device_size_in_bytes(buffer::Buffer) -> Integer

Return the number of bytes that `buffer` occupies on its device.

Backs `PJRT_Buffer_OnDeviceSizeInBytes`.
"""
on_device_size_in_bytes(::Buffer)

@pjrt_catch_return function buffer_on_device_size_in_bytes(
    args::Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    _size = on_device_size_in_bytes(buffer)
    unsafe_store!(args.on_device_size_in_bytes, Csize_t(_size))
    return ok()
end

"""
    copy_to_device(dst::Device, buffer::Buffer) -> Buffer

Return a new buffer with a copy of the data of `buffer` placed on `dst`.
Must fail if `buffer` already resides on `dst`.
The returned buffer is GC rooted by the caller.

Backs `PJRT_Buffer_CopyToDevice`.
"""
copy_to_device(::Device, ::Buffer)

@pjrt_catch_return function buffer_copy_to_device(
    args::Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dst_dev = unsafe_pointer_to_objref(unsafe_load(args.dst_device))::Device

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    dst_buffer = copy_to_device(dst_dev, buffer)::Buffer
    root!(dst_buffer)
    ptr = Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(dst_buffer))
    unsafe_store!(args.dst_buffer, ptr)
    return ok()
end

"""
    copy_to_memory(dst::Memory, buffer::Buffer) -> Buffer

Return a new buffer with a copy of the data of `buffer` placed on memory space `dst`.
Unlike the PJRT C API header says, `dst` may be the memory of `buffer`: XLA's CPU client allows it and Reactant's
`CopyBufferToDevice` relies on it.
The returned buffer is GC rooted by the caller.

Backs `PJRT_Buffer_CopyToMemory`.
"""
copy_to_memory(::Memory, ::Buffer)

@pjrt_catch_return function buffer_copy_to_memory(
    args::Ptr{CAPI.PJRT_Buffer_CopyToMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dst_mem = unsafe_pointer_to_objref(unsafe_load(args.dst_memory))::Memory

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    dst_buffer = copy_to_memory(dst_mem, buffer)::Buffer
    root!(dst_buffer)
    ptr = Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(dst_buffer))
    unsafe_store!(args.dst_buffer, ptr)
    return ok()
end

"""
    to_host_buffer(dst::Ptr, buffer::Buffer)

Copy the data of `buffer` to the host memory pointed by `dst`, which must hold at least
`on_device_size_in_bytes(buffer)` bytes.

!!! todo
The copy is _currently_ synchronous.

Backs `PJRT_Buffer_ToHostBuffer`.
"""
to_host_buffer(::Ptr, ::Buffer)

@pjrt_catch_return function buffer_to_host_buffer(
    args::Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.src))::Buffer
    dst = unsafe_load(args.dst)
    src_size = on_device_size_in_bytes(buffer)

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    # size query shortcut
    if dst == C_NULL
        _size = src_size
        unsafe_store!(args.dst_size, Csize_t(_size))
        return ok()
    end

    dst_size = unsafe_load(args.dst_size)
    if dst_size < src_size
        throw(
            invalid_argument(
                "passed a host buffer of $dst_size bytes when $src_size bytes at least are required",
            ),
        )
    end

    # TODO better support for layouts: relayout is not supported, so `host_layout` must match the dense layout of `src`
    host_layout = unsafe_load(args.host_layout)
    if host_layout != C_NULL
        type = unsafe_load(host_layout.type)
        type != CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled &&
            throw(unimplemented("different layout not yet supported"))

        tiled = unsafe_load(host_layout.tiled)
        tiled.num_tiles != 0 && throw(unimplemented())
        _ndims = length(dimensions(buffer))
        tiled.minor_to_major_size != _ndims &&
            throw(invalid_argument("different layout not yet supported"))
        for i in 1:_ndims
            m2m = unsafe_load(tiled.minor_to_major, i)
            # TODO replace `.minor_to_major` for a function call
            m2m != buffer.minor_to_major[i] &&
                throw(unimplemented("different layout not yet supported"))
        end
    end

    to_host_buffer(dst, buffer)

    event = root!(Event())
    notify(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    is_on_cpu(buffer::Buffer) -> Bool

Return whether the data of `buffer` resides in host memory.

Backs `PJRT_Buffer_IsOnCpu`.
"""
is_on_cpu(::Buffer)

@pjrt_catch_return function buffer_is_on_cpu(
    args::Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    pred = is_on_cpu(buffer)::Bool
    unsafe_store!(args.is_on_cpu, pred)
    return ok()
end

"""
    ready_event(buffer::Buffer) -> Event

Return a new [`Event`](@ref) that is set once the data of `buffer` is ready, or with an [`Error`](@ref) if
producing it failed. A new event is needed on each call because the caller owns it and destroys it.
The returned event is GC rooted by the caller.

If `buffer` is deleted, the C wrapper returns an event set with an error without calling `ready_event`.

Backs `PJRT_Buffer_ReadyEvent`.
"""
ready_event(::Buffer)

@pjrt_catch_return function buffer_ready_event(
    args::Ptr{CAPI.PJRT_Buffer_ReadyEvent_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer

    if isdeleted(buffer)
        event = Event()
        notify(event, invalid_argument("buffer is deleted"))
    else
        event = ready_event(buffer)::Event
    end
    root!(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    unsafe_pointer(buffer::Buffer) -> Ptr

Return the raw pointer to the data of `buffer`. The pointer is only valid until `buffer` is deleted or destroyed.

Backs `PJRT_Buffer_UnsafePointer`.
"""
unsafe_pointer(::Buffer)

@pjrt_catch_return function buffer_unsafe_pointer(
    args::Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    ptr = unsafe_pointer(buffer)::Ptr
    unsafe_store!(args.buffer_pointer, UInt(ptr))
    return ok()
end

"""
    Executable

A compiled program, independent of the devices it runs on. Returned by `executable(::LoadedExecutable)`.

### Interface

- `name`
- `num_replicas`
- `num_partitions`
- `num_outputs`
- `size_of_generated_code_in_bytes`
- `fingerprint`
- `output_element_types`
- `optimized_program`
- `serialize`
- `compile_options`
"""
abstract type Executable end

@pjrt_catch_return function executable_destroy(
    args::Ptr{CAPI.PJRT_Executable_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.executable)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

# frees the rooted bytes handed out by `PJRT_LoadedExecutable_GetDeviceAssignment`, `PJRT_Executable_Serialize` and
# `PJRT_Executable_GetCompileOptions`
function serialized_bytes_deleter(ptr::Ptr{Cvoid})::Cvoid
    ptr != C_NULL && unroot!(ptr)
    return nothing
end

"""
    name(exe::Executable) -> String

Return a string that identifies `exe` (e.g. the name of the compiled module).

!!! warning
The string is handed to C by pointer, so it must be owned by `exe`.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_Name`.
"""
name(::Executable)

@pjrt_catch_return function executable_name(
    args::Ptr{CAPI.PJRT_Executable_Name_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    str = name(exe)::String
    unsafe_store!(args.executable_name, Cstring(pointer(str)))
    unsafe_store!(args.executable_name_size, sizeof(str))
    return ok()
end

"""
    num_replicas(exe::Executable) -> Integer

Return the number of replicas of `exe`, i.e. `executable_build_options.num_replicas` of the `CompileOptionsProto` it
was compiled with.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_NumReplicas`.
"""
num_replicas(::Executable)

@pjrt_catch_return function executable_num_replicas(
    args::Ptr{CAPI.PJRT_Executable_NumReplicas_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.num_replicas, num_replicas(exe))
    return ok()
end

"""
    num_partitions(exe::Executable) -> Integer

Return the number of partitions of `exe`, i.e. `executable_build_options.num_partitions` of the `CompileOptionsProto`
it was compiled with.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_NumPartitions`.
"""
num_partitions(::Executable)

@pjrt_catch_return function executable_num_partitions(
    args::Ptr{CAPI.PJRT_Executable_NumPartitions_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.num_partitions, num_partitions(exe))
    return ok()
end

"""
    num_outputs(exe::Executable) -> Integer

Return the number of outputs that `exe` produces per device.

`PjRtCApiLoadedExecutable` uses it to allocate the output lists filled by `execute`, so it must match the length of
each output vector that `execute` returns.

Backs `PJRT_Executable_NumOutputs`.
"""
num_outputs(::Executable)

@pjrt_catch_return function executable_num_outputs(
    args::Ptr{CAPI.PJRT_Executable_NumOutputs_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.num_outputs, num_outputs(exe))
    return ok()
end

"""
    size_of_generated_code_in_bytes(exe::Executable) -> Integer

Return the size in bytes of the code generated for `exe`, or `0` if unknown.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_SizeOfGeneratedCodeInBytes`.
"""
size_of_generated_code_in_bytes(::Executable)

@pjrt_catch_return function executable_size_of_generated_code_in_bytes(
    args::Ptr{CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.size_in_bytes, size_of_generated_code_in_bytes(exe))
    return ok()
end

"""
    fingerprint(exe::Executable) -> String

Return a fingerprint of `exe`. Executables compiled from identical inputs (program, compile options, compiler
version, ...) must have the same fingerprint. Throw `unimplemented` if not supported.

!!! warning
The string is handed to C by pointer, so it must be owned by `exe`.

Backs `PJRT_Executable_Fingerprint`.
"""
fingerprint(::Executable)

@pjrt_catch_return function executable_fingerprint(
    args::Ptr{CAPI.PJRT_Executable_Fingerprint_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    str = fingerprint(exe)::String
    unsafe_store!(args.executable_fingerprint, Cstring(pointer(str)))
    unsafe_store!(args.executable_fingerprint_size, sizeof(str))
    return ok()
end

"""
    output_element_types(exe::Executable) -> Vector{CAPI.PJRT_Buffer_Type}

Return the element type of each output of `exe`.

!!! warning
The vector is handed to C by pointer, so it must be owned by `exe` and never be resized.

Backs `PJRT_Executable_OutputElementTypes`.
"""
output_element_types(::Executable)

@pjrt_catch_return function executable_output_element_types(
    args::Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    types = output_element_types(exe)::Vector{CAPI.PJRT_Buffer_Type}
    unsafe_store!(args.output_types, pointer(types))
    unsafe_store!(args.num_output_types, length(types))
    return ok()
end

"""
    optimized_program(exe::Executable) -> String

Return the optimized program of `exe` as MLIR text (StableHLO dialect).

`PJRT_Executable_OptimizedProgram` is called twice per query (first for the size, then for the data), so this must
return the same string every time.

Backs `PJRT_Executable_OptimizedProgram`, which `PjRtCApiExecutable` uses for `GetHloModules` and for the default
output and parameter shardings.
"""
optimized_program(::Executable)

@pjrt_catch_return function executable_optimized_program(
    args::Ptr{CAPI.PJRT_Executable_OptimizedProgram_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    code = optimized_program(exe)::String
    program = unsafe_load(args.program)

    # string literals are never freed, so the implementation keeps owning `format`
    format = "mlir"
    unsafe_store!(program.format, Cstring(pointer(format)))
    unsafe_store!(program.format_size, sizeof(format))

    # first call: only report the size of the buffer that the caller must allocate
    code_ptr = Ptr{UInt8}(unsafe_load(program.code))
    if code_ptr == C_NULL
        unsafe_store!(program.code_size, sizeof(code))
        return ok()
    end

    # second call: `code_size` holds the size of the caller's buffer
    capacity = unsafe_load(program.code_size)
    if capacity < sizeof(code)
        throw(
            invalid_argument(
                "program buffer of $capacity bytes is smaller than the $(sizeof(code)) bytes required",
            ),
        )
    end
    GC.@preserve code unsafe_copyto!(code_ptr, pointer(code), sizeof(code))
    unsafe_store!(program.code_size, sizeof(code))
    return ok()
end

"""
    serialize(exe::Executable) -> Vector{UInt8}

Return a platform-specific serialization of `exe`. It doesn't need to be stable across versions.

The returned vector is GC rooted until C calls its deleter, so it must be a new vector that `exe` doesn't hand out
anywhere else.

Backs `PJRT_Executable_Serialize`.
"""
serialize(::Executable)

@pjrt_catch_return function executable_serialize(
    args::Ptr{CAPI.PJRT_Executable_Serialize_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    bytes = root!(serialize(exe)::Vector{UInt8})
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(args.serialized_executable, pointer_from_objref(bytes))
    unsafe_store!(
        args.serialized_executable_deleter,
        @cfunction(serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

"""
    compile_options(exe::Executable) -> Reactant.Proto.xla.CompileOptionsProto

Return the compile options that `exe` was compiled with, i.e. the ones given to `compile!`.

Backs `PJRT_Executable_GetCompileOptions`.
"""
compile_options(::Executable)

@pjrt_catch_return function executable_get_compile_options(
    args::Ptr{CAPI.PJRT_Executable_GetCompileOptions_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    proto = compile_options(exe)::Reactant.Proto.xla.CompileOptionsProto
    bytes = root!(Reactant.ProtoUtils.proto_to_bytes(proto))
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(args.serialized_compile_options, pointer_from_objref(bytes))
    unsafe_store!(
        args.serialized_compile_options_deleter,
        @cfunction(serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

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

@pjrt_catch_return function loaded_executable_destroy(
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

@pjrt_catch_return function loaded_executable_get_executable(
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

@pjrt_catch_return function loaded_executable_delete(
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

@pjrt_catch_return function loaded_executable_is_deleted(
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

@pjrt_catch_return function loaded_executable_addressable_devices(
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

@pjrt_catch_return function loaded_executable_addressable_device_logical_ids(
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

@pjrt_catch_return function loaded_executable_get_device_assignment(
    args::Ptr{CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args}
)::Ptr{CAPI.PJRT_Error}
    exec = unsafe_pointer_to_objref(unsafe_load(args.executable))::LoadedExecutable
    proto = device_assignment(exec)::Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}

    # XLA calls the deleter even if there is no device assignment
    deleter = @cfunction(serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
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
    execute(exec::LoadedExecutable, arguments::Vector{Vector{Buffer}}, dev::Union{Nothing,Device}) -> Vector{Vector{Buffer}}

Run `exec` and return its outputs.

`arguments[i]` holds the input buffers for the `i`-th device of `addressable_devices(exec)`. The result must follow the
same layout: one vector per device, each with as many new buffers as `exec` has outputs.

If `dev` is not `nothing`, run only on `dev` (one of `addressable_devices(exec)`). Then `arguments` and the result
have a single entry, for `dev`.

Execution is synchronous: the device-complete events are signaled as soon as this returns. The caller GC-roots the
returned buffers and destroys each of them, so they must all be distinct.

!!! todo
`PJRT_ExecuteOptions` (e.g. non-donatable inputs, send/recv callbacks) are ignored.

Backs `PJRT_LoadedExecutable_Execute`.
"""
execute(::LoadedExecutable, args...)

@pjrt_catch_return function loaded_executable_execute(
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

    outputs = execute(exec, arguments, dev)::Vector{Vector{Buffer}}
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

"""
    Client

## Interface

- `create`
- `destroy!`
- `platform_name`
- `platform_version`
- `process_index`
- `devices`
- `addressable_devices`
- `addressable_memories`
- `buffer_from_host_buffer`
- `compile!`
"""
abstract type Client end

@pjrt_catch_return function client_destroy(
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

@pjrt_catch_return function client_platform_name(
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

@pjrt_catch_return function client_platform_version(
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

@pjrt_catch_return function client_process_index(
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

@pjrt_catch_return function client_devices(
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

@pjrt_catch_return function client_addressable_devices(
    args::Ptr{CAPI.PJRT_Client_AddressableDevices_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    devs = addressable_devices(client)::Vector{Device}
    unsafe_store!(args.addressable_devices, Ptr{Ptr{CAPI.PJRT_Device}}(pointer(devs)))
    unsafe_store!(args.num_addressable_devices, length(devs))
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

@pjrt_catch_return function client_addressable_memories(
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
        client::Client, dev::Device, mem::Memory, data::Ptr, type, dims, byte_strides, device_layout, host_buffer_semantics
    ) -> Buffer

Return a new buffer placed on memory `mem` of device `dev`, with a copy of the host `data`. The new buffer has element
type `type::CAPI.PJRT_Buffer_Type` and dimensions `dims::Vector{Int64}`.

## Arguments
- `mem` is always one of `addressable_memories(dev)`. PJRT may pass only one of the two, so the caller fills in the other:

  - only `dev` given: `mem` is `default_memory(dev)`.
  - only `mem` given: `dev` is one of `addressable_devices(mem)`.

- `byte_strides::Vector{Int64}` is empty for dense data in major-to-minor order.
- `device_layout::Ptr{CAPI.PJRT_Buffer_MemoryLayout}` is `C_NULL` for the default layout.
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

@pjrt_catch_return function client_buffer_from_host_buffer(
    args::Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client

    data = unsafe_load(args.data)
    _eltype = unsafe_load(args.type)

    _ndims = unsafe_load(args.num_dims)
    _dims_ptr = unsafe_load(args.dims)
    _size = copy(unsafe_wrap(Vector{Int64}, _dims_ptr, _ndims))

    byte_strides_ptr = unsafe_load(args.byte_strides)
    nbyte_strides = unsafe_load(args.num_byte_strides)

    # TODO byte strides not yet supported
    if byte_strides_ptr != C_NULL
        throw(unimplemented("Buffer with byte strides not yet supported"))
    end
    byte_strides = unsafe_wrap(Vector{Int64}, byte_strides_ptr, (nbyte_strides,))

    _device_layout = unsafe_load(args.device_layout)
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
            byte_strides,
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
            byte_strides,
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
    compile!(client::Client, code::Reactant.MLIR.IR.Module, options::Vector{UInt8})

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

@pjrt_catch_return function client_compile(
    args::Ptr{CAPI.PJRT_Client_Compile_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    program = Ptr{CAPI.PJRT_Program}(unsafe_load(args.program))
    format = unsafe_string(
        Ptr{Cchar}(unsafe_load(program.format)), unsafe_load(program.format_size)
    )
    format == "mlir" || throw(unimplemented("code in $format format is not supported"))
    # code = unsafe_string(
    #     Ptr{Cchar}(unsafe_load(program.code)), unsafe_load(program.code_size)
    # )
    code_str_ref = Reactant.MLIR.API.MlirStringRef(
        unsafe_load(program.code), unsafe_load(program.code_size)
    )
    ctx = Reactant.ReactantContext()
    mlir_mod = Reactant.MLIR.IR.Module(
        Reactant.MLIR.API.stablehloDeserializePortableArtifactNoError(code_str_ref, ctx)
    )
    # options = unsafe_string(
    #     Ptr{Cchar}(unsafe_load(args.compile_options)),
    #     unsafe_load(args.compile_options_size),
    # )
    opts_ptr = Ptr{UInt8}(unsafe_load(args.compile_options))
    opts_size = unsafe_load(args.compile_options_size)
    opts_bytes = unsafe_wrap(Vector{UInt8}, opts_ptr, opts_size)
    options = Reactant.ProtoUtils.proto_from_bytes(
        Reactant.Proto.xla.CompileOptionsProto, opts_bytes
    )
    loaded_exec = compile!(client, mlir_mod, options)::LoadedExecutable
    root!(loaded_exec)
    unsafe_store!(
        args.executable, Ptr{CAPI.PJRT_LoadedExecutable}(pointer_from_objref(loaded_exec))
    )
    return ok()
end

## HostMemory
mutable struct HostMemory <: Memory
    const vtable::Ptr{CAPI.PJRT_Memory_FunctionTable}
    const id::Cint
    const kind::String
    const debug_string::String
    const to_string::String
    const devices::Vector{Device}

    function HostMemory(id, kind, devices)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Memory_FunctionTable}, vtable_memory),
            id,
            kind,
            "Memory(id=$id, kind=$kind, platform=julia)",
            "Memory managed by Julia",
            devices,
        )
    end
end

id(mem::HostMemory) = mem.id
kind(mem::HostMemory) = mem.kind
kind_id(::HostMemory) = 0
debug_string(mem::HostMemory) = mem.debug_string
to_string(mem::HostMemory) = mem.to_string
addressable_devices(mem::HostMemory) = mem.devices

## DeviceDescription
mutable struct HostDeviceDescription <: DeviceDescription
    const id::Cint
    const pid::Cint
    const debug_string::String
    const to_string::String
end

function HostDeviceDescription()
    id = 0
    pid = 0
    return HostDeviceDescription(id, pid, "julia: id=$id", "JuliaDevice(id=$id)")
end

id(desc::HostDeviceDescription) = desc.id
process_index(desc::HostDeviceDescription) = desc.pid
kind(desc::HostDeviceDescription) = "julia-cpu"
debug_string(desc::HostDeviceDescription) = desc.debug_string
to_string(desc::HostDeviceDescription) = desc.to_string
# attributes(desc::HostDeviceDescription) = desc.

## Device
mutable struct HostDevice <: Device
    const description::DeviceDescription # TODO or HostDeviceDescription
    const local_hardware_id::Cint
    const memories::Vector{Memory}
end

description(dev::HostDevice) = dev.description
is_addressable(dev::HostDevice) = true
local_hardware_id(dev::HostDevice) = dev.local_hardware_id
addressable_memories(dev::HostDevice) = dev.memories
default_memory(dev::HostDevice) = dev.memories[1]
# memory_stats(dev::HostDevice) = 

## HostBuffer
mutable struct HostBuffer <: Buffer
    data::Ptr{Cvoid}
    const size_in_bytes::Csize_t
    const type::CAPI.PJRT_Buffer_Type
    const dims::Vector{Int64}
    const minor_to_major::Vector{Int64}
    const device::Device
    const memory::Memory

    function HostBuffer(args...)
        obj = new(args...)
        return Base.finalizer(obj) do x
            if !isdeleted(x)
                Libc.free(x.data)
            end
        end
    end
end

function Base.delete!(buffer::HostBuffer)
    if !isdeleted(buffer)
        Libc.free(buffer.data)
        buffer.data = C_NULL
    end
end

isdeleted(buffer::HostBuffer) = buffer.data == C_NULL
element_type(buffer::HostBuffer) = buffer.type
dimensions(buffer::HostBuffer) = buffer.dims
device(buffer::HostBuffer) = buffer.device
memory(buffer::HostBuffer) = buffer.memory
on_device_size_in_bytes(buffer::HostBuffer) = buffer.size_in_bytes

function copy_to_device(dev::HostDevice, buffer::HostBuffer)
    if dev === device(buffer)
        throw(invalid_argument("HostBuffer is already on HostDevice"))
    end
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    new_buffer = HostBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        default_memory(dev),
    )
    return new_buffer
end

function copy_to_memory(mem::HostMemory, buffer::HostBuffer)
    devs = addressable_devices(mem)
    isempty(devs) && throw(invalid_argument("HostMemory has no addressable devices"))
    dev = device(buffer) in devs ? device(buffer) : first(devs)
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    return HostBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        mem,
    )
end

function to_host_buffer(dst::Ptr{Cvoid}, buffer::HostBuffer)
    Libc.memcpy(dst, buffer.data, buffer.size_in_bytes)
    return nothing
end

is_on_cpu(::HostBuffer) = true

# `HostBuffer` data is written synchronously before the buffer is handed out, so it is always ready
function ready_event(::HostBuffer)
    event = Event()
    notify(event)
    return event
end

unsafe_pointer(buffer::HostBuffer) = buffer.data

## HostExecutable
mutable struct HostExecutable <: Executable
    const name::String
    const program::String
    const output_types::Vector{CAPI.PJRT_Buffer_Type}
    const options::Reactant.Proto.xla.CompileOptionsProto
end

name(exe::HostExecutable) = exe.name

function num_replicas(exe::HostExecutable)
    build_options = exe.options.executable_build_options
    return isnothing(build_options) ? 1 : max(build_options.num_replicas, 1)
end

function num_partitions(exe::HostExecutable)
    build_options = exe.options.executable_build_options
    return isnothing(build_options) ? 1 : max(build_options.num_partitions, 1)
end

num_outputs(exe::HostExecutable) = length(exe.output_types)
size_of_generated_code_in_bytes(::HostExecutable) = 0
fingerprint(::HostExecutable) = throw(unimplemented("`HostExecutable` has no fingerprint"))
output_element_types(exe::HostExecutable) = exe.output_types
optimized_program(exe::HostExecutable) = exe.program
function serialize(::HostExecutable)
    throw(unimplemented("serialization of `HostExecutable` is not supported yet"))
end
compile_options(exe::HostExecutable) = exe.options

## HostLoadedExecutable
# `Tag` is unique per compiled program, so that `compile!` can define an `execute` method for each one
mutable struct HostLoadedExecutable{Tag} <: LoadedExecutable
    const executable::HostExecutable
    const devices::Vector{Device}
    const logical_ids::Vector{CAPI.PJRT_LogicalDeviceIds}
    const device_assignment::Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}
    deleted::Bool
end

executable(exec::HostLoadedExecutable) = exec.executable

function Base.delete!(exec::HostLoadedExecutable)
    exec.deleted = true
    return nothing
end

isdeleted(exec::HostLoadedExecutable) = exec.deleted
addressable_devices(exec::HostLoadedExecutable) = exec.devices
addressable_device_logical_ids(exec::HostLoadedExecutable) = exec.logical_ids
device_assignment(exec::HostLoadedExecutable) = exec.device_assignment

"""
    handle(::Val{opname}, operands::Tuple, attributes::NamedTuple, results::Tuple)

Run the MLIR op `opname` (e.g. `Symbol("stablehlo.add")`) and return its results.

- `operands`: Julia arrays, i.e. row-major MLIR tensors seen as column-major arrays with reversed dimensions.
- `attributes`: the op attributes, decoded by `decode_attribute`.
- `results`: the type of each result, as `(; eltype, size)` with `size` in Julia order (i.e. reversed).

Like MLIR values, the return value depends on the number of results: it is ignored if there are none, it is the
result itself if there is one (`%N`), and it must be a tuple with one entry per result if there are several (`%N:2`,
bound to `%N#0` and `%N#1`).
"""
function handle end

# element-wise ops use `map` instead of broadcasting: StableHLO operands always have the same shape, and broadcasting
# would return a scalar over 0-dimensional arrays and a `BitArray` for booleans instead of an `Array`
handle(::Val{Symbol("stablehlo.add")}, (a, b), _, _) = map(+, a, b)
handle(::Val{Symbol("stablehlo.subtract")}, (a, b), _, _) = map(-, a, b)
handle(::Val{Symbol("stablehlo.multiply")}, (a, b), _, _) = map(*, a, b)
handle(::Val{Symbol("stablehlo.maximum")}, (a, b), _, _) = map(max, a, b)
handle(::Val{Symbol("stablehlo.minimum")}, (a, b), _, _) = map(min, a, b)
handle(::Val{Symbol("stablehlo.negate")}, (a,), _, _) = map(-, a)
handle(::Val{Symbol("stablehlo.abs")}, (a,), _, _) = map(abs, a)
handle(::Val{Symbol("stablehlo.floor")}, (a,), _, _) = map(floor, a)
handle(::Val{Symbol("stablehlo.and")}, (a, b), _, _) = map(&, a, b)
handle(::Val{Symbol("stablehlo.or")}, (a, b), _, _) = map(|, a, b)
handle(::Val{Symbol("stablehlo.not")}, (a,), _, _) = map(~, a)

# integer division truncates towards zero, like `div`
function handle(::Val{Symbol("stablehlo.divide")}, (a, b), _, (result,))
    return map(result.eltype <: Integer ? div : /, a, b)
end

handle(::Val{Symbol("stablehlo.constant")}, _, attributes, _) = attributes.value

# each element is its 0-based index along `iota_dimension`, an MLIR dimension, which is Julia dimension `N - d`
function handle(::Val{Symbol("stablehlo.iota")}, _, attributes, (result,))
    j = length(result.size) - attributes.iota_dimension
    return [result.eltype(I[j] - 1) for I in CartesianIndices(result.size)]
end

# Julia throws `DomainError` where StableHLO returns NaN (e.g. `sin(Inf)`, `log(-1.0)`, `sqrt(-1.0)`), so real
# inputs outside of the domain are mapped to NaN explicitly
function handle(::Val{Symbol("stablehlo.sine")}, (a,), _, _)
    return map(x -> x isa Real && isinf(x) ? oftype(x, NaN) : sin(x), a)
end

function handle(::Val{Symbol("stablehlo.cosine")}, (a,), _, _)
    return map(x -> x isa Real && isinf(x) ? oftype(x, NaN) : cos(x), a)
end

function handle(::Val{Symbol("stablehlo.log")}, (a,), _, _)
    return map(x -> x isa Real && x < 0 ? oftype(x, NaN) : log(x), a)
end

function handle(::Val{Symbol("stablehlo.sqrt")}, (a,), _, _)
    return map(x -> x isa Real && x < 0 ? oftype(x, NaN) : sqrt(x), a)
end

function handle(::Val{Symbol("stablehlo.rsqrt")}, (a,), _, _)
    return map(x -> x isa Real && x < 0 ? oftype(x, NaN) : inv(sqrt(x)), a)
end

# like XLA, integer powers with negative exponents are 0 except for bases 1 and -1 (which Julia's `^` handles), and
# negative real bases with finite non-integer exponents are NaN
function handle(::Val{Symbol("stablehlo.power")}, (a, b), _, _)
    return map(a, b) do x, y
        if x isa Integer
            y < 0 && abs(x) != 1 ? zero(x) : x^y
        elseif x isa Real && x < 0 && isfinite(y) && !isinteger(y)
            oftype(x, NaN)
        else
            x^y
        end
    end
end

# the sign of the result follows the dividend, like `rem`. like XLA, an integer remainder by zero is the dividend
function handle(::Val{Symbol("stablehlo.remainder")}, (a, b), _, _)
    return map((x, y) -> x isa Integer && iszero(y) ? x : rem(x, y), a, b)
end

# like XLA, integers wrap around, floats are truncated towards zero and saturate when converted to integers (NaN
# becomes 0), and complex numbers lose their imaginary part when converted to real numbers
function handle(::Val{Symbol("stablehlo.convert")}, (a,), _, (result,))
    T = result.eltype
    return map(a) do x
        if T === Bool
            !iszero(x)
        elseif x isa Complex && !(T <: Complex)
            convert(T, real(x))
        elseif T <: Integer && x isa Integer
            x % T
        elseif T <: Integer
            if isnan(x)
                zero(T)
            elseif x <= typemin(T)
                typemin(T)
            elseif x >= typemax(T)
                typemax(T)
            else
                trunc(T, x)
            end
        else
            convert(T, x)
        end
    end
end

# `compare_type` defaults to the natural order of the element type (IEEE for floats, so comparisons with NaN are false
# except for `NE`). with `TOTALORDER`, floats are compared by their bits as -NaN < -Inf < ... < -0.0 < +0.0 < ... <
# +Inf < +NaN, which is done by flipping all but the sign bit of negative floats
function handle(::Val{Symbol("stablehlo.compare")}, (a, b), attributes, _)
    direction = attributes.comparison_direction
    f = if direction === :EQ
        ==
    elseif direction === :NE
        !=
    elseif direction === :GE
        >=
    elseif direction === :GT
        >
    elseif direction === :LE
        <=
    elseif direction === :LT
        <
    else
        throw(invalid_argument("unknown comparison direction $direction"))
    end

    if get(attributes, :compare_type, :NOTYPE) === :TOTALORDER
        key =
            x -> if x isa Base.IEEEFloat
                (bits = reinterpret(Signed, x); bits < 0 ? bits ⊻ typemax(bits) : bits)
            else
                x
            end
        return map((x, y) -> f(key(x), key(y)), a, b)
    end
    return map(f, a, b)
end

# a 0-dimensional predicate selects whole operands, otherwise elements are selected one by one
function handle(::Val{Symbol("stablehlo.select")}, (pred, on_true, on_false), _, _)
    ndims(pred) == 0 && return pred[] ? on_true : on_false
    return map(ifelse, pred, on_true, on_false)
end

# the result has the batching dimensions, then the free (i.e. neither batching nor contracting) dimensions of `lhs`
# and then those of `rhs`, all in MLIR order. on the reversed arrays this is `rhs * lhs` for each batch: `rhs` is
# permuted and reshaped into a (rhs free × contracting × batch) array, `lhs` into a (contracting × lhs free × batch)
# array, and the product of each batch slice is written into a (rhs free × lhs free × batch) array.
# `dot_dimension_numbers` is decoded by `decode_attribute` with 0-based MLIR dimensions.
function handle(::Val{Symbol("stablehlo.dot_general")}, (lhs, rhs), attributes, (result,))
    dims = attributes.dot_dimension_numbers
    jdim(x, d) = ndims(x) - d

    # batching dimensions are reversed so that they flatten in the order of the result
    lb = [jdim(lhs, d) for d in reverse(dims.lhs_batching_dimensions)]
    rb = [jdim(rhs, d) for d in reverse(dims.rhs_batching_dimensions)]
    lc = [jdim(lhs, d) for d in dims.lhs_contracting_dimensions]
    rc = [jdim(rhs, d) for d in dims.rhs_contracting_dimensions]

    # free dimensions keep their MLIR order, which reversed is the increasing order of Julia dimensions
    lf = [j for j in 1:ndims(lhs) if j ∉ lb && j ∉ lc]
    rf = [j for j in 1:ndims(rhs) if j ∉ rb && j ∉ rc]

    nbatch = prod(j -> size(lhs, j), lb; init=1)
    k = prod(j -> size(lhs, j), lc; init=1)
    m = prod(j -> size(lhs, j), lf; init=1)
    n = prod(j -> size(rhs, j), rf; init=1)

    l = reshape(permutedims(lhs, [lc; lf; lb]), k, m, nbatch)
    r = reshape(permutedims(rhs, [rf; rc; rb]), n, k, nbatch)
    out = Array{result.eltype}(undef, n, m, nbatch)
    for (o, x, y) in zip(eachslice(out; dims=3), eachslice(r; dims=3), eachslice(l; dims=3))
        mul!(o, x, y)
    end
    return reshape(out, result.size)
end

# shape handlers. arrays have reversed dimensions, so MLIR dimension `d` (0-based) of a rank-`N` tensor is Julia
# dimension `N - d`, and attributes indexed by MLIR dimension (e.g. `permutation`) are read in reverse

# reshapes follow the row-major order of MLIR tensors, which is the column-major order of the reversed arrays
function handle(::Val{Symbol("stablehlo.reshape")}, (a,), _, (result,))
    return reshape(a, result.size)
end

function handle(::Val{Symbol("stablehlo.transpose")}, (a,), attributes, _)
    perm = attributes.permutation
    N = length(perm)
    return permutedims(a, ntuple(j -> N - perm[N - j + 1], N))
end

# `start_indices` are 0-based and `limit_indices` are exclusive
function handle(::Val{Symbol("stablehlo.slice")}, (a,), attributes, _)
    (; start_indices, limit_indices, strides) = attributes
    N = length(start_indices)
    ranges = ntuple(N) do j
        d = N - j + 1
        (start_indices[d] + 1):strides[d]:limit_indices[d]
    end
    # indexing a 0-dimensional array with no indices would return a scalar
    return copy(view(a, ranges...))
end

# like in StableHLO, start indices are clamped so that the slice is always within bounds
function handle(::Val{Symbol("stablehlo.dynamic_slice")}, operands, attributes, _)
    a = first(operands)
    starts = Base.tail(operands)
    sizes = attributes.slice_sizes
    N = length(sizes)
    ranges = ntuple(N) do j
        d = N - j + 1
        start = clamp(Int(starts[d][]), 0, size(a, j) - sizes[d])
        (start + 1):(start + sizes[d])
    end
    return copy(view(a, ranges...))
end

function handle(::Val{Symbol("stablehlo.dynamic_update_slice")}, operands, _, _)
    a, update = operands[1], operands[2]
    starts = operands[3:end]
    N = ndims(a)
    ranges = ntuple(N) do j
        start = clamp(Int(starts[N - j + 1][]), 0, size(a, j) - size(update, j))
        (start + 1):(start + size(update, j))
    end

    # the operand may be an input or a constant, so the update goes into a copy
    result = copy(a)
    view(result, ranges...) .= update
    return result
end

# MLIR dimension `i` of the operand is MLIR dimension `broadcast_dimensions[i]` of the result. the operand is permuted
# so that its dimensions follow the order of the result ones, reshaped with size 1 for the missing dimensions and
# broadcast into the result
function handle(::Val{Symbol("stablehlo.broadcast_in_dim")}, (a,), attributes, (result,))
    bdims = attributes.broadcast_dimensions
    N = length(result.size)
    M = ndims(a)

    # result Julia dimension of each operand Julia dimension
    rdims = [N - bdims[M - j + 1] for j in 1:M]
    perm = sortperm(rdims)
    shape = ntuple(Val(N)) do k
        j = findfirst(==(k), rdims)
        isnothing(j) ? 1 : size(a, j)
    end

    out = Array{result.eltype}(undef, result.size)
    out .= reshape(permutedims(a, perm), shape)
    return out
end

function handle(::Val{Symbol("stablehlo.concatenate")}, operands, attributes, (result,))
    N = length(result.size)
    j = N - attributes.dimension
    out = Array{result.eltype}(undef, result.size)
    offset = 0
    for x in operands
        n = size(x, j)
        view(
            out,
            ntuple(k -> k == j ? ((offset + 1):(offset + n)) : (1:size(out, k)), Val(N))...,
        ) .= x
        offset += n
    end
    return out
end

# element `k` (1-based) of the operand along a dimension goes to `low + (k - 1) * (interior + 1) + 1` of the result.
# negative `low` or `high` crop the operand, so only the elements that land within the result are copied
function handle(::Val{Symbol("stablehlo.pad")}, (a, padding_value), attributes, (result,))
    (; edge_padding_low, interior_padding) = attributes
    N = length(result.size)
    out = fill(convert(result.eltype, padding_value[]), result.size)

    src = ntuple(Val(N)) do j
        d = N - j + 1
        lo, step = edge_padding_low[d], interior_padding[d] + 1
        kmin = max(1, cld(-lo, step) + 1)
        kmax = min(size(a, j), fld(size(out, j) - 1 - lo, step) + 1)
        kmin:kmax
    end
    dst = ntuple(Val(N)) do j
        d = N - j + 1
        lo, step = edge_padding_low[d], interior_padding[d] + 1
        (lo + (first(src[j]) - 1) * step + 1):step:(lo + (last(src[j]) - 1) * step + 1)
    end
    view(out, dst...) .= view(a, src...)
    return out
end

function __gather_scatter_julia_index(idx, ::Val{N}) where {N}
    return CartesianIndex(ntuple(j -> idx[N - j + 1] + 1, Val(N)))
end

# for each result index, the operand index is the clamped start index (read from `start_indices` along
# `index_vector_dim` and placed by `start_index_map`), plus the batching index and the offset within the slice
function handle(
    ::Val{Symbol("stablehlo.gather")}, (operand, start_indices), attributes, (result,)
)
    dn = attributes.dimension_numbers
    slice_sizes = attributes.slice_sizes
    ivd = dn.index_vector_dim
    out = Array{result.eltype}(undef, result.size)

    batch_dims = [d for d in 0:(ndims(out) - 1) if d ∉ dn.offset_dims]
    offset_operand_dims = [
        d for d in 0:(ndims(operand) - 1) if
        d ∉ dn.collapsed_slice_dims && d ∉ dn.operand_batching_dims
    ]

    operand_index = zeros(Int, ndims(operand))
    sindex = zeros(Int, ndims(start_indices))
    for I in CartesianIndices(out)
        # `gather` follow the StableHLO spec, which is written in 0-based MLIR indices
        # these convert them from and to the Julia indices of the reversed arrays
        ridx = reverse(Tuple(I)) .- 1
        fill!(operand_index, 0)

        # the batch index, with a gap at `index_vector_dim` for the component of the start index
        for (k, d) in enumerate(batch_dims)
            sindex[k <= ivd ? k : k + 1] = ridx[d + 1]
        end
        for (k, d_operand) in enumerate(dn.start_index_map)
            ivd < ndims(start_indices) && (sindex[ivd + 1] = k - 1)
            start = Int(
                start_indices[__gather_scatter_julia_index(
                    sindex, Val(ndims(start_indices))
                )],
            )
            mlir_size = size(operand, ndims(operand) - d_operand)
            operand_index[d_operand + 1] = clamp(
                start, 0, mlir_size - slice_sizes[d_operand + 1]
            )
        end
        for (i, d_operand) in enumerate(dn.operand_batching_dims)
            d_start = dn.start_indices_batching_dims[i]
            operand_index[d_operand + 1] += ridx[batch_dims[d_start - (d_start < ivd ? 0 : 1) + 1] + 1]
        end
        for (k, d) in enumerate(dn.offset_dims)
            operand_index[offset_operand_dims[k] + 1] += ridx[d + 1]
        end

        out[I] = operand[__gather_scatter_julia_index(operand_index, Val(ndims(operand)))]
    end
    return out
end

# operands are the `inputs`, `scatter_indices` and `updates`, and `update_computation` combines an element of each
# input with the corresponding element of each update (all as 0-dimensional arrays). like in the spec, updates whose
# index falls outside of the inputs are dropped, and they are applied sequentially in index order
function handle(
    ::Val{Symbol("stablehlo.scatter")}, operands, attributes, results, (update_computation,)
)
    n = length(results)
    outs = ntuple(i -> copy(operands[i]), Val(n))
    scatter_indices = operands[n + 1]
    updates = ntuple(i -> operands[n + 1 + i], Val(n))

    dn = attributes.scatter_dimension_numbers
    ivd = dn.index_vector_dim
    R = ndims(first(outs))
    update_scatter_dims = [
        d for d in 0:(ndims(first(updates)) - 1) if d ∉ dn.update_window_dims
    ]
    window_input_dims = [
        d for d in 0:(R - 1) if d ∉ dn.inserted_window_dims && d ∉ dn.input_batching_dims
    ]

    result_index = zeros(Int, R)
    sindex = zeros(Int, ndims(scatter_indices))
    for I in CartesianIndices(first(updates))
        # `scatter` follow the StableHLO spec, which is written in 0-based MLIR indices
        # these convert them from and to the Julia indices of the reversed arrays
        uidx = reverse(Tuple(I)) .- 1
        fill!(result_index, 0)

        for (k, d) in enumerate(update_scatter_dims)
            sindex[k <= ivd ? k : k + 1] = uidx[d + 1]
        end
        for (k, d_input) in enumerate(dn.scatter_dims_to_operand_dims)
            ivd < ndims(scatter_indices) && (sindex[ivd + 1] = k - 1)
            result_index[d_input + 1] = Int(
                scatter_indices[__gather_scatter_julia_index(
                    sindex, Val(ndims(scatter_indices))
                )],
            )
        end
        for (i, d_input) in enumerate(dn.input_batching_dims)
            d_start = dn.scatter_indices_batching_dims[i]
            result_index[d_input + 1] += uidx[update_scatter_dims[d_start - (d_start < ivd ? 0 : 1) + 1] + 1]
        end
        for (k, d) in enumerate(dn.update_window_dims)
            result_index[window_input_dims[k] + 1] += uidx[d + 1]
        end

        mlir_size = (d) -> size(first(outs), ndims(first(outs)) - d)
        all(d -> 0 <= result_index[d + 1] < mlir_size(d), 0:(R - 1)) || continue
        J = __gather_scatter_julia_index(result_index, Val(R))
        new = update_computation(
            map(o -> fill(o[J]), outs)..., map(u -> fill(u[I]), updates)...
        )
        foreach((o, v) -> o[J] = v[], outs, new)
    end
    return n == 1 ? only(outs) : outs
end

# both branches take no arguments and return the results of the op
function handle(
    ::Val{Symbol("stablehlo.if")}, (pred,), _, results, (true_branch, false_branch)
)
    rets = pred[] ? true_branch() : false_branch()
    length(results) == 0 && return nothing
    return length(results) == 1 ? only(rets) : rets
end

# `cond` and `body` take the loop-carried values; `cond` returns a 0-dimensional boolean and `body` the next values
function handle(::Val{Symbol("stablehlo.while")}, operands, _, results, (cond, body))
    vals = operands
    while only(cond(vals...))[]
        vals = body(vals...)
    end
    length(results) == 0 && return nothing
    return length(results) == 1 ? only(vals) : vals
end

# operands are the `inputs` and their `init_values`, and `body` combines the accumulated values with an element of each
# input (all as 0-dimensional arrays). `dimensions` are the reduced MLIR dimensions, and the result keeps the other
# dimensions in order, which on the reversed arrays are the remaining Julia dimensions in increasing order
function handle(::Val{Symbol("stablehlo.reduce")}, operands, attributes, results, (body,))
    n = length(results)
    inputs = ntuple(i -> operands[i], Val(n))
    inits = ntuple(i -> operands[n + i], Val(n))
    outs = ntuple(
        i -> fill(convert(results[i].eltype, inits[i][]), results[i].size), Val(n)
    )

    N = ndims(first(inputs))
    reduced = [N - d for d in attributes.dimensions]
    kept = [j for j in 1:N if j ∉ reduced]
    M = Val(length(first(results).size))
    for I in CartesianIndices(first(inputs))
        J = CartesianIndex(ntuple(k -> I[kept[k]], M))
        new = body(map(o -> fill(o[J]), outs)..., map(x -> fill(x[I]), inputs)...)
        foreach((o, v) -> o[J] = v[], outs, new)
    end
    return n == 1 ? only(outs) : outs
end

# convert an MLIR attribute into a Julia value that doesn't depend on its MLIR context.
# dense elements become arrays with reversed dimensions (like operands), dense arrays keep MLIR order (e.g. dimension
# indices are 0-based MLIR dimensions), enums become a `Symbol` with their value (e.g.
# `#stablehlo<comparison_direction LT>` becomes `:LT`), and attributes of any other kind become their textual form.
function decode_attribute(attr)
    # reads a list of dimensions through the size and element getters of the C API
    getdims(getsize, getelem) = Int64[getelem(attr, i) for i in 0:(getsize(attr) - 1)]

    if IR.isdenseelements(attr)
        T = IR.julia_type(eltype(IR.type(attr)))
        dims = reverse(size(IR.type(attr)))
        if IR.issplat(attr)
            splat = IR.Attribute(API.mlirDenseElementsAttrGetSplatValue(attr))
            x = if T === Bool
                Bool(splat)
            elseif T <: Integer
                T(Int64(splat))
            else
                T(Float64(splat))
            end
            return fill(x, dims)
        end

        # dense boolean attributes are bit-packed
        T === Bool &&
            throw(unimplemented("non-splat boolean dense attributes are not supported"))
        return copy(
            unsafe_wrap(Array, Ptr{T}(API.mlirDenseElementsAttrGetRawData(attr)), dims)
        )
    elseif IR.isdensearray(attr, Int64)
        return Int64[API.mlirDenseI64ArrayGetElement(attr, i) for i in 0:(length(attr) - 1)]
    elseif IR.isbool(attr)
        return Bool(attr)
    elseif IR.isinteger(attr)
        return Int64(attr)
    elseif IR.isfloat(attr)
        return Float64(attr)
    elseif IR.isstring(attr)
        return String(attr)
    elseif API.stablehloAttributeIsADotDimensionNumbers(attr)
        return (;
            lhs_batching_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize,
                API.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem,
            ),
            rhs_batching_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize,
                API.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem,
            ),
            lhs_contracting_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize,
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem,
            ),
            rhs_contracting_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize,
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem,
            ),
        )
    elseif API.stablehloAttributeIsAGatherDimensionNumbers(attr)
        return (;
            offset_dims=getdims(
                API.stablehloGatherDimensionNumbersGetOffsetDimsSize,
                API.stablehloGatherDimensionNumbersGetOffsetDimsElem,
            ),
            collapsed_slice_dims=getdims(
                API.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize,
                API.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem,
            ),
            operand_batching_dims=getdims(
                API.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize,
                API.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem,
            ),
            start_indices_batching_dims=getdims(
                API.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize,
                API.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem,
            ),
            start_index_map=getdims(
                API.stablehloGatherDimensionNumbersGetStartIndexMapSize,
                API.stablehloGatherDimensionNumbersGetStartIndexMapElem,
            ),
            index_vector_dim=API.stablehloGatherDimensionNumbersGetIndexVectorDim(attr),
        )
    elseif API.stablehloAttributeIsAScatterDimensionNumbers(attr)
        return (;
            update_window_dims=getdims(
                API.stablehloScatterDimensionNumbersGetUpdateWindowDimsSize,
                API.stablehloScatterDimensionNumbersGetUpdateWindowDimsElem,
            ),
            inserted_window_dims=getdims(
                API.stablehloScatterDimensionNumbersGetInsertedWindowDimsSize,
                API.stablehloScatterDimensionNumbersGetInsertedWindowDimsElem,
            ),
            input_batching_dims=getdims(
                API.stablehloScatterDimensionNumbersGetInputBatchingDimsSize,
                API.stablehloScatterDimensionNumbersGetInputBatchingDimsElem,
            ),
            scatter_indices_batching_dims=getdims(
                API.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize,
                API.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem,
            ),
            scatter_dims_to_operand_dims=getdims(
                API.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize,
                API.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem,
            ),
            # the C API exports the scatter getter of `index_vector_dim` under this name
            index_vector_dim=API.stablehloDimensionNumbersGetIndexVectorDim(attr),
        )
    elseif IR.isarray(attr)
        return Any[
            decode_attribute(IR.Attribute(API.mlirArrayAttrGetElement(attr, i))) for
            i in 0:(API.mlirArrayAttrGetNumElements(attr) - 1)
        ]
    end

    str = string(attr)
    m = match(r"^#\w+<\w+ (\w+)>$", str)
    return isnothing(m) ? str : Symbol(m[1])
end

## HostClient
mutable struct HostClient <: Client
    devices::Vector{Device}
    memories::Vector{Memory}
end

function HostClient()
    devs = Vector{Device}(undef, 1)
    mems = Vector{Memory}(undef, 1)

    dev = HostDevice(HostDeviceDescription(), -1, mems)
    mem = HostMemory(0, "device", devs)

    devs[1] = dev
    mems[1] = mem

    return HostClient(devs, mems)
end

# used by `PJRT_Client_Create`
function HostClient(args::Ptr{CAPI.PJRT_Client_Create_Args})::Ptr{CAPI.PJRT_Error}
    client = root!(HostClient())
    unsafe_store!(args.client, Ptr{CAPI.PJRT_Client}(pointer_from_objref(client)))
    return ok()
end

platform_name(::HostClient) = "julia-cpu"
platform_version(::HostClient) = string(VERSION)
process_index(::HostClient) = 0
devices(client::HostClient) = client.devices
addressable_devices(client::HostClient) = devices(client)
addressable_memories(client::HostClient) = client.memories

function buffer_from_host_buffer(
    ::HostClient,
    dev::HostDevice,
    mem::HostMemory,
    data,
    _eltype,
    _size,
    byte_strides,
    _layout,
    _host_buffer_semantics,
)
    _elsize = pjrt_elsize(_eltype)
    _ndims = length(_size)

    size_in_bytes = prod(_size; init=_elsize)
    minor_to_major = collect((_ndims - 1):-1:0)

    # TODO support byte strides, layout and host buffer semantics

    # only dense major-to-minor host data is supported; strides of size-1 dims are irrelevant
    # num_byte_strides = Int(unsafe_load(args.num_byte_strides))
    # if num_byte_strides != 0
    #     num_byte_strides != num_dims && throw(invalid_argument())
    #     byte_strides = unsafe_load(args.byte_strides)
    #     stride = elsize
    #     for i in num_dims:-1:1
    #         dim = unsafe_load(host_dims, i)
    #         dim != 1 && unsafe_load(byte_strides, i) != stride && throw(unimplemented())
    #         stride *= dim
    #     end
    # end

    # relayout is not supported, so `device_layout` must be dense major-to-minor
    # if device_layout != C_NULL
    #     layout_type = unsafe_load(device_layout.type)
    #     layout_type != CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled && throw(unimplemented())

    #     tiled = unsafe_load(device_layout.tiled)
    #     tiled.num_tiles != 0 && throw(unimplemented())
    #     tiled.minor_to_major_size != num_dims && throw(invalid_argument())
    #     for i in 1:num_dims
    #         unsafe_load(tiled.minor_to_major, i) != num_dims - i && throw(unimplemented())
    #     end
    # end

    # the copy is synchronous, so all `host_buffer_semantics` are honored
    new_data = Libc.malloc(size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())

    Libc.memcpy(new_data, data, size_in_bytes)

    return HostBuffer(new_data, size_in_bytes, _eltype, _size, minor_to_major, dev, mem)
end

# each op becomes a call to its handler, `%N = handle(Val(opname), operands, attributes, results)`
# like in MLIR, an op with several results binds `%N#0`, `%N#1`, ... instead.
# ops with regions also get `regions`, a tuple with an anonymous function per region, which takes the block
# arguments and returns a tuple with the operands of the region terminator.
# `codegen!` translates the ops of `block` into `exprs` and returns the operands of its terminator. `vars` maps MLIR
# values to the names of their Julia variables, and `nvals` counts the ops with results to name them
function codegen!(exprs, block, vars, nvals)
    for op in block
        opname = IR.name(op)
        operands = [vars[IR.operand(op, i).ref.ptr] for i in 1:IR.noperands(op)]
        opname in ("func.return", "stablehlo.return") && return operands
        codegen_op!(exprs, op, opname, operands, vars, nvals)
    end
    throw(invalid_argument("block has no terminator"))
end

function codegen_op!(exprs, op, opname, operands, vars, nvals)
    opval = Val(Symbol(opname))
    nregions = IR.nregions(op)
    signature = if nregions == 0
        Tuple{typeof(opval),Tuple,NamedTuple,Tuple}
    else
        Tuple{typeof(opval),Tuple,NamedTuple,Tuple,Tuple}
    end
    hasmethod(handle, signature) ||
        throw(unimplemented("`$opname` is not supported by `HostClient`"))

    # attributes are decoded here, since the MLIR context doesn't outlive `compile!`
    attributes = (;
        (
            Symbol(String(Reactant.MLIR.API.mlirIdentifierStr(na.ref.name))) =>
                decode_attribute(IR.Attribute(na.ref.attribute)) for
            na in (IR.getattr(op, i) for i in 1:IR.nattrs(op))
        )...
    )
    nres = IR.nresults(op)
    results = ntuple(nres) do i
        type = IR.type(IR.result(op, i))
        IR.istensor(type) || throw(
            unimplemented("`$opname` returns a non-tensor value, which is not supported"),
        )
        T = IR.julia_type(eltype(type))
        dims = reverse(size(type))

        # `eltype` is typed as `Type{T}` instead of `DataType`, so that handlers that build their result from it
        # (e.g. `Array{result.eltype}`) are type-stable
        NamedTuple{(:eltype, :size),Tuple{Type{T},typeof(dims)}}((T, dims))
    end

    call = :(handle($opval, ($(operands...),), $attributes, $results))
    if nregions > 0
        id = nvals[]
        nvals[] += 1
        regions = map(1:nregions) do r
            rblock = IR.first_block(IR.region(op, r))
            args = map(1:IR.nargs(rblock)) do i
                vars[IR.argument(rblock, i).ref.ptr] = Symbol(
                    "%", id, "^", r - 1, "#arg", i - 1
                )
            end
            rexprs = Expr(:block)
            rets = codegen!(rexprs, rblock, vars, nvals)
            Expr(
                :->,
                Expr(:tuple, args...),
                Expr(:block, rexprs.args..., :(return ($(rets...),))),
            )
        end
        push!(call.args, Expr(:tuple, regions...))
    end

    if nres == 0
        push!(exprs.args, call)
    elseif nres == 1
        var = vars[IR.result(op, 1).ref.ptr] = Symbol("%", nvals[])
        push!(exprs.args, :($var = $call))
        nvals[] += 1
    else
        resvars = map(1:nres) do i
            vars[IR.result(op, i).ref.ptr] = Symbol("%", nvals[], "#", i - 1)
        end
        push!(exprs.args, :(($(resvars...),) = $call::NTuple{$nres,Any}))
        nvals[] += 1
    end
    return nothing
end

function compile!(client::HostClient, code, options)
    # `code` lives in a context owned by the caller, so only copies of its contents are kept
    main = IR.lookup(IR.SymbolTable(code), "main")
    isnothing(main) && throw(invalid_argument("program has no `main` function"))
    ftype = IR.FunctionType(main)
    output_types = map(1:IR.nresults(ftype)) do i
        pjrt_eltype(IR.julia_type(eltype(IR.result(ftype, i))))
    end

    name_attr = IR.getattr(IR.Operation(code), "sym_name")
    _name = isnothing(name_attr) ? "main" : String(name_attr)

    exe = HostExecutable(_name, string(code), output_types, options)
    if num_replicas(exe) * num_partitions(exe) != 1
        throw(unimplemented("`HostClient` only supports single-device executables"))
    end

    # run on the device given by the device assignment, or on the first one if there is none
    build_options = options.executable_build_options
    assignment = isnothing(build_options) ? nothing : build_options.device_assignment
    dev = if isnothing(assignment)
        first(client.devices)
    else
        dev_id = only(only(assignment.computation_devices).replica_device_ids)
        idx = findfirst(d -> id(description(d)) == dev_id, client.devices)
        isnothing(idx) &&
            throw(invalid_argument("device $dev_id is not addressable by `HostClient`"))
        client.devices[idx]
    end

    # translate `@main` op by op into the body of an `execute` method specific to this executable.
    # buffers are row-major, so they are seen from Julia as column-major arrays with reversed dimensions.
    block = IR.first_block(IR.region(main, 1))
    nargs = IR.nargs(block)
    vars = Dict{Ptr{Cvoid},Symbol}()
    body = quote
        length(arguments) == 1 || throw(
            invalid_argument(
                string("expected arguments for 1 device, got ", length(arguments))
            ),
        )
        isnothing(dev) ||
            dev === exec.devices[1] ||
            throw(invalid_argument("cannot execute on another device"))
        buffers = only(arguments)
        length(buffers) == $nargs || throw(
            invalid_argument(
                string("expected ", $nargs, " arguments, got ", length(buffers))
            ),
        )
    end

    # wrap operand buffers into Array without copying
    # TODO support wrapping other MLIR types apart of `tensor`
    for i in 1:nargs
        arg = IR.argument(block, i)
        T = IR.julia_type(eltype(IR.type(arg)))
        dims = reverse(size(IR.type(arg)))
        var = vars[arg.ref.ptr] = Symbol("%arg", i - 1)
        push!(
            body.args,
            quote
                isdeleted(buffers[$i]) &&
                    throw(invalid_argument(string("argument ", $i, " is deleted")))
                $var = unsafe_wrap(Array, Ptr{$T}((buffers[$i]::HostBuffer).data), $dims)
            end,
        )
    end

    # each op becomes a call to its handler
    outputs = codegen!(body, block, vars, Ref(0))

    # outputs are copied into new buffers, since they may alias inputs or embedded constants
    # TODO support buffer donation?
    push!(
        body.args,
        quote
            out_dev = exec.devices[1]
            out_mem = default_memory(out_dev)
            results = Buffer[]
            for out in ($(outputs...),)
                nbytes = sizeof(out::Array)
                data = Libc.malloc(max(nbytes, 1))
                data == C_NULL && throw(resource_exhausted())
                GC.@preserve out unsafe_copyto!(
                    Ptr{UInt8}(data), Ptr{UInt8}(pointer(out)), nbytes
                )
                dims = collect(Int64, reverse(size(out)))
                minor_to_major = collect(Int64, (ndims(out) - 1):-1:0)
                push!(
                    results,
                    HostBuffer(
                        data,
                        nbytes,
                        pjrt_eltype(eltype(out)),
                        dims,
                        minor_to_major,
                        out_dev,
                        out_mem,
                    ),
                )
            end
            return [results]
        end,
    )

    tag = gensym(:main)
    @eval function execute(
        exec::HostLoadedExecutable{$(QuoteNode(tag))},
        arguments::Vector{Vector{Buffer}},
        dev,
    )
        return $body
    end

    # single-device executables run as replica 0 of partition 0
    logical_ids = [CAPI.PJRT_LogicalDeviceIds(0, 0)]
    return HostLoadedExecutable{tag}(exe, Device[dev], logical_ids, assignment, false)
end

unimplemented_c_hook() = Ptr{CAPI.PJRT_Error}(pointer_from_objref(root!(unimplemented())))

function __init__()
    # generic handler for unimplemented PJRT API functions
    c_unimplemented = @cfunction(unimplemented_c_hook, Ptr{CAPI.PJRT_Error}, ())

    #! format: off
    vtable_error[] = CAPI.PJRT_Error_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Error_FunctionTable),
        #= instance_size =# csizeof(CAPI.PJRT_Error),
        #= extension_start =# C_NULL,
        #= destroy =# @cfunction(error_destroy_obj, Cvoid, (Ptr{Error},)),
        #= message =# @cfunction(error_message_obj, Cvoid, (Ptr{Error}, Ptr{Cstring}, Ptr{Csize_t})),
        #= get_code =# @cfunction(error_getcode_obj, CAPI.PJRT_Error_Code, (Ptr{Error},)),
        #= for_each_payload =# @cfunction(error_for_each_payload_obj, Cvoid, (Ptr{Error}, CAPI.PJRT_Error_PayloadVisitor, Ptr{Cvoid})),
    )

    vtable_memory[] = CAPI.PJRT_Memory_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Memory_FunctionTable),
        #= extension_start =# C_NULL,
        #= instance_struct_size =# csizeof(CAPI.PJRT_Memory),
        #= get_user_data =# C_NULL,
        #= set_user_data =# C_NULL,
    )

    vtable_api[] = CAPI.PJRT_Api(
        #= struct_size =# csizeof(CAPI.PJRT_Api),
        #= extension_start =# C_NULL,
        #= pjrt_api_version =# CAPI.PJRT_Api_Version(
            #= struct_size =# csizeof(CAPI.PJRT_Api_Version),
            #= extension_start =# C_NULL,
            #= major_version =# CAPI.PJRT_API_MAJOR,
            #= minor_version =# CAPI.PJRT_API_MINOR,
        ),
        #= PJRT_Error_Destroy =# @cfunction(error_destroy, Cvoid, (Ptr{CAPI.PJRT_Error_Destroy_Args},)),
        #= PJRT_Error_Message =# @cfunction(error_message, Cvoid, (Ptr{CAPI.PJRT_Error_Message_Args},)),
        #= PJRT_Error_GetCode =# @cfunction(error_getcode, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_GetCode_Args},)),
        #= PJRT_Plugin_Initialize =# @cfunction(plugin_initialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Plugin_Initialize_Args},)),
        #= PJRT_Plugin_Attributes =# @cfunction(plugin_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Plugin_Attributes_Args},)),
        #= PJRT_Event_Destroy =# @cfunction(event_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Destroy_Args},)),
        #= PJRT_Event_IsReady =# @cfunction(event_is_ready, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_IsReady_Args},)),
        #= PJRT_Event_Error =# @cfunction(event_error, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Error_Args},)),
        #= PJRT_Event_Await =# @cfunction(event_await, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Await_Args},)),
        #= PJRT_Event_OnReady =# @cfunction(event_on_ready, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_OnReady_Args},)),
        #= PJRT_Client_Create =# @cfunction(HostClient, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Create_Args},)),
        #= PJRT_Client_Destroy =# @cfunction(client_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Destroy_Args},)),
        #= PJRT_Client_PlatformName =# @cfunction(client_platform_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformName_Args},)),
        #= PJRT_Client_ProcessIndex =# @cfunction(client_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_ProcessIndex_Args},)),
        #= PJRT_Client_PlatformVersion =# @cfunction(client_platform_version, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformVersion_Args},)),
        #= PJRT_Client_Devices =# @cfunction(client_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Devices_Args},)),
        #= PJRT_Client_AddressableDevices =# @cfunction(client_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableDevices_Args},)),
        #= PJRT_Client_LookupDevice =# c_unimplemented,
        #= PJRT_Client_LookupAddressableDevice =# c_unimplemented,
        #= PJRT_Client_AddressableMemories =# @cfunction(client_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableMemories_Args},)),
        #= PJRT_Client_Compile =# @cfunction(client_compile, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Compile_Args},)),
        #= PJRT_Client_DefaultDeviceAssignment =# c_unimplemented,
        #= PJRT_Client_BufferFromHostBuffer =# @cfunction(client_buffer_from_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args},)),
        #= PJRT_DeviceDescription_Id =# @cfunction(device_description_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Id_Args},)),
        #= PJRT_DeviceDescription_ProcessIndex =# @cfunction(device_description_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args},)),
        #= PJRT_DeviceDescription_Attributes =# @cfunction(device_description_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args},)),
        #= PJRT_DeviceDescription_Kind =# @cfunction(device_description_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Kind_Args},)),
        #= PJRT_DeviceDescription_DebugString =# @cfunction(device_description_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args},)),
        #= PJRT_DeviceDescription_ToString =# @cfunction(device_description_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ToString_Args},)),
        #= PJRT_Device_GetDescription =# @cfunction(device_get_description, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_GetDescription_Args},)),
        #= PJRT_Device_IsAddressable =# @cfunction(device_is_addressable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_IsAddressable_Args},)),
        #= PJRT_Device_LocalHardwareId =# @cfunction(device_local_hardware_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_LocalHardwareId_Args},)),
        #= PJRT_Device_AddressableMemories =# @cfunction(device_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_AddressableMemories_Args},)),
        #= PJRT_Device_DefaultMemory =# @cfunction(device_default_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_DefaultMemory_Args},)),
        #= PJRT_Device_MemoryStats =# @cfunction(device_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_MemoryStats_Args},)),
        #= PJRT_Memory_Id =# @cfunction(memory_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Id_Args},)),
        #= PJRT_Memory_Kind =# @cfunction(memory_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Args},)),
        #= PJRT_Memory_DebugString =# @cfunction(memory_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_DebugString_Args},)),
        #= PJRT_Memory_ToString =# @cfunction(memory_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_ToString_Args},)),
        #= PJRT_Memory_AddressableByDevices =# @cfunction(memory_addressable_by_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args},)),
        #= PJRT_Executable_Destroy =# @cfunction(executable_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Destroy_Args},)),
        #= PJRT_Executable_Name =# @cfunction(executable_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Name_Args},)),
        #= PJRT_Executable_NumReplicas =# @cfunction(executable_num_replicas, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumReplicas_Args},)),
        #= PJRT_Executable_NumPartitions =# @cfunction(executable_num_partitions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumPartitions_Args},)),
        #= PJRT_Executable_NumOutputs =# @cfunction(executable_num_outputs, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumOutputs_Args},)),
        #= PJRT_Executable_SizeOfGeneratedCodeInBytes =# @cfunction(executable_size_of_generated_code_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args},)),
        #= PJRT_Executable_GetCostAnalysis =# c_unimplemented,
        #= PJRT_Executable_OutputMemoryKinds =# c_unimplemented,
        #= PJRT_Executable_OptimizedProgram =# @cfunction(executable_optimized_program, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OptimizedProgram_Args},)),
        #= PJRT_Executable_Serialize =# @cfunction(executable_serialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Serialize_Args},)),
        #= PJRT_LoadedExecutable_Destroy =# @cfunction(loaded_executable_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Destroy_Args},)),
        #= PJRT_LoadedExecutable_GetExecutable =# @cfunction(loaded_executable_get_executable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_GetExecutable_Args},)),
        #= PJRT_LoadedExecutable_AddressableDevices =# @cfunction(loaded_executable_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_AddressableDevices_Args},)),
        #= PJRT_LoadedExecutable_Delete =# @cfunction(loaded_executable_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Delete_Args},)),
        #= PJRT_LoadedExecutable_IsDeleted =# @cfunction(loaded_executable_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_IsDeleted_Args},)),
        #= PJRT_LoadedExecutable_Execute =# @cfunction(loaded_executable_execute, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Execute_Args},)),
        #= PJRT_Executable_DeserializeAndLoad =# c_unimplemented,
        #= PJRT_LoadedExecutable_Fingerprint =# c_unimplemented,
        #= PJRT_Buffer_Destroy =# @cfunction(buffer_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Destroy_Args},)),
        #= PJRT_Buffer_ElementType =# @cfunction(buffer_element_type, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ElementType_Args},)),
        #= PJRT_Buffer_Dimensions =# @cfunction(buffer_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Dimensions_Args},)),
        #= PJRT_Buffer_UnpaddedDimensions =# c_unimplemented,
        #= PJRT_Buffer_DynamicDimensionIndices =# c_unimplemented,
        #= PJRT_Buffer_GetMemoryLayout =# c_unimplemented,
        #= PJRT_Buffer_OnDeviceSizeInBytes =# @cfunction(buffer_on_device_size_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args},)),
        #= PJRT_Buffer_Device =# @cfunction(buffer_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Device_Args},)),
        #= PJRT_Buffer_Memory =# @cfunction(buffer_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Memory_Args},)),
        #= PJRT_Buffer_Delete =# @cfunction(buffer_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Delete_Args},)),
        #= PJRT_Buffer_IsDeleted =# @cfunction(buffer_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsDeleted_Args},)),
        #= PJRT_Buffer_CopyToDevice =# @cfunction(buffer_copy_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args},)),
        #= PJRT_Buffer_ToHostBuffer =# @cfunction(buffer_to_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args},)),
        #= PJRT_Buffer_IsOnCpu =# @cfunction(buffer_is_on_cpu, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args},)),
        #= PJRT_Buffer_ReadyEvent =# @cfunction(buffer_ready_event, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ReadyEvent_Args},)),
        #= PJRT_Buffer_UnsafePointer =# @cfunction(buffer_unsafe_pointer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args},)),
        #= PJRT_Buffer_IncreaseExternalReferenceCount =# c_unimplemented,
        #= PJRT_Buffer_DecreaseExternalReferenceCount =# c_unimplemented,
        #= PJRT_Buffer_OpaqueDeviceMemoryDataPointer =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_Destroy =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_AddChunk =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_TotalBytes =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_GranuleSize =# c_unimplemented,
        #= PJRT_CopyToDeviceStream_CurrentBytes =# c_unimplemented,
        #= PJRT_TopologyDescription_Create =# c_unimplemented,
        #= PJRT_TopologyDescription_Destroy =# c_unimplemented,
        #= PJRT_TopologyDescription_PlatformName =# c_unimplemented,
        #= PJRT_TopologyDescription_PlatformVersion =# c_unimplemented,
        #= PJRT_TopologyDescription_GetDeviceDescriptions =# c_unimplemented,
        #= PJRT_TopologyDescription_Serialize =# c_unimplemented,
        #= PJRT_TopologyDescription_Attributes =# c_unimplemented,
        #= PJRT_Compile =# c_unimplemented,
        #= PJRT_Executable_OutputElementTypes =# @cfunction(executable_output_element_types, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args},)),
        #= PJRT_Executable_OutputDimensions =# c_unimplemented,
        #= PJRT_Buffer_CopyToMemory =# @cfunction(buffer_copy_to_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToMemory_Args},)),
        #= PJRT_Client_CreateViewOfDeviceBuffer =# c_unimplemented,
        #= PJRT_Executable_Fingerprint =# @cfunction(executable_fingerprint, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Fingerprint_Args},)),
        #= PJRT_Client_TopologyDescription =# c_unimplemented,
        #= PJRT_Executable_GetCompiledMemoryStats =# c_unimplemented,
        #= PJRT_Memory_Kind_Id =# @cfunction(memory_kind_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Id_Args},)),
        #= PJRT_ExecuteContext_Create =# c_unimplemented,
        #= PJRT_ExecuteContext_Destroy =# c_unimplemented,
        #= PJRT_Buffer_CopyRawToHost =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_Destroy =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_TransferData =# c_unimplemented,
        #= PJRT_Client_CreateBuffersForAsyncHostToDevice =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_Device =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_BufferCount =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_BufferSize =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_SetBufferError =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_AddMetadata =# c_unimplemented,
        #= PJRT_Client_DmaMap =# c_unimplemented,
        #= PJRT_Client_DmaUnmap =# c_unimplemented,
        #= PJRT_Client_CreateUninitializedBuffer =# c_unimplemented,
        #= PJRT_Client_UpdateGlobalProcessInfo =# c_unimplemented,
        #= PJRT_TopologyDescription_Deserialize =# c_unimplemented,
        #= PJRT_Client_CreateAliasBuffer =# c_unimplemented,
        #= PJRT_Client_FulfillAliasBuffer =# c_unimplemented,
        #= PJRT_LoadedExecutable_GetDeviceAssignment =# @cfunction(loaded_executable_get_device_assignment, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args},)),
        #= PJRT_Client_CreateErrorBuffer =# c_unimplemented,
        #= PJRT_AsyncHostToDeviceTransferManager_TransferLiteral =# c_unimplemented,
        #= PJRT_Buffer_CopyRawToHostFuture =# c_unimplemented,
        #= PJRT_Device_PoisonExecution =# c_unimplemented,
        #= PJRT_Device_CreateAsyncTrackingEvent =# c_unimplemented,
        #= PJRT_AsyncTrackingEvent_Destroy =# c_unimplemented,
        #= PJRT_Executable_GetCompileOptions =# @cfunction(executable_get_compile_options, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCompileOptions_Args},)),
        #= PJRT_Buffer_DonateWithControlDependency =# c_unimplemented,
        #= PJRT_Event_Create =# @cfunction(event_create, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Create_Args},)),
        #= PJRT_Event_Set =# @cfunction(event_set, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Set_Args},)),
        #= PJRT_Device_GetAttributes =# C_NULL,
        #= PJRT_Client_Load =# c_unimplemented,
        #= PJRT_LoadedExecutable_AddressableDeviceLogicalIds =# @cfunction(loaded_executable_addressable_device_logical_ids, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args},)),
        #= PJRT_Buffer_Bitcast =# c_unimplemented,
        #= PJRT_Error_ForEachPayload =# @cfunction(error_for_each_payload, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_ForEachPayload_Args},)),
        #= PJRT_TopologyDescription_Fingerprint =# c_unimplemented,
        #= PJRT_Executable_ParameterMemoryKinds =# c_unimplemented,
        #= PJRT_Device_ClearMemoryStats =# c_unimplemented,
        #= PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace =# c_unimplemented,
        #= PJRT_TopologyDescription_GetMemorySpaceKindIds =# c_unimplemented,
    )
    #! format: on

    return Reactant.Accelerators.Registration.register_backend(
        "julia"; priority=100, pjrt_initialize_function=make_pjrt_client
    )
end

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert node_id == 0
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert isnothing(distributed_runtime_client) "`make_pjrt_client` does not \
                                                    support distributed_runtime_client"
    @assert isnothing(allowed_devices)

    errstr = Ref{Cstring}()
    api = unsafe_convert(Ptr{CAPI.PJRT_Api}, vtable_api)
    ptr = Reactant.MLIR.API.MakeClientFromApi(api, "julia", "JULIA", errstr)
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.PJRT.Client(ptr)
end

end
