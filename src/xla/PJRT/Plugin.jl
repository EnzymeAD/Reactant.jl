module Plugin

# When dealing with memory ownership at the Julia FFI boundary, there are two different designs that can be followed:
# - rely on Julia's GC, which requires to...
#   - globally mark/root objects
#   - refer to them via `pointer_from_objref`, `unsafe_pointer_to_objref`
#   - use `mutable` objects only (immutable objects do not have a stable memory address)
# - ...or manually allocate/free memory, which requires to...
#   - use `Libc`
#   - only work with `isbits` objects (i.e. not mutable nor immutable with mutable fields)
#
# After experimenting with both designs, the "globally GC rooted" design was ultimately chosen because...
# - it's simpler and easier to read
# - it's easier to extend from the Julia side
# - safer against memory bugs (as memory is tracked by the GC)
# - almost no overhead compared to manual allocations (only a dynamic dispatch for offering a Julia interface)
#
# This code is designed in such a way that PJRT plugins developed in Julia do not need to interface with the FFI
# boundary. All interaction with the PJRT FFI should be located here and abstract away through an interface that Julia
# plugins can implement elsewhere.
#
# When extending the following code or implementing the interface, take these points into considerations:
# - All objects passed through the XLA/PJRT boundary need to be GC rooted (i.e. use `(un)root!`), which requires objects
#   to be `mutable` in order to have a stable pointer address. Objects are GC rooted/unrooted here and should be
#   transparent to the user.
# - Some Julia objects require as a first field a pointer to their vtable (e.g. Error, Memory, ...). This is required by
#   the PJRT Plugin API. Keep in mind that it should point to the correct implementation for your plugin.
# - If for some PJRT object, the API requires it to return a string that is auto-generated (not a string literal),
#   store the string inside the object so that it's correctly freed (owner is the object).
# - If for some PJRT object, the API requires it to return a list, store the list as a `Vector{X}` where `X` is an
#   abstract type.
#   - The Julia array ABI layout states that an array with abstract eltype is represented as a pointer of pointers
#     (no inline storage). Check https://docs.julialang.org/en/v1/manual/embedding/#Working-with-Arrays
#   - The stored pointers are of type `jl_value_t*` which points to the object data (the type tag is stored before that
#     pointer address). Check https://docs.julialang.org/en/v1/devdocs/object/#Object-layout-(jl_value_t)
#   - Once the object is created, DO NOT resize the array.
# - DO NOT use moving GC (e.g. MMTk) as it would break `pointer_from_objref`.
# - The following objects do not need to be reimplemented on different accelerators: Error, Event.
# - There is are a couple of exception to the FFI abstraction:
#   - Entrypoint to the API, 

using Base: unsafe_convert
using ..Reactant: Reactant, WeakKeyIdDict
using Reactant.XLA.CAPI: CAPI, csizeof
using Reactant.MLIR: IR, API
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
function add_chunk! end
function add_metadata! end
function addressable_device_logical_ids end
function addressable_devices end
function addressable_memories end
function attributes end
function bitcast! end
function buffer_count end
function buffer_from_host_buffer end
function buffer_size end
function clear_memory_stats! end
function compile end
function compile_options end
function compile! end
function compiled_memory_stats end
function copy_raw_to_host! end
function copy_to_device end
function copy_to_memory end
function cost_analysis end
function create_alias_buffer end
function create_async_tracking_event end
function create_buffers_for_async_host_to_device end
function create_client end
function create_error_buffer end
function create_topology_description end
function create_uninitialized_buffer end
function current_bytes end
function debug_string end
function decrease_external_reference_count! end
function default_device_assignment end
function default_memory end
function default_memory_description end
function description end
function deserialize_and_load end
function deserialize_topology_description end
function device_assignment end
function device end
function device_descriptions end
function devices end
function dimensions end
function dma_map! end
function dma_unmap! end
function donate_with_control_dependency! end
function dynamic_dimension_indices end
function element_type end
function executable end
function execute end
function fingerprint end
function granule_size end
function id end
function increase_external_reference_count! end
function initialize end
function is_addressable end
function is_on_cpu end
function isdeleted end
function kind_id end
function kind end
function load end
function local_hardware_id end
function lookup_addressable_device end
function lookup_device end
function make_canonical_shape_for_memory_space end
function memory end
function memory_descriptions end
function memory_layout end
function memory_space_kind_ids end
function name end
function num_outputs end
function num_partitions end
function num_replicas end
function on_device_size_in_bytes end
function opaque_device_memory_data_pointer end
function optimized_program end
function output_dimensions end
function output_element_types end
function output_memory_kinds end
function parameter_memory_kinds end
function platform_name end
function platform_version end
function poison_execution! end
function process_index end
function ready_event end
function retrieve_buffer! end
function set_buffer_error! end
function size_of_generated_code_in_bytes end
function to_host_buffer end
function to_string end
function topology_description end
function total_bytes end
function transfer_data! end
function transfer_literal! end
function unpadded_dimensions end
function unsafe_pointer end
function update_global_process_info! end

export unimplemented, invalid_argument, resource_exhausted

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

## vtables
const vtable_error = Ref{CAPI.PJRT_Error_FunctionTable}()
const vtable_memory = Ref{CAPI.PJRT_Memory_FunctionTable}()

## helpers
const ProcessState = CAPI.PJRT_ProcessState
const ProcessInfo = CAPI.PJRT_ProcessInfo

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

# build an `Error` from the status fields of a C struct; the message is owned by C, so it's copied
function Error(code::CAPI.PJRT_Error_Code, message::Cstring, message_size::Integer)
    ptr = Ptr{UInt8}(message)
    return Error(code, ptr == C_NULL ? "" : unsafe_string(ptr, message_size))
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
function pjrt_error_destroy(args::Ptr{CAPI.PJRT_Error_Destroy_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    pjrt_error_destroy_obj(error)
    return nothing
end

function pjrt_error_message(args::Ptr{CAPI.PJRT_Error_Message_Args})::Cvoid
    error = Ptr{Error}(unsafe_load(args.error))
    pjrt_error_message_obj(error, args.message, args.message_size)
    return nothing
end

function pjrt_error_getcode(args::Ptr{CAPI.PJRT_Error_GetCode_Args})::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    code = pjrt_error_getcode_obj(error)
    unsafe_store!(args.code, code)
    return ok()
end

function pjrt_error_for_each_payload(
    args::Ptr{CAPI.PJRT_Error_ForEachPayload_Args}
)::Ptr{CAPI.PJRT_Error}
    error = Ptr{Error}(unsafe_load(args.error))
    visitor = unsafe_load(args.visitor)
    user_arg = unsafe_load(args.user_arg)
    pjrt_error_for_each_payload_obj(error, visitor, user_arg)
    return ok()
end

# this are required by CAPI.PJRT_Error_FunctionTable
function pjrt_error_destroy_obj(error::Ptr{Error})::Cvoid
    error != C_NULL && unroot!(error)
    return nothing
end

function pjrt_error_message_obj(
    error::Ptr{Error}, message::Ptr{Cstring}, message_size::Ptr{Csize_t}
)::Cvoid
    error == C_NULL && return nothing
    err = unsafe_pointer_to_objref(error)::Error
    msg = err.message
    unsafe_store!(message, Cstring(pointer(msg)))
    unsafe_store!(message_size, sizeof(msg))
    return nothing
end

function pjrt_error_getcode_obj(error::Ptr{Error})::CAPI.PJRT_Error_Code
    error == C_NULL && return CAPI.PJRT_Error_Code_OK
    err = unsafe_pointer_to_objref(error)::Error
    return err.code
end

function pjrt_error_for_each_payload_obj(
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

@pjrt_catch_return function pjrt_event_destroy(
    args::Ptr{CAPI.PJRT_Event_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.event)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

@pjrt_catch_return function pjrt_event_is_ready(
    args::Ptr{CAPI.PJRT_Event_IsReady_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    unsafe_store!(args.is_ready, isready(event))
    return ok()
end

@pjrt_catch_return function pjrt_event_error(
    args::Ptr{CAPI.PJRT_Event_Error_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    return event_error_ptr(event)
end

@pjrt_catch_return function pjrt_event_await(
    args::Ptr{CAPI.PJRT_Event_Await_Args}
)::Ptr{CAPI.PJRT_Error}
    event = unsafe_pointer_to_objref(unsafe_load(args.event))::Event
    wait(event)
    return event_error_ptr(event)
end

@pjrt_catch_return function pjrt_event_on_ready(
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

@pjrt_catch_return function pjrt_event_create(
    args::Ptr{CAPI.PJRT_Event_Create_Args}
)::Ptr{CAPI.PJRT_Error}
    event = root!(Event())
    unsafe_store!(args.event, pointer_from_objref(event))
    return ok()
end

@pjrt_catch_return function pjrt_event_set(
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

## Named values
# read a list of `PJRT_NamedValue`s owned by C into a `Dict`, copying their contents
function unsafe_load_pjrt_named_values(ptr::Ptr{CAPI.PJRT_NamedValue}, n::Integer)
    values = Dict{String,Any}()
    for i in 1:n
        nv = ptr + (i - 1) * sizeof(CAPI.PJRT_NamedValue)
        name = unsafe_string(Ptr{UInt8}(unsafe_load(nv.name)), unsafe_load(nv.name_size))
        type = unsafe_load(nv.type)
        size = unsafe_load(nv.value_size)
        values[name] = if type == CAPI.PJRT_NamedValue_kString
            unsafe_string(Ptr{UInt8}(unsafe_load(nv.string_value)), size)
        elseif type == CAPI.PJRT_NamedValue_kInt64
            unsafe_load(nv.int64_value)
        elseif type == CAPI.PJRT_NamedValue_kInt64List
            copy(unsafe_wrap(Vector{Int64}, unsafe_load(nv.int64_array_value), size))
        elseif type == CAPI.PJRT_NamedValue_kFloat
            unsafe_load(nv.float_value)
        elseif type == CAPI.PJRT_NamedValue_kBool
            unsafe_load(nv.bool_value)
        else
            throw(invalid_argument("named value `$name` has unknown type $type"))
        end
    end
    return values
end

# build the `PJRT_NamedValue`s of the `name => value` pairs in `pairs`; also return the Julia objects they point to,
# which must be kept alive for as long as C uses them
function pjrt_named_values(pairs)
    roots = Any[]
    values = map(collect(pairs)) do (name, value)
        name = String(name)
        push!(roots, name)
        r = Ref{CAPI.PJRT_NamedValue}()
        GC.@preserve r begin
            nv = Base.unsafe_convert(Ptr{CAPI.PJRT_NamedValue}, r)
            unsafe_store!(nv.struct_size, csizeof(CAPI.PJRT_NamedValue))
            unsafe_store!(nv.extension_start, C_NULL)
            unsafe_store!(nv.name, Cstring(pointer(name)))
            unsafe_store!(nv.name_size, sizeof(name))
            if value isa AbstractString
                str = String(value)
                push!(roots, str)
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kString)
                unsafe_store!(nv.string_value, Cstring(pointer(str)))
                unsafe_store!(nv.value_size, sizeof(str))
            elseif value isa Bool # before `Integer`, since `Bool <: Integer`
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kBool)
                unsafe_store!(nv.bool_value, value)
                unsafe_store!(nv.value_size, 1)
            elseif value isa Integer
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kInt64)
                unsafe_store!(nv.int64_value, Int64(value))
                unsafe_store!(nv.value_size, 1)
            elseif value isa AbstractVector{<:Integer}
                list = Vector{Int64}(value)
                push!(roots, list)
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kInt64List)
                unsafe_store!(nv.int64_array_value, pointer(list))
                unsafe_store!(nv.value_size, length(list))
            elseif value isa AbstractFloat
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kFloat)
                unsafe_store!(nv.float_value, Float32(value))
                unsafe_store!(nv.value_size, 1)
            else
                msg = "value of `$name` has type $(typeof(value)), which can't be a `PJRT_NamedValue`"
                throw(Error(CAPI.PJRT_Error_Code_INTERNAL, msg))
            end
        end
        r[]
    end
    return values, roots
end

## Plugin
"""
    AbstractPlugin

A type representing a PJRT plugin. Used for dispatching on plugin entrypoints, like client creation or the topologies
created without a client.

Pass an instance to [`make_api`](@ref) to build its `PJRT_Api`.

It must be stateless: its type `P` must be `Base.issingletontype(P)` with an empty constructor `P()`. Plugin-wide state
(e.g. a registry of FFI handlers) belongs in globals of the module of the plugin.

### Interface

- `initialize` (optional)
- `attributes` (optional)
- `create_client`
- `create_topology_description`
- `deserialize_topology_description`
"""
abstract type AbstractPlugin end

"""
    initialize(plugin::AbstractPlugin)

One-time setup of `plugin`, called by C before any other function of the plugin. Defaults to doing nothing.

Backs `PJRT_Plugin_Initialize`.
"""
initialize(::AbstractPlugin) = nothing

struct PluginInitialize{P<:AbstractPlugin} end

@pjrt_catch_return function (::PluginInitialize{P})(
    args::Ptr{CAPI.PJRT_Plugin_Initialize_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    initialize(P())
    return ok()
end

"""
    attributes(plugin::AbstractPlugin)

Return the attributes of `plugin` as `name => value` pairs (e.g. a `Dict`). Common names are `"xla_version"`,
`"stablehlo_current_version"` and `"stablehlo_minimum_version"`. Values must be `AbstractString`, `Bool`, `Integer`,
`AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as `String`, `Bool`, `Int64`, `Vector{Int64}` and
`Float32`. Defaults to no attributes.

It is called at most once per plugin type: the C wrapper keeps a copy for the lifetime of the process, since C
keeps pointers to it.

Backs `PJRT_Plugin_Attributes`.
"""
attributes(::AbstractPlugin) = ()

# the C view of the attributes of each plugin type, computed on the first call since the attributes may depend on
# `initialize`, and never freed since C keeps pointers to it
const plugin_attributes = IdDict{DataType,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}}()
const plugin_attributes_lock = ReentrantLock()

struct PluginAttributes{P<:AbstractPlugin} end

@pjrt_catch_return function (::PluginAttributes{P})(
    args::Ptr{CAPI.PJRT_Plugin_Attributes_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    values, _ = @lock plugin_attributes_lock begin
        get!(() -> pjrt_named_values(attributes(P())), plugin_attributes, P)
    end
    unsafe_store!(args.attributes, pointer(values))
    unsafe_store!(args.num_attributes, length(values))
    return ok()
end

"""
    create_client(plugin::AbstractPlugin, options::Dict{String,Any}) -> Client

Return a new client of `plugin`. `options` holds platform-specific options, whose values are `String`, `Int64`,
`Vector{Int64}`, `Float32` or `Bool`.

The caller GC-roots the returned client until `PJRT_Client_Destroy`.

Backs `PJRT_Client_Create`.
"""
create_client(::AbstractPlugin, ::Dict{String,Any})

struct ClientCreate{P<:AbstractPlugin} end

@pjrt_catch_return function (::ClientCreate{P})(
    args::Ptr{CAPI.PJRT_Client_Create_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    options = unsafe_load_pjrt_named_values(
        unsafe_load(args.create_options), unsafe_load(args.num_options)
    )
    client = root!(create_client(P(), options)::Client)
    unsafe_store!(args.client, Ptr{CAPI.PJRT_Client}(pointer_from_objref(client)))
    return ok()
end

"""
    create_topology_description(plugin::AbstractPlugin, name::String, options::Dict{String,Any}) -> TopologyDescription

Return a new description of the topology named `name` of the platform of `plugin`, without creating a client.
`options` holds platform-specific options, whose values are `String`, `Int64`, `Vector{Int64}`, `Float32` or `Bool`.

The caller GC-roots the returned topology until `PJRT_TopologyDescription_Destroy`.

Backs `PJRT_TopologyDescription_Create`.
"""
create_topology_description(::AbstractPlugin, ::String, ::Dict{String,Any})

struct TopologyDescriptionCreate{P<:AbstractPlugin} end

@pjrt_catch_return function (::TopologyDescriptionCreate{P})(
    args::Ptr{CAPI.PJRT_TopologyDescription_Create_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    name = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.topology_name)), unsafe_load(args.topology_name_size)
    )
    options = unsafe_load_pjrt_named_values(
        unsafe_load(args.create_options), unsafe_load(args.num_options)
    )
    topology = create_topology_description(P(), name, options)::TopologyDescription
    root!(topology)
    unsafe_store!(
        args.topology, Ptr{CAPI.PJRT_TopologyDescription}(pointer_from_objref(topology))
    )
    return ok()
end

"""
    deserialize_topology_description(plugin::AbstractPlugin, bytes::Vector{UInt8}) -> TopologyDescription

Return the topology description of the platform of `plugin` serialized in `bytes` by `serialize`.

The caller GC-roots the returned topology until `PJRT_TopologyDescription_Destroy`.

Backs `PJRT_TopologyDescription_Deserialize`.
"""
deserialize_topology_description(::AbstractPlugin, ::Vector{UInt8})

struct TopologyDescriptionDeserialize{P<:AbstractPlugin} end

@pjrt_catch_return function (::TopologyDescriptionDeserialize{P})(
    args::Ptr{CAPI.PJRT_TopologyDescription_Deserialize_Args}
)::Ptr{CAPI.PJRT_Error} where {P}
    # the serialized topology is owned by C, so copy it in case the implementation stores it
    bytes_ptr = Ptr{UInt8}(unsafe_load(args.serialized_topology))
    bytes = copy(
        unsafe_wrap(Vector{UInt8}, bytes_ptr, unsafe_load(args.serialized_topology_size))
    )
    topology = deserialize_topology_description(P(), bytes)::TopologyDescription
    root!(topology)
    unsafe_store!(
        args.topology, Ptr{CAPI.PJRT_TopologyDescription}(pointer_from_objref(topology))
    )
    return ok()
end

@pjrt_catch_return function pjrt_topology_description_destroy(
    args::Ptr{CAPI.PJRT_TopologyDescription_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.topology)
    ptr != C_NULL && unroot!(ptr)
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

@pjrt_catch_return function pjrt_memory_id(
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

@pjrt_catch_return function pjrt_memory_kind(
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

@pjrt_catch_return function pjrt_memory_kind_id(
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

@pjrt_catch_return function pjrt_memory_debug_string(
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

@pjrt_catch_return function pjrt_memory_to_string(
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

@pjrt_catch_return function pjrt_memory_addressable_by_devices(
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

@pjrt_catch_return function pjrt_device_description_id(
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

@pjrt_catch_return function pjrt_device_description_process_index(
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

@pjrt_catch_return function pjrt_device_description_kind(
    args::Ptr{CAPI.PJRT_DeviceDescription_Kind_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = Ptr{DeviceDescription}(unsafe_load(args.device_description))
    desc = unsafe_pointer_to_objref(ptr)::DeviceDescription
    str = kind(desc)::String
    unsafe_store!(args.device_kind, Cstring(pointer(str)))
    unsafe_store!(args.device_kind_size, sizeof(str))
    return ok()
end

# C views of the results of `attributes(::DeviceDescription)`, which C keeps for as long as the description is alive
const device_description_attributes_cache = WeakKeyIdDict{
    DeviceDescription,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}
}()

"""
    attributes(desc::DeviceDescription)

Return the device-specific attributes of the device described by `desc` as `name => value` pairs (e.g. a `Dict`).
Values must be `AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as
`String`, `Bool`, `Int64`, `Vector{Int64}` and `Float32`.

It is called at most once per `desc`: the C wrapper keeps a copy for as long as `desc` is alive, since C keeps pointers
to it.

!!! warning
`PjRtCApiClient` copies the attributes once, when it wraps the device description, and treats any error as fatal.

Backs `PJRT_DeviceDescription_Attributes`.
"""
attributes(::DeviceDescription)

@pjrt_catch_return function pjrt_device_description_attributes(
    args::Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    desc = unsafe_pointer_to_objref(unsafe_load(args.device_description))::DeviceDescription
    values, _ = get!(device_description_attributes_cache, desc) do
        pjrt_named_values(attributes(desc))
    end
    unsafe_store!(args.num_attributes, length(values))
    unsafe_store!(args.attributes, pointer(values))
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

@pjrt_catch_return function pjrt_device_description_debug_string(
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

@pjrt_catch_return function pjrt_device_description_to_string(
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
- `clear_memory_stats!`
- `create_async_tracking_event`
- `poison_execution!`
"""
abstract type Device end

"""
    description(dev::Device) -> DeviceDescription

Return the description of `dev`. The description must be kept alive by `dev`.

Backs `PJRT_Device_GetDescription`.
"""
description(::Device)

@pjrt_catch_return function pjrt_device_get_description(
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

@pjrt_catch_return function pjrt_device_is_addressable(
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

@pjrt_catch_return function pjrt_device_local_hardware_id(
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

@pjrt_catch_return function pjrt_device_addressable_memories(
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

@pjrt_catch_return function pjrt_device_default_memory(
    args::Ptr{CAPI.PJRT_Device_DefaultMemory_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    mem = default_memory(device)::Memory
    unsafe_store!(args.memory, Ptr{CAPI.PJRT_Memory}(pointer_from_objref(mem)))
    return ok()
end

# TODO document and add interface function
@pjrt_catch_return function pjrt_device_memory_stats(
    args::Ptr{CAPI.PJRT_Device_MemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    throw(unimplemented())
end

"""
    clear_memory_stats!(dev::Device)

Reset the memory statistics of `dev` (e.g. the peaks reported by `PJRT_Device_MemoryStats`).

Backs `PJRT_Device_ClearMemoryStats`.
"""
clear_memory_stats!(::Device)

@pjrt_catch_return function pjrt_device_clear_memory_stats(
    args::Ptr{CAPI.PJRT_Device_ClearMemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    clear_memory_stats!(device)
    return ok()
end

"""
    AsyncTrackingEvent

A scoped event that tells a device that there is asynchronous work depending on its activity, from its creation by
`create_async_tracking_event` until it is closed.

### Interface

- `close`
"""
abstract type AsyncTrackingEvent end

"""
    close(event::AsyncTrackingEvent)

Mark the end of the asynchronous work tracked by `event`. Defaults to doing nothing.

Called by `PJRT_AsyncTrackingEvent_Destroy` before `event` is unrooted.
"""
Base.close(::AsyncTrackingEvent) = nothing

@pjrt_catch_return function pjrt_async_tracking_event_destroy(
    args::Ptr{CAPI.PJRT_AsyncTrackingEvent_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.event)
    ptr == C_NULL && return ok()
    event = unsafe_pointer_to_objref(ptr)::AsyncTrackingEvent
    close(event)
    unroot!(event)
    return ok()
end

"""
    create_async_tracking_event(dev::Device, description::String) -> Union{Nothing,AsyncTrackingEvent}

Return a new [`AsyncTrackingEvent`](@ref) that tracks asynchronous work depending on the activity of `dev`, or
`nothing` if `dev` doesn't support tracking it.

The caller GC-roots the returned event.

Backs `PJRT_Device_CreateAsyncTrackingEvent`.
"""
create_async_tracking_event(::Device, ::String)

@pjrt_catch_return function pjrt_device_create_async_tracking_event(
    args::Ptr{CAPI.PJRT_Device_CreateAsyncTrackingEvent_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    desc = unsafe_string(
        Ptr{UInt8}(unsafe_load(args.description)), unsafe_load(args.description_size)
    )
    event = create_async_tracking_event(device, desc)::Union{Nothing,AsyncTrackingEvent}
    if isnothing(event)
        unsafe_store!(args.event, C_NULL)
        return ok()
    end
    root!(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_AsyncTrackingEvent}(pointer_from_objref(event)))
    return ok()
end

"""
    poison_execution!(dev::Device, launch_id::Int32, error::Error) -> Bool

Make the outputs of the earliest unfinished execution on `dev` with launch id `launch_id` fail with `error`.

Return `true` if they have been poisoned, or `false` if there is no such execution (e.g. it already finished).

!!! todo
Error payloads are not supported yet: if C passes any, `PJRT_Device_PoisonExecution` returns `UNIMPLEMENTED`.

Backs `PJRT_Device_PoisonExecution`.
"""
poison_execution!(::Device, ::Int32, ::Error)

@pjrt_catch_return function pjrt_device_poison_execution(
    args::Ptr{CAPI.PJRT_Device_PoisonExecution_Args}
)::Ptr{CAPI.PJRT_Error}
    device = unsafe_pointer_to_objref(Ptr{Device}(unsafe_load(args.device)))::Device
    unsafe_load(args.num_payload) == 0 ||
        throw(unimplemented("error payloads are not supported yet"))
    error = Error(
        unsafe_load(args.error_code),
        unsafe_load(args.error_message),
        unsafe_load(args.error_message_size),
    )
    poisoned = poison_execution!(device, unsafe_load(args.launch_id), error)::Bool
    unsafe_store!(args.poisoned, poisoned)
    return ok()
end

"""
    Buffer

### Interface

- `destroy`
- `delete!`
- `isdeleted`
- `element_type` -> eltype
- `dimensions` -> size
- `unpadded_dimensions`
- `dynamic_dimension_indices`
- `memory_layout`
- `on_device_size_in_bytes` -> sizeof?
- `device`
- `memory`
- `copy_to_device` -> copyto!?
- `copy_to_memory`
- `to_host_buffer` -> copyto!?
- `copy_raw_to_host!`
- `bitcast!`
- `donate_with_control_dependency!`
- `opaque_device_memory_data_pointer`
- `increase_external_reference_count!`
- `decrease_external_reference_count!`
- `is_on_cpu`
- `ready_event`
- `unsafe_pointer`
"""
abstract type Buffer end

@pjrt_catch_return function pjrt_buffer_destroy(
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

@pjrt_catch_return function pjrt_buffer_delete(
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

@pjrt_catch_return function pjrt_buffer_is_deleted(
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

@pjrt_catch_return function pjrt_buffer_element_type(
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

@pjrt_catch_return function pjrt_buffer_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_Dimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dims = dimensions(buffer)::Vector{Int}
    unsafe_store!(args.dims, pointer(dims))
    unsafe_store!(args.num_dims, length(dims))
    return ok()
end

"""
    unpadded_dimensions(buffer::Buffer) -> Vector{Int64}

Return the dimensions of `buffer` without padding. They usually are `dimensions(buffer)`, but dynamically-sized
dimensions implemented by padding to a fixed size may be smaller.

Defaults to `dimensions(buffer)`, i.e. no dimension is padded.

!!! warning
The returned vector is handed to C by pointer, so it must be owned by `buffer` and never be resized.

Backs `PJRT_Buffer_UnpaddedDimensions`.
"""
unpadded_dimensions(buffer::Buffer) = dimensions(buffer)

@pjrt_catch_return function pjrt_buffer_unpadded_dimensions(
    args::Ptr{CAPI.PJRT_Buffer_UnpaddedDimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dims = unpadded_dimensions(buffer)::Vector{Int64}
    unsafe_store!(args.unpadded_dims, pointer(dims))
    unsafe_store!(args.num_dims, length(dims))
    return ok()
end

"""
    dynamic_dimension_indices(buffer::Buffer) -> Vector{Csize_t}

Return the 0-based indices of the dynamically-sized dimensions of `buffer`, or an empty vector if all are static.

!!! warning
The returned vector is handed to C by pointer, so it must be owned by `buffer` and never be resized.

Backs `PJRT_Buffer_DynamicDimensionIndices`.
"""
dynamic_dimension_indices(::Buffer)

@pjrt_catch_return function pjrt_buffer_dynamic_dimension_indices(
    args::Ptr{CAPI.PJRT_Buffer_DynamicDimensionIndices_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    indices = dynamic_dimension_indices(buffer)::Vector{Csize_t}
    unsafe_store!(args.dynamic_dim_indices, pointer(indices))
    unsafe_store!(args.num_dynamic_dims, length(indices))
    return ok()
end

"""
    MemoryLayout

Layout of an array in memory.
"""
abstract type MemoryLayout end

"""
    TiledMemoryLayout(minor_to_major, tiles=Vector{Int64}[])

Tiled layout of an array in memory, as described by `PJRT_Buffer_MemoryLayout` (of type `Tiled`).

## Fields

- `minor_to_major::Vector{Int64}`: the 0-based order of the dimensions, from the fastest-varying to the slowest-varying
  one (e.g. `[1, 0]` for a row-major matrix like C/numpy, `[0, 1]` for column-major matrix like Julia/Fortran).
- `tiles::Vector{Vector{Int64}}`: the sizes of each level of tiling, in the order they are applied to the minor-most
  dimensions (e.g. `[[8, 128], [2, 1]]` for XLA's `T(8,128)(2,1)`). Leave empty if the array is not tiled.
"""
struct TiledMemoryLayout <: MemoryLayout
    minor_to_major::Vector{Int64}
    tiles::Vector{Vector{Int64}}

    function TiledMemoryLayout(minor_to_major, tiles=Vector{Int64}[])
        # XLA can't handle such layouts, and aborts on some of them
        isperm(minor_to_major .+ 1) || throw(
            ArgumentError(
                "minor_to_major must be a permutation of 0:$(length(minor_to_major) - 1), got $minor_to_major",
            ),
        )
        all(tile -> all(>(0), tile), tiles) ||
            throw(ArgumentError("tile sizes must be positive, got $tiles"))
        return new(minor_to_major, tiles)
    end
end

function Base.:(==)(a::TiledMemoryLayout, b::TiledMemoryLayout)
    return a.minor_to_major == b.minor_to_major && a.tiles == b.tiles
end
function Base.hash(layout::TiledMemoryLayout, h::UInt)
    return hash(layout.tiles, hash(layout.minor_to_major, h))
end

"""
    StridedMemoryLayout(byte_strides)

Strided layout of an array in memory, as described by `PJRT_Buffer_MemoryLayout` (of type `Strides`):
`byte_strides::Vector{Int64}` holds the number of bytes to traverse along each dimension (e.g. `[8n, 8]` for a row-major
`m×n` matrix of `Float64`). Strides can be negative.
"""
struct StridedMemoryLayout <: MemoryLayout
    byte_strides::Vector{Int64}
end

function Base.:(==)(a::StridedMemoryLayout, b::StridedMemoryLayout)
    return a.byte_strides == b.byte_strides
end
Base.hash(layout::StridedMemoryLayout, h::UInt) = hash(layout.byte_strides, h)

# converts a layout passed by C, which must not be `C_NULL`
function MemoryLayout(c_layout::Ptr{CAPI.PJRT_Buffer_MemoryLayout})
    type = unsafe_load(c_layout.type)
    if type == CAPI.PJRT_Buffer_MemoryLayout_Type_Strides
        strides = unsafe_load(c_layout.strides)
        return StridedMemoryLayout(
            copy(unsafe_wrap(Vector{Int64}, strides.byte_strides, strides.num_byte_strides))
        )
    elseif type == CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled
        tiled = unsafe_load(c_layout.tiled)
        minor_to_major = copy(
            unsafe_wrap(Vector{Int64}, tiled.minor_to_major, tiled.minor_to_major_size)
        )
        # C passes the sizes of all tiles back to back, plus the number of dimensions of each tile
        tiles = Vector{Vector{Int64}}(undef, tiled.num_tiles)
        offset = 0
        for i in eachindex(tiles)
            n = Int(unsafe_load(tiled.tile_dim_sizes, i))
            tiles[i] = Int64[unsafe_load(tiled.tile_dims, offset + j) for j in 1:n]
            offset += n
        end
        # an invalid layout is an error of the caller
        try
            return TiledMemoryLayout(minor_to_major, tiles)
        catch err
            err isa ArgumentError && throw(invalid_argument(err.msg))
            rethrow()
        end
    else
        throw(invalid_argument("unknown layout type $type"))
    end
end

"""
    memory_layout(buffer::Buffer) -> MemoryLayout

Return the layout of `buffer` in memory (e.g. `TiledMemoryLayout(n-1:-1:0)` for XLA's default row-major layout of an
array with `n` dimensions).

If the plugin advertises [`LayoutsExtension`](@ref), it must return a [`TiledMemoryLayout`](@ref): XLA aborts if the
layout of a buffer can't be serialized.

It is called at most once per `buffer`: the C wrapper keeps a copy for as long as `buffer` is alive, since C keeps
pointers to it.

Backs `PJRT_Buffer_GetMemoryLayout` and `PJRT_Layouts_PJRT_Buffer_MemoryLayout`.
"""
memory_layout(::Buffer)

# C borrows the arrays of `PJRT_Buffer_MemoryLayout`, so they are kept for as long as the buffer is alive (like in XLA's
# reference plugin, which keeps them in the buffer)
const memory_layout_cache = WeakKeyIdDict{
    Buffer,Tuple{MemoryLayout,Vector{Int64},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_buffer_get_memory_layout(
    args::Ptr{CAPI.PJRT_Buffer_GetMemoryLayout_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    # for tiled layouts, C expects the sizes of all tiles back to back, plus the number of dimensions of each tile
    layout, tile_dims, tile_dim_sizes = get!(memory_layout_cache, buffer) do
        buffer_layout = memory_layout(buffer)::MemoryLayout
        if buffer_layout isa StridedMemoryLayout
            return (buffer_layout, Int64[], Csize_t[])
        elseif buffer_layout isa TiledMemoryLayout
            (
                buffer_layout,
                Int64[d for tile in buffer_layout.tiles for d in tile],
                Csize_t[length(tile) for tile in buffer_layout.tiles],
            )
        else
            throw(unimplemented("unsupported layout type $(typeof(buffer_layout))"))
        end
    end
    c_layout = args.layout
    unsafe_store!(c_layout.struct_size, csizeof(CAPI.PJRT_Buffer_MemoryLayout))
    unsafe_store!(c_layout.extension_start, C_NULL)
    if layout isa StridedMemoryLayout
        unsafe_store!(c_layout.type, CAPI.PJRT_Buffer_MemoryLayout_Type_Strides)
        unsafe_store!(
            c_layout.strides,
            CAPI.PJRT_Buffer_MemoryLayout_Strides(
                csizeof(CAPI.PJRT_Buffer_MemoryLayout_Strides), #= struct_size =#
                C_NULL, #= extension_start =#
                pointer(layout.byte_strides), #= byte_strides =#
                length(layout.byte_strides), #= num_byte_strides =#
            ),
        )
    else
        unsafe_store!(c_layout.type, CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled)
        unsafe_store!(
            c_layout.tiled,
            CAPI.PJRT_Buffer_MemoryLayout_Tiled(
                csizeof(CAPI.PJRT_Buffer_MemoryLayout_Tiled), #= struct_size =#
                C_NULL, #= extension_start =#
                pointer(layout.minor_to_major), #= minor_to_major =#
                length(layout.minor_to_major), #= minor_to_major_size =#
                pointer(tile_dims), #= tile_dims =#
                pointer(tile_dim_sizes), #= tile_dim_sizes =#
                length(tile_dim_sizes), #= num_tiles =#
            ),
        )
    end
    return ok()
end

"""
    device(buffer::Buffer) -> Device

Return the device where `buffer` resides. The device must be kept alive by its client.

Backs `PJRT_Buffer_Device`.
"""
device(::Buffer)

@pjrt_catch_return function pjrt_buffer_device(
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

@pjrt_catch_return function pjrt_buffer_memory(
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

@pjrt_catch_return function pjrt_buffer_on_device_size_in_bytes(
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

@pjrt_catch_return function pjrt_buffer_copy_to_device(
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

@pjrt_catch_return function pjrt_buffer_copy_to_memory(
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

@pjrt_catch_return function pjrt_buffer_to_host_buffer(
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

    # TODO relayout is not supported, so `host_layout` must be the layout of `buffer`. If it's `C_NULL`, the layout of
    # `buffer` is used.
    c_host_layout = unsafe_load(args.host_layout)
    if c_host_layout != C_NULL
        host_layout = MemoryLayout(c_host_layout)
        buffer_layout = memory_layout(buffer)::MemoryLayout
        host_layout == buffer_layout || throw(
            unimplemented("relayout from $buffer_layout to $host_layout is not supported"),
        )
    end

    to_host_buffer(dst, buffer)

    event = root!(Event())
    notify(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    copy_raw_to_host!(dst::Ptr, buffer::Buffer, offset::Integer, transfer_size::Integer)

Copy `transfer_size` bytes of the raw on-device data of `buffer`, starting at byte `offset`, to the host memory pointed
by `dst`. The C wrapper checks that the range is within `on_device_size_in_bytes(buffer)`.

!!! todo
The copy is _currently_ synchronous.

Backs `PJRT_Buffer_CopyRawToHost`.
"""
copy_raw_to_host!(::Ptr, ::Buffer, ::Integer, ::Integer)

@pjrt_catch_return function pjrt_buffer_copy_raw_to_host(
    args::Ptr{CAPI.PJRT_Buffer_CopyRawToHost_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    dst = unsafe_load(args.dst)
    offset = unsafe_load(args.offset)
    transfer_size = unsafe_load(args.transfer_size)

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    src_size = on_device_size_in_bytes(buffer)
    if offset < 0 || transfer_size < 0 || offset + transfer_size > src_size
        throw(
            invalid_argument(
                "copy range [$offset, $(offset + transfer_size)) is out of the $src_size bytes of the buffer",
            ),
        )
    end

    copy_raw_to_host!(dst, buffer, offset, transfer_size)

    event = root!(Event())
    notify(event)
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    bitcast!(buffer::Buffer, type, dims, layout) -> Buffer

Return a new buffer that takes over the on-device data of `buffer`, reinterpreted with element type
`type::CAPI.PJRT_Buffer_Type`, dimensions `dims::Vector{Int64}` and layout `layout::Union{Nothing,MemoryLayout}`
(`nothing` for the default layout, not the layout of `buffer`).

It must not copy any data, and the new buffer must have the same `on_device_size_in_bytes` as `buffer`. `buffer` is
donated: afterwards, `isdeleted(buffer)` must return `true`, but its data must not be freed.

The caller GC-roots the returned buffer.

Backs `PJRT_Buffer_Bitcast`.
"""
bitcast!(::Buffer, args...)

@pjrt_catch_return function pjrt_buffer_bitcast(
    args::Ptr{CAPI.PJRT_Buffer_Bitcast_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    _eltype = unsafe_load(args.element_type)
    _size = copy(
        unsafe_wrap(Vector{Int64}, unsafe_load(args.dims), unsafe_load(args.num_dims))
    )
    c_layout = unsafe_load(args.device_layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    out_buffer = bitcast!(buffer, _eltype, _size, layout)::Buffer
    root!(out_buffer)
    unsafe_store!(args.out_buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(out_buffer)))
    return ok()
end

# called by C when the destination of a `PJRT_Buffer_CopyRawToHostFuture` is ready (or failed): finishes the copy from
# the staging copy of the data
function pjrt_buffer_copy_raw_to_host_future_ready(
    args::Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Callback_Args}
)::Cvoid
    data = unsafe_pointer_to_objref(unsafe_load(args.callback_data))
    unroot!(data)
    staging, event = data[]::Tuple{Vector{UInt8},Event}

    code = unsafe_load(args.error_code)
    if code != CAPI.PJRT_Error_Code_OK
        error = Error(
            code, unsafe_load(args.error_message), unsafe_load(args.error_message_size)
        )
        notify(event, error)
        return nothing
    end

    dst = Ptr{UInt8}(unsafe_load(args.dst))
    GC.@preserve staging unsafe_copyto!(dst, pointer(staging), length(staging))
    notify(event)
    return nothing
end

@pjrt_catch_return function pjrt_buffer_copy_raw_to_host_future(
    args::Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    offset = unsafe_load(args.offset)
    transfer_size = unsafe_load(args.transfer_size)

    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    src_size = on_device_size_in_bytes(buffer)
    if offset < 0 || transfer_size < 0 || offset + transfer_size > src_size
        throw(
            invalid_argument(
                "copy range [$offset, $(offset + transfer_size)) is out of the $src_size bytes of the buffer",
            ),
        )
    end

    # the copy must happen even if `buffer` is deleted before the destination is ready, so the data is copied now
    # and moved to the destination by `pjrt_buffer_copy_raw_to_host_future_ready`
    staging = Vector{UInt8}(undef, transfer_size)
    GC.@preserve staging copy_raw_to_host!(pointer(staging), buffer, offset, transfer_size)

    event = root!(Event())
    data = root!(Ref((staging, event)))
    unsafe_store!(args.event, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    unsafe_store!(args.callback_data, pointer_from_objref(data))
    unsafe_store!(
        args.future_ready_callback,
        @cfunction(
            pjrt_buffer_copy_raw_to_host_future_ready,
            Cvoid,
            (Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Callback_Args},)
        ),
    )
    return ok()
end

"""
    donate_with_control_dependency!(buffer::Buffer, dependency::Event) -> Buffer

Return a new buffer that takes over the on-device data of `buffer`, and which is ready only once both `buffer` and
`dependency` are ready. If either of them fails, the new buffer fails too.

`buffer` is donated: afterwards, `isdeleted(buffer)` must return `true`, but its data must not be freed.

`dependency` is set by the C wrapper when C signals it, possibly after this returns, so it must not be waited on here.
The caller GC-roots the returned buffer.

Backs `PJRT_Buffer_DonateWithControlDependency`.
"""
donate_with_control_dependency!(::Buffer, ::Event)

# called by C when the dependency of a `PJRT_Buffer_DonateWithControlDependency` is ready (or failed)
function pjrt_buffer_donate_dependency_ready(
    args::Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Callback_Args}
)::Cvoid
    data = unsafe_pointer_to_objref(unsafe_load(args.callback_data))
    unroot!(data)
    dependency = data[]::Event
    code = unsafe_load(args.error_code)
    if code == CAPI.PJRT_Error_Code_OK
        notify(dependency)
    else
        error = Error(
            code, unsafe_load(args.error_message), unsafe_load(args.error_message_size)
        )
        notify(dependency, error)
    end
    return nothing
end

@pjrt_catch_return function pjrt_buffer_donate_with_control_dependency(
    args::Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))

    dependency = Event()
    out_buffer = donate_with_control_dependency!(buffer, dependency)::Buffer
    root!(out_buffer)

    # rooted in a `Ref` of its own, so that the callback unroots it even if the plugin hands `dependency` out to C
    data = root!(Ref(dependency))
    unsafe_store!(args.out_buffer, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(out_buffer)))
    unsafe_store!(args.callback_data, pointer_from_objref(data))
    unsafe_store!(
        args.dependency_ready_callback,
        @cfunction(
            pjrt_buffer_donate_dependency_ready,
            Cvoid,
            (Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Callback_Args},)
        ),
    )
    return ok()
end

"""
    opaque_device_memory_data_pointer(buffer::Buffer) -> Ptr

Return the opaque pointer to the device memory of `buffer`. It may become invalid at any point unless `buffer` has an
external reference (see `increase_external_reference_count!`).

Defaults to `unsafe_pointer(buffer)`.

Backs `PJRT_Buffer_OpaqueDeviceMemoryDataPointer`.
"""
opaque_device_memory_data_pointer(buffer::Buffer) = unsafe_pointer(buffer)

@pjrt_catch_return function pjrt_buffer_opaque_device_memory_data_pointer(
    args::Ptr{CAPI.PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    isdeleted(buffer) && throw(invalid_argument("buffer is deleted"))
    ptr = opaque_device_memory_data_pointer(buffer)::Ptr
    unsafe_store!(args.device_memory_ptr, ptr)
    return ok()
end

"""
    increase_external_reference_count!(buffer::Buffer)

Increment the number of external references to `buffer`, which mean that its data is shared with another framework
(e.g. through DLPack). While there is any, the data of `buffer` must be neither freed (even by `delete!`) nor moved.

Backs `PJRT_Buffer_IncreaseExternalReferenceCount`.
"""
increase_external_reference_count!(::Buffer)

@pjrt_catch_return function pjrt_buffer_increase_external_reference_count(
    args::Ptr{CAPI.PJRT_Buffer_IncreaseExternalReferenceCount_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    increase_external_reference_count!(buffer)
    return ok()
end

"""
    decrease_external_reference_count!(buffer::Buffer)

Decrement the number of external references to `buffer` (see `increase_external_reference_count!`). Must throw
[`invalid_argument`](@ref) if there is none.

Backs `PJRT_Buffer_DecreaseExternalReferenceCount`.
"""
decrease_external_reference_count!(::Buffer)

@pjrt_catch_return function pjrt_buffer_decrease_external_reference_count(
    args::Ptr{CAPI.PJRT_Buffer_DecreaseExternalReferenceCount_Args}
)::Ptr{CAPI.PJRT_Error}
    buffer = unsafe_pointer_to_objref(unsafe_load(args.buffer))::Buffer
    decrease_external_reference_count!(buffer)
    return ok()
end

"""
    is_on_cpu(buffer::Buffer) -> Bool

Return whether the data of `buffer` resides in host memory.

Backs `PJRT_Buffer_IsOnCpu`.
"""
is_on_cpu(::Buffer)

@pjrt_catch_return function pjrt_buffer_is_on_cpu(
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

@pjrt_catch_return function pjrt_buffer_ready_event(
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

@pjrt_catch_return function pjrt_buffer_unsafe_pointer(
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
- `cost_analysis`
- `output_element_types`
- `output_dimensions`
- `parameter_memory_kinds`
- `output_memory_kinds`
- `optimized_program`
- `serialize`
- `compile_options`
- `compiled_memory_stats`
"""
abstract type Executable end

@pjrt_catch_return function pjrt_executable_destroy(
    args::Ptr{CAPI.PJRT_Executable_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.executable)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

function pjrt_serialized_bytes_deleter(ptr::Ptr{Cvoid})::Cvoid
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

@pjrt_catch_return function pjrt_executable_name(
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

@pjrt_catch_return function pjrt_executable_num_replicas(
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

@pjrt_catch_return function pjrt_executable_num_partitions(
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

@pjrt_catch_return function pjrt_executable_num_outputs(
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

@pjrt_catch_return function pjrt_executable_size_of_generated_code_in_bytes(
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

@pjrt_catch_return function pjrt_executable_fingerprint(
    args::Ptr{CAPI.PJRT_Executable_Fingerprint_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    str = fingerprint(exe)::String
    unsafe_store!(args.executable_fingerprint, Cstring(pointer(str)))
    unsafe_store!(args.executable_fingerprint_size, sizeof(str))
    return ok()
end

# C views of the results of `cost_analysis`, which C keeps for as long as the executable is alive
const cost_analysis_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}
}()

"""
    cost_analysis(exe::Executable)

Return the cost analysis of `exe` as `name => value` pairs (e.g. `"flops" => 1.0e6`, `"bytes accessed" => 4096`).
Values must be `AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as
`String`, `Bool`, `Int64`, `Vector{Int64}` and `Float32`.

It is called at most once per `exe`: the C wrapper keeps a copy for as long as `exe` is alive, since C keeps pointers
to it.

Backs `PJRT_Executable_GetCostAnalysis`.
"""
cost_analysis(::Executable)

@pjrt_catch_return function pjrt_executable_get_cost_analysis(
    args::Ptr{CAPI.PJRT_Executable_GetCostAnalysis_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    props, _ = get!(cost_analysis_cache, exe) do
        pjrt_named_values(cost_analysis(exe))
    end
    unsafe_store!(args.num_properties, length(props))
    unsafe_store!(args.properties, isempty(props) ? C_NULL : pointer(props))
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

@pjrt_catch_return function pjrt_executable_output_element_types(
    args::Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    types = output_element_types(exe)::Vector{CAPI.PJRT_Buffer_Type}
    unsafe_store!(args.output_types, pointer(types))
    unsafe_store!(args.num_output_types, length(types))
    return ok()
end

"""
    output_dimensions(exe::Executable) -> Vector{Vector{Int64}}

Return the dimensions of each output of `exe`.

It is called at most once per `exe`: the C wrapper keeps a copy for as long as `exe` is alive, since C keeps pointers
to it.

Backs `PJRT_Executable_OutputDimensions`.
"""
output_dimensions(::Executable)

const output_dimensions_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{Int64},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_executable_output_dimensions(
    args::Ptr{CAPI.PJRT_Executable_OutputDimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    # C expects the dimensions of all outputs back to back, plus the number of dimensions of each output
    dims, ranks = get!(output_dimensions_cache, exe) do
        shapes = output_dimensions(exe)::Vector{Vector{Int64}}
        (
            Int64[d for shape in shapes for d in shape],
            Csize_t[length(shape) for shape in shapes],
        )
    end
    unsafe_store!(args.num_outputs, length(ranks))
    unsafe_store!(args.dims, pointer(dims))
    unsafe_store!(args.dim_sizes, pointer(ranks))
    return ok()
end

"""
    parameter_memory_kinds(exe::Executable) -> Vector{String}

Return the memory kind of each parameter of `exe` (e.g. `"device"`, `"pinned_host"`).

It is called at most once per `exe`: the C wrapper keeps the strings for as long as `exe` is alive, since C keeps
pointers to them.

Backs `PJRT_Executable_ParameterMemoryKinds`.
"""
parameter_memory_kinds(::Executable)

const parameter_memory_kinds_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{String},Vector{Cstring},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_executable_parameter_memory_kinds(
    args::Ptr{CAPI.PJRT_Executable_ParameterMemoryKinds_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    _, ptrs, sizes = get!(parameter_memory_kinds_cache, exe) do
        kinds = parameter_memory_kinds(exe)::Vector{String}
        (kinds, [Cstring(pointer(s)) for s in kinds], Csize_t[sizeof(s) for s in kinds])
    end
    unsafe_store!(args.num_parameters, length(ptrs))
    unsafe_store!(args.memory_kinds, pointer(ptrs))
    unsafe_store!(args.memory_kind_sizes, pointer(sizes))
    return ok()
end

"""
    output_memory_kinds(exe::Executable) -> Vector{String}

Return the memory kind of each output of `exe` (e.g. `"device"`, `"pinned_host"`).

It is called at most once per `exe`: the C wrapper keeps the strings for as long as `exe` is alive, since C keeps
pointers to them.

Backs `PJRT_Executable_OutputMemoryKinds`.
"""
output_memory_kinds(::Executable)

const output_memory_kinds_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{String},Vector{Cstring},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_executable_output_memory_kinds(
    args::Ptr{CAPI.PJRT_Executable_OutputMemoryKinds_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    _, ptrs, sizes = get!(output_memory_kinds_cache, exe) do
        kinds = output_memory_kinds(exe)::Vector{String}
        (kinds, [Cstring(pointer(s)) for s in kinds], Csize_t[sizeof(s) for s in kinds])
    end
    unsafe_store!(args.num_outputs, length(ptrs))
    unsafe_store!(args.memory_kinds, pointer(ptrs))
    unsafe_store!(args.memory_kind_sizes, pointer(sizes))
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

@pjrt_catch_return function pjrt_executable_optimized_program(
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

@pjrt_catch_return function pjrt_executable_serialize(
    args::Ptr{CAPI.PJRT_Executable_Serialize_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    bytes = root!(serialize(exe)::Vector{UInt8})
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(args.serialized_executable, pointer_from_objref(bytes))
    unsafe_store!(
        args.serialized_executable_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

"""
    compile_options(exe::Executable) -> Reactant.Proto.xla.CompileOptionsProto

Return the compile options that `exe` was compiled with, i.e. the ones given to `compile!`.

Backs `PJRT_Executable_GetCompileOptions`.
"""
compile_options(::Executable)

@pjrt_catch_return function pjrt_executable_get_compile_options(
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
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

"""
    compiled_memory_stats(exe::Executable)

Return the memory usage of `exe`, mirroring `xla::CompiledMemoryStats`, as an object (e.g. a `NamedTuple`) with the
following integer properties:

- device memory: `generated_code_size_in_bytes`, `argument_size_in_bytes`, `output_size_in_bytes`,
  `alias_size_in_bytes` (how much of the arguments is reused for the outputs), `temp_size_in_bytes`
- host memory: `host_generated_code_size_in_bytes`, `host_argument_size_in_bytes`, `host_output_size_in_bytes`,
  `host_alias_size_in_bytes`, `host_temp_size_in_bytes`
- device memory totals: `peak_memory_in_bytes`, `total_size_in_bytes`, `total_allocation_bytes`,
  `indefinite_allocations`, `peak_unpadded_heap_bytes`

Backs `PJRT_Executable_GetCompiledMemoryStats`.
"""
compiled_memory_stats(::Executable)

@pjrt_catch_return function pjrt_executable_get_compiled_memory_stats(
    args::Ptr{CAPI.PJRT_Executable_GetCompiledMemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    stats = compiled_memory_stats(exe)
    # skip `struct_size`, `extension_start` and `executable`
    for field in fieldnames(CAPI.PJRT_Executable_GetCompiledMemoryStats_Args)[4:end]
        unsafe_store!(getproperty(args, field), getproperty(stats, field))
    end
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

"""
    AsyncHostToDeviceTransferManager

Manages the transfer of host data to a set of new buffers, as returned by `create_buffers_for_async_host_to_device`.

Buffers are identified by their 1-based index (C's 0-based index + 1), which the C wrapper checks to be within
`1:buffer_count(manager)`. Once a buffer has received its last transfer, or an error with `set_buffer_error!`, no more
transfers into it are made.

!!! todo
Transfers are _currently_ synchronous: the events returned to C are set as soon as the transfer methods return, so
they must be done reading the host data by then.

### Interface

- `buffer_count`
- `device`
- `retrieve_buffer!`
- `buffer_size`
- `transfer_data!`
- `transfer_literal!`
- `set_buffer_error!`
- `add_metadata!`
"""
abstract type AsyncHostToDeviceTransferManager end

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_destroy(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

# convert C's 0-based `buffer_index` into the 1-based index of a buffer of `manager`, checking that it exists
function checked_buffer_index(manager::AsyncHostToDeviceTransferManager, c_index::Integer)
    n = buffer_count(manager)
    if !(0 <= c_index < n)
        throw(invalid_argument("buffer index $c_index is out of range for $n buffers"))
    end
    return Int(c_index) + 1
end

"""
    buffer_count(manager::AsyncHostToDeviceTransferManager) -> Integer

Return the number of buffers that `manager` transfers data into.

Backs `PJRT_AsyncHostToDeviceTransferManager_BufferCount`.
"""
buffer_count(::AsyncHostToDeviceTransferManager)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_buffer_count(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    unsafe_store!(args.buffer_count, buffer_count(manager))
    return ok()
end

"""
    device(manager::AsyncHostToDeviceTransferManager) -> Device

Return the device that `manager` transfers data to. The device must be kept alive by its client.

Backs `PJRT_AsyncHostToDeviceTransferManager_Device`.
"""
device(::AsyncHostToDeviceTransferManager)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_device(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Device_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    dev = device(manager)::Device
    unsafe_store!(args.device_out, Ptr{CAPI.PJRT_Device}(pointer_from_objref(dev)))
    return ok()
end

"""
    retrieve_buffer!(manager::AsyncHostToDeviceTransferManager, index::Int) -> Buffer

Return the `index`-th buffer of `manager`, handing it over to the caller. It may be retrieved before its transfers are
done, in which case it becomes ready once they are. C retrieves each buffer at most once.

The caller GC-roots the returned buffer.

Backs `PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer`.
"""
retrieve_buffer!(::AsyncHostToDeviceTransferManager, ::Int)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_retrieve_buffer(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    buffer = retrieve_buffer!(manager, index)::Buffer
    root!(buffer)
    unsafe_store!(args.buffer_out, Ptr{CAPI.PJRT_Buffer}(pointer_from_objref(buffer)))
    return ok()
end

"""
    buffer_size(manager::AsyncHostToDeviceTransferManager, index::Int) -> Integer

Return the on-device size in bytes of the `index`-th buffer of `manager`.

Backs `PJRT_AsyncHostToDeviceTransferManager_BufferSize`.
"""
buffer_size(::AsyncHostToDeviceTransferManager, ::Int)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_buffer_size(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    unsafe_store!(args.buffer_size, buffer_size(manager, index))
    return ok()
end

"""
    transfer_data!(manager::AsyncHostToDeviceTransferManager, index::Int, data::Ptr{Cvoid}, offset, size, is_last::Bool)

Copy `size` bytes of host memory at `data` into the `index`-th buffer of `manager`, starting at byte `offset` of its
on-device data. `data` is already laid out in the on-device format (e.g. as returned by `copy_raw_to_host!`). The C
wrapper checks that the range is within `buffer_size(manager, index)`.

If `is_last`, the buffer becomes ready once the copy is done, and no more transfers into it are made.

Backs `PJRT_AsyncHostToDeviceTransferManager_TransferData`.
"""
transfer_data!(::AsyncHostToDeviceTransferManager, ::Int, args...)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_transfer_data(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    offset = unsafe_load(args.offset)
    transfer_size = unsafe_load(args.transfer_size)

    dst_size = buffer_size(manager, index)
    if offset < 0 || transfer_size < 0 || offset + transfer_size > dst_size
        throw(
            invalid_argument(
                "transfer range [$offset, $(offset + transfer_size)) is out of the $dst_size bytes of the buffer",
            ),
        )
    end

    data = unsafe_load(args.data)
    transfer_data!(
        manager, index, data, offset, transfer_size, unsafe_load(args.is_last_transfer)
    )

    event = root!(Event())
    notify(event)
    unsafe_store!(
        args.done_with_h2d_transfer, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event))
    )
    return ok()
end

"""
    transfer_literal!(manager::AsyncHostToDeviceTransferManager, index::Int, data::Ptr{Cvoid}, type, dims, layout)

Copy the host array at `data`, with element type `type::CAPI.PJRT_Buffer_Type`, dimensions `dims::Vector{Int64}` and
host layout `layout::Union{Nothing,MemoryLayout}` (`nothing` for the default layout), into the whole `index`-th
buffer of `manager`, converting it to the on-device format. The buffer becomes ready once the copy is done, and no more
transfers into it are made.

Backs `PJRT_AsyncHostToDeviceTransferManager_TransferLiteral`.
"""
transfer_literal!(::AsyncHostToDeviceTransferManager, ::Int, args...)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_transfer_literal(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))

    _eltype = unsafe_load(args.shape_element_type)
    _size = copy(
        unsafe_wrap(
            Vector{Int64}, unsafe_load(args.shape_dims), unsafe_load(args.shape_num_dims)
        ),
    )
    c_layout = unsafe_load(args.shape_layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)
    transfer_literal!(manager, index, unsafe_load(args.data), _eltype, _size, layout)

    event = root!(Event())
    notify(event)
    unsafe_store!(
        args.done_with_h2d_transfer, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event))
    )
    return ok()
end

"""
    set_buffer_error!(manager::AsyncHostToDeviceTransferManager, index::Int, error::Error)

Make the `index`-th buffer of `manager` fail with `error` instead of receiving data. No more transfers into it are
made.

Backs `PJRT_AsyncHostToDeviceTransferManager_SetBufferError`.
"""
set_buffer_error!(::AsyncHostToDeviceTransferManager, ::Int, ::Error)

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_set_buffer_error(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    index = checked_buffer_index(manager, unsafe_load(args.buffer_index))
    error = Error(
        unsafe_load(args.error_code),
        unsafe_load(args.error_message),
        unsafe_load(args.error_message_size),
    )
    set_buffer_error!(manager, index, error)
    return ok()
end

"""
    add_metadata!(manager::AsyncHostToDeviceTransferManager, metadata::Dict{String,String})

Attach `metadata` to the transfers of `manager`, e.g. a handle to identify them when debugging.

Backs `PJRT_AsyncHostToDeviceTransferManager_AddMetadata`.
"""
add_metadata!(::AsyncHostToDeviceTransferManager, ::Dict{String,String})

@pjrt_catch_return function pjrt_async_host_to_device_transfer_manager_add_metadata(
    args::Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.transfer_manager)
    manager = unsafe_pointer_to_objref(ptr)::AsyncHostToDeviceTransferManager
    values = unsafe_load_pjrt_named_values(
        unsafe_load(args.transfer_metadata), unsafe_load(args.num_metadata)
    )
    metadata = Dict{String,String}()
    for (key, value) in values
        value isa String ||
            throw(invalid_argument("transfer metadata `$key` is not a string"))
        metadata[key] = value
    end
    add_metadata!(manager, metadata)
    return ok()
end

"""
    CopyToDeviceStream

A stream of chunks of host data to be copied to a device, which expects `total_bytes` bytes in total, in chunks whose
size is a multiple of `granule_size`. It is complete once `current_bytes == total_bytes`, and never changes afterwards.

The C wrapper checks those conditions before calling `add_chunk!`.

!!! todo
`execute` ignores the receive callbacks of `PJRT_ExecuteOptions`, which is how C gets a stream, so nothing can hand
one to C yet.

### Interface

- `add_chunk!`
- `total_bytes`
- `granule_size`
- `current_bytes`
"""
abstract type CopyToDeviceStream end

@pjrt_catch_return function pjrt_copy_to_device_stream_destroy(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.stream)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

"""
    add_chunk!(stream::CopyToDeviceStream, data::Ptr{Cvoid}, size::Integer)

Copy the `size` bytes of host memory at `data` to the device as the next chunk of `stream`, adding `size` to
`current_bytes(stream)`.

`data` is freed right after this returns, so the copy must be done by then.

!!! todo
The copy is _currently_ synchronous: the event returned to C is set as soon as this returns.

Backs `PJRT_CopyToDeviceStream_AddChunk`.
"""
add_chunk!(::CopyToDeviceStream, ::Ptr{Cvoid}, ::Integer)

@pjrt_catch_return function pjrt_copy_to_device_stream_add_chunk(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_AddChunk_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    chunk_ptr = unsafe_load(args.chunk)
    chunk_ptr == C_NULL && throw(invalid_argument("chunk is null"))
    # `PJRT_Chunk` has no field accessors on pointers, so it's loaded by value
    chunk = unsafe_load(chunk_ptr)

    # the stream takes ownership of the chunk, so the chunk is freed whatever happens; invalid chunks and failed
    # transfers are reported through the returned event, like in XLA
    error = try
        total = total_bytes(stream)
        current = current_bytes(stream)
        granule = granule_size(stream)
        if current == total
            throw(invalid_argument("stream is already complete"))
        elseif chunk.size % granule != 0
            throw(
                invalid_argument(
                    "chunk of $(chunk.size) bytes is not a multiple of the $granule-byte granule",
                ),
            )
        elseif current + chunk.size > total
            throw(
                invalid_argument(
                    "chunk of $(chunk.size) bytes exceeds the $(total - current) bytes left in the stream",
                ),
            )
        end
        add_chunk!(stream, chunk.data, chunk.size)
        nothing
    catch err
        err isa Error || rethrow()
        err
    finally
        if chunk.deleter != C_NULL
            ccall(chunk.deleter, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), chunk.data, chunk.deleter_arg)
        end
    end

    event = root!(Event())
    notify(event, error)
    unsafe_store!(args.transfer_complete, Ptr{CAPI.PJRT_Event}(pointer_from_objref(event)))
    return ok()
end

"""
    total_bytes(stream::CopyToDeviceStream) -> Integer

Return the total number of bytes that `stream` expects to be transferred.

Backs `PJRT_CopyToDeviceStream_TotalBytes`.
"""
total_bytes(::CopyToDeviceStream)

@pjrt_catch_return function pjrt_copy_to_device_stream_total_bytes(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_TotalBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    unsafe_store!(args.total_bytes, total_bytes(stream))
    return ok()
end

"""
    granule_size(stream::CopyToDeviceStream) -> Integer

Return the granule size in bytes of `stream`: the size of every chunk must be a multiple of it.

Backs `PJRT_CopyToDeviceStream_GranuleSize`.
"""
granule_size(::CopyToDeviceStream)

@pjrt_catch_return function pjrt_copy_to_device_stream_granule_size(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_GranuleSize_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    unsafe_store!(args.granule_size_in_bytes, granule_size(stream))
    return ok()
end

"""
    current_bytes(stream::CopyToDeviceStream) -> Integer

Return the number of bytes that `stream` has transferred or buffered to transfer so far.

Backs `PJRT_CopyToDeviceStream_CurrentBytes`.
"""
current_bytes(::CopyToDeviceStream)

@pjrt_catch_return function pjrt_copy_to_device_stream_current_bytes(
    args::Ptr{CAPI.PJRT_CopyToDeviceStream_CurrentBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    stream = unsafe_pointer_to_objref(unsafe_load(args.stream))::CopyToDeviceStream
    unsafe_store!(args.current_bytes, current_bytes(stream))
    return ok()
end

"""
    create_buffers_for_async_host_to_device(client::Client, specs::Vector, layouts::Vector, mem::Memory)
        -> AsyncHostToDeviceTransferManager

Return a new [`AsyncHostToDeviceTransferManager`](@ref) for a set of new buffers placed on memory `mem`, one per
element of `specs`, which are `NamedTuple`s with fields `element_type::CAPI.PJRT_Buffer_Type` and `dims::Vector{Int64}`.

`layouts::Vector{Union{Nothing,MemoryLayout}}` is either empty, for the default layouts, or holds the layout of each
buffer (`nothing` for its default layout).

The caller GC-roots the returned manager.

Backs `PJRT_Client_CreateBuffersForAsyncHostToDevice`.
"""
create_buffers_for_async_host_to_device(::Client, ::Vector, ::Vector, ::Memory)

@pjrt_catch_return function pjrt_client_create_buffers_for_async_host_to_device(
    args::Ptr{CAPI.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    mem_ptr = Ptr{Memory}(unsafe_load(args.memory))
    mem_ptr == C_NULL && throw(invalid_argument("No memory has been assigned"))
    mem = unsafe_pointer_to_objref(mem_ptr)::Memory

    c_specs = unsafe_load(args.shape_specs)
    specs = map(1:unsafe_load(args.num_shape_specs)) do i
        spec = unsafe_load(c_specs, i)
        (;
            element_type=spec.element_type,
            dims=copy(unsafe_wrap(Vector{Int64}, spec.dims, spec.num_dims)),
        )
    end

    num_layouts = unsafe_load(args.num_device_layouts)
    if num_layouts != 0 && num_layouts != length(specs)
        throw(
            invalid_argument("got $num_layouts device layouts for $(length(specs)) buffers")
        )
    end
    c_layouts = unsafe_load(args.device_layouts)
    layouts = Vector{Union{Nothing,MemoryLayout}}(undef, num_layouts)
    for i in eachindex(layouts)
        c_layout = unsafe_load(c_layouts, i)
        layouts[i] = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)
    end

    manager = create_buffers_for_async_host_to_device(
        client, specs, layouts, mem
    )::AsyncHostToDeviceTransferManager
    root!(manager)

    unsafe_store!(
        args.transfer_manager,
        Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager}(pointer_from_objref(manager)),
    )
    return ok()
end

"""
    TopologyDescription

The description of the devices of a platform and how they are connected, used by `compile` to compile programs without
a `Client`. C gets one from a client with `topology_description`, or without a client through
`create_topology_description` and `deserialize_topology_description`.

### Interface

- `platform_name`
- `platform_version`
- `device_descriptions`
- `serialize`
- `attributes`
- `fingerprint`
- `make_canonical_shape_for_memory_space`
- `memory_space_kind_ids`
"""
abstract type TopologyDescription end

"""
    platform_name(topology::TopologyDescription) -> String

Return a string that identifies the platform of `topology` (e.g. `"cpu"`, `"gpu"`, `"tpu"`).

!!! warning
The string is handed to C by pointer, so it must be owned by `topology`.

Backs `PJRT_TopologyDescription_PlatformName`.
"""
platform_name(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_platform_name(
    args::Ptr{CAPI.PJRT_TopologyDescription_PlatformName_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    str = platform_name(topology)::String
    unsafe_store!(args.platform_name, Cstring(pointer(str)))
    unsafe_store!(args.platform_name_size, sizeof(str))
    return ok()
end

"""
    platform_version(topology::TopologyDescription) -> String

Return a string containing the platform version of `topology`.

!!! warning
The string is handed to C by pointer, so it must be owned by `topology`.

Backs `PJRT_TopologyDescription_PlatformVersion`.
"""
platform_version(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_platform_version(
    args::Ptr{CAPI.PJRT_TopologyDescription_PlatformVersion_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    str = platform_version(topology)::String
    unsafe_store!(args.platform_version, Cstring(pointer(str)))
    unsafe_store!(args.platform_version_size, sizeof(str))
    return ok()
end

"""
    device_descriptions(topology::TopologyDescription) -> Vector{DeviceDescription}

Return the descriptions of all the devices in `topology`, in any order but the same one on every call.

!!! warning
The vector is handed to C by pointer, so it must be owned by `topology` and never be resized.

Backs `PJRT_TopologyDescription_GetDeviceDescriptions`.
"""
device_descriptions(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_get_device_descriptions(
    args::Ptr{CAPI.PJRT_TopologyDescription_GetDeviceDescriptions_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    descs = device_descriptions(topology)::Vector{DeviceDescription}
    unsafe_store!(args.descriptions, Ptr{Ptr{CAPI.PJRT_DeviceDescription}}(pointer(descs)))
    unsafe_store!(args.num_descriptions, length(descs))
    return ok()
end

"""
    serialize(topology::TopologyDescription) -> Vector{UInt8}

Return a serialization of `topology` that `deserialize_topology_description` can read back, also used in cache keys.

The returned vector is GC rooted until C calls its deleter, so it must be a new vector that `topology` doesn't hand out
anywhere else.

Backs `PJRT_TopologyDescription_Serialize`.
"""
serialize(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_serialize(
    args::Ptr{CAPI.PJRT_TopologyDescription_Serialize_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    bytes = root!(serialize(topology)::Vector{UInt8})
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(
        args.serialized_topology,
        Ptr{CAPI.PJRT_SerializedTopology}(pointer_from_objref(bytes)),
    )
    unsafe_store!(
        args.serialized_topology_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

# C views of the results of `attributes(::TopologyDescription)`, which C keeps for as long as the topology is alive
const topology_attributes_cache = WeakKeyIdDict{
    TopologyDescription,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}
}()

"""
    attributes(topology::TopologyDescription)

Return the platform-specific attributes of `topology` as `name => value` pairs (e.g. a `Dict`). Values must be
`AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as `String`,
`Bool`, `Int64`, `Vector{Int64}` and `Float32`.

It is called at most once per `topology`: the C wrapper keeps a copy for as long as `topology` is alive, since C keeps
pointers to it.

Backs `PJRT_TopologyDescription_Attributes`.
"""
attributes(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_attributes(
    args::Ptr{CAPI.PJRT_TopologyDescription_Attributes_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    values, _ = get!(topology_attributes_cache, topology) do
        pjrt_named_values(attributes(topology))
    end
    unsafe_store!(args.attributes, pointer(values))
    unsafe_store!(args.num_attributes, length(values))
    return ok()
end

"""
    fingerprint(topology::TopologyDescription) -> UInt64

Return a fingerprint of `topology` for use in cache keys. It doesn't need to be stable across versions.

Backs `PJRT_TopologyDescription_Fingerprint`.
"""
fingerprint(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_fingerprint(
    args::Ptr{CAPI.PJRT_TopologyDescription_Fingerprint_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    unsafe_store!(args.fingerprint, fingerprint(topology)::UInt64)
    return ok()
end

"""
    make_canonical_shape_for_memory_space(topology::TopologyDescription, kind_id::Integer, type, dims, layout)
        -> Reactant.Proto.xla.ShapeProto

Return the shape, including its layout, that a buffer with element type `type::CAPI.PJRT_Buffer_Type`, dimensions
`dims::Vector{Int64}` and layout `layout::Union{Nothing,MemoryLayout}` (`nothing` if none is given) takes once
transferred to a memory space of kind id `kind_id` (see `memory_space_kind_ids`).

Backs `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace`.
"""
make_canonical_shape_for_memory_space(::TopologyDescription, ::Integer, args...)

@pjrt_catch_return function pjrt_topology_description_make_canonical_shape_for_memory_space(
    args::Ptr{CAPI.PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    kind_id = unsafe_load(args.memory_space_kind_id)
    _eltype = unsafe_load(args.element_type)
    _size = copy(
        unsafe_wrap(Vector{Int64}, unsafe_load(args.dims), unsafe_load(args.num_dims))
    )
    c_layout = unsafe_load(args.layout)
    layout = c_layout == C_NULL ? nothing : MemoryLayout(c_layout)

    shape = make_canonical_shape_for_memory_space(
        topology, kind_id, _eltype, _size, layout
    )::Reactant.Proto.xla.ShapeProto
    bytes = Reactant.ProtoUtils.proto_to_bytes(shape)

    # the deleter only gets the pointer to the data, so it's malloc'ed and freed by `free` instead of GC rooted
    ptr = Ptr{UInt8}(Libc.malloc(max(length(bytes), 1)))
    ptr == C_NULL && throw(resource_exhausted())
    GC.@preserve bytes unsafe_copyto!(ptr, pointer(bytes), length(bytes))
    unsafe_store!(args.serialized_shape, Cstring(ptr))
    unsafe_store!(args.serialized_shape_size, length(bytes))
    unsafe_store!(args.serialized_shape_deleter, cglobal(:free))
    return ok()
end

"""
    memory_space_kind_ids(topology::TopologyDescription) -> Vector{Cint}

Return the kind ids of the memory spaces in `topology` (see `kind_id(::Memory)`). The first one must be the kind id
of the default memory space.

!!! warning
The vector is handed to C by pointer, so it must be owned by `topology` and never be resized.

Backs `PJRT_TopologyDescription_GetMemorySpaceKindIds`.
"""
memory_space_kind_ids(::TopologyDescription)

@pjrt_catch_return function pjrt_topology_description_get_memory_space_kind_ids(
    args::Ptr{CAPI.PJRT_TopologyDescription_GetMemorySpaceKindIds_Args}
)::Ptr{CAPI.PJRT_Error}
    topology = unsafe_pointer_to_objref(unsafe_load(args.topology))::TopologyDescription
    ids = memory_space_kind_ids(topology)::Vector{Cint}
    unsafe_store!(args.memory_space_kind_ids, pointer(ids))
    unsafe_store!(args.num_memory_space_kind_ids, length(ids))
    return ok()
end

"""
    topology_description(client::Client) -> TopologyDescription

Return the description of the topology of the devices of `client`.

The topology must be kept alive by `client`: C doesn't destroy it.

Backs `PJRT_Client_TopologyDescription`.
"""
topology_description(::Client)

@pjrt_catch_return function pjrt_client_topology_description(
    args::Ptr{CAPI.PJRT_Client_TopologyDescription_Args}
)::Ptr{CAPI.PJRT_Error}
    client = unsafe_pointer_to_objref(unsafe_load(args.client))::Client
    topology = topology_description(client)::TopologyDescription
    unsafe_store!(
        args.topology, Ptr{CAPI.PJRT_TopologyDescription}(pointer_from_objref(topology))
    )
    return ok()
end

"""
    compile(topology::TopologyDescription, code::MLIR.IR.Module, options, client::Union{Nothing,Client}) -> Executable

Compile `code` for the devices described by `topology` and return an executable, which must be loaded with `load`
before it can run.

`code` and `options::Reactant.Proto.xla.CompileOptionsProto` are like in `compile!`. `client` is an optional client of
the same platform, for performance-guided optimizations.

The caller GC-roots the returned executable.

Backs `PJRT_Compile`.
"""
compile(::TopologyDescription, args...)

@pjrt_catch_return function pjrt_compile(
    args::Ptr{CAPI.PJRT_Compile_Args}
)::Ptr{CAPI.PJRT_Error}
    topology_ptr = unsafe_load(args.topology)
    topology_ptr == C_NULL && throw(invalid_argument("No topology has been given"))
    topology = unsafe_pointer_to_objref(topology_ptr)::TopologyDescription

    client_ptr = unsafe_load(args.client)
    client = client_ptr == C_NULL ? nothing : unsafe_pointer_to_objref(client_ptr)::Client

    mlir_mod = unsafe_load_program(Ptr{CAPI.PJRT_Program}(unsafe_load(args.program)))
    options = unsafe_load_compile_options(
        unsafe_load(args.compile_options), unsafe_load(args.compile_options_size)
    )
    exe = compile(topology, mlir_mod, options, client)::Executable
    root!(exe)
    unsafe_store!(args.executable, Ptr{CAPI.PJRT_Executable}(pointer_from_objref(exe)))
    return ok()
end

unimplemented_c_hook() = Ptr{CAPI.PJRT_Error}(pointer_from_objref(root!(unimplemented())))

## Extensions
"""
    Extension

A PJRT extension that a plugin implements: an optional set of C functions that XLA looks up by type in the extension
chain of the `PJRT_Api`. Pass its type to [`make_api`](@ref) to advertise it to XLA. Since XLA only uses an extension
when it is in the chain, a plugin must pass only the extensions whose interface it implements.

Each subtype `E` is a `mutable struct` whose first field is the C struct of the extension (e.g.
`CAPI.PJRT_MemoryDescriptions_Extension`), so that a pointer to an `E` is a pointer to its C struct. Julia-only fields
(e.g. the `Base.CFunction`s of the C struct) can follow it.

### Interface

- `extension_type`
- `E(plugin::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base})`: build the C struct, whose `PJRT_Extension_Base`
  has the type of `E` and `next` as the next extension of the chain
"""
abstract type Extension end

"""
    extension_type(::Type{E}) where {E<:Extension} -> CAPI.PJRT_Extension_Type

Return the type that XLA identifies extensions of type `E` by. Each extension type can be passed only once to
[`make_api`](@ref).
"""
function extension_type end

include("Extensions/MemoryDescriptions.jl")
include("Extensions/FFI.jl")
include("Extensions/Layouts.jl")

"""
    PluginApi

A `PJRT_Api` function table bound to a specific [`AbstractPlugin`](@ref) and set of [`Extension`](@ref)s. Create it
with [`make_api`](@ref).

!!! warning
It must be kept alive for as long as any client or topology created through it exists.
"""
struct PluginApi
    api::Base.RefValue{CAPI.PJRT_Api}

    # plugin entrypoints
    plugin_initialize::Base.CFunction
    plugin_attributes::Base.CFunction
    client_create::Base.CFunction
    topology_create::Base.CFunction
    topology_deserialize::Base.CFunction

    # keepalive
    extensions::Vector{Extension}
end

function Base.unsafe_convert(::Type{Ptr{CAPI.PJRT_Api}}, x::PluginApi)
    return unsafe_convert(Ptr{CAPI.PJRT_Api}, x.api)
end

function __init__()
    #! format: off
    vtable_error[]= CAPI.PJRT_Error_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Error_FunctionTable),
        #= instance_size =# csizeof(CAPI.PJRT_Error),
        #= extension_start =# C_NULL,
        #= destroy =# @cfunction(pjrt_error_destroy_obj, Cvoid, (Ptr{Error},)),
        #= message =# @cfunction(pjrt_error_message_obj, Cvoid, (Ptr{Error}, Ptr{Cstring}, Ptr{Csize_t})),
        #= get_code =# @cfunction(pjrt_error_getcode_obj, CAPI.PJRT_Error_Code, (Ptr{Error},)),
        #= for_each_payload =# @cfunction(pjrt_error_for_each_payload_obj, Cvoid, (Ptr{Error}, CAPI.PJRT_Error_PayloadVisitor, Ptr{Cvoid})),
    )

    vtable_memory[] = CAPI.PJRT_Memory_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Memory_FunctionTable),
        #= extension_start =# C_NULL,
        #= instance_struct_size =# csizeof(CAPI.PJRT_Memory),
        #= get_user_data =# C_NULL,
        #= set_user_data =# C_NULL,
    )
    #! format: on
    return nothing
end

"""
    make_api(plugin::AbstractPlugin; extensions=()) -> PluginApi

Build the `PJRT_Api` function table of `plugin`.

`extensions` is a collection of [`Extension`](@ref) types, which are built as `E(plugin, next)` and advertised to XLA
through the extension chain of the table, in the given order, e.g.
```julia
make_api(MyPlugin(); extensions=[MemoryDescriptionsExtension, FFIExtension])
```
Each extension type can be given only once.

Store the result in a global (e.g. a `const Ref`) set from the plugin's `__init__`, and pass
`unsafe_convert(Ptr{CAPI.PJRT_Api}, api)` to XLA.
"""
function make_api(plugin::P; extensions=()) where {P<:AbstractPlugin}
    extensions = collect(Type{<:Extension}, extensions)
    types = map(extension_type, extensions)
    allunique(types) ||
        throw(ArgumentError("each extension type can be given only once, got $types"))
    Base.issingletontype(P) || throw(
        ArgumentError("plugin must be stateless (a `struct` without fields), got a `$P`"),
    )

    # build the chain back to front, so that each extension gets the pointer to the next one
    exts = Vector{Extension}(undef, length(extensions))
    extension_start = Ptr{CAPI.PJRT_Extension_Base}(C_NULL)
    for i in reverse(eachindex(extensions))
        E = extensions[i]
        ext = E(plugin, extension_start)
        ext isa E || throw(ArgumentError("constructor of $E returned a `$(typeof(ext))`"))
        ismutable(ext) || throw(ArgumentError("$E must be a mutable struct"))
        exts[i] = ext
        extension_start = Ptr{CAPI.PJRT_Extension_Base}(pointer_from_objref(ext))
    end

    # generic handler for unimplemented PJRT API functions
    c_unimplemented = @cfunction(unimplemented_c_hook, Ptr{CAPI.PJRT_Error}, ())
    c_plugin_initialize = @cfunction(
        $(PluginInitialize{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_Plugin_Initialize_Args},)
    )
    c_plugin_attributes = @cfunction(
        $(PluginAttributes{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_Plugin_Attributes_Args},)
    )
    c_client_create = @cfunction(
        $(ClientCreate{P}()), Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Create_Args},)
    )
    c_topology_create = @cfunction(
        $(TopologyDescriptionCreate{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_TopologyDescription_Create_Args},)
    )
    c_topology_deserialize = @cfunction(
        $(TopologyDescriptionDeserialize{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_TopologyDescription_Deserialize_Args},)
    )

    #! format: off
    api = Ref(CAPI.PJRT_Api(
        #= struct_size =# csizeof(CAPI.PJRT_Api),
        #= extension_start =# extension_start,
        #= pjrt_api_version =# CAPI.PJRT_Api_Version(
            #= struct_size =# csizeof(CAPI.PJRT_Api_Version),
            #= extension_start =# C_NULL,
            #= major_version =# CAPI.PJRT_API_MAJOR,
            #= minor_version =# CAPI.PJRT_API_MINOR,
        ),
        #= PJRT_Error_Destroy =# @cfunction(pjrt_error_destroy, Cvoid, (Ptr{CAPI.PJRT_Error_Destroy_Args},)),
        #= PJRT_Error_Message =# @cfunction(pjrt_error_message, Cvoid, (Ptr{CAPI.PJRT_Error_Message_Args},)),
        #= PJRT_Error_GetCode =# @cfunction(pjrt_error_getcode, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_GetCode_Args},)),
        #= PJRT_Plugin_Initialize =# unsafe_convert(Ptr{Cvoid}, c_plugin_initialize),
        #= PJRT_Plugin_Attributes =# unsafe_convert(Ptr{Cvoid}, c_plugin_attributes),
        #= PJRT_Event_Destroy =# @cfunction(pjrt_event_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Destroy_Args},)),
        #= PJRT_Event_IsReady =# @cfunction(pjrt_event_is_ready, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_IsReady_Args},)),
        #= PJRT_Event_Error =# @cfunction(pjrt_event_error, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Error_Args},)),
        #= PJRT_Event_Await =# @cfunction(pjrt_event_await, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Await_Args},)),
        #= PJRT_Event_OnReady =# @cfunction(pjrt_event_on_ready, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_OnReady_Args},)),
        #= PJRT_Client_Create =# unsafe_convert(Ptr{Cvoid}, c_client_create),
        #= PJRT_Client_Destroy =# @cfunction(pjrt_client_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Destroy_Args},)),
        #= PJRT_Client_PlatformName =# @cfunction(pjrt_client_platform_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformName_Args},)),
        #= PJRT_Client_ProcessIndex =# @cfunction(pjrt_client_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_ProcessIndex_Args},)),
        #= PJRT_Client_PlatformVersion =# @cfunction(pjrt_client_platform_version, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformVersion_Args},)),
        #= PJRT_Client_Devices =# @cfunction(pjrt_client_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Devices_Args},)),
        #= PJRT_Client_AddressableDevices =# @cfunction(pjrt_client_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableDevices_Args},)),
        #= PJRT_Client_LookupDevice =# @cfunction(pjrt_client_lookup_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_LookupDevice_Args},)),
        #= PJRT_Client_LookupAddressableDevice =# @cfunction(pjrt_client_lookup_addressable_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_LookupAddressableDevice_Args},)),
        #= PJRT_Client_AddressableMemories =# @cfunction(pjrt_client_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableMemories_Args},)),
        #= PJRT_Client_Compile =# @cfunction(pjrt_client_compile, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Compile_Args},)),
        #= PJRT_Client_DefaultDeviceAssignment =# @cfunction(pjrt_client_default_device_assignment, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_DefaultDeviceAssignment_Args},)),
        #= PJRT_Client_BufferFromHostBuffer =# @cfunction(pjrt_client_buffer_from_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args},)),
        #= PJRT_DeviceDescription_Id =# @cfunction(pjrt_device_description_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Id_Args},)),
        #= PJRT_DeviceDescription_ProcessIndex =# @cfunction(pjrt_device_description_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args},)),
        #= PJRT_DeviceDescription_Attributes =# @cfunction(pjrt_device_description_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args},)),
        #= PJRT_DeviceDescription_Kind =# @cfunction(pjrt_device_description_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Kind_Args},)),
        #= PJRT_DeviceDescription_DebugString =# @cfunction(pjrt_device_description_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args},)),
        #= PJRT_DeviceDescription_ToString =# @cfunction(pjrt_device_description_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ToString_Args},)),
        #= PJRT_Device_GetDescription =# @cfunction(pjrt_device_get_description, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_GetDescription_Args},)),
        #= PJRT_Device_IsAddressable =# @cfunction(pjrt_device_is_addressable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_IsAddressable_Args},)),
        #= PJRT_Device_LocalHardwareId =# @cfunction(pjrt_device_local_hardware_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_LocalHardwareId_Args},)),
        #= PJRT_Device_AddressableMemories =# @cfunction(pjrt_device_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_AddressableMemories_Args},)),
        #= PJRT_Device_DefaultMemory =# @cfunction(pjrt_device_default_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_DefaultMemory_Args},)),
        #= PJRT_Device_MemoryStats =# @cfunction(pjrt_device_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_MemoryStats_Args},)),
        #= PJRT_Memory_Id =# @cfunction(pjrt_memory_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Id_Args},)),
        #= PJRT_Memory_Kind =# @cfunction(pjrt_memory_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Args},)),
        #= PJRT_Memory_DebugString =# @cfunction(pjrt_memory_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_DebugString_Args},)),
        #= PJRT_Memory_ToString =# @cfunction(pjrt_memory_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_ToString_Args},)),
        #= PJRT_Memory_AddressableByDevices =# @cfunction(pjrt_memory_addressable_by_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args},)),
        #= PJRT_Executable_Destroy =# @cfunction(pjrt_executable_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Destroy_Args},)),
        #= PJRT_Executable_Name =# @cfunction(pjrt_executable_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Name_Args},)),
        #= PJRT_Executable_NumReplicas =# @cfunction(pjrt_executable_num_replicas, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumReplicas_Args},)),
        #= PJRT_Executable_NumPartitions =# @cfunction(pjrt_executable_num_partitions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumPartitions_Args},)),
        #= PJRT_Executable_NumOutputs =# @cfunction(pjrt_executable_num_outputs, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumOutputs_Args},)),
        #= PJRT_Executable_SizeOfGeneratedCodeInBytes =# @cfunction(pjrt_executable_size_of_generated_code_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args},)),
        #= PJRT_Executable_GetCostAnalysis =# @cfunction(pjrt_executable_get_cost_analysis, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCostAnalysis_Args},)),
        #= PJRT_Executable_OutputMemoryKinds =# @cfunction(pjrt_executable_output_memory_kinds, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputMemoryKinds_Args},)),
        #= PJRT_Executable_OptimizedProgram =# @cfunction(pjrt_executable_optimized_program, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OptimizedProgram_Args},)),
        #= PJRT_Executable_Serialize =# @cfunction(pjrt_executable_serialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Serialize_Args},)),
        #= PJRT_LoadedExecutable_Destroy =# @cfunction(pjrt_loaded_executable_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Destroy_Args},)),
        #= PJRT_LoadedExecutable_GetExecutable =# @cfunction(pjrt_loaded_executable_get_executable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_GetExecutable_Args},)),
        #= PJRT_LoadedExecutable_AddressableDevices =# @cfunction(pjrt_loaded_executable_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_AddressableDevices_Args},)),
        #= PJRT_LoadedExecutable_Delete =# @cfunction(pjrt_loaded_executable_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Delete_Args},)),
        #= PJRT_LoadedExecutable_IsDeleted =# @cfunction(pjrt_loaded_executable_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_IsDeleted_Args},)),
        #= PJRT_LoadedExecutable_Execute =# @cfunction(pjrt_loaded_executable_execute, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Execute_Args},)),
        #= PJRT_Executable_DeserializeAndLoad =# @cfunction(pjrt_executable_deserialize_and_load, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_DeserializeAndLoad_Args},)),
        #= PJRT_LoadedExecutable_Fingerprint =# c_unimplemented, # deprecated
        #= PJRT_Buffer_Destroy =# @cfunction(pjrt_buffer_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Destroy_Args},)),
        #= PJRT_Buffer_ElementType =# @cfunction(pjrt_buffer_element_type, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ElementType_Args},)),
        #= PJRT_Buffer_Dimensions =# @cfunction(pjrt_buffer_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Dimensions_Args},)),
        #= PJRT_Buffer_UnpaddedDimensions =# @cfunction(pjrt_buffer_unpadded_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_UnpaddedDimensions_Args},)),
        #= PJRT_Buffer_DynamicDimensionIndices =# @cfunction(pjrt_buffer_dynamic_dimension_indices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_DynamicDimensionIndices_Args},)),
        #= PJRT_Buffer_GetMemoryLayout =# @cfunction(pjrt_buffer_get_memory_layout, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_GetMemoryLayout_Args},)),
        #= PJRT_Buffer_OnDeviceSizeInBytes =# @cfunction(pjrt_buffer_on_device_size_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args},)),
        #= PJRT_Buffer_Device =# @cfunction(pjrt_buffer_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Device_Args},)),
        #= PJRT_Buffer_Memory =# @cfunction(pjrt_buffer_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Memory_Args},)),
        #= PJRT_Buffer_Delete =# @cfunction(pjrt_buffer_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Delete_Args},)),
        #= PJRT_Buffer_IsDeleted =# @cfunction(pjrt_buffer_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsDeleted_Args},)),
        #= PJRT_Buffer_CopyToDevice =# @cfunction(pjrt_buffer_copy_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args},)),
        #= PJRT_Buffer_ToHostBuffer =# @cfunction(pjrt_buffer_to_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args},)),
        #= PJRT_Buffer_IsOnCpu =# @cfunction(pjrt_buffer_is_on_cpu, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args},)),
        #= PJRT_Buffer_ReadyEvent =# @cfunction(pjrt_buffer_ready_event, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ReadyEvent_Args},)),
        #= PJRT_Buffer_UnsafePointer =# @cfunction(pjrt_buffer_unsafe_pointer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args},)),
        #= PJRT_Buffer_IncreaseExternalReferenceCount =# @cfunction(pjrt_buffer_increase_external_reference_count, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IncreaseExternalReferenceCount_Args},)),
        #= PJRT_Buffer_DecreaseExternalReferenceCount =# @cfunction(pjrt_buffer_decrease_external_reference_count, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_DecreaseExternalReferenceCount_Args},)),
        #= PJRT_Buffer_OpaqueDeviceMemoryDataPointer =# @cfunction(pjrt_buffer_opaque_device_memory_data_pointer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args},)),
        #= PJRT_CopyToDeviceStream_Destroy =# @cfunction(pjrt_copy_to_device_stream_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_Destroy_Args},)),
        #= PJRT_CopyToDeviceStream_AddChunk =# @cfunction(pjrt_copy_to_device_stream_add_chunk, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_AddChunk_Args},)),
        #= PJRT_CopyToDeviceStream_TotalBytes =# @cfunction(pjrt_copy_to_device_stream_total_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_TotalBytes_Args},)),
        #= PJRT_CopyToDeviceStream_GranuleSize =# @cfunction(pjrt_copy_to_device_stream_granule_size, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_GranuleSize_Args},)),
        #= PJRT_CopyToDeviceStream_CurrentBytes =# @cfunction(pjrt_copy_to_device_stream_current_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_CurrentBytes_Args},)),
        #= PJRT_TopologyDescription_Create =# unsafe_convert(Ptr{Cvoid}, c_topology_create),
        #= PJRT_TopologyDescription_Destroy =# @cfunction(pjrt_topology_description_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Destroy_Args},)),
        #= PJRT_TopologyDescription_PlatformName =# @cfunction(pjrt_topology_description_platform_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_PlatformName_Args},)),
        #= PJRT_TopologyDescription_PlatformVersion =# @cfunction(pjrt_topology_description_platform_version, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_PlatformVersion_Args},)),
        #= PJRT_TopologyDescription_GetDeviceDescriptions =# @cfunction(pjrt_topology_description_get_device_descriptions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_GetDeviceDescriptions_Args},)),
        #= PJRT_TopologyDescription_Serialize =# @cfunction(pjrt_topology_description_serialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Serialize_Args},)),
        #= PJRT_TopologyDescription_Attributes =# @cfunction(pjrt_topology_description_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Attributes_Args},)),
        #= PJRT_Compile =# @cfunction(pjrt_compile, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Compile_Args},)),
        #= PJRT_Executable_OutputElementTypes =# @cfunction(pjrt_executable_output_element_types, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args},)),
        #= PJRT_Executable_OutputDimensions =# @cfunction(pjrt_executable_output_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputDimensions_Args},)),
        #= PJRT_Buffer_CopyToMemory =# @cfunction(pjrt_buffer_copy_to_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToMemory_Args},)),
        #= PJRT_Client_CreateViewOfDeviceBuffer =# c_unimplemented,
        #= PJRT_Executable_Fingerprint =# @cfunction(pjrt_executable_fingerprint, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Fingerprint_Args},)),
        #= PJRT_Client_TopologyDescription =# @cfunction(pjrt_client_topology_description, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_TopologyDescription_Args},)),
        #= PJRT_Executable_GetCompiledMemoryStats =# @cfunction(pjrt_executable_get_compiled_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCompiledMemoryStats_Args},)),
        #= PJRT_Memory_Kind_Id =# @cfunction(pjrt_memory_kind_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Id_Args},)),
        #= PJRT_ExecuteContext_Create =# c_unimplemented,
        #= PJRT_ExecuteContext_Destroy =# c_unimplemented,
        #= PJRT_Buffer_CopyRawToHost =# @cfunction(pjrt_buffer_copy_raw_to_host, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyRawToHost_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_Destroy =# @cfunction(pjrt_async_host_to_device_transfer_manager_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_TransferData =# @cfunction(pjrt_async_host_to_device_transfer_manager_transfer_data, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args},)),
        #= PJRT_Client_CreateBuffersForAsyncHostToDevice =# @cfunction(pjrt_client_create_buffers_for_async_host_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer =# @cfunction(pjrt_async_host_to_device_transfer_manager_retrieve_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_Device =# @cfunction(pjrt_async_host_to_device_transfer_manager_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Device_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_BufferCount =# @cfunction(pjrt_async_host_to_device_transfer_manager_buffer_count, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_BufferSize =# @cfunction(pjrt_async_host_to_device_transfer_manager_buffer_size, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_SetBufferError =# @cfunction(pjrt_async_host_to_device_transfer_manager_set_buffer_error, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_AddMetadata =# @cfunction(pjrt_async_host_to_device_transfer_manager_add_metadata, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args},)),
        #= PJRT_Client_DmaMap =# @cfunction(pjrt_client_dma_map, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_DmaMap_Args},)),
        #= PJRT_Client_DmaUnmap =# @cfunction(pjrt_client_dma_unmap, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_DmaUnmap_Args},)),
        #= PJRT_Client_CreateUninitializedBuffer =# @cfunction(pjrt_client_create_uninitialized_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateUninitializedBuffer_Args},)),
        #= PJRT_Client_UpdateGlobalProcessInfo =# @cfunction(pjrt_client_update_global_process_info, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_UpdateGlobalProcessInfo_Args},)),
        #= PJRT_TopologyDescription_Deserialize =# unsafe_convert(Ptr{Cvoid}, c_topology_deserialize),
        #= PJRT_Client_CreateAliasBuffer =# @cfunction(pjrt_client_create_alias_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateAliasBuffer_Args},)),
        #= PJRT_Client_FulfillAliasBuffer =# @cfunction(pjrt_client_fulfill_alias_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_FulfillAliasBuffer_Args},)),
        #= PJRT_LoadedExecutable_GetDeviceAssignment =# @cfunction(pjrt_loaded_executable_get_device_assignment, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args},)),
        #= PJRT_Client_CreateErrorBuffer =# @cfunction(pjrt_client_create_error_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateErrorBuffer_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_TransferLiteral =# @cfunction(pjrt_async_host_to_device_transfer_manager_transfer_literal, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args},)),
        #= PJRT_Buffer_CopyRawToHostFuture =# @cfunction(pjrt_buffer_copy_raw_to_host_future, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Args},)),
        #= PJRT_Device_PoisonExecution =# @cfunction(pjrt_device_poison_execution, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_PoisonExecution_Args},)),
        #= PJRT_Device_CreateAsyncTrackingEvent =# @cfunction(pjrt_device_create_async_tracking_event, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_CreateAsyncTrackingEvent_Args},)),
        #= PJRT_AsyncTrackingEvent_Destroy =# @cfunction(pjrt_async_tracking_event_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncTrackingEvent_Destroy_Args},)),
        #= PJRT_Executable_GetCompileOptions =# @cfunction(pjrt_executable_get_compile_options, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCompileOptions_Args},)),
        #= PJRT_Buffer_DonateWithControlDependency =# @cfunction(pjrt_buffer_donate_with_control_dependency, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Args},)),
        #= PJRT_Event_Create =# @cfunction(pjrt_event_create, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Create_Args},)),
        #= PJRT_Event_Set =# @cfunction(pjrt_event_set, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Set_Args},)),
        #= PJRT_Device_GetAttributes =# C_NULL,
        #= PJRT_Client_Load =# @cfunction(pjrt_client_load, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Load_Args},)),
        #= PJRT_LoadedExecutable_AddressableDeviceLogicalIds =# @cfunction(pjrt_loaded_executable_addressable_device_logical_ids, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args},)),
        #= PJRT_Buffer_Bitcast =# @cfunction(pjrt_buffer_bitcast, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Bitcast_Args},)),
        #= PJRT_Error_ForEachPayload =# @cfunction(pjrt_error_for_each_payload, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_ForEachPayload_Args},)),
        #= PJRT_TopologyDescription_Fingerprint =# @cfunction(pjrt_topology_description_fingerprint, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Fingerprint_Args},)),
        #= PJRT_Executable_ParameterMemoryKinds =# @cfunction(pjrt_executable_parameter_memory_kinds, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_ParameterMemoryKinds_Args},)),
        #= PJRT_Device_ClearMemoryStats =# @cfunction(pjrt_device_clear_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_ClearMemoryStats_Args},)),
        #= PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace =# @cfunction(pjrt_topology_description_make_canonical_shape_for_memory_space, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args},)),
        #= PJRT_TopologyDescription_GetMemorySpaceKindIds =# @cfunction(pjrt_topology_description_get_memory_space_kind_ids, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_GetMemorySpaceKindIds_Args},)),
    ))
    #! format: on

    return PluginApi(
        api,
        c_plugin_initialize,
        c_plugin_attributes,
        c_client_create,
        c_topology_create,
        c_topology_deserialize,
        exts,
    )
end

end
