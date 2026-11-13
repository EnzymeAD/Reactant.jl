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

"""
    AsyncTrackingEvent(description::String)

A scoped event that tells a device that there is asynchronous work depending on its activity, from its creation by
[`async_tracking_event`](@ref) until C destroys it.

`description` is the one given by C (e.g. for tracing), and `done` is an [`Event`](@ref) that is set once the work is
done, i.e. when `PJRT_AsyncTrackingEvent_Destroy` is called. A plugin that supports tracking events can keep them, and
`wait` for (or check with `isready`) their `done` event.

Backs `PJRT_AsyncTrackingEvent` opaque type.
"""
mutable struct AsyncTrackingEvent
    const description::String
    const done::Event
end

AsyncTrackingEvent(description::String) = AsyncTrackingEvent(description, Event())

@pjrt_catch_return function pjrt_async_tracking_event_destroy(
    args::Ptr{CAPI.PJRT_AsyncTrackingEvent_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.event)
    ptr == C_NULL && return ok()
    event = unsafe_pointer_to_objref(ptr)::AsyncTrackingEvent
    notify(event.done)
    unroot!(event)
    return ok()
end
