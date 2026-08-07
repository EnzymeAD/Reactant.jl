# Event.jl — PJRT event callbacks

function _event_destroy(args::Ptr{CAPI.PJRT_Event_Destroy_Args})::Ptr{Cvoid}
    return C_NULL
end

function _event_is_ready(args::Ptr{CAPI.PJRT_Event_IsReady_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, true, Val{:is_ready}())
    return C_NULL
end

function _event_await(args::Ptr{CAPI.PJRT_Event_Await_Args})::Ptr{Cvoid}
    return C_NULL  # immediate: events are always ready
end

# PJRT_Event_OnReadyCallback = void (*)(PJRT_Error* error, void* user_arg)
# Our events are always ready (no error), so call the callback immediately.
function _event_on_ready(args::Ptr{CAPI.PJRT_Event_OnReady_Args})::Ptr{Cvoid}
    callback = Reactant.unsafe_load_field(args, Val{:callback}())
    user_arg = Reactant.unsafe_load_field(args, Val{:user_arg}())
    if callback != C_NULL
        ccall(callback, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, user_arg)
    end
    return C_NULL
end
