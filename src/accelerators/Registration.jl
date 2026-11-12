module Registration

using ..Reactant: Reactant

const BackendRegistrationLock = ReentrantLock()

struct RegisteredBackend
    platform_name::String
    priority::Int64 # Higher Priority is used to determine the default device
    pjrt_initialize_function
    ifrt_initialize_function
    pjrt_client_count::Base.RefValue{Int64}
    ifrt_client_count::Base.RefValue{Int64}
    preinitialize_setup_function # this is run before any of the init functions are called
end

const RegisteredBackends = Vector{RegisteredBackend}()

function __init__()
    empty!(RegisteredBackends)
    return nothing
end

function register_backend(
    platform_name::String;
    priority::Int,
    pjrt_initialize_function=nothing,
    ifrt_initialize_function=nothing,
    preinitialize_setup_function=Returns(nothing),
)
    @assert pjrt_initialize_function !== nothing || ifrt_initialize_function !== nothing "atleast one of pjrt_initialize_function or ifrt_initialize_function must be provided."

    for backend in RegisteredBackends
        @assert backend.platform_name != platform_name "Backend with platform_name: \
                                                        $(platform_name) was already \
                                                        registered."
    end

    @debug "Registering Backend $(platform_name) with Priority $(priority)."
    @lock BackendRegistrationLock begin
        backend = RegisteredBackend(
            platform_name,
            priority,
            pjrt_initialize_function,
            ifrt_initialize_function,
            Ref(Int64(0)),
            Ref(Int64(0)),
            preinitialize_setup_function,
        )
        push!(RegisteredBackends, backend)
    end

    return nothing
end

for (runtime, countsym, initfn) in (
    (:IFRT, :ifrt_client_count, :ifrt_initialize_function),
    (:PJRT, :pjrt_client_count, :pjrt_initialize_function),
)
    @eval function make_client(
        ::Val{$(Meta.quot(runtime))},
        backend::RegisteredBackend,
        args...;
        checkcount::Bool=true,
        kwargs...,
    )
        if checkcount
            @assert backend.$(countsym)[] == 0
        end
        @assert backend.$(initfn) !== nothing
        client = backend.$(initfn)(args...; kwargs...)
        if checkcount
            backend.$(countsym)[] += 1
        end
        return client
    end
end

registered_platform_names() = [backend.platform_name for backend in RegisteredBackends]

function is_registered_backend(platform_name::String)
    return any(backend.platform_name == platform_name for backend in RegisteredBackends)
end

is_null_client(client) = client.client == C_NULL

"""
    initialize_backends(runtime, state, was_initialized, args...; allow_initialization, kwargs...)

Initialize (or re-initialize) the clients of all registered backends for which
`allow_initialization(backend)` returns `true`, storing them in `state.clients`.

The default client (`state.default_client`) is only (re)computed when it needs to be,
i.e. when there is no default client yet or when the client it refers to was freed and
re-created during this call. In particular, lazily initializing an additional backend
(e.g. via `Reactant.XLA.client("cpu")`) never silently replaces the current default.

This function never throws when a backend fails to initialize (the failure is logged with
`@error`), callers are expected to report a precise error to the user if the client they
asked for is missing afterwards.
"""
function initialize_backends(
    runtime::Val,
    state,
    was_initialized::Bool,
    args...;
    allow_initialization=Returns(true),
    kwargs...,
)
    # Whether the default client has to be (re)computed at the end of this call.
    default_needs_update = !was_initialized || is_null_client(state.default_client)

    for backend in RegisteredBackends
        allow_initialization(backend) || continue

        @debug "Initializing client: $(backend.platform_name)"

        try
            if was_initialized && haskey(state.clients, backend.platform_name)
                old_client = state.clients[backend.platform_name]
                # The default client is about to be freed, so it must be recomputed.
                if old_client == state.default_client
                    default_needs_update = true
                end
                # Remove the entry before freeing so that a failed re-initialization
                # below can't leave a dangling (already freed) client in the state.
                delete!(state.clients, backend.platform_name)
                Reactant.XLA.free_client(old_client)
                if runtime isa Val{:IFRT}
                    backend.ifrt_client_count[] -= 1
                elseif runtime isa Val{:PJRT}
                    backend.pjrt_client_count[] -= 1
                else
                    error("Unknown runtime: $(runtime)")
                end
            end

            backend.preinitialize_setup_function()
            client = make_client(runtime, backend, args...; kwargs...)
            state.clients[backend.platform_name] = client
        catch err
            @error "Failed to initialize client: $(backend.platform_name)" exception = (
                err, catch_backtrace()
            )
        end
    end

    default_needs_update || return nothing

    # Set the default to the live client with the highest priority
    best_priority = typemin(Int64)
    best_client = ""
    for backend in RegisteredBackends
        haskey(state.clients, backend.platform_name) || continue
        if best_priority < backend.priority
            best_priority = backend.priority
            best_client = backend.platform_name
        end
    end

    if isempty(best_client)
        @debug "No functional client was found. Not setting a default client."
        return nothing
    end

    @debug "Reactant will use platform_name: $(best_client) by default."
    state.default_client = state.clients[best_client]
    return nothing
end

end
