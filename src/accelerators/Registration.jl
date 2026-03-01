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
    priority::Int64,
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
            Ref(0),
            Ref(0),
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

function initialize_backends(
    runtime::Val,
    state,
    was_initialized::Bool,
    args...;
    allow_initialization=Returns(true),
    kwargs...,
)
    successful_initializations = falses(length(RegisteredBackends))

    for (i, backend) in enumerate(RegisteredBackends)
        allow_initialization(backend) || continue

        @debug "Initializing client: $(backend.platform_name)"

        try
            if was_initialized && haskey(state.clients, backend.platform_name)
                Reactant.XLA.free_client(state.clients[backend.platform_name])
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
            successful_initializations[i] = true
        catch err
            @error "Failed to initialize client: $(backend.platform_name)" exception = (
                err, catch_backtrace()
            )
        end
    end

    # Set the default to the client with the highest priority
    best_priority = -1
    best_client = ""
    for (i, backend) in enumerate(RegisteredBackends)
        if successful_initializations[i] && best_priority < backend.priority
            best_priority = backend.priority
            best_client = backend.platform_name
        end
    end

    @assert !isempty(best_client) "No functional client was found."
    @debug "Reactant will use platform_name: $(best_client) by default."
    state.default_client = state.clients[best_client]
    return nothing
end

end
