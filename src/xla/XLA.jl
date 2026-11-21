module XLA

using ..Reactant: Reactant, MLIR
using Reactant_jll: Reactant_jll
using LLVM: LLVM
using Libdl: Libdl
using EnumX: @enumx
using Enzyme: Compiler
using Preferences: load_preference
using UUIDs: UUID
using ScopedValues: ScopedValue, with

using Setfield: Setfield, @set!

const XLA_REACTANT_GPU_MEM_FRACTION = Ref{Float64}(0.75)
const XLA_REACTANT_GPU_PREALLOCATE = Ref{Bool}(true)
const REACTANT_XLA_RUNTIME = load_preference(
    UUID("3c362404-f566-11ee-1572-e11a4b42c853"), "xla_runtime", "PJRT"
)

const CUDA_DATA_DIR = Ref(
    isdefined(Reactant_jll, :ptxas_path) ? dirname(dirname(Reactant_jll.ptxas_path)) : ""
)

# The ROCm root `enzymexla-gpu-module-to-binary` assembles and links kernels with
# (`ld.lld` and `amdgcn/bitcode` under it). Refreshed in `__init__` so the
# precompile-time `ENV["ROCM_PATH"]` does not get baked into the pkgimage; empty
# leaves it to the pass, which reads `ROCM_PATH`, `ROCM_ROOT` and `ROCM_HOME`.
const ROCM_PATH = Ref("")

function LLVMclopts(opts...)
    args = ["", opts...]
    return MLIR.API.ReactantLLVMParseCommandLineOptions(length(args), args, C_NULL)
end

include("CAPI/CAPI.jl")
include("Distributed.jl")
include("Client.jl")
include("Device.jl")
include("Sharding.jl")
include("LoadedExecutable.jl")
include("Future.jl")
include("Buffer.jl")
include("Stats.jl")
include("Utils.jl")
include("Memory.jl")

include("IR/Module.jl")
include("IR/Instruction.jl")
include("IR/Computation.jl")
include("IR/PerformanceModel.jl")

include("PJRT/PJRT.jl")

include("IFRT/IFRT.jl")

include("CompileOptions.jl")

const BACKENDS_TO_INITIALIZE = ScopedValue{Union{Missing,Set{String}}}(missing)

abstract type AbstractBackendState end

function finalize_backend_state end

for runtime in (:PJRT, :IFRT)
    backend_state = Symbol(runtime, :BackendState)

    @eval mutable struct $(backend_state) <: AbstractBackendState
        initialized::Bool
        clients::Dict{String,$(runtime).Client}
        default_client::$(runtime).Client
        attempted_backends::Set{String}

        function $(backend_state)(
            initialized::Bool=false,
            clients::Dict{String,$(runtime).Client}=Dict{String,$(runtime).Client}(),
            default_client::$(runtime).Client=$(runtime).NullClient,
            attempted_backends::Set{String}=Set{String}(),
        )
            return finalizer(
                finalize_backend_state,
                new(initialized, clients, default_client, attempted_backends),
            )
        end
    end

    @eval function finalize_backend_state(state::$(backend_state))
        state.initialized || return nothing

        @debug "[GETPID $(getpid())] Finalizing backend state, $state"
        for (_, client) in state.clients
            free_client(client)
        end
        empty!(state.clients)
        empty!(state.attempted_backends)
        state.default_client = $(runtime).NullClient
        return nothing
    end
end

function Base.getproperty(bs::AbstractBackendState, sym::Symbol)
    (sym === :initialized || bs.initialized) && return getfield(bs, sym)
    initialize_default_clients!(bs)
    return getfield(bs, sym)
end

function Base.setproperty!(bs::AbstractBackendState, sym::Symbol, val)
    (sym === :initialized || bs.initialized) && return setfield!(bs, sym, val)
    initialize_default_clients!(bs)
    return setfield!(bs, sym, val)
end

const global_pjrt_backend_state = PJRTBackendState()
const global_ifrt_backend_state = IFRTBackendState()
const PREFERRED_DEFAULT_BACKEND = Ref{Union{Nothing,String}}(nothing)

const global_backend_state = if REACTANT_XLA_RUNTIME == "PJRT"
    global_pjrt_backend_state
elseif REACTANT_XLA_RUNTIME == "IFRT"
    global_ifrt_backend_state
else
    error("Unsupported REACTANT_XLA_RUNTIME (set via `xla_runtime` preference): \
           $(REACTANT_XLA_RUNTIME)")
end
const global_state = State()

backend_state() = global_backend_state
backend_state(::Val{:PJRT}) = global_pjrt_backend_state
backend_state(::Val{:IFRT}) = global_ifrt_backend_state

struct _InitDefaultClientsThunk{BS<:AbstractBackendState}
    bs::BS
end
(f::_InitDefaultClientsThunk)() = initialize_default_clients!(f.bs)

const is_live = Threads.Atomic{Bool}(true)

function cleanup_backend_state()
    @debug "[GETPID $(getpid())] Cleanup Backend State, $global_pjrt_backend_state, $global_ifrt_backend_state, $global_state"
    finalize_backend_state(global_pjrt_backend_state)
    finalize_backend_state(global_ifrt_backend_state)
    shutdown(global_state)
    is_live[] = false
    return nothing
end

const GPU_BACKEND_NAMES = ("cuda", "metal", "rocm")

function normalize_backend_name(backend::String)
    backend == "gpu" && return Set(GPU_BACKEND_NAMES)
    return Set([backend])
end

function registered_backend_names()
    return Reactant.Accelerators.Registration.registered_platform_names()
end

"""
    validate_backend_name(backend::String)

Throw an `ArgumentError` naming the requested backend and listing the backends that are
valid in this session if `backend` is neither `"gpu"` nor a registered backend. Backend
names are case-sensitive (e.g. `"cpu"`, not `"CPU"`).
"""
function validate_backend_name(backend::String)
    backend == "gpu" && return nothing
    Reactant.Accelerators.Registration.is_registered_backend(backend) && return nothing

    registered = registered_backend_names()
    valid = join(["\"$(b)\"" for b in [registered..., "gpu"]], ", ")
    msg = "Requested backend \"$(backend)\" is not valid. Valid backends in this session \
           are: $(valid) (\"gpu\" selects the first available of \
           $(join(["\"$(b)\"" for b in GPU_BACKEND_NAMES], ", "))). Backend names are \
           case-sensitive."

    # Case-only mismatch (e.g. "CPU" instead of "cpu") is by far the most common mistake.
    suggestion_idx = findfirst(
        b -> lowercase(b) == lowercase(backend), [registered..., "gpu"]
    )
    if suggestion_idx !== nothing
        suggestion = [registered..., "gpu"][suggestion_idx]
        msg *= " Did you mean \"$(suggestion)\"?"
    elseif lowercase(backend) in GPU_BACKEND_NAMES
        msg *= " The \"$(lowercase(backend))\" backend is not registered, which usually \
                means that `Reactant_jll` selected an artifact without \
                \"$(lowercase(backend))\" support. See \
                https://enzymead.github.io/Reactant.jl/stable/introduction/configuration \
                for how to force a specific artifact and debug the installation."
    end
    throw(ArgumentError(msg))
end

function client(backend::String, bs::AbstractBackendState=global_backend_state)
    validate_backend_name(backend)

    if backend == "gpu"
        for b in GPU_BACKEND_NAMES
            if haskey(bs.clients, b)
                return bs.clients[b]
            end
        end

        # If none found, check if we've attempted all of them
        gpu_backends = normalize_backend_name("gpu")
        missing_gpu_backend = false
        for b in gpu_backends
            if b ∉ bs.attempted_backends
                missing_gpu_backend = true
                break
            end
        end
        if missing_gpu_backend
            with(_InitDefaultClientsThunk(bs), BACKENDS_TO_INITIALIZE => gpu_backends)
            union!(bs.attempted_backends, gpu_backends)
            return client("gpu", bs) # Try again
        end

        registered_gpu_backends = filter(in(registered_backend_names()), GPU_BACKEND_NAMES)
        if isempty(registered_gpu_backends)
            error(
                "No GPU client found. No GPU backend ($(join(GPU_BACKEND_NAMES, ", "))) \
                 is registered in this session, which usually means that `Reactant_jll` \
                 selected an artifact without GPU support. See \
                 https://enzymead.github.io/Reactant.jl/stable/introduction/configuration \
                 for how to force a specific artifact and debug the installation."
            )
        else
            error("No GPU client found. Initialization of the registered GPU backend(s) \
                   ($(join(registered_gpu_backends, ", "))) failed, see the error(s) \
                   logged above.")
        end
    end

    if (!haskey(bs.clients, backend) && backend ∉ bs.attempted_backends)
        backends = normalize_backend_name(backend)
        with(_InitDefaultClientsThunk(bs), BACKENDS_TO_INITIALIZE => backends)
        union!(bs.attempted_backends, backends)
    end

    haskey(bs.clients, backend) ||
        error("Initialization of the \"$(backend)\" backend failed, see the error(s) \
               logged above.")
    return bs.clients[backend]
end
function client(backend::String, rt::Union{Val{:PJRT},Val{:IFRT}})
    return client(backend, backend_state(rt))
end

function default_backend(bs::AbstractBackendState=global_backend_state)
    preferred = PREFERRED_DEFAULT_BACKEND[]
    if !getfield(bs, :initialized) && preferred !== nothing
        bs.default_client = client(preferred, bs)
        return bs.default_client
    end

    default_client = bs.default_client
    Reactant.Accelerators.Registration.is_null_client(default_client) ||
        return default_client

    # A previous (filtered) initialization, e.g. `set_default_backend("gpu")` on a machine
    # without a GPU, didn't yield any functional client. Try the backends that we haven't
    # attempted yet before giving up.
    remaining = Set(filter(∉(bs.attempted_backends), registered_backend_names()))
    if !isempty(remaining)
        with(_InitDefaultClientsThunk(bs), BACKENDS_TO_INITIALIZE => remaining)
        default_client = bs.default_client
    end

    Reactant.Accelerators.Registration.is_null_client(default_client) &&
        error("No functional XLA client could be initialized (registered backends: \
               $(join(registered_backend_names(), ", "))). See the error(s) logged above.")
    return default_client
end
default_backend(rt::Union{Val{:PJRT},Val{:IFRT}}) = default_backend(backend_state(rt))
default_device() = default_device(default_backend())
default_device(rt::Union{Val{:PJRT},Val{:IFRT}}) = default_device(default_backend(rt))
process_index() = process_index(default_backend())

runtime(::Nothing) = runtime()
runtime() = runtime(global_backend_state)
runtime(::PJRTBackendState) = Val(:PJRT)
runtime(::IFRTBackendState) = Val(:IFRT)
runtime(::PJRT.Client) = Val(:PJRT)
runtime(::IFRT.Client) = Val(:IFRT)

function set_default_backend(backend::PJRT.Client)
    global_pjrt_backend_state.default_client = backend
    global_backend_state === global_pjrt_backend_state &&
        (PREFERRED_DEFAULT_BACKEND[] = nothing)
    return nothing
end

function set_default_backend(backend::IFRT.Client)
    global_ifrt_backend_state.default_client = backend
    global_backend_state === global_ifrt_backend_state &&
        (PREFERRED_DEFAULT_BACKEND[] = nothing)
    return nothing
end

struct _SetDefaultBackendThunk{BS<:AbstractBackendState}
    bs::BS
    backend::String
end
function (f::_SetDefaultBackendThunk)()
    f.bs.default_client = client(f.backend, f.bs)
    return nothing
end

function set_default_backend(backend::String)
    validate_backend_name(backend)
    PREFERRED_DEFAULT_BACKEND[] = backend
    backends = normalize_backend_name(backend)
    with(
        _SetDefaultBackendThunk(global_backend_state, backend),
        BACKENDS_TO_INITIALIZE => backends,
    )
    other_bs = if global_backend_state === global_pjrt_backend_state
        global_ifrt_backend_state
    else
        global_pjrt_backend_state
    end
    if getfield(other_bs, :initialized)
        with(_SetDefaultBackendThunk(other_bs, backend), BACKENDS_TO_INITIALIZE => backends)
    end
    return nothing
end

function update_global_state!(args...; kwargs...)
    update!(global_state, args...; kwargs...)
    # We conditionally initialize for now, since a lot of options that are set are not
    # necessarily supported by PJRT.
    if getfield(global_pjrt_backend_state, :initialized)
        # We need to update the clients based on the new state
        initialize_default_clients!(global_pjrt_backend_state)
    end
    if getfield(global_ifrt_backend_state, :initialized)
        initialize_default_clients!(global_ifrt_backend_state)
    end
    return nothing
end

"""
    has_initialized_client(backend)

Return whether a client for `backend` already exists without triggering lazy
backend initialization.
"""
function has_initialized_client(backend::String)
    clients = getfield(global_backend_state, :clients)
    return any(haskey(clients, name) for name in normalize_backend_name(backend))
end

function __init__()
    ROCM_PATH[] = Reactant.precompiling() ? "" : get(ENV, "ROCM_PATH", "")
    if Reactant_jll.is_available()
        @debug "Using libReactantExtra from $(Reactant_jll.libReactantExtra_path)"

        # This must be the very first thing initialized (otherwise we can't throw errors)
        errptr = cglobal(
            (:ReactantThrowError, Reactant_jll.libReactantExtra), Ptr{Ptr{Cvoid}}
        )
        unsafe_store!(errptr, @cfunction(reactant_err, Cvoid, (Cstring,)))

        # Also sets the log level, from `TF_CPP_MIN_LOG_LEVEL` or to WARNING;
        # see `XLA.SetLogLevel` to change it later.
        initLogs = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "InitializeLogs")
        ccall(initLogs, Cvoid, ())

        if haskey(ENV, "XLA_REACTANT_GPU_MEM_FRACTION")
            XLA_REACTANT_GPU_MEM_FRACTION[] = parse(
                Float64, ENV["XLA_REACTANT_GPU_MEM_FRACTION"]
            )
            @debug "XLA_REACTANT_GPU_MEM_FRACTION: " XLA_REACTANT_GPU_MEM_FRACTION[] maxlog =
                1
            if XLA_REACTANT_GPU_MEM_FRACTION[] > 1 || XLA_REACTANT_GPU_MEM_FRACTION[] < 0
                error("XLA_REACTANT_GPU_MEM_FRACTION must be between 0 and 1")
            end
        end

        if haskey(ENV, "XLA_REACTANT_GPU_PREALLOCATE")
            XLA_REACTANT_GPU_PREALLOCATE[] = parse(
                Bool, ENV["XLA_REACTANT_GPU_PREALLOCATE"]
            )
            @debug "XLA_REACTANT_GPU_PREALLOCATE: " XLA_REACTANT_GPU_PREALLOCATE[] maxlog =
                1
        end

        if haskey(ENV, "REACTANT_VISIBLE_GPU_DEVICES")
            global_state.local_gpu_device_ids =
                parse.(Int, split(ENV["REACTANT_VISIBLE_GPU_DEVICES"], ","))
            @debug "REACTANT_VISIBLE_GPU_DEVICES: " global_state.local_gpu_device_ids maxlog =
                1
        end

        @debug "REACTANT_XLA_RUNTIME: " REACTANT_XLA_RUNTIME maxlog = 1

        MLIR.API.RegisterEnzymeXLACPUHandler()
        MLIR.API.RegisterEnzymeXLAGPUHandler()
        MLIR.API.registerEnzymeJaXXLAFFI()
        MLIR.API.registerReactantXLAFFI()

        @static if !Sys.isapple()
            lljit = LLVM.JuliaOJIT()
            jd_main = LLVM.JITDylib(lljit)

            for name in
                ("XLAExecute", "XLAExecuteSharded", "ifrt_loaded_executable_execute")
                ptr = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, name)
                LLVM.define(
                    jd_main,
                    Compiler.JIT.absolute_symbol_materialization(
                        LLVM.mangle(lljit, name), ptr
                    ),
                )
            end
        end

        # Force nothing to ensure these dont get baked in at precompile
        DEFAULT_XLA_DEBUG_OPTIONS[] = nothing
        DEFAULT_XLA_COMPILE_OPTIONS[] = nothing
    end

    # Register the cleanup function
    atexit(cleanup_backend_state)

    return nothing
end

for runtime in (:PJRT, :IFRT)
    @eval function initialize_default_clients!(state::$(Symbol(runtime, :BackendState)))
        backends_to_initialize = BACKENDS_TO_INITIALIZE[]

        was_initialized = state.initialized
        state.initialized = true
        distributed_runtime_client = if global_state.num_processes > 1
            @assert global_state.client !== nothing
            global_state.client
        else
            nothing
        end

        Reactant.Accelerators.Registration.initialize_backends(
            Val($(Meta.quot(runtime))),
            state,
            was_initialized;
            allow_initialization=backend -> begin
                Reactant.precompiling() && return backend.platform_name == "cpu"
                will_try_initializing = false
                if backends_to_initialize === missing
                    will_try_initializing = true
                else
                    will_try_initializing = backend.platform_name in backends_to_initialize
                end
                will_try_initializing &&
                    push!(state.attempted_backends, backend.platform_name)
                return will_try_initializing
            end,
            node_id=global_state.process_id,
            num_nodes=global_state.num_processes,
            distributed_runtime_client,
            allowed_devices=global_state.local_gpu_device_ids,
        )

        return nothing
    end
end

end
