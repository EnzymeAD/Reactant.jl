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

function LLVMclopts(opts...)
    args = ["", opts...]
    return MLIR.API.ReactantLLVMParseCommandLineOptions(length(args), args, C_NULL)
end

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

const global_backend_state = if REACTANT_XLA_RUNTIME == "PJRT"
    PJRTBackendState()
elseif REACTANT_XLA_RUNTIME == "IFRT"
    IFRTBackendState()
else
    error("Unsupported REACTANT_XLA_RUNTIME (set via `xla_runtime` preference): \
           $(REACTANT_XLA_RUNTIME)")
end
const global_state = State()

const is_live = Threads.Atomic{Bool}(true)

function cleanup_backend_state()
    @debug "[GETPID $(getpid())] Cleanup Backend State, $global_backend_state, $global_state"
    finalize_backend_state(global_backend_state)
    shutdown(global_state)
    is_live[] = false
    return nothing
end

function normalize_backend_name(backend::String)
    backend == "gpu" && return Set(["cuda", "metal", "rocm"])
    return Set([backend])
end

function client(backend::String)
    if backend == "gpu"
        for b in ["cuda", "metal", "rocm"]
            if haskey(global_backend_state.clients, b)
                return global_backend_state.clients[b]
            end
        end

        # If none found, check if we've attempted all of them
        gpu_backends = normalize_backend_name("gpu")
        if any(b ∉ global_backend_state.attempted_backends for b in gpu_backends)
            with(BACKENDS_TO_INITIALIZE => gpu_backends) do
                initialize_default_clients!(global_backend_state)
            end
            union!(global_backend_state.attempted_backends, gpu_backends)
            return client("gpu") # Try again
        end

        error("No GPU client found")
    end

    if (
        !haskey(global_backend_state.clients, backend) &&
        backend ∉ global_backend_state.attempted_backends
    )
        backends = normalize_backend_name(backend)
        with(BACKENDS_TO_INITIALIZE => backends) do
            initialize_default_clients!(global_backend_state)
        end
        union!(global_backend_state.attempted_backends, backends)
    end
    return global_backend_state.clients[backend]
end
default_backend() = global_backend_state.default_client
default_device() = default_device(default_backend())
process_index() = process_index(default_backend())

runtime(::Nothing) = runtime()
runtime() = runtime(global_backend_state)
runtime(::PJRTBackendState) = Val(:PJRT)
runtime(::IFRTBackendState) = Val(:IFRT)
runtime(::PJRT.Client) = Val(:PJRT)
runtime(::IFRT.Client) = Val(:IFRT)

function set_default_backend(backend::AbstractClient)
    global_backend_state.default_client = backend
    return nothing
end

function set_default_backend(backend::String)
    with(BACKENDS_TO_INITIALIZE => normalize_backend_name(backend)) do
        global_backend_state.default_client = client(backend)
    end
    return nothing
end

function update_global_state!(args...; kwargs...)
    update!(global_state, args...; kwargs...)
    # We conditionally initialize for now, since a lot of options that are set are not
    # necessarily supported by PJRT.
    if global_backend_state.initialized
        # We need to update the clients based on the new state
        initialize_default_clients!(global_backend_state)
    end
    return nothing
end

function __init__()
    if Reactant_jll.is_available()
        @debug "Using libReactantExtra from $(Reactant_jll.libReactantExtra_path)"

        # This must be the very first thing initialized (otherwise we can't throw errors)
        errptr = cglobal((:ReactantThrowError, MLIR.API.mlir_c), Ptr{Ptr{Cvoid}})
        unsafe_store!(errptr, @cfunction(reactant_err, Cvoid, (Cstring,)))

        initLogs = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "InitializeLogs")
        ccall(initLogs, Cvoid, ())
        # Add most log level
        # SetLogLevel(0)

        if haskey(ENV, "XLA_REACTANT_GPU_MEM_FRACTION")
            XLA_REACTANT_GPU_MEM_FRACTION[] = parse(
                Float64, ENV["XLA_REACTANT_GPU_MEM_FRACTION"]
            )
            @debug "XLA_REACTANT_GPU_MEM_FRACTION: " XLA_REACTANT_GPU_MEM_FRACTION[] maxlog =
                1
            if XLA_REACTANT_GPU_MEM_FRACTION[] < 0
                error("XLA_REACTANT_GPU_MEM_FRACTION must be not be negative")
            elseif XLA_REACTANT_GPU_MEM_FRACTION[] > 1
                if get(ENV, "TF_FORCE_UNIFIED_MEMORY", "0") != "1"
                    error("XLA_REACTANT_GPU_MEM_FRACTION must be not greater than 1 without \
                        TF_FORCE_UNIFIED_MEMORY=1")
                end
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
