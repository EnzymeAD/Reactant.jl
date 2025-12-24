module XLA

using ..Reactant: Reactant, MLIR, Accelerators
using Reactant_jll: Reactant_jll
using LLVM: LLVM
using Libdl: Libdl
using EnumX: @enumx
using Enzyme: Compiler
using Preferences: load_preference

const XLA_REACTANT_GPU_MEM_FRACTION = Ref{Float64}(0.75)
const XLA_REACTANT_GPU_PREALLOCATE = Ref{Bool}(true)
const REACTANT_XLA_RUNTIME = load_preference(Reactant, "xla_runtime", "PJRT")

const CUDA_DATA_DIR = Ref(
    isdefined(Reactant_jll, :ptxas_path) ? dirname(dirname(Reactant_jll.ptxas_path)) : ""
)

function LLVMclopts(opts...)
    args = ["", opts...]
    @ccall MLIR.API.mlir_c.ReactantLLVMParseCommandLineOptions(
        length(args)::Cint, args::Ptr{Cstring}, C_NULL::Ptr{Cvoid}
    )::Cvoid
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

abstract type AbstractBackendState end

for runtime in (:PJRT, :IFRT)
    backend_state = Symbol(runtime, :BackendState)

    @eval @kwdef mutable struct $(backend_state) <: AbstractBackendState
        initialized::Bool = false
        clients::Dict{String,$(runtime).Client} = Dict{String,$(runtime).Client}()
        default_client::$(runtime).Client = $(runtime).NullClient
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

function client(backend::String)
    if backend == "gpu"
        if haskey(global_backend_state.clients, "cuda")
            backend = "cuda"
        elseif haskey(global_backend_state.clients, "metal")
            backend = "metal"
        else
            error("No GPU client found")
        end
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
    global_backend_state.default_client = client(backend)
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

        @ccall MLIR.API.mlir_c.RegisterEnzymeXLACPUHandler()::Cvoid
        @ccall MLIR.API.mlir_c.RegisterEnzymeXLAGPUHandler()::Cvoid
        @ccall MLIR.API.mlir_c.registerReactantXLAFFI()::Cvoid

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
    end

    return nothing
end

for runtime in (:PJRT, :IFRT)
    @eval function initialize_default_clients!(state::$(Symbol(runtime, :BackendState)))
        was_initialized = state.initialized
        state.initialized = true
        distributed_runtime_client = if global_state.num_processes > 1
            @assert global_state.client !== nothing
            global_state.client
        else
            nothing
        end
        common_kwargs = (;
            node_id=global_state.process_id,
            num_nodes=global_state.num_processes,
            distributed_runtime_client,
        )

        # CPU
        if was_initialized && haskey(state.clients, "cpu")
            free_client(state.clients["cpu"])
            $(runtime).cpu_client_count[] -= 1
        end
        cpu = $(runtime).CPUClient(; common_kwargs..., asynchronous=true)
        state.clients["cpu"] = cpu
        state.default_client = cpu

        # Try TPU if possible, then try GPU (CUDA)
        if !Reactant.precompiling()
            @static if !Sys.isapple()
                if Accelerators.TPU.has_tpu()
                    Accelerators.TPU.download_libtpu_if_needed()
                    try
                        if was_initialized && haskey(state.clients, "tpu")
                            free_client(state.clients["tpu"])
                            $(runtime).tpu_client_count[] -= 1
                        end
                        tpu = $(runtime).TPUClient(;
                            tpu_path=Accelerators.TPU.get_libtpu_path(), common_kwargs...
                        )
                        state.clients["tpu"] = tpu
                        state.default_client = tpu
                    catch e
                        println(stdout, e)
                    end
                elseif Accelerators.TT.has_tt()
                    @debug "TT accelerator detected, setting it up"
                    try
                        if was_initialized && haskey(state.clients, "tt")
                            free_client(state.clients["tt"])
                            $(runtime).tt_client_count[] -= 1
                        end
                        # The env var `TT_METAL_RUNTIME_ROOT` must be set before creating the client.
                        tt_metal_runtime_root = get(ENV, "TT_METAL_RUNTIME_ROOT", nothing)
                        if isnothing(tt_metal_runtime_root)
                            tt_metal_path_in_wheel = joinpath(
                                dirname(Accelerators.TT.get_tt_pjrt_plugin_path()),
                                "tt-metal",
                            )
                            if ispath(tt_metal_path_in_wheel)
                                @debug "Setting environment variable 'TT_METAL_RUNTIME_ROOT' to '$(tt_metal_path_in_wheel)'"
                                ENV["TT_METAL_RUNTIME_ROOT"] = tt_metal_path_in_wheel
                            else
                                error(
                                    "`TT_METAL_RUNTIME_ROOT` environment variable not set and we could not automatically determine it",
                                )
                            end
                        else
                            @debug "Environment variable 'TT_METAL_RUNTIME_ROOT' already set to to '$(tt_metal_runtime_root)'"
                        end

                        tt = $(runtime).TTClient(;
                            tt_pjrt_plugin_path=Accelerators.TT.get_tt_pjrt_plugin_path(),
                            common_kwargs...,
                        )
                        state.clients["tt"] = tt
                        state.default_client = tt
                    catch e
                        println(stdout, e)
                    end
                elseif Reactant_jll.host_platform.tags["gpu"] != "none"
                    try
                        if was_initialized && haskey(state.clients, "cuda")
                            free_client(state.clients["cuda"])
                            $(runtime).cuda_client_count[] -= 1
                        end
                        gpu = $(runtime).CUDAClient(;
                            common_kwargs...,
                            allowed_devices=global_state.local_gpu_device_ids,
                        )
                        state.clients["cuda"] = gpu
                        state.default_client = gpu
                    catch e
                        println(stdout, e)
                    end
                end
            else
                try
                    #=
                    if was_initialized && haskey(state.clients, "metal")
                        free_client(state.clients["metal"])
                        $(runtime).metal_client_count[] -= 1
                    end
                    gpu = $(runtime).MetalClient(;
                        metal_pjrt_plugin_path=Accelerators.Metal.get_metal_pjrt_plugin_path(),
                        common_kwargs...,
                    )
                    state.clients["metal"] = gpu
                    # Don't put this in the default_client since metal support is fairly
                    # limited
                    =#
                    # Metal PJRT plugin is not yet compatible with latest OpenXLA
                catch e
                    println(stdout, e)
                end
            end
        end

        return nothing
    end
end

end
