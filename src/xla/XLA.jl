module XLA

using ..Reactant: Reactant, MLIR
using Reactant_jll
using Libdl
using Scratch, Downloads
using EnumX: @enumx

const XLA_REACTANT_GPU_MEM_FRACTION = Ref{Float64}(0.75)
const XLA_REACTANT_GPU_PREALLOCATE = Ref{Bool}(true)

using Reactant_jll
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
include("HloModule.jl")
include("Memory.jl")

include("PJRT/PJRT.jl")

@kwdef mutable struct BackendState
    initialized::Bool = false
    clients::Dict{String,PJRT.Client} = Dict{String,PJRT.Client}()
    default_client::PJRT.Client = PJRT.Client(C_NULL; skip_check=true)
end

function Base.getproperty(bs::BackendState, sym::Symbol)
    (sym === :initialized || bs.initialized) && return getfield(bs, sym)
    initialize_default_clients!(bs)
    return getfield(bs, sym)
end

function Base.setproperty!(bs::BackendState, sym::Symbol, val)
    (sym === :initialized || bs.initialized) && return setfield!(bs, sym, val)
    initialize_default_clients!(bs)
    return setfield!(bs, sym, val)
end

const global_backend_state = BackendState()
const global_state = State()

client(backend::String) = global_backend_state.clients[backend]
default_backend() = global_backend_state.default_client
process_index() = process_index(default_backend())

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
    # We need to update the clients based on the new state
    initialize_default_clients!(global_backend_state)
    return nothing
end

function __init__()
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
        @debug "XLA_REACTANT_GPU_MEM_FRACTION: " XLA_REACTANT_GPU_MEM_FRACTION[]
    end

    if haskey(ENV, "XLA_REACTANT_GPU_PREALLOCATE")
        XLA_REACTANT_GPU_PREALLOCATE[] = parse(Bool, ENV["XLA_REACTANT_GPU_PREALLOCATE"])
        @debug "XLA_REACTANT_GPU_PREALLOCATE: " XLA_REACTANT_GPU_PREALLOCATE[]
    end

    if haskey(ENV, "REACTANT_VISIBLE_GPU_DEVICES")
        global_state.local_device_ids =
            parse.(Int, split(ENV["REACTANT_VISIBLE_GPU_DEVICES"], ","))
        @debug "REACTANT_VISIBLE_GPU_DEVICES: " global_state.local_device_ids
    end

    @ccall MLIR.API.mlir_c.RegisterEnzymeXLACPUHandler()::Cvoid
    @ccall MLIR.API.mlir_c.RegisterEnzymeXLAGPUHandler()::Cvoid
    return nothing
end

function initialize_default_clients!(state::BackendState)
    was_initialized = state.initialized
    state.initialized = true

    # CPU
    if was_initialized && haskey(state.clients, "cpu")
        XLA.free_client(state.clients["cpu"])
        XLA.PJRT.cpu_client_count[] -= 1
    end
    cpu = PJRT.CPUClient(global_state.process_id, global_state.num_processes)
    state.clients["cpu"] = cpu
    state.default_client = cpu

    # Try TPU if possible, then try GPU (CUDA)
    @static if !Sys.isapple()
        if Reactant.has_tpu()
            dataset_dir = @get_scratch!("libtpu")
            if !isfile(dataset_dir * "/libtpu.so")
                Downloads.download(
                    "https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20240829-py3-none-any.whl",
                    dataset_dir * "/tpu.zip",
                )
                run(`unzip -qq $(dataset_dir*"/tpu.zip") -d $(dataset_dir)/tmp`)
                run(`mv $(dataset_dir)/tmp/libtpu/libtpu.so $(dataset_dir)/libtpu.so`)
                rm(dataset_dir * "/tmp"; recursive=true)
                rm(dataset_dir * "/tpu.zip"; recursive=true)
            end
            try
                if was_initialized && haskey(state.clients, "tpu")
                    XLA.free_client(state.clients["tpu"])
                    XLA.PJRT.tpu_client_count[] -= 1
                end
                # XXX: process_id? num_processes?
                tpu = PJRT.TPUClient(dataset_dir * "/libtpu.so")
                state.clients["tpu"] = tpu
                state.default_client = tpu
            catch e
                println(stdout, e)
            end
        else
            if !Reactant.precompiling()
                try
                    distributed_runtime_client = if global_state.num_processes > 1
                        @assert global_state.client !== nothing
                        global_state.client
                    else
                        nothing
                    end

                    if was_initialized && haskey(state.clients, "gpu")
                        XLA.free_client(state.clients["gpu"])
                        XLA.PJRT.gpu_client_count[] -= 1
                    end
                    gpu = PJRT.GPUClient(
                        global_state.process_id,
                        global_state.num_processes;
                        allowed_devices=global_state.local_device_ids,
                        distributed_runtime_client,
                    )
                    state.clients["gpu"] = gpu
                    state.default_client = gpu
                catch e
                    println(stdout, e)
                end
            end
        end
    end

    return nothing
end

end
