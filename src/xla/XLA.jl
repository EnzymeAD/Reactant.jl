module XLA

using ..Reactant: Reactant, MLIR
using Reactant_jll
using Libdl
using Scratch, Downloads
using EnumX: @enumx

const XLA_REACTANT_GPU_MEM_FRACTION = Ref{Float64}(0.75)
const XLA_REACTANT_GPU_PREALLOCATE = Ref{Bool}(true)
const REACTANT_XLA_RUNTIME = Ref{String}("PJRT")

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

include("IFRT/IFRT.jl")

for runtime in (:PJRT, :IFRT)
    backend_state = Symbol(runtime, :BackendState)

    @eval begin
        @kwdef mutable struct $(backend_state)
            initialized::Bool = false
            clients::Dict{String,$(runtime).Client} = Dict{String,$(runtime).Client}()
            default_client::$(runtime).Client = $(runtime).Client(C_NULL; skip_check=true)
        end

        function Base.getproperty(bs::$(backend_state), sym::Symbol)
            (sym === :initialized || bs.initialized) && return getfield(bs, sym)
            initialize_default_clients!(bs)
            return getfield(bs, sym)
        end

        function Base.setproperty!(bs::$(backend_state), sym::Symbol, val)
            (sym === :initialized || bs.initialized) && return setfield!(bs, sym, val)
            initialize_default_clients!(bs)
            return setfield!(bs, sym, val)
        end
    end
end

const global_backend_state = Ref{Union{Nothing,PJRTBackendState,IFRTBackendState}}(nothing)
const global_state = State()

client(backend::String) = global_backend_state[].clients[backend]
default_backend() = global_backend_state[].default_client
default_device() = default_device(default_backend())
process_index() = process_index(default_backend())

runtime(::Nothing) = runtime()
runtime() = runtime(default_backend())
runtime(::PJRT.Client) = Val(:PJRT)
runtime(::IFRT.Client) = Val(:IFRT)

function set_default_backend(backend::AbstractClient)
    global_backend_state[].default_client = backend
    return nothing
end

function set_default_backend(backend::String)
    global_backend_state[].default_client = client(backend)
    return nothing
end

function update_global_state!(args...; kwargs...)
    update!(global_state, args...; kwargs...)
    # We conditionally initialize for now, since a lot of options that are set are not
    # necessarily supported by PJRT. This makes testing for IFRT quite hard.
    # Once we move to IFRT completely, we can remove this.
    if global_backend_state[].initialized
        # We need to update the clients based on the new state
        initialize_default_clients!(global_backend_state[])
    end
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
        if XLA_REACTANT_GPU_MEM_FRACTION[] > 1 || XLA_REACTANT_GPU_MEM_FRACTION[] < 0
            error("XLA_REACTANT_GPU_MEM_FRACTION must be between 0 and 1")
        end
    end

    if haskey(ENV, "XLA_REACTANT_GPU_PREALLOCATE")
        XLA_REACTANT_GPU_PREALLOCATE[] = parse(Bool, ENV["XLA_REACTANT_GPU_PREALLOCATE"])
        @debug "XLA_REACTANT_GPU_PREALLOCATE: " XLA_REACTANT_GPU_PREALLOCATE[]
    end

    if haskey(ENV, "REACTANT_VISIBLE_GPU_DEVICES")
        global_state.local_gpu_device_ids =
            parse.(Int, split(ENV["REACTANT_VISIBLE_GPU_DEVICES"], ","))
        @debug "REACTANT_VISIBLE_GPU_DEVICES: " global_state.local_gpu_device_ids
    end

    if haskey(ENV, "REACTANT_XLA_RUNTIME")
        REACTANT_XLA_RUNTIME[] = ENV["REACTANT_XLA_RUNTIME"]
        @debug "REACTANT_XLA_RUNTIME: " REACTANT_XLA_RUNTIME[]
    end

    # Set the runtime backend
    if REACTANT_XLA_RUNTIME[] == "PJRT"
        global_backend_state[] = PJRTBackendState()
    elseif REACTANT_XLA_RUNTIME[] == "IFRT"
        global_backend_state[] = IFRTBackendState()
    else
        error("Unsupported REACTANT_XLA_RUNTIME: $(REACTANT_XLA_RUNTIME[])")
    end

    @ccall MLIR.API.mlir_c.RegisterEnzymeXLACPUHandler()::Cvoid
    @ccall MLIR.API.mlir_c.RegisterEnzymeXLAGPUHandler()::Cvoid
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
            XLA.free_client(state.clients["cpu"])
            XLA.$(runtime).cpu_client_count[] -= 1
        end
        cpu = $(runtime).CPUClient(; common_kwargs..., asynchronous=true)
        state.clients["cpu"] = cpu
        state.default_client = cpu

        # Try TPU if possible, then try GPU (CUDA)
        @static if !Sys.isapple()
            if Reactant.has_tpu()
                dataset_dir = @get_scratch!("libtpu")
                download_tpu(dataset_dir)
                try
                    if was_initialized && haskey(state.clients, "tpu")
                        XLA.free_client(state.clients["tpu"])
                        XLA.$(runtime).tpu_client_count[] -= 1
                    end
                    tpu = $(runtime).TPUClient(;
                        tpu_path=dataset_dir * "/libtpu.so", common_kwargs...
                    )
                    state.clients["tpu"] = tpu
                    state.default_client = tpu
                catch e
                    println(stdout, e)
                end
            else
                if !Reactant.precompiling()
                    try
                        if was_initialized && haskey(state.clients, "gpu")
                            XLA.free_client(state.clients["gpu"])
                            XLA.$(runtime).gpu_client_count[] -= 1
                        end
                        gpu = $(runtime).GPUClient(;
                            common_kwargs...,
                            allowed_devices=global_state.local_gpu_device_ids,
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

function download_tpu(dataset_dir::String)
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
end

end
