module XLA

using ..Reactant: Reactant, MLIR
using Reactant_jll
using Libdl
using Scratch, Downloads

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

include("Client.jl")
include("Device.jl")
include("LoadedExecutable.jl")
include("Future.jl")
include("Buffer.jl")
include("Stats.jl")
include("Utils.jl")

const backends = Dict{String,Client}()
const default_backend = Ref{Client}()
const default_device_idx = Ref{Int}(0)

function __init__()
    initLogs = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "InitializeLogs")
    ccall(initLogs, Cvoid, ())
    # Add most log level
    # SetLogLevel(0)
    cpu = CPUClient()
    backends["cpu"] = cpu
    default_backend[] = cpu

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
                tpu = TPUClient(dataset_dir * "/libtpu.so")
                backends["tpu"] = tpu
                default_backend[] = tpu
            catch e
                println(stdout, e)
            end
        else
            if !Reactant.precompiling()
                try
                    gpu = GPUClient()
                    backends["gpu"] = gpu
                    default_backend[] = gpu
                catch e
                    println(stdout, e)
                end
            end
        end
    end

    @ccall MLIR.API.mlir_c.RegisterEnzymeXLACPUHandler()::Cvoid
    @ccall MLIR.API.mlir_c.RegisterEnzymeXLAGPUHandler()::Cvoid

    # This wasn't properly exported on macos, we'll remove the try once macOS JLL
    # has the fix.
    errptr = cglobal((:ReactantThrowError, MLIR.API.mlir_c), Ptr{Ptr{Cvoid}})
    unsafe_store!(errptr, @cfunction(reactant_err, Cvoid, (Cstring,)))
    return nothing
end

include("IFRT/IFRT.jl")

end
