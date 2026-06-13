module ReactantMetalExt

using Metal
using Metal.MPS
using Metal: MtlArray

# ObjectiveC primitives needed by @objc call sites in XLACompiler.jl
using Metal.MTL: @objc, id, nil, NSString, NSArray, NSDictionary, reinterpret

# MTLBuffer types for Buffer.jl helpers
using Metal.MTL:
    MTLDevice, MTLBuffer, NSUInteger, contents

# Descriptor types needed by @objc [T alloc] calls (macro requires bare identifiers)
using Metal.MPSGraphs:
    MPSGraphConvolution2DOpDescriptor,
    MPSGraphConvolution3DOpDescriptor,
    MPSGraphPooling2DOpDescriptor,
    MPSGraphPooling4DOpDescriptor

# Reactant's in-tree MLIR modules — no parameter injection needed
using Reactant: Reactant, MLIR
using Reactant.MLIR: IR, API

const CAPI = Reactant.XLA.PJRT.CAPI

include("Buffer.jl")
include("Executable.jl")

# Phase-1 PJRT plugin: 30 @cfunction callbacks + PJRT_Api struct + make_client()
include("PJRTPlugin.jl")

# @objc bindings for MPSGraph ops not wrapped by Metal.jl,
# plus julia_to_mps_dtype and mps_reshape helpers
include("XLACompiler.jl")

# MLIR walker: compile_mlir_module, MetalExecutable, execute!
include("MLIRWalker.jl")

export compile_mlir_module, MetalExecutable, execute!

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    api_ptr = Ptr{Cvoid}(_PJRT_API_MEM)
    errstr = Ref{Cstring}()
    GC.@preserve errstr begin
        client = Reactant.MLIR.API.MakeClientFromApi(
            Ptr{CAPI.PJRT_Api}(api_ptr), "metal", "METAL", errstr
        )
    end
    client == C_NULL && throw(AssertionError(unsafe_string(errstr[])))
    return Reactant.XLA.PJRT.Client(client)
end

function __init__()
    @static if Sys.isapple()
        if Metal.functional()
            try
                init_pjrt_handles!()
                Reactant.Accelerators.Registration.register_backend(
                    "metal";
                    priority=400,
                    pjrt_initialize_function=make_pjrt_client,
                )
            catch e
                @warn "Metal backend initialization failed" exception = e
            end
        end
    end
    return nothing
end

end # module ReactantMetalExt
