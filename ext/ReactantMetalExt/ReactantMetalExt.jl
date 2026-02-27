module ReactantMetalExt

using Metal
using Metal.MPS
using Metal: MtlArray

# ObjectiveC primitives needed by @objc call sites in XLACompiler.jl
using Metal.MTL: @objc, id, nil, NSString, NSArray, NSDictionary, reinterpret

# MTLTensor types for Buffer.jl helpers (macOS 26+)
using Metal.MTL:
    MTLDevice, MTLBuffer, NSUInteger, contents,
    MTLTensor, MTLTensorExtents, MTLTensorDescriptor,
    MTLTensorDataType,
    MTLTensorDataTypeFloat32, MTLTensorDataTypeFloat16, MTLTensorDataTypeBFloat16,
    MTLTensorDataTypeInt8, MTLTensorDataTypeUInt8,
    MTLTensorDataTypeInt16, MTLTensorDataTypeUInt16,
    MTLTensorDataTypeInt32, MTLTensorDataTypeUInt32,
    MTLStorageModeShared, MTLTensorUsageCompute

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

function __init__()
    @static if Sys.isapple()
        if Metal.functional()
            try
                # Register PJRT callback trampolines so MetalClient() can be created.
                init_pjrt_handles!()
                Reactant.XLA.PJRT._metal_pjrt_api_ptr[] = Ptr{Cvoid}(_PJRT_API_MEM)

                # TODO(upstream): The client registration below should be replaced
                # with a generic Reactant.XLA.register_backend!() API that handles
                # late-loading backends (loaded after initialize_default_clients!
                # has already run).  Ref: wsmoses review on PR #2489.
                state = Reactant.XLA.global_backend_state
                if state.initialized
                    # Backends were already initialized before Metal.jl loaded.
                    # Manually register since initialize_default_clients! won't
                    # re-run automatically.
                    if !haskey(state.clients, "metal")
                        metal = Reactant.XLA.PJRT.MetalClient(; checkcount=false)
                        Reactant.XLA.PJRT.metal_client_count[] += 1
                        state.clients["metal"] = metal
                    end
                    state.default_client = state.clients["metal"]
                end
                # If !state.initialized, initialize_default_clients! will handle
                # Metal registration when global_backend_state is first accessed.
            catch e
                @warn "Metal backend initialization failed" exception = e
            end
        end
    end
    return nothing
end

end # module ReactantMetalExt
