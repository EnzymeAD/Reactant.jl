module ReactantMetalExt

using Metal
using Metal.MPS
using Metal: MtlArray

# ObjectiveC primitives needed by @objc call sites in XLACompiler.jl
using Metal.MTL: @objc, id, nil, NSString, NSArray, NSDictionary, reinterpret

# Descriptor types needed by @objc [T alloc] calls (macro requires bare identifiers)
using Metal.MPSGraphs: MPSGraphConvolution2DOpDescriptor,
                       MPSGraphConvolution3DOpDescriptor,
                       MPSGraphPooling2DOpDescriptor,
                       MPSGraphPooling4DOpDescriptor

# Reactant's in-tree MLIR modules — no parameter injection needed
using Reactant: Reactant, MLIR
using Reactant.MLIR: IR, API

# Phase-1 PJRT plugin: 30 @cfunction callbacks + PJRT_Api struct + make_client()
include("ReactantMetalExt/PJRTPlugin.jl")

# @objc bindings for MPSGraph ops not wrapped by Metal.jl,
# plus julia_to_mps_dtype and mps_reshape helpers
include("ReactantMetalExt/XLACompiler.jl")

# MLIR walker: compile_mlir_module, MetalExecutable, execute!
include("ReactantMetalExt/MLIRWalker.jl")

export compile_mlir_module, MetalExecutable, execute!

function __init__()
    @static if Sys.isapple()
        if Metal.functional()
            # Initialize @cfunction handles and register the PJRT_Api pointer
            # so PJRT.MakeMetalClient() (no-args) can be called from XLA.jl.
            try
                init_pjrt_handles!()
                # Expose the PJRT_Api struct pointer to Reactant's Client.jl
                Reactant.XLA.PJRT._metal_pjrt_api_ptr[] = Ptr{Cvoid}(_PJRT_API_MEM)

                # Create client via the shared PJRT.MetalClient() path (checkcount=false
                # because initialize_default_clients! may not have run yet and the counter
                # won't have been touched).
                state = Reactant.XLA.global_backend_state
                if haskey(state.clients, "metal")
                    # Already registered (e.g., XLA.jl's init block ran first).
                    state.default_client = state.clients["metal"]
                else
                    metal = Reactant.XLA.PJRT.MetalClient(checkcount=false)
                    Reactant.XLA.PJRT.metal_client_count[] += 1
                    state.clients["metal"] = metal
                    state.default_client = metal
                end
            catch e
                if e isa ErrorException && contains(e.msg, "MakeClientFromApi")
                    @warn "Metal PJRT backend requires rebuilt libReactantExtra. Run: julia --project=deps deps/build_local.jl"
                else
                    @warn "Metal backend initialization failed" exception = e
                end
            end
        end
    end
    return nothing
end

end # module ReactantMetalExt
