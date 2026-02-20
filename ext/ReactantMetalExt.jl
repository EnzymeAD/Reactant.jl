# Metal backend for Reactant — not precompiled because it overrides
# Base.convert, Reactant.XLA.free_buffer, and Reactant.XLA.to_host to add
# thread-safety for host buffer transfers.
# (Julia disallows method overwriting during precompilation.)
__precompile__(false)

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

# ─── Thread-safe Metal PJRT buffer operations ────────────────────────────────
#
# Julia 1.9+ runs GC finalizers in a dedicated finalizer thread.  When the Julia
# GC triggers, old PJRT buffer wrappers from previous @jit calls are finalized:
#
#   finalizer thread: free_buffer → PjRtBufferFree → delete PjRtCApiBuffer
#                     → ~PjRtCApiBuffer() → accesses PjRtCApiClient shared state
#
#   main thread:      BufferToHost → PjRtCApiBuffer::ToLiteralSync()
#                     → accesses PjRtCApiClient shared state
#
# XLA's PjRtCApiClient is NOT thread-safe.  Concurrent access from the finalizer
# thread and the main thread causes heap corruption (std::bad_alloc).
#
# Fix: METAL_XLA_LOCK serializes every call that enters XLA's C++ wrapper layer.
# Both free_buffer (finalizer thread) and to_host (main thread) must hold the
# lock before calling PjRtBufferFree / BufferToHost respectively.
#
# Note: GC.enable_finalizers(false/true) alone is insufficient because it only
# prevents NEW finalizers from being dequeued — already-running finalizers
# continue concurrently.  A proper mutex is needed.

const METAL_XLA_LOCK = ReentrantLock()

# Override Base.convert for ConcretePJRTArray.
# Disabling finalizers here reduces lock contention: no new finalizers can start
# between the output-buffer allocation and the BufferToHost call, so METAL_XLA_LOCK
# inside to_host is almost never contended.
function Base.convert(::Type{<:Array}, X::Reactant.ConcretePJRTArray{T,N}) where {T,N}
    GC.enable_finalizers(false)
    try
        if Reactant.has_padding(X)
            padding = Reactant.get_padding(X)
            data = Array{T,N}(undef, (size(X) .+ padding)...)
            Reactant.write_to_host_buffer!(data, X)
            return view(data, [1:size(X, i) for i in 1:ndims(X)]...)
        else
            data = Array{T,N}(undef, size(X)...)
            Reactant.write_to_host_buffer!(data, X)
            return data
        end
    finally
        GC.enable_finalizers(true)
    end
end

# Override free_buffer so that PjRtBufferFree (called from the Julia GC finalizer
# thread) cannot overlap with BufferToHost on the main thread.
function Reactant.XLA.free_buffer(buffer::Reactant.XLA.PJRT.Buffer)
    sbuffer = buffer.buffer
    if sbuffer != C_NULL && Reactant.XLA.is_live[]
        @lock METAL_XLA_LOCK begin
            @ccall Reactant.MLIR.API.mlir_c.PjRtBufferFree(sbuffer::Ptr{Cvoid})::Cvoid
        end
    end
end

# Override to_host so that BufferToHost cannot overlap with PjRtBufferFree.
function Reactant.XLA.to_host(
    buffer::Reactant.XLA.PJRT.Buffer,
    data,
    sharding,
)
    @assert buffer.buffer !== C_NULL
    @lock METAL_XLA_LOCK begin
        GC.@preserve buffer data begin
            @ccall Reactant.MLIR.API.mlir_c.BufferToHost(
                buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
            )::Cvoid
        end
    end
    return data
end

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
