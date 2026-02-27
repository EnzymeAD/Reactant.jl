# Executable.jl — MetalExecutableData struct + executable PJRT callbacks
#
# C-layout struct for executable metadata, allocated via Libc.malloc.
# All ObjC objects (MPSGraph, MPSGraphTensor, MTLBuffer) stored as raw UInt64 ids
# with explicit retain on freeze, release on destroy.
# ZERO Julia GC objects — per Billy: "the executable object itself should not have
# julia GC objects, if possible"

# ============================================================
# MetalExecutableData struct
# ============================================================

struct MetalExecutableData
    # Core MPSGraph (ObjC id stored as UInt64, manually retained)
    graph_id::UInt64

    # Input metadata
    input_placeholder_ids::Ptr{UInt64}  # C array of retained ObjC ids (MPSGraphTensor)
    input_shapes::Ptr{Ptr{Int64}}       # C array of Libc.malloc'd Int64 arrays
    input_shape_ranks::Ptr{Int32}       # rank of each input
    input_dtypes::Ptr{UInt32}           # PJRT element type enum values
    n_inputs::Int32

    # Output metadata
    output_tensor_ids::Ptr{UInt64}      # C array of retained ObjC ids (MPSGraphTensor)
    output_shapes::Ptr{Ptr{Int64}}      # C array of Libc.malloc'd Int64 arrays
    output_shape_ranks::Ptr{Int32}      # rank of each output
    output_dtypes::Ptr{UInt32}          # PJRT element type enum values
    n_outputs::Int32

    # Constants (fed at every execute! call)
    const_placeholder_ids::Ptr{UInt64}  # C array of retained ObjC ids (MPSGraphTensor)
    const_mtl_buf_ids::Ptr{UInt64}      # C array of raw MTLBuffer ObjC ids (from Metal.alloc)
    const_shapes::Ptr{Ptr{Int64}}       # C array of Libc.malloc'd Int64 arrays (Julia shape)
    const_shape_ranks::Ptr{Int32}       # rank of each constant
    const_dtypes::Ptr{UInt32}           # PJRT element type enum values
    n_consts::Int32

    num_ops::Int32

    # MLIR text for _exec_optimized_program
    mlir_text_ptr::Ptr{UInt8}           # Libc.malloc'd copy of string bytes
    mlir_text_len::Int
end

"""
    freeze_executable(exec) -> Ptr{MetalExecutableData}

Convert a Julia `MetalExecutable` into a C-allocated `MetalExecutableData` struct.

Retains all ObjC objects (graph, placeholders, output tensors).
Copies all metadata to `Libc.malloc`'d C arrays.
Allocates fresh MTLBuffers for constant data (independent of MtlArray GC).
The returned pointer IS the PJRT executable handle.

Libc.malloc count (for `n_in` inputs, `n_out` outputs, `n_c` constants):
- Fixed: 15 allocations (arrays + struct)
- Per-input/output/const shape: `n_in + n_out + n_c` allocations
- Plus: `n_c` MTLBuffer allocations via `Metal.alloc` (freed via `Metal.free`)
"""
function freeze_executable(exec)::Ptr{MetalExecutableData}
    n_in = Int32(length(exec.input_placeholders))
    n_out = Int32(length(exec.output_tensors))
    n_c = Int32(length(exec.const_placeholders))

    # --- Retain all ObjC objects (prevent dealloc when Julia wrappers are GC'd) ---
    ObjC = Metal.MTL.ObjectiveC
    ObjC.retain(exec.graph)
    for ph in exec.input_placeholders
        ObjC.retain(ph)
    end
    for t in exec.output_tensors
        ObjC.retain(t)
    end
    for ph in exec.const_placeholders
        ObjC.retain(ph)
    end
    # const_mtl_values are MtlArrays (Julia GC) — we allocate fresh MTLBuffers below

    # --- Extract graph id ---
    graph_id = UInt64(pointer(exec.graph))

    # --- Input placeholder ids ---                         # malloc #1
    input_ph_ids = Ptr{UInt64}(Libc.malloc(max(1, n_in) * sizeof(UInt64)))
    for i in 1:n_in
        unsafe_store!(input_ph_ids, UInt64(pointer(exec.input_placeholders[i])), i)
    end

    # --- Input shapes (outer array + per-input shape arrays) ---
    input_shapes_p = Ptr{Ptr{Int64}}(Libc.malloc(max(1, n_in) * sizeof(Ptr{Int64})))  # malloc #2
    input_ranks_p = Ptr{Int32}(Libc.malloc(max(1, n_in) * sizeof(Int32)))             # malloc #3
    for i in 1:n_in
        shape = exec.input_shapes[i]
        rank = Int32(length(shape))
        unsafe_store!(input_ranks_p, rank, i)
        sp = Ptr{Int64}(Libc.malloc(max(1, rank) * sizeof(Int64)))  # malloc #(3+i)
        for j in 1:rank
            unsafe_store!(sp, Int64(shape[j]), j)
        end
        unsafe_store!(input_shapes_p, sp, i)
    end

    # --- Input dtypes ---                                  # malloc #(4+n_in)
    input_dtypes_p = Ptr{UInt32}(Libc.malloc(max(1, n_in) * sizeof(UInt32)))
    for i in 1:n_in
        unsafe_store!(input_dtypes_p, julia_type_to_pjrt(exec.input_dtypes[i]), i)
    end

    # --- Output tensor ids ---                             # malloc #(5+n_in)
    output_ids_p = Ptr{UInt64}(Libc.malloc(max(1, n_out) * sizeof(UInt64)))
    for i in 1:n_out
        unsafe_store!(output_ids_p, UInt64(pointer(exec.output_tensors[i])), i)
    end

    # --- Output shapes ---
    output_shapes_p = Ptr{Ptr{Int64}}(Libc.malloc(max(1, n_out) * sizeof(Ptr{Int64})))  # malloc #(6+n_in)
    output_ranks_p = Ptr{Int32}(Libc.malloc(max(1, n_out) * sizeof(Int32)))              # malloc #(7+n_in)
    for i in 1:n_out
        shape = exec.output_shapes[i]
        rank = Int32(length(shape))
        unsafe_store!(output_ranks_p, rank, i)
        sp = Ptr{Int64}(Libc.malloc(max(1, rank) * sizeof(Int64)))  # malloc #(7+n_in+i)
        for j in 1:rank
            unsafe_store!(sp, Int64(shape[j]), j)
        end
        unsafe_store!(output_shapes_p, sp, i)
    end

    # --- Output dtypes ---                                 # malloc #(8+n_in+n_out)
    output_dtypes_p = Ptr{UInt32}(Libc.malloc(max(1, n_out) * sizeof(UInt32)))
    for i in 1:n_out
        unsafe_store!(output_dtypes_p, julia_type_to_pjrt(exec.output_dtypes[i]), i)
    end

    # --- Const placeholder ids ---                         # malloc #(9+n_in+n_out)
    const_ph_ids_p = Ptr{UInt64}(Libc.malloc(max(1, n_c) * sizeof(UInt64)))
    for i in 1:n_c
        unsafe_store!(const_ph_ids_p, UInt64(pointer(exec.const_placeholders[i])), i)
    end

    # --- Const MTLBuffers + shapes + dtypes ---
    # Allocate fresh MTLBuffers for constant data (avoids MtlArray GC dependency).
    # Metal.alloc returns +1 retained — released via Metal.free in destroy.
    const_buf_ids_p = Ptr{UInt64}(Libc.malloc(max(1, n_c) * sizeof(UInt64)))    # malloc #(10+n_in+n_out)
    const_shapes_p = Ptr{Ptr{Int64}}(Libc.malloc(max(1, n_c) * sizeof(Ptr{Int64})))  # malloc #(11+n_in+n_out)
    const_ranks_p = Ptr{Int32}(Libc.malloc(max(1, n_c) * sizeof(Int32)))       # malloc #(12+n_in+n_out)
    const_dtypes_p = Ptr{UInt32}(Libc.malloc(max(1, n_c) * sizeof(UInt32)))     # malloc #(13+n_in+n_out)
    dev = Metal.current_device()
    for i in 1:n_c
        mtl_arr = exec.const_mtl_values[i]
        julia_arr = Array(mtl_arr)  # copy GPU → CPU
        dt = eltype(julia_arr)
        s = size(julia_arr)
        nbytes = length(julia_arr) * sizeof(dt)

        # Allocate raw MTLBuffer and copy data (SharedStorage = unified memory)
        mtl_buf = Metal.alloc(dev, max(nbytes, 1); storage=Metal.SharedStorage)
        gpu_ptr = Metal.MTL.contents(mtl_buf)
        GC.@preserve julia_arr begin
            unsafe_copyto!(Ptr{UInt8}(gpu_ptr), Ptr{UInt8}(pointer(julia_arr)), nbytes)
        end

        unsafe_store!(const_buf_ids_p, UInt64(pointer(mtl_buf)), i)

        # Store shape
        rank = Int32(length(s))
        unsafe_store!(const_ranks_p, rank, i)
        sp = Ptr{Int64}(Libc.malloc(max(1, rank) * sizeof(Int64)))  # malloc per const
        for j in 1:rank
            unsafe_store!(sp, Int64(s[j]), j)
        end
        unsafe_store!(const_shapes_p, sp, i)

        # Store dtype (as PJRT enum)
        unsafe_store!(const_dtypes_p, julia_type_to_pjrt(dt), i)
    end

    # --- MLIR text ---                                     # malloc #(14+n_in+n_out+n_c)
    mlir_bytes = Vector{UInt8}(exec.mlir_text)
    mlir_len = length(mlir_bytes)
    mlir_ptr = Ptr{UInt8}(Libc.malloc(max(1, mlir_len)))
    if mlir_len > 0
        unsafe_copyto!(mlir_ptr, pointer(mlir_bytes), mlir_len)
    end

    # --- Allocate and fill the struct ---                  # malloc #(15+n_in+n_out+n_c)
    meta_ptr = Ptr{MetalExecutableData}(Libc.malloc(sizeof(MetalExecutableData)))
    unsafe_store!(
        meta_ptr,
        MetalExecutableData(
            graph_id,
            input_ph_ids,
            input_shapes_p,
            input_ranks_p,
            input_dtypes_p,
            n_in,
            output_ids_p,
            output_shapes_p,
            output_ranks_p,
            output_dtypes_p,
            n_out,
            const_ph_ids_p,
            const_buf_ids_p,
            const_shapes_p,
            const_ranks_p,
            const_dtypes_p,
            n_c,
            Int32(exec.num_ops),
            mlir_ptr,
            mlir_len,
        ),
    )

    return meta_ptr
end

# ============================================================
# Executable metadata callbacks
# ============================================================

const EXEC_NAME_STR = "metal_exec"  # Julia string — pointer is process-stable
const MLIR_FORMAT_STR = "mlir"        # format identifier for PJRT_Program
const EMPTY_MLIR_MOD = "module {}"   # minimal valid MLIR module (parsed by XLA for GetHloModules)

function _exec_name(args::Ptr{CAPI.PJRT_Executable_Name_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, pointer(EXEC_NAME_STR), Val{:executable_name}())
    Reactant.unsafe_store_field!(
        args, Csize_t(length(EXEC_NAME_STR)), Val{:executable_name_size}()
    )
    return C_NULL
end

function _exec_num_replicas(args::Ptr{CAPI.PJRT_Executable_NumReplicas_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_replicas}())
    return C_NULL
end

function _exec_num_partitions(
    args::Ptr{CAPI.PJRT_Executable_NumPartitions_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_partitions}())
    return C_NULL
end

function _exec_num_outputs(args::Ptr{CAPI.PJRT_Executable_NumOutputs_Args})::Ptr{Cvoid}
    handle = Reactant.unsafe_load_field(args, Val{:executable}())
    meta = unsafe_load(Ptr{MetalExecutableData}(handle))
    Reactant.unsafe_store_field!(args, Csize_t(meta.n_outputs), Val{:num_outputs}())
    return C_NULL
end

function _exec_code_size(
    args::Ptr{CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Int64(0), Val{:size_in_bytes}())
    return C_NULL
end

# XLA calls this twice (1st: code=NULL to get size; 2nd: code=buffer to fill).
# We return the MLIR text in "mlir" format so XLA can build an HloModule.
function _exec_optimized_program(
    args::Ptr{CAPI.PJRT_Executable_OptimizedProgram_Args}
)::Ptr{Cvoid}
    exec_handle = Reactant.unsafe_load_field(args, Val{:executable}())
    meta = unsafe_load(Ptr{MetalExecutableData}(exec_handle))

    prog_ptr = Ptr{CAPI.PJRT_Program}(Reactant.unsafe_load_field(args, Val{:program}()))
    code_ptr = Reactant.unsafe_load_field(prog_ptr, Val{:code}())
    # Set format on both calls
    Reactant.unsafe_store_field!(prog_ptr, pointer(MLIR_FORMAT_STR), Val{:format}())
    Reactant.unsafe_store_field!(
        prog_ptr, Csize_t(length(MLIR_FORMAT_STR)), Val{:format_size}()
    )

    if meta.mlir_text_len > 0
        # MLIR text stored in C memory (Libc.malloc'd) — stable pointer
        if code_ptr == C_NULL
            Reactant.unsafe_store_field!(
                prog_ptr, Csize_t(meta.mlir_text_len), Val{:code_size}()
            )
        else
            unsafe_copyto!(Ptr{UInt8}(code_ptr), meta.mlir_text_ptr, meta.mlir_text_len)
        end
    else
        # No MLIR text — use empty module fallback
        code_len = length(codeunits(EMPTY_MLIR_MOD))
        if code_ptr == C_NULL
            Reactant.unsafe_store_field!(prog_ptr, Csize_t(code_len), Val{:code_size}())
        else
            unsafe_copyto!(
                Ptr{UInt8}(code_ptr), pointer(codeunits(EMPTY_MLIR_MOD)), code_len
            )
        end
    end
    return C_NULL
end

function _exec_output_memory_kinds(
    args::Ptr{CAPI.PJRT_Executable_OutputMemoryKinds_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:num_outputs}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:memory_kinds}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:memory_kind_sizes}())
    return C_NULL
end

function _exec_output_element_types(
    args::Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, C_NULL, Val{:output_types}())
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:num_output_types}())
    return C_NULL
end

function _exec_output_dimensions(
    args::Ptr{CAPI.PJRT_Executable_OutputDimensions_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:num_outputs}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:dims}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:dim_sizes}())
    return C_NULL
end

# ============================================================
# Loaded executable callbacks
# ============================================================

function _loaded_exec_fingerprint(
    args::Ptr{CAPI.PJRT_LoadedExecutable_Fingerprint_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, C_NULL, Val{:executable_fingerprint}())
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:executable_fingerprint_size}())
    return C_NULL
end

function _loaded_exec_get_executable(
    args::Ptr{CAPI.PJRT_LoadedExecutable_GetExecutable_Args}
)::Ptr{Cvoid}
    handle = Reactant.unsafe_load_field(args, Val{:loaded_executable}())
    Reactant.unsafe_store_field!(args, handle, Val{:executable}())
    return C_NULL
end

function _loaded_exec_addressable_devices(
    args::Ptr{CAPI.PJRT_LoadedExecutable_AddressableDevices_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, DEVICE_PTR_ARRAY, Val{:addressable_devices}())
    Reactant.unsafe_store_field!(args, Csize_t(1), Val{:num_addressable_devices}())
    return C_NULL
end

# No-op deleter for PJRT_LoadedExecutable_GetDeviceAssignment.
# XLA always calls serialized_device_assignment_deleter via absl::Cleanup,
# even when serialized_bytes_size == 0 (early return path).
function _noop_free(p::Ptr{Cvoid})::Cvoid
    return nothing
end

# Strategy: set serialized_bytes_size = 0, which signals to XLA that this
# is a "portable" executable with no device assignment (device_assignment_ = nullptr).
function _loaded_exec_get_device_assignment(
    args::Ptr{CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args}
)::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, C_NULL, Val{:serialized_bytes}())
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:serialized_bytes_size}())
    Reactant.unsafe_store_field!(args, C_NULL, Val{:serialized_device_assignment}())
    Reactant.unsafe_store_field!(
        args, cfn_noop_free, Val{:serialized_device_assignment_deleter}()
    )
    return C_NULL
end

# ============================================================
# Compile callback
# ============================================================

function _client_compile(args::Ptr{CAPI.PJRT_Client_Compile_Args})::Ptr{Cvoid}
    program_ptr = Ptr{CAPI.PJRT_Program}(Reactant.unsafe_load_field(args, Val{:program}()))
    code_ptr = Ptr{UInt8}(Reactant.unsafe_load_field(program_ptr, Val{:code}()))
    code_size = Reactant.unsafe_load_field(program_ptr, Val{:code_size}())

    # Deserialize the StableHLO portable artifact sent by XLA's PjRtCApiClient.
    # XLA serializes the MLIR module via SerializeUsingNativeBytecode, which
    # produces a portable artifact using VHLO dialect internally.
    # mlirModuleCreateParse fails with "dialect 'vhlo' is unknown" because VHLO
    # is not in Reactant's registry.  stablehloDeserializePortableArtifactNoError
    # handles VHLO internally (parses VHLO, upgrades to StableHLO) and returns
    # a StableHLO module in our context.
    exec, mlir_text = IR.with_context() do _ctx
        str_ref = API.MlirStringRef(Ptr{Cchar}(code_ptr), Csize_t(code_size))
        mlir_mod_ref = API.stablehloDeserializePortableArtifactNoError(str_ref, _ctx)
        mod = IR.Module(mlir_mod_ref)
        # Serialize MLIR text BEFORE compile_mlir_module modifies/consumes the module.
        io = IOBuffer()
        show(io, mod)
        mlir_str = String(take!(io))
        exec_inner = compile_mlir_module(mod)
        (exec_inner, mlir_str)
    end

    # Store MLIR text in the executable struct (for _exec_optimized_program)
    exec.mlir_text = mlir_text

    @debug "Metal PJRT compile: op_count = $(exec.num_ops), nout = $(length(exec.output_dtypes))"

    # Freeze the Julia MetalExecutable into a C-allocated MetalExecutableData struct.
    # This retains all ObjC objects, copies metadata to C arrays, and returns a stable
    # Libc.malloc'd pointer with explicit lifetime management (release+free in destroy).
    meta_ptr = freeze_executable(exec)
    handle = Ptr{Cvoid}(meta_ptr)

    Reactant.unsafe_store_field!(args, handle, Val{:executable}())

    return C_NULL
end

# ============================================================
# Destroy callback — release ObjC objects + free C arrays
# ============================================================

function _loaded_exec_destroy(
    args::Ptr{CAPI.PJRT_LoadedExecutable_Destroy_Args}
)::Ptr{Cvoid}
    handle = Reactant.unsafe_load_field(args, Val{:executable}())
    handle == C_NULL && return C_NULL

    meta_ptr = Ptr{MetalExecutableData}(handle)
    meta = unsafe_load(meta_ptr)

    ObjC_id = Metal.MTL.ObjectiveC.id
    ObjC = Metal.MTL.ObjectiveC
    n_in = Int(meta.n_inputs)
    n_out = Int(meta.n_outputs)
    n_c = Int(meta.n_consts)

    # --- Release ObjC objects (balances retain in freeze_executable) ---

    # Graph
    graph = Metal.MPSGraphs.MPSGraphInstance(
        ObjC_id{Metal.MPSGraphs.MPSGraph}(meta.graph_id)
    )
    ObjC.release(graph)

    # Input placeholders
    for i in 1:n_in
        ph = Metal.MPSGraphs.MPSGraphTensorInstance(
            ObjC_id{Metal.MPSGraphs.MPSGraphTensor}(
                unsafe_load(meta.input_placeholder_ids, i)
            ),
        )
        ObjC.release(ph)
    end

    # Output tensors
    for i in 1:n_out
        t = Metal.MPSGraphs.MPSGraphTensorInstance(
            ObjC_id{Metal.MPSGraphs.MPSGraphTensor}(unsafe_load(meta.output_tensor_ids, i))
        )
        ObjC.release(t)
    end

    # Const placeholders
    for i in 1:n_c
        ph = Metal.MPSGraphs.MPSGraphTensorInstance(
            ObjC_id{Metal.MPSGraphs.MPSGraphTensor}(
                unsafe_load(meta.const_placeholder_ids, i)
            ),
        )
        ObjC.release(ph)
    end

    # Const MTLBuffers (allocated via Metal.alloc — freed via Metal.free)
    mtl_id = Metal.MTL.ObjectiveC.id
    for i in 1:n_c
        mtl_buf = Metal.MTL.MTLBufferInstance(
            mtl_id{Metal.MTL.MTLBuffer}(unsafe_load(meta.const_mtl_buf_ids, i))
        )
        Metal.free(mtl_buf)
    end

    # --- Free per-element shape arrays ---                 # n_in + n_out + n_c frees

    for i in 1:n_in
        Libc.free(unsafe_load(meta.input_shapes, i))
    end
    for i in 1:n_out
        Libc.free(unsafe_load(meta.output_shapes, i))
    end
    for i in 1:n_c
        Libc.free(unsafe_load(meta.const_shapes, i))
    end

    # --- Free top-level C arrays ---                       # 14 frees

    # Input arrays (4)
    Libc.free(meta.input_placeholder_ids)
    Libc.free(meta.input_shapes)
    Libc.free(meta.input_shape_ranks)
    Libc.free(meta.input_dtypes)

    # Output arrays (4)
    Libc.free(meta.output_tensor_ids)
    Libc.free(meta.output_shapes)
    Libc.free(meta.output_shape_ranks)
    Libc.free(meta.output_dtypes)

    # Const arrays (5)
    Libc.free(meta.const_placeholder_ids)
    Libc.free(meta.const_mtl_buf_ids)
    Libc.free(meta.const_shapes)
    Libc.free(meta.const_shape_ranks)
    Libc.free(meta.const_dtypes)

    # MLIR text (1)
    Libc.free(meta.mlir_text_ptr)

    # --- Free the struct itself ---                        # 1 free
    Libc.free(Ptr{Cvoid}(meta_ptr))

    # Total frees: 15 + n_in + n_out + n_c (matches freeze_executable malloc count)
    # Plus: n_c Metal.free calls (matches Metal.alloc count)

    return C_NULL
end

# ============================================================
# Execute callback
# ============================================================

# PJRT_LoadedExecutable_Execute_Args:
#   executable* — Ptr{MetalExecutableData} (Libc.malloc'd C struct)
#   argument_lists*** — [num_devices][num_args] MTLTensor handles (ObjC ids)
#   output_lists*** — [num_devices][num_outputs] MTLTensor handles to fill
#   device_complete_events** — completion events to fill
function _loaded_exec_execute(
    args::Ptr{CAPI.PJRT_LoadedExecutable_Execute_Args}
)::Ptr{Cvoid}
    exec_handle = Reactant.unsafe_load_field(args, Val{:executable}())
    num_args = Int(Reactant.unsafe_load_field(args, Val{:num_args}()))

    arglist_outer = Reactant.unsafe_load_field(args, Val{:argument_lists}())
    device0_arglist = unsafe_load(Ptr{Ptr{Cvoid}}(arglist_outer))

    # Load MetalExecutableData from C struct
    meta = unsafe_load(Ptr{MetalExecutableData}(exec_handle))

    ObjC_id = Metal.MTL.ObjectiveC.id

    # Reconstruct MPSGraph from retained ObjC id
    graph = Metal.MPSGraphs.MPSGraphInstance(
        ObjC_id{Metal.MPSGraphs.MPSGraph}(meta.graph_id)
    )

    # Reconstruct input MTLBuffers from PJRT argument handles (MTLTensor ObjC ids)
    input_bufs = Metal.MTL.MTLBuffer[]
    for i in 0:(num_args - 1)
        buf_handle = unsafe_load(Ptr{Ptr{Cvoid}}(device0_arglist + 8 * i))
        tensor = MTLTensor(id{MTLTensor}(UInt64(buf_handle)))
        push!(input_bufs, tensor.buffer)
    end

    # Build feeds dictionary from C struct metadata
    feeds = Dict{Metal.MPSGraphs.MPSGraphTensor,Metal.MPSGraphs.MPSGraphTensorData}()
    for k in 1:Int(meta.n_inputs)
        # Read shape from C arrays (stored as IR/MLIR order)
        rank = Int(unsafe_load(meta.input_shape_ranks, k))
        ir_shape = [Int(unsafe_load(unsafe_load(meta.input_shapes, k), j)) for j in 1:rank]
        input_dtype = pjrt_type_to_julia(unsafe_load(meta.input_dtypes, k))
        mps_dtype = julia_to_mps_dtype(input_dtype)
        mps_shape = convert(Metal.MPS.MPSShape, ir_shape)
        ph = Metal.MPSGraphs.MPSGraphTensorInstance(
            ObjC_id{Metal.MPSGraphs.MPSGraphTensor}(
                unsafe_load(meta.input_placeholder_ids, k)
            ),
        )
        feeds[ph] = Metal.MPSGraphs.MPSGraphTensorData(input_bufs[k], mps_shape, mps_dtype)
    end

    # Add constant feeds (from frozen MTLBuffers + stored shape/dtype)
    # Const shapes are stored as Julia shapes; MPSGraphTensorData needs IR order (reversed for 2D+)
    for i in 1:Int(meta.n_consts)
        ph = Metal.MPSGraphs.MPSGraphTensorInstance(
            ObjC_id{Metal.MPSGraphs.MPSGraphTensor}(
                unsafe_load(meta.const_placeholder_ids, i)
            ),
        )
        buf_id = unsafe_load(meta.const_mtl_buf_ids, i)
        mtl_buf = Metal.MTL.MTLBufferInstance(id{Metal.MTL.MTLBuffer}(buf_id))
        rank = Int(unsafe_load(meta.const_shape_ranks, i))
        julia_shape = [
            Int(unsafe_load(unsafe_load(meta.const_shapes, i), j)) for j in 1:rank
        ]
        # Convert Julia shape → IR shape (same convention as input feeds)
        ir_shape_const = rank >= 2 ? reverse(julia_shape) : julia_shape
        dtype = pjrt_type_to_julia(unsafe_load(meta.const_dtypes, i))
        mps_dtype = julia_to_mps_dtype(dtype)
        mps_shape = convert(Metal.MPS.MPSShape, ir_shape_const)
        feeds[ph] = Metal.MPSGraphs.MPSGraphTensorData(mtl_buf, mps_shape, mps_dtype)
    end

    # Reconstruct output tensors from retained ObjC ids
    n_out = Int(meta.n_outputs)
    output_tensors = [
        Metal.MPSGraphs.MPSGraphTensorInstance(
            ObjC_id{Metal.MPSGraphs.MPSGraphTensor}(unsafe_load(meta.output_tensor_ids, i))
        ) for i in 1:n_out
    ]

    # Run MPSGraph
    feeds_ns = NSDictionary(feeds)
    targets_ns = NSArray(output_tensors)
    dev = Metal.device()
    queue = Metal.global_queue(dev)
    results = Metal.MPSGraphs.run(graph, queue, feeds_ns, targets_ns)
    Metal.synchronize()

    # Extract output data from MPSGraph results into MTLTensor-backed buffer handles
    outlist_outer = Reactant.unsafe_load_field(args, Val{:output_lists}())
    device0_outlist = unsafe_load(Ptr{Ptr{Cvoid}}(outlist_outer))

    for (i, output_tensor) in enumerate(output_tensors)
        result_ptr = results[output_tensor]
        if result_ptr != nil
            result_data = reinterpret(Metal.MPSGraphs.MPSGraphTensorData, result_ptr)
            ndarray = Metal.MPS.MPSNDArray(result_data)

            # Read output shape/dtype from C struct
            out_rank = Int(unsafe_load(meta.output_shape_ranks, i))
            out_shape = [
                Int(unsafe_load(unsafe_load(meta.output_shapes, i), j)) for j in 1:out_rank
            ]
            out_dtype = pjrt_type_to_julia(unsafe_load(meta.output_dtypes, i))
            mps_dtype_jl = julia_to_mps_dtype(out_dtype)
            out_pjrt_type = julia_type_to_pjrt(mps_dtype_jl)
            out_nbytes = max(1, prod(out_shape) * sizeof(mps_dtype_jl))

            out_buf = Metal.alloc(dev, out_nbytes; storage=Metal.SharedStorage)
            cmdbuf = Metal.MTL.MTLCommandBuffer(queue) do cmdbuf
                Metal.MPS.exportDataWithCommandBuffer(
                    ndarray, cmdbuf, out_buf, mps_dtype_jl, UInt(0)
                )
            end
            Metal.MTL.wait_completed(cmdbuf)

            # Wrap output buffer in MTLTensor + cache dims for PJRT queries
            out_dims = Int64.(out_shape)
            tensor = _wrap_buffer_as_tensor(dev, out_buf, out_dims, out_pjrt_type)
            out_handle = UInt64(pointer(tensor))
            _BUFFER_DIMS_CACHE[out_handle] = out_dims
            unsafe_store!(
                Ptr{Ptr{Cvoid}}(device0_outlist + 8 * (i - 1)), Ptr{Cvoid}(out_handle)
            )
        end
    end

    # Write device_complete_events (immediately ready)
    events_arr = Reactant.unsafe_load_field(args, Val{:device_complete_events}())
    if events_arr != C_NULL
        unsafe_store!(Ptr{Ptr{Cvoid}}(events_arr), READY_EVENT_HANDLE)
    end

    return C_NULL
end
