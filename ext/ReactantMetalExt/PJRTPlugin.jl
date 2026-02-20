# PJRTPlugin.jl — Phase 1 PJRT callbacks for Metal backend
# METAL-420: Provides 30 @cfunction callbacks that implement the PJRT_Api
# for PjRtCApiClient initialization (plugin load + client discovery).
#
# Design:
#   - All handles use Libc.malloc (C memory, stable across GC)
#   - All @cfunction pointers are module-level consts (stable LLVM stubs)
#   - PJRT_Api struct stored in C memory via Libc.malloc (stable pointer)
#   - Raw byte offsets used for field writes (no CAPI.jl dependency)
#   - PLATFORM_NAME = "METAL" (uppercase, matches test assertion)
#
# Validated in METAL-402: option-a-prototype.jl proved these callbacks work
# end-to-end with PjRtCApiClient (platform_name="METAL", 1 device).

# ============================================================
# Minimal PJRT_Api struct definitions
# ABI-compatible with CAPI.jl (same field layout, all fn ptrs as Ptr{Cvoid})
# ============================================================

struct MetalPJRT_Api_Version
    struct_size::UInt64        # offset 0,  8 bytes
    extension_start::Ptr{Cvoid} # offset 8,  8 bytes
    major_version::Int32       # offset 16, 4 bytes
    minor_version::Int32       # offset 20, 4 bytes
end  # 24 bytes total (0x18)

struct MetalPJRT_Api
    struct_size::UInt64              # offset 0
    extension_start::Ptr{Cvoid}     # offset 8
    pjrt_api_version::MetalPJRT_Api_Version  # offset 16, 24 bytes
    fns::NTuple{128,Ptr{Cvoid}}     # offset 40, 1024 bytes
end  # Total: 8 + 8 + 24 + 1024 = 1064 bytes (0x428)

# ============================================================
# Opaque handles (C memory — GC-stable, never moved)
# Initialized at runtime in init_pjrt_handles!() to avoid
# dangling pointers from precompilation serialization.
# ============================================================

CLIENT_HANDLE  = Ptr{Cvoid}(0)
DEVICE_HANDLE  = Ptr{Cvoid}(0)
DEVDESC_HANDLE = Ptr{Cvoid}(0)
MEMORY_HANDLE  = Ptr{Cvoid}(0)

DEVICE_PTR_ARRAY = Ptr{Cvoid}(0)
MEMORY_PTR_ARRAY = Ptr{Cvoid}(0)

# Static strings (Julia global consts — stable pointers for the process lifetime)
const PLATFORM_NAME        = "METAL"   # uppercase — test asserts == "METAL"
const PLATFORM_VERSION     = "1.0.0"
const DEVICE_KIND          = "Metal GPU"
const DEVICE_DEBUG_STRING  = "Metal Apple GPU Device"
const DEVICE_TO_STRING     = "Metal:0"
const MEMORY_KIND_STR      = "device"
const MEMORY_DEBUG_STR     = "Metal Device Memory"
const MEMORY_TO_STR        = "Metal:0:device"
const UNIMPL_MESSAGE       = "UNIMPLEMENTED: Operation not supported by Metal PJRT plugin"

# Error handle for UNIMPLEMENTED responses
# PJRT_Client_TopologyDescription is called WITHOUT NULL check — must return non-NULL
UNIMPL_ERROR_HANDLE = Ptr{Cvoid}(0)

# GC-rooted storage for compiled MetalExecutable objects.
# Maps C handle address (UInt64) -> MetalExecutable.
# Using Any to avoid dependency on MLIRWalker.jl (included after this file).
const LOADED_EXECUTABLES = Dict{UInt64, Any}()

# GC-rooted MLIR text for each compiled executable.
# Serialized before MPSGraph compilation so XLA can build HloModules for get_hlo_modules.
const LOADED_EXECUTABLE_MLIR = Dict{UInt64, String}()

# Buffer storage: handle (UInt64) → NamedTuple (data, dims_c, ndims, dtype, nbytes)
# dims_c: C-allocated Int64 array (stable pointer, freed in _buffer_destroy)
const METAL_BUFFERS = Dict{UInt64, Any}()

# Pre-allocated "ready" event handle — event_is_ready always returns true for this
READY_EVENT_HANDLE = Ptr{Cvoid}(0)

# Global lock serializing all PJRT callbacks that access shared Julia state
# (METAL_BUFFERS, LOADED_EXECUTABLES, etc.).
#
# Julia's Dict is NOT thread-safe.  Julia 1.9+ runs finalizers in a dedicated
# finalizer thread, which may call _buffer_destroy (→ delete! METAL_BUFFERS)
# concurrently with the main thread calling _buffer_dimensions / _buffer_to_host
# (→ get METAL_BUFFERS).  The data race corrupts the Dict's internal hash table
# and causes std::bad_alloc in subsequent C++ heap allocations.
#
# Using ReentrantLock (not SpinLock) so that the finalizer thread can block
# without starving other Julia tasks.
const PJRT_LOCK = ReentrantLock()

# Pool for PJRT opaque handles (64-byte mallocs) — avoids malloc/free per call
const _HANDLE_POOL = Ptr{Cvoid}[]
const _HANDLE_POOL_LOCK = ReentrantLock()

function _handle_alloc()
    @lock _HANDLE_POOL_LOCK begin
        if !isempty(_HANDLE_POOL)
            return pop!(_HANDLE_POOL)
        end
    end
    return Libc.malloc(64)
end

function _handle_recycle(ptr::Ptr{Cvoid})
    @lock _HANDLE_POOL_LOCK begin
        push!(_HANDLE_POOL, ptr)
    end
end

# ============================================================
# Generic stubs
# ============================================================

# Returns C_NULL (success, no error). Safe for callbacks whose output fields are not read.
function _stub(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL
end

# Returns non-NULL error handle. Used for PJRT_Client_TopologyDescription
# which is called without NULL-checking the function pointer.
function _unimpl(args::Ptr{Cvoid})::Ptr{Cvoid}
    return UNIMPL_ERROR_HANDLE
end

# No-op deleter for PJRT_LoadedExecutable_GetDeviceAssignment.
# XLA always calls serialized_device_assignment_deleter via absl::Cleanup,
# even when serialized_bytes_size == 0 (early return path).
# We set serialized_device_assignment = NULL, so the deleter just ignores it.
function _noop_free(p::Ptr{Cvoid})::Cvoid
    return nothing
end

# ============================================================
# Phase-1 PJRT Callbacks
#
# All use raw byte offsets — PJRT_*_Args struct layout:
#   offset 0:  struct_size (8 bytes)
#   offset 8:  extension_start (8 bytes)
#   offset 16: first field (input or output, 8 bytes)
#   offset 24: second field (input or output, 8 bytes)
#   ...
#
# Most callbacks: header(16) + 1 input(8) = 24 bytes to first output.
# ============================================================

# --- Error handling ---
function _error_destroy(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL
end

function _error_message(args::Ptr{Cvoid})::Ptr{Cvoid}
    # PJRT_Error_Message_Args: struct_size(8) + extension_start(8) + error*(8) = 24 bytes to message
    # field 4 (offset 24) = message ptr, field 5 (offset 32) = message_size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(UNIMPL_MESSAGE))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(UNIMPL_MESSAGE)))
    return C_NULL
end

function _error_getcode(args::Ptr{Cvoid})::Ptr{Cvoid}
    # PJRT_Error_GetCode_Args: header(16) + error*(8) + code(4-at-24)
    # field 4 (offset 24) = error_code (UInt32)
    unsafe_store!(Ptr{UInt32}(args + 24), UInt32(12))  # PJRT_Error_Code_UNIMPLEMENTED
    return C_NULL
end

# --- Plugin lifecycle ---
function _plugin_initialize(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL
end

function _plugin_attributes(args::Ptr{Cvoid})::Ptr{Cvoid}
    # PJRT_Plugin_Attributes_Args: header(16) + attributes*(8) + num_attributes(8)
    # field 3 (offset 16) = attributes ptr, field 4 (offset 24) = num_attributes
    # Set num_attributes = 0 (no plugin attributes)
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(0))
    return C_NULL
end

# --- Events ---
function _event_destroy(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL
end

function _event_is_ready(args::Ptr{Cvoid})::Ptr{Cvoid}
    # PJRT_Event_IsReady_Args: header(16) + event*(8) + is_ready(bool-at-24)
    # field 4 (offset 24) = is_ready (Bool)
    unsafe_store!(Ptr{Bool}(args + 24), true)
    return C_NULL
end

function _event_await(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL  # immediate: events are always ready
end

# PJRT_Event_OnReady_Args:
#   offset 24: callback (PJRT_Event_OnReadyCallback — INPUT)
#   offset 32: user_arg (Ptr{Cvoid} — INPUT)
# PJRT_Event_OnReadyCallback = void (*)(PJRT_Error* error, void* user_arg)
# Our events are always ready (no error), so call the callback immediately
# with C_NULL error pointer and the provided user_arg.
function _event_on_ready(args::Ptr{Cvoid})::Ptr{Cvoid}
    callback = unsafe_load(Ptr{Ptr{Cvoid}}(args + 24))
    user_arg = unsafe_load(Ptr{Ptr{Cvoid}}(args + 32))
    if callback != C_NULL
        # Two-arg callback: (PJRT_Error* error, void* user_arg)
        ccall(callback, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, user_arg)
    end
    return C_NULL
end

# ============================================================
# Executable metadata callbacks (METAL-421)
# These prevent XLA from reading garbage string/array output fields
# after calling a stub (which leaves output fields uninitialized on stack).
# ============================================================

const EXEC_NAME_STR    = "metal_exec"  # Julia string — pointer is process-stable
const MLIR_FORMAT_STR  = "mlir"        # format identifier for PJRT_Program
const EMPTY_MLIR_MOD   = "module {}"   # minimal valid MLIR module (parsed by XLA for GetHloModules)

# PJRT_Executable_Name_Args: offset 24=name ptr (out), offset 32=name size (out)
function _exec_name(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), Ptr{UInt8}(pointer(EXEC_NAME_STR)))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(EXEC_NAME_STR)))
    return C_NULL
end

# PJRT_Executable_NumReplicas_Args: offset 24=num_replicas (out, Csize_t)
function _exec_num_replicas(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(1))
    return C_NULL
end

# PJRT_Executable_NumPartitions_Args: offset 24=num_partitions (out, Csize_t)
function _exec_num_partitions(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(1))
    return C_NULL
end

# PJRT_Executable_NumOutputs_Args: offset 16=exec* (in), offset 24=num_outputs (out, Csize_t)
function _exec_num_outputs(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    nout = @lock PJRT_LOCK begin
        exec = get(LOADED_EXECUTABLES, UInt64(handle), nothing)
        exec !== nothing ? length(exec.output_dtypes) : 1
    end
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(nout))
    return C_NULL
end

# PJRT_Executable_SizeOfGeneratedCodeInBytes_Args: offset 24=size (out, Int64)
function _exec_code_size(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Int64}(args + 24), Int64(0))
    return C_NULL
end

# PJRT_Executable_OptimizedProgram_Args:
#   offset 16: executable* (8, input)
#   offset 24: program* (8, PJRT_Program in/out)
#
# PJRT_Program:
#   offset 16: code (char*, in/out) — NULL on first call, filled on second
#   offset 24: code_size (Csize_t, out on 1st call)
#   offset 32: format (const char*, out)
#   offset 40: format_size (Csize_t, out)
#
# XLA calls this twice (1st: code=NULL to get size; 2nd: code=buffer to fill).
# We return a minimal "module {}" in "mlir" format so XLA can build an HloModule.
function _exec_optimized_program(args::Ptr{Cvoid})::Ptr{Cvoid}
    exec_handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    mlir_text   = @lock PJRT_LOCK begin
        get(LOADED_EXECUTABLE_MLIR, UInt64(exec_handle), EMPTY_MLIR_MOD)
    end
    prog_ptr    = unsafe_load(Ptr{Ptr{Cvoid}}(args + 24))
    code_ptr    = unsafe_load(Ptr{Ptr{UInt8}}(prog_ptr + 16))
    # Set format on both calls
    unsafe_store!(Ptr{Ptr{UInt8}}(prog_ptr + 32), pointer(MLIR_FORMAT_STR))
    unsafe_store!(Ptr{Csize_t}(prog_ptr + 40), Csize_t(length(MLIR_FORMAT_STR)))
    code_bytes = codeunits(mlir_text)
    code_len   = length(code_bytes)
    if code_ptr == C_NULL
        # First call: report how many bytes the code occupies
        unsafe_store!(Ptr{Csize_t}(prog_ptr + 24), Csize_t(code_len))
    else
        # Second call: XLA has allocated a buffer of code_len bytes; fill it
        GC.@preserve mlir_text begin
            unsafe_copyto!(code_ptr, pointer(code_bytes), code_len)
        end
    end
    return C_NULL
end

# PJRT_Executable_OutputMemoryKinds_Args:
#   offset 24=num_outputs (Csize_t), offset 32=memory_kinds ptr (out), offset 40=kind_sizes ptr (out)
function _exec_output_memory_kinds(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(0))
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 32), C_NULL)
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 40), C_NULL)
    return C_NULL
end

# PJRT_LoadedExecutable_Fingerprint_Args:
#   offset 24=fingerprint ptr (out, Cstring), offset 32=fingerprint_size (out, Csize_t)
function _loaded_exec_fingerprint(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), C_NULL)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(0))
    return C_NULL
end

# PJRT_Executable_OutputElementTypes_Args:
#   offset 16=exec* (in), offset 24=output_types ptr (out, Ptr{UInt32}), offset 32=num_output_types (out, Csize_t)
function _exec_output_element_types(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), C_NULL)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(0))
    return C_NULL
end

# PJRT_Executable_OutputDimensions_Args:
#   offset 16=exec* (in), offset 24=num_outputs (Csize_t out), offset 32=dims ptr (out), offset 40=dim_sizes ptr (out)
function _exec_output_dimensions(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(0))
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 32), C_NULL)
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 40), C_NULL)
    return C_NULL
end

# --- Client lifecycle ---
function _client_create(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 64), CLIENT_HANDLE)
    return C_NULL
end

function _client_destroy(args::Ptr{Cvoid})::Ptr{Cvoid}
    return C_NULL
end

function _client_platform_name(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(PLATFORM_NAME))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(PLATFORM_NAME)))
    return C_NULL
end

function _client_process_index(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = process_index (Cint)
    unsafe_store!(Ptr{Cint}(args + 24), Cint(0))
    return C_NULL
end

function _client_platform_version(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = version ptr, field 5 (offset 32) = size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(PLATFORM_VERSION))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(PLATFORM_VERSION)))
    return C_NULL
end

function _client_devices(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), DEVICE_PTR_ARRAY)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(1))
    return C_NULL
end

function _client_addressable_devices(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = devices ptr, field 5 (offset 32) = count
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), DEVICE_PTR_ARRAY)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(1))
    return C_NULL
end

function _client_addressable_memories(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = memories ptr, field 5 (offset 32) = count
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), MEMORY_PTR_ARRAY)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(1))
    return C_NULL
end

# --- Device operations ---
function _device_get_description(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), DEVDESC_HANDLE)
    return C_NULL
end

function _device_is_addressable(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = is_addressable (Bool)
    unsafe_store!(Ptr{Bool}(args + 24), true)
    return C_NULL
end

function _device_local_hardware_id(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = local_hardware_id (Cint)
    unsafe_store!(Ptr{Cint}(args + 24), Cint(0))
    return C_NULL
end

function _device_addressable_memories(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = memories ptr, field 5 (offset 32) = count
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), MEMORY_PTR_ARRAY)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(1))
    return C_NULL
end

function _device_default_memory(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = memory ptr
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), MEMORY_HANDLE)
    return C_NULL
end

# --- Device description ---
function _devdesc_id(args::Ptr{Cvoid})::Ptr{Cvoid}
    # header(16) + device_description*(8) = offset 24 for id
    # field 4 (offset 24) = id (Cint)
    unsafe_store!(Ptr{Cint}(args + 24), Cint(0))
    return C_NULL
end

function _devdesc_process_index(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = process_index (Cint)
    unsafe_store!(Ptr{Cint}(args + 24), Cint(0))
    return C_NULL
end

function _devdesc_attributes(args::Ptr{Cvoid})::Ptr{Cvoid}
    # PJRT_DeviceDescription_Attributes_Args:
    # header(16) + device_description*(8) = offset 24 for num_attributes
    # field 4 (offset 24) = num_attributes (Csize_t), field 5 (offset 32) = attributes ptr
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(0))
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 32), C_NULL)
    return C_NULL
end

function _devdesc_kind(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = kind ptr, field 5 (offset 32) = kind_size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(DEVICE_KIND))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(DEVICE_KIND)))
    return C_NULL
end

function _devdesc_debug_string(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = debug_string ptr, field 5 (offset 32) = size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(DEVICE_DEBUG_STRING))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(DEVICE_DEBUG_STRING)))
    return C_NULL
end

function _devdesc_to_string(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = to_string ptr, field 5 (offset 32) = size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(DEVICE_TO_STRING))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(DEVICE_TO_STRING)))
    return C_NULL
end

# --- Memory operations ---
function _memory_id(args::Ptr{Cvoid})::Ptr{Cvoid}
    # PJRT_Memory_Id_Args: header(16) + memory*(8) = offset 24 for id (Cint)
    unsafe_store!(Ptr{Cint}(args + 24), Cint(0))
    return C_NULL
end

function _memory_kind(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = kind ptr, field 5 (offset 32) = size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(MEMORY_KIND_STR))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(MEMORY_KIND_STR)))
    return C_NULL
end

function _memory_kind_id(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = kind_id (Cint)
    unsafe_store!(Ptr{Cint}(args + 24), Cint(0))
    return C_NULL
end

function _memory_debug_string(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = debug_string ptr, field 5 (offset 32) = size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(MEMORY_DEBUG_STR))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(MEMORY_DEBUG_STR)))
    return C_NULL
end

function _memory_to_string(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = to_string ptr, field 5 (offset 32) = size
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), pointer(MEMORY_TO_STR))
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(length(MEMORY_TO_STR)))
    return C_NULL
end

function _memory_addressable_by_devices(args::Ptr{Cvoid})::Ptr{Cvoid}
    # field 4 (offset 24) = devices ptr, field 5 (offset 32) = num_devices
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), DEVICE_PTR_ARRAY)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(1))
    return C_NULL
end

# ============================================================
# Phase-2 PJRT Callbacks: compile
#
# METAL-421: Wire PJRT_Client_Compile → compile_mlir_module (MLIRWalker.jl)
#
# PJRT_Client_Compile_Args layout (PJRT_Client_Compile_Args_STRUCT_SIZE = 0x38):
#   offset 0:  struct_size (8)
#   offset 8:  extension_start (8)
#   offset 16: client* (8)
#   offset 24: program* (8) — pointer to PJRT_Program
#   offset 32: compile_options (8)
#   offset 40: compile_options_size (8)
#   offset 48: executable* (output, 8) — write our LoadedExecutable handle here
#
# PJRT_Program layout (PJRT_Program_STRUCT_SIZE = 0x30):
#   offset 0:  struct_size (8)
#   offset 8:  extension_start (8)
#   offset 16: code (Ptr{UInt8}) — MLIR bytecode
#   offset 24: code_size (Csize_t)
#   offset 32: format (Cstring) — "mlir"
#   offset 40: format_size (Csize_t)
# ============================================================

function _client_compile(args::Ptr{Cvoid})::Ptr{Cvoid}
    program_ptr = unsafe_load(Ptr{Ptr{Cvoid}}(args + 24))
    code_ptr  = unsafe_load(Ptr{Ptr{UInt8}}(program_ptr + 16))
    code_size = unsafe_load(Ptr{Csize_t}(program_ptr + 24))

    # Deserialize the StableHLO portable artifact sent by XLA's PjRtCApiClient.
    # XLA serializes the MLIR module via SerializeUsingNativeBytecode, which
    # produces a portable artifact using VHLO dialect internally.
    # mlirModuleCreateParse fails with "dialect 'vhlo' is unknown" because VHLO
    # is not in Reactant's registry.  stablehloDeserializePortableArtifactNoError
    # handles VHLO internally (parses VHLO, upgrades to StableHLO) and returns
    # a StableHLO module in our context.
    exec, mlir_text = IR.with_context() do _ctx
        str_ref = API.MlirStringRef(Ptr{Cchar}(code_ptr), Csize_t(code_size))
        mlir_mod_ref = @ccall MLIR.API.mlir_c.stablehloDeserializePortableArtifactNoError(
            str_ref::API.MlirStringRef,
            _ctx::API.MlirContext
        )::API.MlirModule
        mod = IR.Module(mlir_mod_ref)
        # Serialize MLIR text BEFORE compile_mlir_module modifies/consumes the module.
        # Stored for _exec_optimized_program (called by XLA's get_hlo_modules).
        io = IOBuffer()
        show(io, mod)
        mlir_str = String(take!(io))
        exec_inner = compile_mlir_module(mod)
        (exec_inner, mlir_str)
    end

    @debug "Metal PJRT compile: op_count = $(exec.num_ops), nout = $(length(exec.output_dtypes))"

    # Pre-populate output MtlArray pool so first execute! calls avoid GPU allocation
    for (shape, dtype) in zip(exec.output_shapes, exec.output_dtypes)
        for _ in 1:3
            pool_return!(MtlArray{dtype}(undef, shape...))
        end
    end

    # Allocate opaque C handle and GC-root the executable in LOADED_EXECUTABLES.
    handle = Libc.malloc(64)
    @lock PJRT_LOCK begin
        LOADED_EXECUTABLES[UInt64(handle)] = exec
        LOADED_EXECUTABLE_MLIR[UInt64(handle)] = mlir_text
    end

    # Write handle to output field (args + 48 = executable* out).
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 48), handle)

    return C_NULL
end

# PJRT_LoadedExecutable_Destroy_Args:
#   offset 16: executable* (input, 8) — our handle
function _loaded_exec_destroy(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    @lock PJRT_LOCK begin
        delete!(LOADED_EXECUTABLES, UInt64(handle))
        delete!(LOADED_EXECUTABLE_MLIR, UInt64(handle))
    end
    Libc.free(handle)
    return C_NULL
end

# PJRT_LoadedExecutable_GetExecutable_Args:
#   offset 16: loaded_executable* (input, 8) — our handle
#   offset 24: executable* (output, 8) — return same handle as Executable
function _loaded_exec_get_executable(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), handle)
    return C_NULL
end

# PJRT_LoadedExecutable_AddressableDevices_Args:
#   offset 24: devices** (output) — array of device pointers
#   offset 32: num_devices (output)
function _loaded_exec_addressable_devices(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), DEVICE_PTR_ARRAY)
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(1))
    return C_NULL
end

# PJRT_LoadedExecutable_GetDeviceAssignment_Args:
#   offset 0:  struct_size (8)
#   offset 8:  extension_start (8)
#   offset 16: executable* (8, input)
#   offset 24: serialized_bytes (char*, 8, output)
#   offset 32: serialized_bytes_size (Csize_t, 8, output)
#   offset 40: serialized_device_assignment (void*, 8, output)
#   offset 48: serialized_device_assignment_deleter (fn ptr, 8, output)
#
# Strategy: set serialized_bytes_size = 0, which signals to XLA that this
# is a "portable" executable with no device assignment (device_assignment_ = nullptr).
# XLA still calls the deleter via absl::Cleanup on all exit paths, so
# serialized_device_assignment_deleter MUST be a valid non-NULL function pointer.
# We use cfn_noop_free (initialized before cfn_loaded_exec_get_device_assignment).
function _loaded_exec_get_device_assignment(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{UInt8}}(args + 24), C_NULL)           # serialized_bytes = NULL
    unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(0))          # serialized_bytes_size = 0
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 40), C_NULL)           # serialized_device_assignment = NULL
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 48), cfn_noop_free)    # deleter (non-NULL, handles NULL arg)
    return C_NULL
end

# ============================================================
# Phase-3 PJRT Callbacks: buffers + execute
#
# METAL-421: Minimal buffer support to allow to_rarray + @jit to work end-to-end.
#
# Helper: PJRT_Buffer_Type enum value → Julia element type
# PJRT_Buffer_Type::UInt32: PRED=1,S8=2,S16=3,S32=4,S64=5,U8=6,F16=10,F32=11,F64=22
# ============================================================

function pjrt_type_to_julia(t::UInt32)
    t == 11 ? Float32 :
    t == 22 ? Float64 :
    t == 10 ? Float16 :
    t == 4  ? Int32   :
    t == 5  ? Int64   : Float32
end

function julia_type_to_pjrt(T)
    T == Float32 ? UInt32(11) :
    T == Float64 ? UInt32(22) :
    T == Float16 ? UInt32(10) :
    T == Int32   ? UInt32(4)  :
    T == Int64   ? UInt32(5)  : UInt32(11)
end

# PJRT_Client_BufferFromHostBuffer_Args layout (0x78 = 120 bytes):
#   offset 24: data*         — host data pointer
#   offset 32: type (UInt32) — PJRT_Buffer_Type
#   offset 40: dims*         — Int64 array of dimension sizes
#   offset 48: num_dims      — Csize_t
#   offset 104: done_with_host_buffer* — event output (write C_NULL = immediate)
#   offset 112: buffer*      — OUTPUT: our buffer handle
function _client_buffer_from_host(args::Ptr{Cvoid})::Ptr{Cvoid}
    data_ptr  = unsafe_load(Ptr{Ptr{UInt8}}(args + 24))
    type_val  = unsafe_load(Ptr{UInt32}(args + 32))
    dims_ptr  = unsafe_load(Ptr{Ptr{Int64}}(args + 40))
    num_dims  = Int(unsafe_load(Ptr{Csize_t}(args + 48)))

    dims = [unsafe_load(dims_ptr, i) for i in 1:num_dims]
    julia_dtype = pjrt_type_to_julia(type_val)
    n_elems = num_dims > 0 ? Int(prod(dims)) : 1
    nbytes = n_elems * sizeof(julia_dtype)

    # Copy host data to GPU immediately — data lives on Metal device
    src = unsafe_wrap(Array, Ptr{julia_dtype}(data_ptr), n_elems; own=false)
    data_gpu = MtlArray(copy(src))

    # Allocate stable C array for dims (returned by PJRT_Buffer_Dimensions)
    dims_c = Libc.malloc(max(1, num_dims) * 8)
    for i in 1:num_dims
        unsafe_store!(Ptr{Int64}(dims_c + 8*(i-1)), dims[i])
    end

    handle = _handle_alloc()
    @lock PJRT_LOCK begin
        METAL_BUFFERS[UInt64(handle)] = (data=data_gpu, dims_c=dims_c, ndims=num_dims,
                                         dtype=type_val, nbytes=nbytes)
    end

    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 104), C_NULL)    # done_with_host_buffer
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 112), handle)    # buffer out
    return C_NULL
end

# PJRT_Buffer_Destroy_Args: offset 16 = buffer handle to destroy
function _buffer_destroy(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    dims_c_to_free = Ptr{Cvoid}(0)
    buf_data = nothing
    @lock PJRT_LOCK begin
        buf = get(METAL_BUFFERS, UInt64(handle), nothing)
        if buf !== nothing
            dims_c_to_free = buf.dims_c
            buf_data = buf.data
            delete!(METAL_BUFFERS, UInt64(handle))
        end
    end
    # Return MtlArray to pool for reuse (avoids GPU allocation on next execute)
    if buf_data isa MtlArray
        pool_return!(buf_data)
    end
    # Free C memory outside the lock (Libc.free is thread-safe)
    dims_c_to_free != Ptr{Cvoid}(0) && Libc.free(dims_c_to_free)
    _handle_recycle(handle)
    return C_NULL
end

# PJRT_Buffer_ElementType_Args: offset 16 = buffer, offset 24 = type (UInt32) out
function _buffer_element_type(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    dtype = @lock PJRT_LOCK begin
        buf = get(METAL_BUFFERS, UInt64(handle), nothing)
        (buf !== nothing) ? buf.dtype : UInt32(11)
    end
    unsafe_store!(Ptr{UInt32}(args + 24), dtype)
    return C_NULL
end

# PJRT_Buffer_Dimensions_Args (0x28): offset 24=dims* out, offset 32=num_dims out
function _buffer_dimensions(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    @lock PJRT_LOCK begin
        buf = get(METAL_BUFFERS, UInt64(handle), nothing)
        if buf !== nothing
            unsafe_store!(Ptr{Ptr{Int64}}(args + 24), Ptr{Int64}(buf.dims_c))
            unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(buf.ndims))
        else
            unsafe_store!(Ptr{Ptr{Int64}}(args + 24), Ptr{Int64}(C_NULL))
            unsafe_store!(Ptr{Csize_t}(args + 32), Csize_t(0))
        end
    end
    return C_NULL
end

# PJRT_Buffer_OnDeviceSizeInBytes_Args (0x20): offset 24=size out
function _buffer_on_device_size(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    nbytes = @lock PJRT_LOCK begin
        buf = get(METAL_BUFFERS, UInt64(handle), nothing)
        (buf !== nothing) ? buf.nbytes : 0
    end
    unsafe_store!(Ptr{Csize_t}(args + 24), Csize_t(nbytes))
    return C_NULL
end

# PJRT_Buffer_Device_Args (0x20): offset 24 = device* out — always our Metal device
function _buffer_device(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), DEVICE_HANDLE)
    return C_NULL
end

# PJRT_Buffer_Memory_Args (0x20): offset 24 = memory* out — our Metal memory space
function _buffer_memory(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), MEMORY_HANDLE)
    return C_NULL
end

# PJRT_Buffer_IsOnCpu_Args (0x19): offset 24 = is_on_cpu (Bool) — always false (Metal GPU)
function _buffer_is_on_cpu(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Bool}(args + 24), false)
    return C_NULL
end

# PJRT_Buffer_ReadyEvent_Args (0x20): offset 24 = event* out
function _buffer_ready_event(args::Ptr{Cvoid})::Ptr{Cvoid}
    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 24), READY_EVENT_HANDLE)
    return C_NULL
end

# PJRT_Buffer_ToHostBuffer_Args (0x38):
#   offset 16: src buffer handle
#   offset 32: dst* — pre-allocated host memory
#   offset 40: dst_size (Csize_t)
#   offset 48: event* out
function _buffer_to_host(args::Ptr{Cvoid})::Ptr{Cvoid}
    handle   = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    dst_ptr  = unsafe_load(Ptr{Ptr{UInt8}}(args + 32))
    dst_size = Int(unsafe_load(Ptr{Csize_t}(args + 40)))

    buf = @lock PJRT_LOCK begin
        get(METAL_BUFFERS, UInt64(handle), nothing)
    end
    if buf !== nothing && dst_ptr != C_NULL
        src_data = buf.data
        # Download from GPU if needed (MtlArray → Array), otherwise use CPU data directly
        cpu_data = src_data isa MtlArray ? Array(src_data) : src_data
        nbytes = min(dst_size, length(cpu_data) * sizeof(eltype(cpu_data)))
        GC.@preserve cpu_data begin
            unsafe_copyto!(dst_ptr, Ptr{UInt8}(pointer(cpu_data)), nbytes)
        end
    end

    unsafe_store!(Ptr{Ptr{Cvoid}}(args + 48), READY_EVENT_HANDLE)
    return C_NULL
end

# PJRT_LoadedExecutable_Execute_Args (0x50 = 80 bytes):
#   offset 16: executable* — our loaded exec handle
#   offset 24: options*
#   offset 32: argument_lists*** — [num_devices][num_args] buffer handles
#   offset 40: num_devices (Csize_t)
#   offset 48: num_args (Csize_t)
#   offset 56: output_lists*** — [num_devices][num_outputs] to fill
#   offset 64: device_complete_events** — array of completion events to fill
#   offset 72: execute_device*
function _loaded_exec_execute(args::Ptr{Cvoid})::Ptr{Cvoid}
    exec_handle = unsafe_load(Ptr{Ptr{Cvoid}}(args + 16))
    num_args = Int(unsafe_load(Ptr{Csize_t}(args + 48)))

    # argument_lists: Ptr to array of [device][arg] buffer pointers
    # For device 0: arglist_ptr[0] is the array of buffer handles
    arglist_outer = unsafe_load(Ptr{Ptr{Cvoid}}(args + 32))   # Ptr → device0_arglist
    device0_arglist = unsafe_load(Ptr{Ptr{Cvoid}}(arglist_outer))  # Ptr → [buf0, buf1, ...]

    # Single lock acquisition for executable + all input buffers
    exec_and_inputs = @lock PJRT_LOCK begin
        e = get(LOADED_EXECUTABLES, UInt64(exec_handle), nothing)
        if e === nothing
            nothing
        else
            arrs = AbstractArray[]
            for i in 0:(num_args - 1)
                buf_handle = unsafe_load(Ptr{Ptr{Cvoid}}(device0_arglist + 8*i))
                buf = get(METAL_BUFFERS, UInt64(buf_handle), nothing)
                if buf !== nothing
                    push!(arrs, buf.data)
                end
            end
            (e, arrs)
        end
    end
    exec_and_inputs === nothing && return C_NULL
    exec, input_arrays = exec_and_inputs

    # execute! on Metal GPU (no try/catch — errors propagate)
    output_arrays = execute!(exec, input_arrays)

    # Store outputs in METAL_BUFFERS and write handles to output_lists
    outlist_outer = unsafe_load(Ptr{Ptr{Cvoid}}(args + 56))    # Ptr → device0_outlist
    device0_outlist = unsafe_load(Ptr{Ptr{Cvoid}}(outlist_outer))  # Ptr → [out0, out1, ...]

    @lock PJRT_LOCK begin
        for (i, out_arr) in enumerate(output_arrays)
            out_handle = _handle_alloc()
            out_dtype  = julia_type_to_pjrt(eltype(out_arr))
            out_dims   = collect(Int64, size(out_arr))
            out_dims_c = Libc.malloc(max(1, length(out_dims)) * 8)
            for j in 1:length(out_dims)
                unsafe_store!(Ptr{Int64}(out_dims_c + 8*(j-1)), out_dims[j])
            end
            METAL_BUFFERS[UInt64(out_handle)] = (data=out_arr, dims_c=out_dims_c,
                                                  ndims=length(out_dims), dtype=out_dtype,
                                                  nbytes=length(out_arr)*sizeof(eltype(out_arr)))
            unsafe_store!(Ptr{Ptr{Cvoid}}(device0_outlist + 8*(i-1)), out_handle)
        end
    end

    # Write device_complete_events (immediately ready)
    events_arr = unsafe_load(Ptr{Ptr{Cvoid}}(args + 64))
    if events_arr != C_NULL
        unsafe_store!(Ptr{Ptr{Cvoid}}(events_arr), READY_EVENT_HANDLE)
    end

    return C_NULL
end

# ============================================================
# @cfunction pointers — initialized at runtime in init_pjrt_handles!()
# @cfunction returns Ptr{Cvoid} which is just a number to Julia's serializer.
# Precompilation serializes the ADDRESS, not the trampoline code.
# Must be re-created at runtime.
# ============================================================

cfn_stub              = Ptr{Cvoid}(0)
cfn_unimpl            = Ptr{Cvoid}(0)
cfn_client_compile          = Ptr{Cvoid}(0)
cfn_loaded_exec_destroy     = Ptr{Cvoid}(0)
cfn_loaded_exec_get_exec    = Ptr{Cvoid}(0)
cfn_loaded_exec_addr_devs   = Ptr{Cvoid}(0)
cfn_loaded_exec_execute     = Ptr{Cvoid}(0)
cfn_client_buffer_from_host = Ptr{Cvoid}(0)
cfn_buffer_destroy          = Ptr{Cvoid}(0)
cfn_buffer_element_type     = Ptr{Cvoid}(0)
cfn_buffer_dimensions       = Ptr{Cvoid}(0)
cfn_buffer_on_device_size   = Ptr{Cvoid}(0)
cfn_buffer_device           = Ptr{Cvoid}(0)
cfn_buffer_memory           = Ptr{Cvoid}(0)
cfn_buffer_is_on_cpu        = Ptr{Cvoid}(0)
cfn_buffer_ready_event      = Ptr{Cvoid}(0)
cfn_buffer_to_host          = Ptr{Cvoid}(0)
cfn_error_destroy     = Ptr{Cvoid}(0)
cfn_error_message     = Ptr{Cvoid}(0)
cfn_error_getcode     = Ptr{Cvoid}(0)
cfn_plugin_initialize = Ptr{Cvoid}(0)
cfn_plugin_attributes = Ptr{Cvoid}(0)
cfn_event_destroy     = Ptr{Cvoid}(0)
cfn_event_is_ready    = Ptr{Cvoid}(0)
cfn_event_await       = Ptr{Cvoid}(0)
cfn_client_create     = Ptr{Cvoid}(0)
cfn_client_destroy    = Ptr{Cvoid}(0)
cfn_client_platform_name    = Ptr{Cvoid}(0)
cfn_client_process_index    = Ptr{Cvoid}(0)
cfn_client_platform_version = Ptr{Cvoid}(0)
cfn_client_devices          = Ptr{Cvoid}(0)
cfn_client_addr_devices     = Ptr{Cvoid}(0)
cfn_client_addr_memories    = Ptr{Cvoid}(0)
cfn_device_get_desc         = Ptr{Cvoid}(0)
cfn_device_is_addressable   = Ptr{Cvoid}(0)
cfn_device_local_hw_id      = Ptr{Cvoid}(0)
cfn_device_addr_memories    = Ptr{Cvoid}(0)
cfn_device_default_memory   = Ptr{Cvoid}(0)
cfn_devdesc_id              = Ptr{Cvoid}(0)
cfn_devdesc_process_index   = Ptr{Cvoid}(0)
cfn_devdesc_attributes      = Ptr{Cvoid}(0)
cfn_devdesc_kind            = Ptr{Cvoid}(0)
cfn_devdesc_debug_string    = Ptr{Cvoid}(0)
cfn_devdesc_to_string       = Ptr{Cvoid}(0)
cfn_memory_id               = Ptr{Cvoid}(0)
cfn_memory_kind             = Ptr{Cvoid}(0)
cfn_memory_kind_id          = Ptr{Cvoid}(0)
cfn_memory_debug_string     = Ptr{Cvoid}(0)
cfn_memory_to_string        = Ptr{Cvoid}(0)
cfn_memory_addr_by_devices  = Ptr{Cvoid}(0)
cfn_event_on_ready          = Ptr{Cvoid}(0)
cfn_exec_name               = Ptr{Cvoid}(0)
cfn_exec_num_replicas       = Ptr{Cvoid}(0)
cfn_exec_num_partitions     = Ptr{Cvoid}(0)
cfn_exec_num_outputs        = Ptr{Cvoid}(0)
cfn_exec_code_size           = Ptr{Cvoid}(0)
cfn_exec_optimized_program   = Ptr{Cvoid}(0)
cfn_exec_output_memory_kinds = Ptr{Cvoid}(0)
cfn_loaded_exec_fingerprint  = Ptr{Cvoid}(0)
cfn_exec_output_element_types = Ptr{Cvoid}(0)
cfn_exec_output_dimensions   = Ptr{Cvoid}(0)
cfn_noop_free                          = Ptr{Cvoid}(0)
cfn_loaded_exec_get_device_assignment  = Ptr{Cvoid}(0)

# ============================================================
# PJRT_Api struct (128 function pointers in flat NTuple)
#
# Function pointer field mapping (1-based index into fns NTuple):
#  [1]  PJRT_Error_Destroy
#  [2]  PJRT_Error_Message
#  [3]  PJRT_Error_GetCode
#  [4]  PJRT_Plugin_Initialize
#  [5]  PJRT_Plugin_Attributes
#  [6]  PJRT_Event_Destroy
#  [7]  PJRT_Event_IsReady
#  [8]  PJRT_Event_Error       (stub)
#  [9]  PJRT_Event_Await
#  [10] PJRT_Event_OnReady     (stub)
#  [11] PJRT_Client_Create
#  [12] PJRT_Client_Destroy
#  [13] PJRT_Client_PlatformName
#  [14] PJRT_Client_ProcessIndex
#  [15] PJRT_Client_PlatformVersion
#  [16] PJRT_Client_Devices
#  [17] PJRT_Client_AddressableDevices
#  [18] PJRT_Client_LookupDevice        (stub)
#  [19] PJRT_Client_LookupAddressableDevice (stub)
#  [20] PJRT_Client_AddressableMemories
#  [21] PJRT_Client_Compile             (METAL-421)
#  [22] PJRT_Client_DefaultDeviceAssignment (stub)
#  [23] PJRT_Client_BufferFromHostBuffer (stub, Phase 3)
#  [24] PJRT_DeviceDescription_Id
#  [25] PJRT_DeviceDescription_ProcessIndex
#  [26] PJRT_DeviceDescription_Attributes
#  [27] PJRT_DeviceDescription_Kind
#  [28] PJRT_DeviceDescription_DebugString
#  [29] PJRT_DeviceDescription_ToString
#  [30] PJRT_Device_GetDescription
#  [31] PJRT_Device_IsAddressable
#  [32] PJRT_Device_LocalHardwareId
#  [33] PJRT_Device_AddressableMemories
#  [34] PJRT_Device_DefaultMemory
#  [35] PJRT_Device_MemoryStats         (stub)
#  [36] PJRT_Memory_Id
#  [37] PJRT_Memory_Kind
#  [38] PJRT_Memory_DebugString
#  [39] PJRT_Memory_ToString
#  [40] PJRT_Memory_AddressableByDevices
#  [41] PJRT_Executable_Destroy         (stub)
#  [42]-[50]  stubs (Executable_Name, NumReplicas, NumPartitions, NumOutputs, ...)
#  [51] PJRT_LoadedExecutable_Destroy   (METAL-421)
#  [52] PJRT_LoadedExecutable_GetExecutable (METAL-421)
#  [53] PJRT_LoadedExecutable_AddressableDevices (METAL-421)
#  [54]-[55]  stubs (Delete, IsDeleted)
#  [56] PJRT_LoadedExecutable_Execute   (stub, METAL-422)
#  [57]-[95]  stubs
#  [96] PJRT_Client_TopologyDescription  (unimpl — returns error, called w/o NULL check)
#  [97] stub
#  [98] PJRT_Memory_Kind_Id
#  [99]-[128] stubs
# ============================================================

# Pointer to C-managed PJRT_Api struct (initialized in init_pjrt_handles!())
_PJRT_API_MEM = Ptr{Cvoid}(0)

"""
    init_pjrt_handles!()

Allocate C memory for opaque handles and the PJRT_Api struct.
Must be called at runtime (from __init__), NOT at precompile time,
because Libc.malloc pointers become dangling after .ji deserialization.
"""
function init_pjrt_handles!()
    global CLIENT_HANDLE  = Libc.malloc(64)
    global DEVICE_HANDLE  = Libc.malloc(64)
    global DEVDESC_HANDLE = Libc.malloc(64)
    global MEMORY_HANDLE  = Libc.malloc(64)

    unsafe_store!(Ptr{Int64}(CLIENT_HANDLE),  Int64(0xDEADBEEF))
    unsafe_store!(Ptr{Int64}(DEVICE_HANDLE),  Int64(0xCAFEBABE))
    unsafe_store!(Ptr{Int64}(DEVDESC_HANDLE), Int64(0xF00DCAFE))
    unsafe_store!(Ptr{Int64}(MEMORY_HANDLE),  Int64(0xFEEDFACE))

    global DEVICE_PTR_ARRAY = Libc.malloc(8)
    unsafe_store!(Ptr{Ptr{Cvoid}}(DEVICE_PTR_ARRAY), DEVICE_HANDLE)

    global MEMORY_PTR_ARRAY = Libc.malloc(8)
    unsafe_store!(Ptr{Ptr{Cvoid}}(MEMORY_PTR_ARRAY), MEMORY_HANDLE)

    global UNIMPL_ERROR_HANDLE = Libc.malloc(8)
    unsafe_store!(Ptr{Int64}(UNIMPL_ERROR_HANDLE), Int64(0xDEAD))

    global READY_EVENT_HANDLE = Libc.malloc(8)
    unsafe_store!(Ptr{Int64}(READY_EVENT_HANDLE), Int64(0x1234CAFE))

    # Create @cfunction trampolines at runtime (addresses are process-specific).
    global cfn_stub              = @cfunction(_stub,              Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_unimpl            = @cfunction(_unimpl,            Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_error_destroy     = @cfunction(_error_destroy,     Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_error_message     = @cfunction(_error_message,     Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_error_getcode     = @cfunction(_error_getcode,     Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_plugin_initialize = @cfunction(_plugin_initialize, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_plugin_attributes = @cfunction(_plugin_attributes, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_event_destroy     = @cfunction(_event_destroy,     Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_event_is_ready    = @cfunction(_event_is_ready,    Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_event_await       = @cfunction(_event_await,       Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_create     = @cfunction(_client_create,     Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_destroy    = @cfunction(_client_destroy,    Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_platform_name    = @cfunction(_client_platform_name,    Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_process_index    = @cfunction(_client_process_index,    Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_platform_version = @cfunction(_client_platform_version, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_devices          = @cfunction(_client_devices,          Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_addr_devices     = @cfunction(_client_addressable_devices, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_addr_memories    = @cfunction(_client_addressable_memories, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_device_get_desc         = @cfunction(_device_get_description,  Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_device_is_addressable   = @cfunction(_device_is_addressable,   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_device_local_hw_id      = @cfunction(_device_local_hardware_id, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_device_addr_memories    = @cfunction(_device_addressable_memories, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_device_default_memory   = @cfunction(_device_default_memory,   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_devdesc_id              = @cfunction(_devdesc_id,              Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_devdesc_process_index   = @cfunction(_devdesc_process_index,   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_devdesc_attributes      = @cfunction(_devdesc_attributes,      Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_devdesc_kind            = @cfunction(_devdesc_kind,            Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_devdesc_debug_string    = @cfunction(_devdesc_debug_string,    Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_devdesc_to_string       = @cfunction(_devdesc_to_string,       Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_memory_id               = @cfunction(_memory_id,              Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_memory_kind             = @cfunction(_memory_kind,            Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_memory_kind_id          = @cfunction(_memory_kind_id,         Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_memory_debug_string     = @cfunction(_memory_debug_string,    Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_memory_to_string        = @cfunction(_memory_to_string,       Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_memory_addr_by_devices  = @cfunction(_memory_addressable_by_devices, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_compile          = @cfunction(_client_compile,               Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_loaded_exec_destroy     = @cfunction(_loaded_exec_destroy,          Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_loaded_exec_get_exec    = @cfunction(_loaded_exec_get_executable,   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_loaded_exec_addr_devs   = @cfunction(_loaded_exec_addressable_devices, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_loaded_exec_execute     = @cfunction(_loaded_exec_execute,             Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_client_buffer_from_host = @cfunction(_client_buffer_from_host,         Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_destroy          = @cfunction(_buffer_destroy,                  Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_element_type     = @cfunction(_buffer_element_type,             Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_dimensions       = @cfunction(_buffer_dimensions,               Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_on_device_size   = @cfunction(_buffer_on_device_size,           Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_device           = @cfunction(_buffer_device,                   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_memory           = @cfunction(_buffer_memory,                   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_is_on_cpu        = @cfunction(_buffer_is_on_cpu,                Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_ready_event      = @cfunction(_buffer_ready_event,              Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_buffer_to_host          = @cfunction(_buffer_to_host,                  Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_event_on_ready           = @cfunction(_event_on_ready,           Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_name                = @cfunction(_exec_name,                Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_num_replicas        = @cfunction(_exec_num_replicas,        Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_num_partitions      = @cfunction(_exec_num_partitions,      Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_num_outputs         = @cfunction(_exec_num_outputs,         Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_code_size           = @cfunction(_exec_code_size,           Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_optimized_program   = @cfunction(_exec_optimized_program,   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_output_memory_kinds  = @cfunction(_exec_output_memory_kinds,  Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_loaded_exec_fingerprint   = @cfunction(_loaded_exec_fingerprint,   Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_output_element_types = @cfunction(_exec_output_element_types, Ptr{Cvoid}, (Ptr{Cvoid},))
    global cfn_exec_output_dimensions    = @cfunction(_exec_output_dimensions,    Ptr{Cvoid}, (Ptr{Cvoid},))
    # Must be initialized before cfn_loaded_exec_get_device_assignment (which embeds it at call time)
    global cfn_noop_free                         = @cfunction(_noop_free, Cvoid, (Ptr{Cvoid},))
    global cfn_loaded_exec_get_device_assignment = @cfunction(_loaded_exec_get_device_assignment, Ptr{Cvoid}, (Ptr{Cvoid},))

    # Rebuild the function pointer tuple and PJRT_Api struct at runtime.
    # @cfunction pointers get re-resolved after precompilation, but the NTuple
    # captured their old addresses — must rebuild from the live cfn_* globals.
    fns = ntuple(Val(128)) do i
        if     i == 1;  cfn_error_destroy
        elseif i == 2;  cfn_error_message
        elseif i == 3;  cfn_error_getcode
        elseif i == 4;  cfn_plugin_initialize
        elseif i == 5;  cfn_plugin_attributes
        elseif i == 6;  cfn_event_destroy
        elseif i == 7;  cfn_event_is_ready
        elseif i == 8;  cfn_stub
        elseif i == 9;  cfn_event_await
        elseif i == 10; cfn_event_on_ready
        elseif i == 11; cfn_client_create
        elseif i == 12; cfn_client_destroy
        elseif i == 13; cfn_client_platform_name
        elseif i == 14; cfn_client_process_index
        elseif i == 15; cfn_client_platform_version
        elseif i == 16; cfn_client_devices
        elseif i == 17; cfn_client_addr_devices
        elseif i == 18; cfn_stub
        elseif i == 19; cfn_stub
        elseif i == 20; cfn_client_addr_memories
        elseif i == 21; cfn_client_compile
        elseif i == 22; cfn_stub
        elseif i == 23; cfn_client_buffer_from_host
        elseif i == 24; cfn_devdesc_id
        elseif i == 25; cfn_devdesc_process_index
        elseif i == 26; cfn_devdesc_attributes
        elseif i == 27; cfn_devdesc_kind
        elseif i == 28; cfn_devdesc_debug_string
        elseif i == 29; cfn_devdesc_to_string
        elseif i == 30; cfn_device_get_desc
        elseif i == 31; cfn_device_is_addressable
        elseif i == 32; cfn_device_local_hw_id
        elseif i == 33; cfn_device_addr_memories
        elseif i == 34; cfn_device_default_memory
        elseif i == 35; cfn_stub
        elseif i == 36; cfn_memory_id
        elseif i == 37; cfn_memory_kind
        elseif i == 38; cfn_memory_debug_string
        elseif i == 39; cfn_memory_to_string
        elseif i == 40; cfn_memory_addr_by_devices
        elseif i == 42; cfn_exec_name
        elseif i == 43; cfn_exec_num_replicas
        elseif i == 44; cfn_exec_num_partitions
        elseif i == 45; cfn_exec_num_outputs
        elseif i == 46; cfn_exec_code_size
        elseif i == 49; cfn_exec_optimized_program
        elseif i == 48; cfn_exec_output_memory_kinds
        elseif i == 51; cfn_loaded_exec_destroy
        elseif i == 52; cfn_loaded_exec_get_exec
        elseif i == 53; cfn_loaded_exec_addr_devs
        elseif i == 56; cfn_loaded_exec_execute
        elseif i == 58; cfn_loaded_exec_fingerprint
        elseif i == 59; cfn_buffer_destroy
        elseif i == 60; cfn_buffer_element_type
        elseif i == 61; cfn_buffer_dimensions
        elseif i == 62; cfn_buffer_dimensions  # UnpaddedDimensions — same layout, must set output fields
        elseif i == 65; cfn_buffer_on_device_size
        elseif i == 66; cfn_buffer_device
        elseif i == 67; cfn_buffer_memory
        elseif i == 71; cfn_buffer_to_host
        elseif i == 72; cfn_buffer_is_on_cpu
        elseif i == 73; cfn_buffer_ready_event
        elseif i == 91; cfn_exec_output_element_types
        elseif i == 92; cfn_exec_output_dimensions
        elseif i == 96; cfn_unimpl
        elseif i == 98; cfn_memory_kind_id
        elseif i == 118; cfn_loaded_exec_get_device_assignment
        else;           cfn_stub
        end
    end

    api_val = MetalPJRT_Api(
        UInt64(1064),
        C_NULL,
        MetalPJRT_Api_Version(UInt64(24), C_NULL, Int32(0), Int32(90)),
        fns,
    )

    global _PJRT_API_MEM = Libc.malloc(sizeof(MetalPJRT_Api))
    unsafe_store!(Ptr{MetalPJRT_Api}(_PJRT_API_MEM), api_val)

    return nothing
end

# ============================================================
# Public API
# ============================================================

"""
    make_client() -> Ptr{Cvoid}

Create a Metal PJRT client by registering our @cfunction callbacks with
libReactantExtra via MakeClientFromApi. Returns a PjRtClient*.

The PJRT_Api struct is stored in C memory and remains valid for the process
lifetime. The @cfunction stubs are LLVM function stubs (stable addresses).
"""
function make_client()
    api_ptr = Ptr{Cvoid}(_PJRT_API_MEM)
    client_ptr = Reactant.XLA.PJRT.MakeMetalClientFromApi(api_ptr)
    return client_ptr
end
