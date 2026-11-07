// Shared C ABI declarations and test utilities for ReactantExtra unit tests.

#ifndef THIRD_PARTY_REACTANT_JL_DEPS_REACTANTEXTRA_TEST_REACTANT_EXTRA_TEST_H_
#define THIRD_PARTY_REACTANT_JL_DEPS_REACTANTEXTRA_TEST_REACTANT_EXTRA_TEST_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// Opaque C++ runtime handle managed by reactantXLAInit and reactantXLADeInit.
struct LinkableRuntime;

// Opaque PJRT C-API plugin function table.
struct PJRT_Api;

// Minimal MLIR C-API opaque handle structs matching upstream C-API layout.
struct MlirContext {
  void *ptr;
};
struct MlirDialectRegistry {
  void *ptr;
};
struct MlirModule {
  void *ptr;
};
struct MlirOperation {
  void *ptr;
};
struct MlirRegion {
  void *ptr;
};
struct MlirBlock {
  void *ptr;
};
struct MlirLocation {
  void *ptr;
};
struct MlirType {
  void *ptr;
};
struct MlirAttribute {
  void *ptr;
};
struct MlirPassManager {
  void *ptr;
};
struct MlirOpPassManager {
  void *ptr;
};
struct MlirLogicalResult {
  int8_t value;
};

// Non-owning string reference passed across the MLIR C-API boundary.
struct MlirStringRef {
  // Pointer to the first character of the string slice.
  const char *data;
  // Number of bytes in the string slice.
  size_t length;
};

// Callback signature used by mlirOperationPrint and mlirAttributePrint.
typedef void (*MlirStringCallback)(MlirStringRef, void *);

// HLO cost analysis properties populated by
// pjrt_hlo_module_cost_analysis_properties and
// ifrt_hlo_module_cost_analysis_properties.
struct JLHloCostAnalysisProperties {
  float flops;
  float transcendentals;
  float bytes_accessed;
  float optimal_seconds;
  float utilization;
  float operand0_utilization;
  float operand1_utilization;
  float operand0_bytes_accessed;
  float operand1_bytes_accessed;
  float output_root_bytes_accessed;
  float reserved0;
};

// Device memory allocator statistics populated by PjRtDeviceGetAllocatorStats.
struct JLAllocatorStats {
  int64_t num_allocs;
  int64_t bytes_in_use;
  int64_t peak_bytes_in_use;
  int64_t largest_alloc_size;
  int64_t bytes_limit;
  int64_t bytes_reserved;
  int64_t peak_bytes_reserved;
  int64_t bytes_reservable_limit;
  int64_t largest_free_block_bytes;
  int64_t pool_bytes;
  int64_t peak_pool_bytes;
};

// Compiled executable memory usage breakdown populated by
// PjRtLoadedExecutableGetCompiledMemoryStats and
// ifrt_loaded_executable_get_compiled_memory_stats.
struct JLCompiledMemoryStats {
  int64_t generated_code_size_in_bytes;
  int64_t argument_size_in_bytes;
  int64_t output_size_in_bytes;
  int64_t alias_size_in_bytes;
  int64_t temp_size_in_bytes;
  int64_t host_generated_code_size_in_bytes;
  int64_t host_argument_size_in_bytes;
  int64_t host_output_size_in_bytes;
  int64_t host_alias_size_in_bytes;
  int64_t host_temp_size_in_bytes;
  int64_t peak_memory_in_bytes;
};

extern "C" {

// Global error callback invoked by ReactantExtra when a C ABI entry point
// encounters a recoverable error.
extern void (*ReactantThrowError)(const char *);

// MLIR C-API functions re-exported by ReactantExtraLib.
MlirContext mlirContextCreate(void);
void mlirContextDestroy(MlirContext context);
MlirDialectRegistry mlirDialectRegistryCreate(void);
void mlirDialectRegistryDestroy(MlirDialectRegistry registry);
void mlirContextAppendDialectRegistry(MlirContext context,
                                      MlirDialectRegistry registry);
MlirLocation mlirLocationUnknownGet(MlirContext context);
MlirModule mlirModuleCreateEmpty(MlirLocation location);
MlirModule mlirModuleCreateParse(MlirContext context, MlirStringRef module);
void mlirModuleDestroy(MlirModule module);
MlirOperation mlirModuleGetOperation(MlirModule module);
MlirBlock mlirModuleGetBody(MlirModule module);
MlirOperation mlirBlockGetFirstOperation(MlirBlock block);
MlirOperation mlirOperationGetNextInBlock(MlirOperation operation);
MlirRegion mlirOperationGetFirstRegion(MlirOperation operation);
MlirBlock mlirRegionGetFirstBlock(MlirRegion region);
void mlirOperationDestroy(MlirOperation operation);
bool mlirOperationVerify(MlirOperation operation);
void mlirOperationPrint(MlirOperation operation, MlirStringCallback callback,
                        void *user_data);
MlirAttribute mlirAttributeParseGet(MlirContext context,
                                    MlirStringRef attribute);
void mlirAttributePrint(MlirAttribute attribute, MlirStringCallback callback,
                        void *user_data);
MlirPassManager mlirPassManagerCreate(MlirContext context);
void mlirPassManagerDestroy(MlirPassManager pass_manager);
MlirOpPassManager
mlirPassManagerGetAsOpPassManager(MlirPassManager pass_manager);
MlirLogicalResult mlirPassManagerRunOnOp(MlirPassManager pass_manager,
                                         MlirOperation operation);

// Process initialization, dialect/pass registration, and FFI (API.cpp,
// xla_ffi.cpp).
void InitializeLogs(void);
void RegisterDialects(MlirContext context);
void InitializePasses(MlirDialectRegistry registry);
void InitializeRegistry(MlirDialectRegistry registry);
char *GetDefaultTargetTriple(void);
void registerReactantXLAFFI(void);
void RegisterCustomCallTarget(const char *name, void *address,
                              const char *platform);

// MLIR, Shardy, and LLVM-to-MLIR utilities (API.cpp).
void mlirPassManagerEnableXLATraceTiming(MlirPassManager pass_manager);
bool mlirOperationInject(MlirContext context, MlirBlock block,
                         MlirStringRef code, MlirLocation location,
                         bool verify_after_parse);
MlirOperation mlirOperationParse(MlirContext context, MlirBlock block,
                                 MlirStringRef code, MlirLocation location,
                                 bool verify_after_parse);
MlirType mlirGetFunctionTypeFromOperation(MlirOperation operation);
bool mlirIsFunctionOpInterface(MlirOperation operation);
void ReactantFuncSetResultAttr(MlirOperation operation, intptr_t position,
                               MlirStringRef name, MlirAttribute attribute);
void ReactantFuncSetArgAttr(MlirOperation operation, intptr_t position,
                            MlirStringRef name, MlirAttribute attribute);
void *mlirGetParentOfTypeFunctionOp(void *operation);
MlirAttribute enzymeActivityAttrGet(MlirContext context, int32_t value);
MlirOperation LinkInModule(MlirModule previous_module, MlirModule new_module,
                           const char *entry_function_name);
MlirModule ConvertLLVMStrToMLIR(const char *llvm_ir_text, MlirContext context);
void addSdyPropagationPipeline(
    MlirOpPassManager pass_manager, uint8_t keep_sharding_rules,
    uint8_t conservative_propagation, uint8_t debug_sharding_origins,
    uint8_t debug_propagation_edge_sharding, uint8_t skip_convert_to_reshard,
    uint8_t skip_inline, uint8_t enable_insert_explicit_collectives);
int32_t ReactantLexMLIR(MlirContext context, const char *input,
                        int32_t input_length, int32_t *token_kinds,
                        int32_t *token_offsets, int32_t *token_lengths,
                        int32_t max_tokens);

// HLO module, computation, instruction, sharding, and cost analysis C ABI.
void *convertMlirModuleToHloModule(MlirModule module);
void *parseAndReturnUnverifiedHloModule(const char *hlo_text);
void *hloModuleGetEntryComputation(void *hlo_module);
const char *hloComputationToString(void *hlo_computation,
                                   int32_t print_options);
int64_t hloComputationInstructionCount(void *hlo_computation);
void hloComputationGetInstructionsPostOrder(void *hlo_computation,
                                            int64_t num_instructions,
                                            void **hlo_instructions);
const char *hloInstructionToString(void *hlo_instruction,
                                   int32_t print_options);
uint8_t hloInstructionHasToApply(void *hlo_instruction);
void *hloInstructionGetToApply(void *hlo_instruction);
uint8_t hloInstructionGetOpcode(void *hlo_instruction);
const char *hloOpcodeToString(uint8_t opcode);
uint8_t hloInstructionIsFusion(void *hlo_instruction);
uint8_t hloInstructionGetFusionKind(void *hlo_instruction);
const char *hloFusionKindToString(uint8_t kind);
void *hloInstructionFusedInstructionsComputation(void *hlo_instruction);

void *hloShardingFromTensorShardingAttr(MlirAttribute sharding_attribute,
                                        MlirAttribute mesh_attribute);
MlirAttribute
hloShardingToTensorShardingAttr(MlirContext context, const void *hlo_sharding,
                                MlirAttribute mesh_name,
                                MlirAttribute mesh_attribute, int64_t rank,
                                const bool *is_closed, const int64_t *priority);
void free_hlo_sharding(void *hlo_sharding);
void *hlo_sharding_from_op_sharding(void *op_sharding);
void *hlo_sharding_to_op_sharding(void *hlo_sharding);
const char *hlo_sharding_to_string(const void *hlo_sharding);
bool hlo_sharding_is_tuple(void *hlo_sharding);
bool hlo_sharding_is_replicated(void *hlo_sharding);
bool hlo_sharding_is_manual(void *hlo_sharding);
bool hlo_sharding_is_unknown(void *hlo_sharding);
bool hlo_sharding_is_tiled(void *hlo_sharding);
bool hlo_sharding_is_maximal(void *hlo_sharding);
bool hlo_sharding_replicate_on_last_tile_dim(void *hlo_sharding);
int32_t hlo_sharding_tile_assignment_dimensions_size(void *hlo_sharding);
int32_t hlo_sharding_tile_assignment_devices_size(void *hlo_sharding);
void hlo_sharding_tile_assignment_dimensions(void *hlo_sharding,
                                             int64_t *dimensions,
                                             int32_t num_dimensions);
void hlo_sharding_tile_assignment_devices(void *hlo_sharding, int64_t *devices,
                                          int32_t num_devices);
bool hlo_sharding_check_eq(void *hlo_sharding, void *other);
bool hlo_sharding_check_eq_ignoring_metadata(void *hlo_sharding, void *other);

void free_op_sharding(void *op_sharding);
int32_t op_sharding_to_op_sharding_type(void *op_sharding);
bool op_sharding_is_shard_group(void *op_sharding);
bool op_sharding_replicate_on_last_tile_dim(void *op_sharding);
bool op_sharding_has_last_tile_dims(void *op_sharding);
bool op_sharding_has_iota_reshape_dims(void *op_sharding);
int32_t op_sharding_iota_reshape_dims_size(void *op_sharding);
void op_sharding_iota_reshape_dims(void *op_sharding,
                                   int32_t *iota_reshape_dims);
bool op_sharding_has_iota_transpose_perm(void *op_sharding);
int32_t op_sharding_iota_transpose_perm_size(void *op_sharding);
void op_sharding_iota_transpose_perm(void *op_sharding,
                                     int32_t *iota_transpose_perm);
bool op_sharding_has_tile_assignment_dimensions(void *op_sharding);
int32_t op_sharding_tile_assignment_dimensions_size(void *op_sharding);
void op_sharding_tile_assignment_dimensions(
    void *op_sharding, int32_t *tile_assignment_dimensions);
bool op_sharding_has_tile_assignment_devices(void *op_sharding);
int32_t op_sharding_tile_assignment_devices_size(void *op_sharding);
void op_sharding_tile_assignment_devices(void *op_sharding,
                                         int32_t *tile_assignment_devices);

void *ifrt_sharding_from_xla_hlo_sharding(void *client, void **device_list,
                                          int32_t num_devices,
                                          void *memory_kind,
                                          void *xla_hlo_sharding);
void *ifrt_sharding_to_xla_hlo_sharding(void *sharding);
void ifrt_sharding_to_index_domains(void *sharding, int64_t *array_size_list,
                                    int32_t array_rank,
                                    int64_t *index_domain_origins,
                                    int64_t *index_domain_shapes);

void pjrt_hlo_module_cost_analysis_properties(
    void *client, void *hlo_module, JLHloCostAnalysisProperties *properties);
void ifrt_hlo_module_cost_analysis_properties(
    void *client, void *hlo_module, JLHloCostAnalysisProperties *properties);
void *ReactantGetDebugOptions(size_t *size_in_bytes);
void *ReactantGetCompileOptions(size_t *size_in_bytes);

// LinkableRuntime C ABI (API.cpp).
void reactantXLAThrow(const char *message);
void reactantXLAInit(LinkableRuntime **runtime, const char *backend);
void reactantXLADeInit(LinkableRuntime **runtime);
void *reactantXLAMalloc(LinkableRuntime **runtime, uint64_t primitive_type,
                        uint64_t rank, uint64_t *shape);
void reactantXLAFree(LinkableRuntime **runtime, void *buffer);
void reactantXLAMemcpy(LinkableRuntime **runtime, void *destination,
                       void *source, size_t size_in_bytes, int32_t direction);
void reactantXLAExec(LinkableRuntime **runtime, const char *module_text,
                     int64_t num_arguments, void **arguments,
                     int64_t num_constants, const int64_t *constants);

// PJRT client and device C ABI (API.cpp).
void *MakeCPUClient(uint8_t asynchronous, int node_id);
void FreeClient(void *client);
const PJRT_Api *GetCpuPjrtApi(void);
void *MakeClientFromApi(const PJRT_Api *api, const char *device_type,
                        const char *client_name, const char **error);
int ClientNumDevices(void *client);
int ClientNumAddressableDevices(void *client);
int ClientProcessIndex(void *client);
void *ClientGetDevice(void *client, int device_id);
void *ClientGetAddressableDevice(void *client, int device_id);
const char *ClientGetPlatformName(void *client);
const char *DeviceGetKind(void *device);
void ClientGetDevices(void *client, void **out_devices);
void ClientGetAddressableDevices(void *client, void **out_devices);
void *DeviceToClient(void *device);
bool pjrt_device_is_addressable(void *device);
int64_t PjRtDeviceGetLocalDeviceId(void *device);
int64_t PjRtDeviceGetGlobalDeviceId(void *device);
int64_t PjRtDeviceGetLocalHardwareId(void *device);

// PJRT buffer C ABI (API.cpp).
void *ArrayFromHostBuffer(void *client, void *data, uint64_t primitive_type,
                          size_t num_dimensions, const int64_t *shape,
                          void *device);
void *UninitPJRTBuffer(void *client, void *device, uint64_t primitive_type,
                       uint64_t rank, uint64_t *shape);
void CopyToBuffer(void *client, void *buffer, void *data,
                  size_t offset_in_bytes, size_t size_in_bytes,
                  void **out_buffer);
void CopyFromBuffer(void *client, void *buffer, void *data,
                    size_t offset_in_bytes, size_t size_in_bytes,
                    void **out_buffer);
void BufferToHost(void *buffer, void *data);
void *UnsafeBufferPointer(void *buffer);
uint8_t BufferOnCPU(void *buffer);
void *CopyBufferToDevice(void *buffer, void *destination_device);
void *BufferToDevice(void *buffer);
void *BufferToClient(void *buffer);
const int64_t *BufferShape(void *buffer);
int64_t BufferNDimensions(void *buffer);
int32_t BufferPrimitiveType(void *buffer);
void PjRtBufferFree(void *buffer);
void *PjRtBufferGetReadyFuture(void *buffer);
void FreeFuture(void *future);
uint8_t FutureIsReady(void *future);
void FutureAwait(void *future);

// PJRT compilation and execution C ABI (API.cpp).
void *ClientCompile(void *client, MlirModule module, int64_t device_id,
                    const int64_t *mesh_ids, int64_t num_mesh_ids,
                    const char *xla_gpu_cuda_data_dir,
                    bool use_shardy_partitioner, int64_t num_replicas,
                    int64_t num_partitions, bool use_spmd_partitioning,
                    bool kernel_cache_enabled, const char *kernel_cache_path,
                    bool autotune_cache_enabled,
                    const char *autotune_cache_path, int process_id,
                    bool enable_enzyme_comms);
void *ClientCompileWithProto(void *client, MlirModule module,
                             const uint8_t *compile_options_proto,
                             size_t compile_options_proto_size);
void ExecutableFree(void *executable);
void *PjRtLoadedExecutableGetClient(void *executable);
int PjRtLoadedExecutableNumReplicas(void *executable);
int PjRtLoadedExecutableNumPartitions(void *executable);
void PjRtLoadedExecutableGetCompiledMemoryStats(
    void *executable, JLCompiledMemoryStats *statistics);
void PjRtLoadedExecutableGetHloModules(void *executable, void **hlo_modules,
                                       int32_t *num_modules);
const char *HloModuleToString(void *hlo_module, int32_t print_options);
void FreeHloModule(void *hlo_module);
uint8_t *PjRtLoadedExecutableSerialize(void *executable, size_t *size_in_bytes);
void *PjRtClientLoadSerializedExecutable(void *client, const uint8_t *data,
                                         size_t size_in_bytes,
                                         const uint8_t *compile_options_proto,
                                         size_t compile_options_proto_size);
void XLAExecuteSharded(void *executable, int num_arguments, void **arguments,
                       void *device, uint8_t *is_argument_donatable,
                       int num_results, void **results, uint8_t *futures,
                       void **future_results);
void XLAExecute(void *executable, int num_arguments, void **arguments,
                uint8_t *is_argument_donatable, int num_results, void **results,
                uint8_t *futures, void **future_results);

// IFRT client, device, memory, array, and execution C ABI (API.cpp).
void *ifrt_make_pjrt_cpu_client(uint8_t asynchronous, int node_id,
                                int num_nodes, void *distributed_runtime_client,
                                const char **error);
void ifrt_FreeClient(void *client);
int ifrt_client_device_count(void *client);
int ifrt_client_addressable_device_count(void *client);
void ifrt_client_devices(void *client, void **out_devices);
void ifrt_client_addressable_devices(void *client, void **out_devices);
void ifrt_client_all_devices(void *client, void **out_devices);
void *ifrt_client_lookup_device(void *client, int device_id);
void *ifrt_client_lookup_addressable_device(void *client,
                                            int local_hardware_id);
int ifrt_ClientProcessIndex(void *client);
const char *ifrt_ClientGetPlatformName(void *client);
void *ifrt_ClientGetDevice(void *client, int index);
void *ifrt_ClientGetAddressableDevice(void *client, int index);
int64_t ifrt_DeviceGetGlobalDeviceId(void *device);
const char *ifrt_DeviceGetKind(void *device);
void *ifrt_DeviceToClient(void *device);
bool ifrt_DeviceIsAddressable(void *device);
int64_t ifrt_DeviceGetLocalHardwareId(void *device);
void *ifrt_DeviceGetDefaultMemory(void *device);
void **ifrt_DeviceGetMemories(void *device, int32_t *num_memories);
void *ifrt_MemoryGetMemoryKind(void *memory);
const char *ifrt_MemoryToString(void *memory);
const char *ifrt_MemoryKindToString(void *memory_kind);
bool ifrt_MemoryKindsAreEqual(void *lhs_memory_kind, void *rhs_memory_kind);
void *ifrt_memory_kind_from_string(const char *name);
void *ifrt_memory_kind_with_optional_memory_space(void);
bool ifrt_memory_kind_has_value(void *memory_kind);
void free_ifrt_memory_kind(void *memory_kind);

void *ifrt_client_make_single_shard_array_from_host_buffer(
    void *client, void *data, int dtype_kind, int num_dimensions,
    const int64_t *shape, int semantics, void *device, void *memory_kind);
void ifrt_free_array(void *array);
int64_t *ifrt_array_shape(void *array);
int64_t ifrt_array_ndims(void *array);
void *ifrt_array_to_client(void *array);
void *ifrt_array_to_sharding(void *array);
void free_ifrt_sharding(void *sharding);
bool ifrt_sharding_is_single_device_sharding(void *sharding);
bool ifrt_sharding_is_fully_replicated(void *sharding);
const char *ifrt_sharding_to_string(void *sharding);
int32_t ifrt_sharding_devices_size(void *sharding);
void ifrt_sharding_to_device_list(void *sharding, void **devices);
void ifrt_array_copy_to_host_buffer(void *array, void *data);
void *ifrt_CopyArrayToHostBuffer(void *array, void *data, int semantics);
void **ifrt_array_disassemble_into_single_device_arrays(
    void *array, int32_t semantics, int32_t single_device_shard_semantics,
    int32_t *num_arrays);
void *ifrt_copy_array(void *array);
void ifrt_free_future(void *future);
uint8_t ifrt_future_is_ready(void *future);
void ifrt_future_await(void *future);

void *ifrt_compile(void *client, MlirModule module, int64_t device_id,
                   const int64_t *mesh_ids, int64_t num_mesh_ids,
                   const char *xla_gpu_cuda_data_dir,
                   bool use_shardy_partitioner, int64_t num_replicas,
                   int64_t num_partitions, bool use_spmd_partitioning,
                   bool kernel_cache_enabled, const char *kernel_cache_path,
                   bool autotune_cache_enabled, const char *autotune_cache_path,
                   int process_id, bool xla_enable_enzyme_comms_opt);
void *ifrt_compile_with_proto(void *client, MlirModule module,
                              const uint8_t *compile_options_proto,
                              size_t compile_options_proto_size);
void ifrt_loaded_executable_dtor(void *executable);
void ifrt_loaded_executable_execute(void *executable, int num_arguments,
                                    void **arguments,
                                    uint8_t *is_argument_donatable,
                                    int num_results, void **results,
                                    uint8_t *futures, void **status);
void *ifrt_loaded_executable_client(void *executable);
int32_t ifrt_loaded_executable_num_devices(void *executable);
void ifrt_loaded_executable_get_compiled_memory_stats(
    void *executable, JLCompiledMemoryStats *statistics);
void ifrt_loaded_executable_get_hlo_modules(void *executable,
                                            void **hlo_modules,
                                            int32_t *num_modules);
uint8_t *ifrt_loaded_executable_serialize(void *executable,
                                          size_t *size_in_bytes);
void *ifrt_client_load_serialized_executable(
    void *client, const uint8_t *data, size_t size_in_bytes,
    const uint8_t *compile_options_proto, size_t compile_options_proto_size);

} // extern "C"

namespace reactant_test {

// XLA PrimitiveType / IFRT DType::Kind numeric codes.
enum XlaPrimitiveType : uint64_t {
  kPred = 1,
  kS8 = 2,
  kS16 = 3,
  kS32 = 4,
  kS64 = 5,
  kU8 = 6,
  kU16 = 7,
  kU32 = 8,
  kU64 = 9,
  kF16 = 10,
  kF32 = 11,
  kF64 = 12,
};

// Transfer direction codes accepted by reactantXLAMemcpy.
enum XlaMemcpyDirection : int32_t {
  kMemcpyHostToDevice = 1,
  kMemcpyDeviceToHost = 2,
  kMemcpyDeviceToDevice = 3,
};

// Wraps a null-terminated C string into an MlirStringRef.
inline MlirStringRef makeStringRef(const char *text) {
  return MlirStringRef{text, std::strlen(text)};
}

// Callback that appends an MlirStringRef chunk to a std::string passed via
// `user_data`.
inline void appendMlirStringRef(MlirStringRef string_ref, void *user_data) {
  std::string *output = static_cast<std::string *>(user_data);
  output->append(string_ref.data, string_ref.length);
}

// Prints an MlirOperation into an owned std::string.
inline std::string printOperationToString(MlirOperation operation) {
  std::string output;
  mlirOperationPrint(operation, &appendMlirStringRef, &output);
  return output;
}

// Prints an MlirAttribute into an owned std::string.
inline std::string printAttributeToString(MlirAttribute attribute) {
  std::string output;
  mlirAttributePrint(attribute, &appendMlirStringRef, &output);
  return output;
}

// RAII guard that installs a non-aborting ReactantThrowError handler and
// records the last error message reported during its scope.
class ScopedErrorHandler {
public:
  ScopedErrorHandler() : previous_handler_(ReactantThrowError) {
    lastError().clear();
    ReactantThrowError = &recordError;
  }
  ~ScopedErrorHandler() { ReactantThrowError = previous_handler_; }

  // Returns the most recent error message captured in this scope, or an empty
  // string if no error was reported.
  static const std::string &error() { return lastError(); }

  // Clears the recorded error message.
  static void clear() { lastError().clear(); }

private:
  static void recordError(const char *message) {
    lastError() = message ? message : "unknown ReactantExtra error";
  }

  static std::string &lastError() {
    static std::string *error = new std::string();
    return *error;
  }

  // Previous error callback restored upon destruction.
  void (*previous_handler_)(const char *);
};

} // namespace reactant_test

#endif // THIRD_PARTY_REACTANT_JL_DEPS_REACTANTEXTRA_TEST_REACTANT_EXTRA_TEST_H_
