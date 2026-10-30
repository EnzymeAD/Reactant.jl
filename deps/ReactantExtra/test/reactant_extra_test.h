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

// Non-owning string reference passed across the MLIR C-API boundary.
struct MlirStringRef {
  // Pointer to the first character of the string slice.
  const char *data;
  // Number of bytes in the string slice.
  size_t length;
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
MlirModule mlirModuleCreateParse(MlirContext context, MlirStringRef module);
void mlirModuleDestroy(MlirModule module);

// Process initialization and dialect/pass registration (API.cpp).
void InitializeLogs(void);
void RegisterDialects(MlirContext context);
void InitializePasses(MlirDialectRegistry registry);
void InitializeRegistry(MlirDialectRegistry registry);
char *GetDefaultTargetTriple(void);

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
