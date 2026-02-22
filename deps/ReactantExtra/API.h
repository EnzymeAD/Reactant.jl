//===-- API.h - Reactant Extra C API --------------------------------------===//
//
// Public C API for libReactantExtra.
//
// All functions here use `extern "C"` linkage, matching the REACTANT_ABI
// functions defined in API.cpp.  The header is designed to be included from
// both C and C++ translation units:
//
//   * In **C mode** (Clang.jl binding generator) every handle is an opaque
//     `void *`, so the generated Julia wrappers use `Ptr{Cvoid}`.
//
//   * In **C++ mode** (API.cpp) the same typedef names resolve to the real
//     concrete types via forward declarations, so the compiler can verify
//     that every REACTANT_ABI definition matches its declaration here.
//
//===----------------------------------------------------------------------===//

#ifndef REACTANT_EXTRA_API_H
#define REACTANT_EXTRA_API_H

#include <stddef.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "llvm-c/TargetMachine.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

//===----------------------------------------------------------------------===//
// Structs exposed to Julia
//
// These are defined here (and only here) so that both the C binding generator
// and the C++ implementation share a single definition.
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif

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

struct DeviceProperties {
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  size_t totalConstMem;
  int major;
  int minor;
  int multiProcessorCount;
  int canMapHostMemory;
  int l2CacheSize;
  int maxThreadsPerMultiProcessor;
};

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

struct JLEstimateRunTimeData {
  int64_t flops;
  int64_t bytes_read;
  int64_t bytes_written;
  int64_t read_time_ns;
  int64_t write_time_ns;
  int64_t compute_time_ns;
  int64_t execution_time_ns;
};

#ifdef __cplusplus
} // extern "C" (structs)
#endif

//===----------------------------------------------------------------------===//
// Opaque handle types
//
// In C mode every handle is just `void *`.
// In C++ mode we forward-declare the real types so that the compiler can
// check that API.cpp definitions match these declarations.
//===----------------------------------------------------------------------===//

#if defined(__cplusplus) && !defined(REACTANT_BINDINGS_GENERATION)

#include <memory>

// ---------- forward declarations (C++ only) ----------
// These must match the concrete types used in API.cpp.
namespace xla {
class PjRtClient;
class PjRtDevice;
class PjRtBuffer;
class PjRtLoadedExecutable;
class HloModule;
class HloComputation;
class HloInstruction;
class OpSharding;
class HloSharding;
class DistributedRuntimeService;
class DistributedRuntimeClient;
namespace ifrt {
class Client;
class Device;
class Memory;
class MemoryKind;
class PjRtClient;
class PjRtLoadedExecutable;
class Sharding;
class LoadedExecutable;
class Array;
namespace proxy {
class GrpcServer;
} // namespace proxy
} // namespace ifrt
} // namespace xla
namespace tsl {
class ProfilerSession;
namespace profiler {
class ProfilerServer;
} // namespace profiler
template <typename T> class RCReference;
} // namespace tsl
namespace stream_executor {
class DeviceDescription;
} // namespace stream_executor
namespace mlir {
class Operation;
} // namespace mlir
struct PJRT_Api;
struct LinkableRuntime;

namespace reactant {
template <typename T> struct HeldValue;
} // namespace reactant

namespace details {
class GPUPerformanceModel;
} // namespace details

// Concrete typedefs matching API.cpp types
typedef xla::PjRtClient *PjRtClientPtr;
typedef xla::PjRtDevice *PjRtDevicePtr;
typedef xla::PjRtBuffer *PjRtBufferPtr;
typedef xla::PjRtLoadedExecutable *PjRtLoadedExecutablePtr;
typedef const PJRT_Api *PJRT_ApiPtr;
typedef tsl::ProfilerSession *ProfilerSessionPtr;
typedef tsl::profiler::ProfilerServer *ProfilerServerPtr;
typedef void *FutureTypePtr;
typedef void *IfRtFutureTypePtr;

typedef reactant::HeldValue<std::shared_ptr<xla::PjRtClient>>
    *HeldPjRtClientPtr;
typedef reactant::HeldValue<std::shared_ptr<xla::PjRtBuffer>>
    *HeldPjRtBufferPtr;
typedef reactant::HeldValue<tsl::RCReference<xla::ifrt::Array>>
    *HeldIfrtArrayPtr;
typedef reactant::HeldValue<std::shared_ptr<xla::HloModule>> *HeldHloModulePtr;
typedef reactant::HeldValue<std::shared_ptr<xla::ifrt::Sharding>>
    *HeldIfrtShardingPtr;
typedef reactant::HeldValue<std::shared_ptr<xla::ifrt::LoadedExecutable>>
    *HeldIfrtLoadedExecutablePtr;
typedef reactant::HeldValue<std::shared_ptr<const xla::ifrt::Sharding>>
    *HeldIfrtConstShardingPtr;
typedef reactant::HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>>
    *HeldDistributedRuntimeClientPtr;

typedef xla::ifrt::Client *IfrtClientPtr;
typedef xla::ifrt::Device *IfrtDevicePtr;
typedef xla::ifrt::Memory *IfrtMemoryPtr;
typedef xla::ifrt::MemoryKind *IfrtMemoryKindPtr;
typedef xla::ifrt::PjRtClient *IfrtPjRtClientPtr;
typedef xla::ifrt::PjRtLoadedExecutable *IfrtPjRtLoadedExecutablePtr;
typedef xla::ifrt::proxy::GrpcServer *IfrtGrpcServerPtr;

typedef xla::OpSharding *OpShardingPtr;
typedef const xla::HloSharding *HloShardingPtr;
typedef xla::HloComputation *HloComputationPtr;
typedef xla::HloInstruction *HloInstructionPtr;
typedef xla::HloModule *HloModulePtr;

typedef stream_executor::DeviceDescription *DeviceDescriptionPtr;
typedef xla::DistributedRuntimeService *DistributedRuntimeServicePtr;
typedef details::GPUPerformanceModel *GPUPerformanceModelPtr;

typedef LinkableRuntime *LinkableRuntimePtr;

#else /* C mode */

typedef void *PjRtClientPtr;
typedef void *PjRtDevicePtr;
typedef void *PjRtBufferPtr;
typedef void *PjRtLoadedExecutablePtr;
typedef void *PJRT_ApiPtr;
typedef void *ProfilerSessionPtr;
typedef void *ProfilerServerPtr;
typedef void *FutureTypePtr;
typedef void *IfRtFutureTypePtr;

typedef void *HeldPjRtClientPtr;
typedef void *HeldPjRtBufferPtr;
typedef void *HeldIfrtArrayPtr;
typedef void *HeldHloModulePtr;
typedef void *HeldIfrtShardingPtr;
typedef void *HeldIfrtLoadedExecutablePtr;
typedef void *HeldIfrtConstShardingPtr;
typedef void *HeldDistributedRuntimeClientPtr;

typedef void *IfrtClientPtr;
typedef void *IfrtDevicePtr;
typedef void *IfrtMemoryPtr;
typedef void *IfrtMemoryKindPtr;
typedef void *IfrtPjRtClientPtr;
typedef void *IfrtPjRtLoadedExecutablePtr;
typedef void *IfrtGrpcServerPtr;

typedef void *OpShardingPtr;
typedef void *HloShardingPtr;
typedef void *HloComputationPtr;
typedef void *HloInstructionPtr;
typedef void *HloModulePtr;

typedef void *DeviceDescriptionPtr;
typedef void *DistributedRuntimeServicePtr;
typedef void *GPUPerformanceModelPtr;

typedef void *LinkableRuntimePtr;

#endif /* defined(__cplusplus) && !defined(REACTANT_BINDINGS_GENERATION) */

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED extern void (*ReactantThrowError)(const char *);

//===----------------------------------------------------------------------===//
// Initialization & Logging
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void InitializeLogs(void);
MLIR_CAPI_EXPORTED void SetLogLevel(int level);
MLIR_CAPI_EXPORTED void SetModuleLogLevel(const char *module_pattern,
                                          int level);
MLIR_CAPI_EXPORTED char *GetDefaultTargetTriple(void);
MLIR_CAPI_EXPORTED void
ReactantLLVMParseCommandLineOptions(int argc, const char *const *argv,
                                    const char *Overview);

//===----------------------------------------------------------------------===//
// CUDA helpers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void ReactantHandleCuResult(uint32_t curesult);
MLIR_CAPI_EXPORTED int32_t ReactantCudaDriverGetVersion(void);
MLIR_CAPI_EXPORTED int32_t ReactantHermeticCudaGetVersion(void);
MLIR_CAPI_EXPORTED int32_t ReactantCudaDeviceGetComputeCapalilityMajor(void);
MLIR_CAPI_EXPORTED int32_t ReactantCudaDeviceGetComputeCapalilityMinor(void);
MLIR_CAPI_EXPORTED int32_t ReactantCudaDeviceGetWarpSizeInThreads(void);
MLIR_CAPI_EXPORTED void
ReactantCudaDeviceGetProperties(struct DeviceProperties *jlprops,
                                int32_t device_id);
MLIR_CAPI_EXPORTED void ReactantCudaGetRegsSpillsMaxThreadsFromBinary(
    const char *binary, const char *fnname, int32_t *regs, int32_t *spills,
    int32_t *maxThreads);
MLIR_CAPI_EXPORTED DeviceDescriptionPtr
CudaGetStreamExecutorDeviceDescription(int32_t device_id);
MLIR_CAPI_EXPORTED const char *
deviceDescriptionToString(DeviceDescriptionPtr device);

//===----------------------------------------------------------------------===//
// MLIR C-API extras
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirOperationInject(MlirContext ctx, MlirBlock block,
                                            MlirStringRef code,
                                            MlirLocation location,
                                            bool verify_after_parse);
MLIR_CAPI_EXPORTED MlirOperation mlirOperationParse(MlirContext ctx,
                                                    MlirBlock block,
                                                    MlirStringRef code,
                                                    MlirLocation location,
                                                    bool verify_after_parse);
MLIR_CAPI_EXPORTED MlirType mlirGetFunctionTypeFromOperation(MlirOperation op);
MLIR_CAPI_EXPORTED bool mlirIsFunctionOpInterface(MlirOperation op);
MLIR_CAPI_EXPORTED void ReactantFuncSetResultAttr(MlirOperation op,
                                                  intptr_t pos,
                                                  MlirStringRef name,
                                                  MlirAttribute attr);
MLIR_CAPI_EXPORTED void ReactantFuncSetArgAttr(MlirOperation op, intptr_t pos,
                                               MlirStringRef name,
                                               MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(MlirContext ctx,
                                                       int32_t val);

MLIR_CAPI_EXPORTED void RegisterDialects(MlirContext cctx);
MLIR_CAPI_EXPORTED void InitializePasses(MlirDialectRegistry creg);
MLIR_CAPI_EXPORTED void InitializeRegistry(MlirDialectRegistry creg);

MLIR_CAPI_EXPORTED MlirOperation LinkInModule(MlirModule prevModC,
                                              MlirModule newModC,
                                              const char *entryfn);

// LLVM IR -> MLIR
MLIR_CAPI_EXPORTED MlirModule ConvertLLVMToMLIR(LLVMModuleRef lmod,
                                                MlirContext cctx);
MLIR_CAPI_EXPORTED MlirModule ConvertLLVMStrToMLIR(const char *lmod,
                                                   MlirContext cctx);

// MLIR helpers
MLIR_CAPI_EXPORTED void dump_op(void *op);
MLIR_CAPI_EXPORTED void dump_mval(MlirValue v);
MLIR_CAPI_EXPORTED void dump_operation(void *op, const char *filename);
MLIR_CAPI_EXPORTED void dump_string(const char *op, const char *filename);
MLIR_CAPI_EXPORTED void *mlirGetParentOfTypeFunctionOp(void *op);

//===----------------------------------------------------------------------===//
// Profiler
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED ProfilerSessionPtr
CreateProfilerSession(uint32_t device_tracer_level, uint32_t host_tracer_level);
MLIR_CAPI_EXPORTED void ProfilerSessionCollectData(ProfilerSessionPtr session,
                                                   const char *path);
MLIR_CAPI_EXPORTED void ProfilerSessionDelete(ProfilerSessionPtr session);
MLIR_CAPI_EXPORTED int64_t ProfilerActivityStart(const char *name, int level);
MLIR_CAPI_EXPORTED void ProfilerActivityEnd(int64_t id);
MLIR_CAPI_EXPORTED ProfilerServerPtr ProfilerServerStart(int32_t port);
MLIR_CAPI_EXPORTED void ProfilerServerStop(ProfilerServerPtr server);

//===----------------------------------------------------------------------===//
// PjRt Client
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED PjRtClientPtr MakeCPUClient(uint8_t asynchronous,
                                               int node_id);
MLIR_CAPI_EXPORTED PjRtClientPtr
MakeGPUClient(int node_id, int num_nodes, int64_t *allowed_devices,
              int64_t num_allowed_devices, double memory_fraction,
              bool preallocate, const char *platform_name, const char **error,
              void *distributed_runtime_client);
MLIR_CAPI_EXPORTED PjRtClientPtr MakeTPUClient(const char *tpu_path,
                                               const char **error);
MLIR_CAPI_EXPORTED PJRT_ApiPtr LoadPjrtPlugin(const char *device_type,
                                              const char *library_path,
                                              const char **error);
MLIR_CAPI_EXPORTED int InitializePjrtPlugin(const char *device_type,
                                            const char **error);
MLIR_CAPI_EXPORTED PjRtClientPtr GetCApiClient(const char *device_type);
MLIR_CAPI_EXPORTED void pjrt_client_register_profiler(PJRT_ApiPtr api);
MLIR_CAPI_EXPORTED PjRtClientPtr
MakeClientUsingPluginAPI(const char *device_type, const char *library_path,
                         const char *client_name, const char **error);
MLIR_CAPI_EXPORTED PjRtClientPtr MakeClientFromApi(PJRT_ApiPtr api,
                                                   const char *device_type,
                                                   const char *client_name,
                                                   const char **error);

MLIR_CAPI_EXPORTED int ClientNumDevices(PjRtClientPtr client);
MLIR_CAPI_EXPORTED int ClientNumAddressableDevices(PjRtClientPtr client);
MLIR_CAPI_EXPORTED int ClientProcessIndex(PjRtClientPtr client);
MLIR_CAPI_EXPORTED PjRtDevicePtr ClientGetDevice(PjRtClientPtr client,
                                                 int device_id);
MLIR_CAPI_EXPORTED PjRtDevicePtr
ClientGetAddressableDevice(PjRtClientPtr client, int device_id);
MLIR_CAPI_EXPORTED const char *ClientGetPlatformName(PjRtClientPtr client);
MLIR_CAPI_EXPORTED void ClientGetDevices(PjRtClientPtr client,
                                         PjRtDevicePtr *out_devices);
MLIR_CAPI_EXPORTED void ClientGetAddressableDevices(PjRtClientPtr client,
                                                    PjRtDevicePtr *out_devices);
MLIR_CAPI_EXPORTED void FreeClient(PjRtClientPtr client);

//===----------------------------------------------------------------------===//
// PjRt Device
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED const char *DeviceGetKind(PjRtDevicePtr device);
MLIR_CAPI_EXPORTED PjRtClientPtr DeviceToClient(PjRtDevicePtr device);
MLIR_CAPI_EXPORTED void
PjRtDeviceGetAllocatorStats(PjRtDevicePtr device,
                            struct JLAllocatorStats *jlstats);
MLIR_CAPI_EXPORTED int64_t PjRtDeviceGetLocalDeviceId(PjRtDevicePtr device);
MLIR_CAPI_EXPORTED int64_t PjRtDeviceGetGlobalDeviceId(PjRtDevicePtr device);
MLIR_CAPI_EXPORTED int64_t PjRtDeviceGetLocalHardwareId(PjRtDevicePtr device);
MLIR_CAPI_EXPORTED bool pjrt_device_is_addressable(PjRtDevicePtr device);

//===----------------------------------------------------------------------===//
// PjRt Buffer
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED PjRtDevicePtr BufferToDevice(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED PjRtClientPtr BufferToClient(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED const int64_t *BufferShape(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED int64_t BufferNDimensions(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED int BufferPrimitiveType(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED void PjRtBufferFree(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED void *UnsafeBufferPointer(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED PjRtBufferPtr ArrayFromHostBuffer(PjRtClientPtr client,
                                                     void *data, uint64_t ptype,
                                                     size_t dim,
                                                     const int64_t *cshape,
                                                     PjRtDevicePtr device);
MLIR_CAPI_EXPORTED void CopyToBuffer(PjRtClientPtr client, PjRtBufferPtr buffer,
                                     void *data, size_t offset, size_t size,
                                     PjRtBufferPtr *bufferP);
MLIR_CAPI_EXPORTED void BufferToHost(PjRtBufferPtr buffer, void *data);
MLIR_CAPI_EXPORTED void CopyFromBuffer(PjRtClientPtr client,
                                       PjRtBufferPtr buffer, void *data,
                                       size_t offset, size_t size,
                                       PjRtBufferPtr *bufferP);
MLIR_CAPI_EXPORTED PjRtBufferPtr UninitPJRTBuffer(PjRtClientPtr client,
                                                  PjRtDevicePtr device,
                                                  uint64_t ptype,
                                                  uint64_t shapeLen,
                                                  uint64_t *shape);
MLIR_CAPI_EXPORTED uint8_t BufferOnCPU(PjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED PjRtBufferPtr CopyBufferToDevice(PjRtBufferPtr buffer,
                                                    PjRtDevicePtr dst_device);
MLIR_CAPI_EXPORTED void
RegisterCustomCallTarget(const char *name, void *address, const char *platform);

//===----------------------------------------------------------------------===//
// PjRt Future
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void FreeFuture(FutureTypePtr future);
MLIR_CAPI_EXPORTED uint8_t FutureIsReady(FutureTypePtr future);
MLIR_CAPI_EXPORTED void FutureAwait(FutureTypePtr future);

//===----------------------------------------------------------------------===//
// PjRt Compilation & Execution
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED PjRtLoadedExecutablePtr
ClientCompile(PjRtClientPtr client, MlirModule cmod, int64_t device_id,
              const int64_t *mesh_ids, int64_t num_mesh_ids,
              const char *xla_gpu_cuda_data_dir, bool use_shardy_partitioner,
              int64_t num_replicas, int64_t num_partitions,
              bool use_spmd_partitioning, bool kernel_cache_enabled,
              const char *kernel_cache_path, bool autotune_cache_enabled,
              const char *autotune_cache_path, int process_id);
MLIR_CAPI_EXPORTED PjRtLoadedExecutablePtr ClientCompileWithProto(
    PjRtClientPtr client, MlirModule cmod, const char *compile_options_proto,
    size_t compile_options_proto_size);

MLIR_CAPI_EXPORTED PjRtClientPtr
PjRtLoadedExecutableGetClient(PjRtLoadedExecutablePtr exec);
MLIR_CAPI_EXPORTED void ExecutableFree(PjRtLoadedExecutablePtr exec);
MLIR_CAPI_EXPORTED int
PjRtLoadedExecutableNumReplicas(PjRtLoadedExecutablePtr exec);
MLIR_CAPI_EXPORTED int
PjRtLoadedExecutableNumPartitions(PjRtLoadedExecutablePtr exec);

MLIR_CAPI_EXPORTED void
PjRtLoadedExecutableGetOuputShardings(PjRtLoadedExecutablePtr exec,
                                      OpShardingPtr *op_shardings,
                                      int32_t num_op_shardings);
MLIR_CAPI_EXPORTED void
PjRtLoadedExecutableGetParameterShardings(PjRtLoadedExecutablePtr exec,
                                          OpShardingPtr *op_shardings,
                                          int32_t num_op_shardings);
MLIR_CAPI_EXPORTED void
PjRtLoadedExecutableGetHloModules(PjRtLoadedExecutablePtr exec,
                                  void **hlo_modules, int32_t *nmodules);

MLIR_CAPI_EXPORTED void XLAExecuteSharded(
    PjRtLoadedExecutablePtr exec, int num_args, PjRtBufferPtr *op_args,
    PjRtDevicePtr device, uint8_t *is_arg_donatable, int num_results,
    PjRtBufferPtr *op_results, uint8_t *futures, FutureTypePtr *future_results);
MLIR_CAPI_EXPORTED void XLAExecute(PjRtLoadedExecutablePtr exec,
                                   int op_args_len, PjRtBufferPtr *op_args,
                                   uint8_t *is_arg_donatable, int num_results,
                                   PjRtBufferPtr *op_results, uint8_t *futures,
                                   FutureTypePtr *future_results);

//===----------------------------------------------------------------------===//
// Held PjRt Client (shared_ptr wrappers)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void pjrt_client_dtor(HeldPjRtClientPtr client);
MLIR_CAPI_EXPORTED int pjrt_client_num_devices(HeldPjRtClientPtr client);
MLIR_CAPI_EXPORTED int
pjrt_client_num_addressable_devices(HeldPjRtClientPtr client);
MLIR_CAPI_EXPORTED int pjrt_client_pid(HeldPjRtClientPtr client);
MLIR_CAPI_EXPORTED PjRtDevicePtr
pjrt_client_get_device(HeldPjRtClientPtr client, int device_id);
MLIR_CAPI_EXPORTED PjRtDevicePtr
pjrt_client_get_addressable_device(HeldPjRtClientPtr client, int device_id);
MLIR_CAPI_EXPORTED const char *
pjrt_client_platform_name(HeldPjRtClientPtr client);

//===----------------------------------------------------------------------===//
// Held PjRt Buffer (shared_ptr wrappers)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED HeldPjRtBufferPtr
pjrt_buffer_from_host(HeldPjRtClientPtr client, void *data, uint64_t ptype,
                      size_t dim, int64_t *cshape, PjRtDevicePtr device);
MLIR_CAPI_EXPORTED void pjrt_buffer_dtor(HeldPjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED void *
pjrt_buffer_unsafe_buffer_pointer(HeldPjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED bool pjrt_buffer_is_on_cpu(HeldPjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED HeldPjRtBufferPtr
pjrt_buffer_copy_to_device(HeldPjRtBufferPtr buffer, PjRtDevicePtr dst_device);
MLIR_CAPI_EXPORTED void pjrt_buffer_to_host(HeldPjRtBufferPtr buffer,
                                            void *data);
MLIR_CAPI_EXPORTED void pjrt_buffer_print(HeldPjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED PjRtDevicePtr
pjrt_buffer_get_device(HeldPjRtBufferPtr buffer);
MLIR_CAPI_EXPORTED HeldPjRtClientPtr
pjrt_buffer_get_client(HeldPjRtBufferPtr buffer);

//===----------------------------------------------------------------------===//
// IFRT Client
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void ifrt_client_dtor(IfrtClientPtr client);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr ifrt_client_make_array_from_host_buffer(
    IfrtClientPtr client, void *data, int dtype_kind, int ndims,
    const int64_t *c_shape, HeldIfrtConstShardingPtr sharding, int c_semantics);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr
ifrt_client_make_single_shard_array_from_host_buffer(
    IfrtClientPtr client, void *data, int dtype_kind, int ndims,
    const int64_t *c_shape, int c_semantics, IfrtDevicePtr device,
    const char *mem_kind);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr
ifrt_client_assemble_array_from_single_shards(
    IfrtClientPtr client, int32_t ndims, const int64_t *c_shape,
    HeldIfrtConstShardingPtr sharding, int32_t narrays,
    HeldIfrtArrayPtr *c_arrays, int32_t c_semantics);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr
ifrt_pjrt_array_create(IfrtPjRtClientPtr client, HeldPjRtBufferPtr buffer);

MLIR_CAPI_EXPORTED IfrtClientPtr ifrt_pjrt_make_client_with_default_kv_store(
    PjRtClientPtr pjrt_client, int node_id, int num_nodes,
    void *distributed_runtime_client, const char **error,
    const char *key_prefix);
MLIR_CAPI_EXPORTED IfrtClientPtr
ifrt_make_pjrt_cpu_client(uint8_t asynchronous, int node_id, int num_nodes,
                          void *distributed_runtime_client, const char **error);
MLIR_CAPI_EXPORTED IfrtClientPtr
ifrt_make_pjrt_gpu_client(int node_id, int num_nodes, int64_t *allowed_devices,
                          int64_t num_allowed_devices, double memory_fraction,
                          bool preallocate, const char *platform_name,
                          const char **error, void *distributed_runtime_client);
MLIR_CAPI_EXPORTED IfrtClientPtr
ifrt_make_pjrt_tpu_client(const char *tpu_path, const char **error, int node_id,
                          int num_nodes, void *distributed_runtime_client);

MLIR_CAPI_EXPORTED void ifrt_FreeClient(IfrtClientPtr client);
MLIR_CAPI_EXPORTED int ifrt_client_device_count(IfrtClientPtr client);
MLIR_CAPI_EXPORTED int
ifrt_client_addressable_device_count(IfrtClientPtr client);
MLIR_CAPI_EXPORTED void ifrt_client_devices(IfrtClientPtr client,
                                            IfrtDevicePtr *out_devices);
MLIR_CAPI_EXPORTED void
ifrt_client_addressable_devices(IfrtClientPtr client,
                                IfrtDevicePtr *out_devices);
MLIR_CAPI_EXPORTED void ifrt_client_all_devices(IfrtClientPtr client,
                                                IfrtDevicePtr *out_devices);
MLIR_CAPI_EXPORTED IfrtDevicePtr ifrt_client_lookup_device(IfrtClientPtr client,
                                                           int dev_id);
MLIR_CAPI_EXPORTED IfrtDevicePtr
ifrt_client_lookup_addressable_device(IfrtClientPtr client, int local_hw_id);
MLIR_CAPI_EXPORTED int ifrt_ClientProcessIndex(IfrtClientPtr client);
MLIR_CAPI_EXPORTED const char *ifrt_ClientGetPlatformName(IfrtClientPtr client);
MLIR_CAPI_EXPORTED IfrtDevicePtr ifrt_ClientGetDevice(IfrtClientPtr client,
                                                      int idx);
MLIR_CAPI_EXPORTED IfrtDevicePtr
ifrt_ClientGetAddressableDevice(IfrtClientPtr client, int idx);

//===----------------------------------------------------------------------===//
// IFRT Device
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED int64_t ifrt_DeviceGetGlobalDeviceId(IfrtDevicePtr device);
MLIR_CAPI_EXPORTED const char *ifrt_DeviceGetKind(IfrtDevicePtr device);
MLIR_CAPI_EXPORTED IfrtClientPtr ifrt_DeviceToClient(IfrtDevicePtr device);
MLIR_CAPI_EXPORTED bool ifrt_DeviceIsAddressable(IfrtDevicePtr device);
MLIR_CAPI_EXPORTED int64_t ifrt_DeviceGetLocalHardwareId(IfrtDevicePtr device);
MLIR_CAPI_EXPORTED IfrtMemoryPtr
ifrt_DeviceGetDefaultMemory(IfrtDevicePtr device);
MLIR_CAPI_EXPORTED IfrtMemoryPtr *ifrt_DeviceGetMemories(IfrtDevicePtr device,
                                                         int32_t *size);
MLIR_CAPI_EXPORTED void
ifrt_device_get_allocator_stats(IfrtDevicePtr device,
                                struct JLAllocatorStats *jlstats);

//===----------------------------------------------------------------------===//
// IFRT Memory
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED IfrtMemoryKindPtr
ifrt_MemoryGetMemoryKind(IfrtMemoryPtr memory);
MLIR_CAPI_EXPORTED const char *ifrt_MemoryToString(IfrtMemoryPtr memory);
MLIR_CAPI_EXPORTED const char *
ifrt_MemoryKindToString(IfrtMemoryKindPtr memory_kind);
MLIR_CAPI_EXPORTED bool ifrt_MemoryKindsAreEqual(IfrtMemoryKindPtr a,
                                                 IfrtMemoryKindPtr b);
MLIR_CAPI_EXPORTED IfrtMemoryKindPtr
ifrt_memory_kind_from_string(const char *c_str);
MLIR_CAPI_EXPORTED IfrtMemoryKindPtr
ifrt_memory_kind_with_optional_memory_space(void);
MLIR_CAPI_EXPORTED bool
ifrt_memory_kind_has_value(IfrtMemoryKindPtr memory_kind);

//===----------------------------------------------------------------------===//
// IFRT Compilation & Execution
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED HeldIfrtLoadedExecutablePtr
ifrt_compile(IfrtClientPtr client, MlirModule cmod, int64_t device_id,
             const int64_t *mesh_ids, int64_t num_mesh_ids,
             const char *xla_gpu_cuda_data_dir, bool use_shardy_partitioner,
             int64_t num_replicas, int64_t num_partitions,
             bool use_spmd_partitioning, bool kernel_cache_enabled,
             const char *kernel_cache_path, bool autotune_cache_enabled,
             const char *autotune_cache_path, int process_id);
MLIR_CAPI_EXPORTED HeldIfrtLoadedExecutablePtr ifrt_compile_with_proto(
    IfrtClientPtr client, MlirModule cmod, const char *compile_options_proto,
    size_t compile_options_proto_size);

MLIR_CAPI_EXPORTED void
ifrt_pjrt_loaded_executable_dtor(IfrtPjRtLoadedExecutablePtr exec);
MLIR_CAPI_EXPORTED void
ifrt_loaded_executable_dtor(HeldIfrtLoadedExecutablePtr exec);
MLIR_CAPI_EXPORTED void ifrt_loaded_executable_execute(
    HeldIfrtLoadedExecutablePtr exec, int num_args, HeldIfrtArrayPtr *op_args,
    uint8_t *is_arg_donatable, int num_results, HeldIfrtArrayPtr *op_results,
    uint8_t *futures, FutureTypePtr *status);
MLIR_CAPI_EXPORTED IfrtClientPtr
ifrt_loaded_executable_client(HeldIfrtLoadedExecutablePtr exec);
MLIR_CAPI_EXPORTED void
ifrt_loaded_executable_get_parameter_shardings(HeldIfrtLoadedExecutablePtr exec,
                                               OpShardingPtr *op_shardings,
                                               int32_t num_op_shardings);
MLIR_CAPI_EXPORTED void
ifrt_loaded_executable_get_output_shardings(HeldIfrtLoadedExecutablePtr exec,
                                            OpShardingPtr *op_shardings,
                                            int32_t num_op_shardings);
MLIR_CAPI_EXPORTED void
ifrt_loaded_executable_get_hlo_modules(HeldIfrtLoadedExecutablePtr exec,
                                       void **hlo_modules, int32_t *nmodules);
MLIR_CAPI_EXPORTED int32_t
ifrt_loaded_executable_num_devices(HeldIfrtLoadedExecutablePtr exec);

//===----------------------------------------------------------------------===//
// IFRT Array
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void ifrt_array_dtor(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED FutureTypePtr
ifrt_CopyArrayToHostBuffer(HeldIfrtArrayPtr array, void *data, int semantics);
MLIR_CAPI_EXPORTED void ifrt_free_array(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED int64_t *ifrt_array_shape(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED int64_t ifrt_array_ndims(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED int ifrt_array_eltype(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED IfrtClientPtr ifrt_array_to_client(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED HeldIfrtConstShardingPtr
ifrt_array_to_sharding(HeldIfrtArrayPtr array);
MLIR_CAPI_EXPORTED void ifrt_array_copy_to_host_buffer(HeldIfrtArrayPtr array,
                                                       void *data);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr *
ifrt_array_disassemble_into_single_device_arrays(
    HeldIfrtArrayPtr array, int32_t c_semantics,
    int32_t c_single_device_shard_semantics, int32_t *narrays);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr ifrt_copy_array(HeldIfrtArrayPtr array);

MLIR_CAPI_EXPORTED HeldIfrtArrayPtr *ifrt_copy_arrays_to_device_with_sharding(
    IfrtClientPtr client, HeldIfrtArrayPtr *arrays, int32_t num_arrays,
    HeldIfrtConstShardingPtr dst_sharding, int32_t c_semantics);
MLIR_CAPI_EXPORTED HeldIfrtArrayPtr ifrt_make_array_from_host_buffer_shards(
    IfrtClientPtr client, const void **host_buffers, int num_buffers,
    const int64_t **host_buffer_shapes,
    const int64_t **addressable_shard_indices,
    const int64_t *addressable_shard_indices_sizes, int dtype_kind, int ndims,
    const int64_t *final_buffer_shape, HeldIfrtConstShardingPtr sharding,
    int32_t c_host_buffer_semantics);

//===----------------------------------------------------------------------===//
// IFRT Future
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void ifrt_free_future(IfRtFutureTypePtr future);
MLIR_CAPI_EXPORTED uint8_t ifrt_future_is_ready(IfRtFutureTypePtr future);
MLIR_CAPI_EXPORTED void ifrt_future_await(IfRtFutureTypePtr future);

//===----------------------------------------------------------------------===//
// IFRT Proxy (gRPC)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void ifrt_proxy_grpc_server_dtor(IfrtGrpcServerPtr server);
MLIR_CAPI_EXPORTED const char *
ifrt_proxy_grpc_server_address(IfrtGrpcServerPtr server);
MLIR_CAPI_EXPORTED void ifrt_proxy_grpc_server_wait(IfrtGrpcServerPtr server);
MLIR_CAPI_EXPORTED IfrtClientPtr ifrt_proxy_create_client(
    const char *c_proxy_server_address, int connection_timeout_in_minutes);

//===----------------------------------------------------------------------===//
// OpSharding
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void free_op_sharding(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_to_op_sharding_type(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_to_shard_group_type(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_to_shard_group_id(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED bool op_sharding_is_shard_group(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED bool
op_sharding_replicate_on_last_tile_dim(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED bool
op_sharding_has_last_tile_dims(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_last_tile_dims_size(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED void op_sharding_last_tile_dims(OpShardingPtr op_sharding,
                                                   int32_t *last_tile_dims);
MLIR_CAPI_EXPORTED bool
op_sharding_has_iota_reshape_dims(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_iota_reshape_dims_size(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED void
op_sharding_iota_reshape_dims(OpShardingPtr op_sharding,
                              int32_t *iota_reshape_dims);
MLIR_CAPI_EXPORTED bool
op_sharding_has_iota_transpose_perm(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_iota_transpose_perm_size(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED void
op_sharding_iota_transpose_perm(OpShardingPtr op_sharding,
                                int32_t *iota_transpose_perm);
MLIR_CAPI_EXPORTED bool
op_sharding_has_tile_assignment_dimensions(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_tile_assignment_dimensions_size(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED void
op_sharding_tile_assignment_dimensions(OpShardingPtr op_sharding,
                                       int32_t *tile_assignment_dimensions);
MLIR_CAPI_EXPORTED bool
op_sharding_has_tile_assignment_devices(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED int32_t
op_sharding_tile_assignment_devices_size(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED void
op_sharding_tile_assignment_devices(OpShardingPtr op_sharding,
                                    int32_t *tile_assignment_devices);

//===----------------------------------------------------------------------===//
// HloSharding
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void free_hlo_sharding(HloShardingPtr hlo_sharding);
MLIR_CAPI_EXPORTED HloShardingPtr
hlo_sharding_from_op_sharding(OpShardingPtr op_sharding);
MLIR_CAPI_EXPORTED OpShardingPtr
hlo_sharding_to_op_sharding(HloShardingPtr hlo_sharding);
MLIR_CAPI_EXPORTED const char *
hlo_sharding_to_string(HloShardingPtr hlo_sharding);
MLIR_CAPI_EXPORTED bool hlo_sharding_is_tuple(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED bool hlo_sharding_is_replicated(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED bool hlo_sharding_is_manual(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED bool hlo_sharding_is_unknown(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED bool hlo_sharding_is_tiled(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED bool hlo_sharding_is_maximal(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED bool
hlo_sharding_replicate_on_last_tile_dim(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED int32_t
hlo_sharding_tile_assignment_dimensions_size(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED int32_t
hlo_sharding_tile_assignment_devices_size(HloShardingPtr hloSharding);
MLIR_CAPI_EXPORTED void
hlo_sharding_tile_assignment_dimensions(HloShardingPtr hloSharding,
                                        int64_t *dims, int32_t size);
MLIR_CAPI_EXPORTED void
hlo_sharding_tile_assignment_devices(HloShardingPtr hloSharding,
                                     int64_t *devices, int32_t size);
MLIR_CAPI_EXPORTED bool hlo_sharding_check_eq(HloShardingPtr hloSharding,
                                              HloShardingPtr other);

//===----------------------------------------------------------------------===//
// IFRT Sharding
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void free_ifrt_sharding(HeldIfrtShardingPtr sharding);
MLIR_CAPI_EXPORTED HeldIfrtShardingPtr ifrt_sharding_from_xla_hlo_sharding(
    IfrtClientPtr client, IfrtDevicePtr *device_list, int32_t num_devices,
    IfrtMemoryKindPtr memory_kind, HloShardingPtr xla_hlo_sharding);
MLIR_CAPI_EXPORTED HloShardingPtr
ifrt_sharding_to_xla_hlo_sharding(HeldIfrtShardingPtr sharding);
MLIR_CAPI_EXPORTED bool
ifrt_sharding_is_single_device_sharding(HeldIfrtShardingPtr sharding);
MLIR_CAPI_EXPORTED bool
ifrt_sharding_is_fully_replicated(HeldIfrtShardingPtr sharding);
MLIR_CAPI_EXPORTED const char *
ifrt_sharding_to_string(HeldIfrtShardingPtr sharding);
MLIR_CAPI_EXPORTED int32_t
ifrt_sharding_devices_size(HeldIfrtShardingPtr sharding);
MLIR_CAPI_EXPORTED void
ifrt_sharding_to_device_list(HeldIfrtShardingPtr sharding,
                             IfrtDevicePtr *devices);
MLIR_CAPI_EXPORTED void
ifrt_sharding_to_index_domains(HeldIfrtShardingPtr sharding,
                               int64_t *array_size_list, int32_t array_size_len,
                               int64_t *index_domain_origins,
                               int64_t *index_domain_shapes);

//===----------------------------------------------------------------------===//
// Distributed Runtime
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED HeldDistributedRuntimeClientPtr GetDistributedRuntimeClient(
    char *c_address, int32_t node_id, int32_t rpc_timeout_in_seconds,
    int32_t init_timeout, int32_t shutdown_timeout_in_minutes,
    int32_t heartbeat_timeout_in_seconds, bool use_compression);
MLIR_CAPI_EXPORTED void
free_distributed_runtime_client(HeldDistributedRuntimeClientPtr client);
MLIR_CAPI_EXPORTED void
distributed_runtime_client_connect(HeldDistributedRuntimeClientPtr client);
MLIR_CAPI_EXPORTED void
distributed_runtime_client_shutdown(HeldDistributedRuntimeClientPtr client);
MLIR_CAPI_EXPORTED DistributedRuntimeServicePtr GetDistributedRuntimeService(
    char *c_address, int num_nodes, int32_t heartbeat_timeout_in_seconds,
    int32_t cluster_register_timeout_in_minutes,
    int32_t shutdown_timeout_in_minutes);
MLIR_CAPI_EXPORTED void
free_distributed_runtime_service(DistributedRuntimeServicePtr service);
MLIR_CAPI_EXPORTED void
distributed_runtime_service_shutdown(DistributedRuntimeServicePtr service);

//===----------------------------------------------------------------------===//
// Shardy
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED HloShardingPtr
hloShardingFromTensorShardingAttr(MlirAttribute cattr, MlirAttribute cmeshAttr);
MLIR_CAPI_EXPORTED MlirAttribute hloShardingToTensorShardingAttr(
    MlirContext cctx, const void *hloSharding, MlirAttribute cmeshName,
    MlirAttribute cmeshAttr, int64_t rank, const bool *isClosed,
    const int64_t *priority);

MLIR_CAPI_EXPORTED void addSdyPropagationPipeline(
    MlirOpPassManager pm, uint8_t keepShardingRules,
    uint8_t conservativePropagation, uint8_t debugShardingOrigins,
    uint8_t debugPropagationEdgeSharding, uint8_t skipConvertToReshard,
    uint8_t skipInline, uint8_t enableInsertExplicitCollectives);

//===----------------------------------------------------------------------===//
// HLO Module / Computation / Instruction
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED HeldHloModulePtr
convertMlirModuleToHloModule(MlirModule mod);
MLIR_CAPI_EXPORTED HeldHloModulePtr
parseAndReturnUnverifiedHloModule(const char *cstr);
MLIR_CAPI_EXPORTED const char *HloModuleToString(HeldHloModulePtr hlo_module,
                                                 int32_t print_options);
MLIR_CAPI_EXPORTED void FreeHloModule(HeldHloModulePtr hlo_module);

MLIR_CAPI_EXPORTED HloComputationPtr
hloModuleGetEntryComputation(HeldHloModulePtr hlo_module);
MLIR_CAPI_EXPORTED void freeHloComputation(HloComputationPtr hlo_computation);
MLIR_CAPI_EXPORTED const char *
hloComputationToString(HloComputationPtr hlo_computation,
                       int32_t print_options);
MLIR_CAPI_EXPORTED int64_t
hloComputationInstructionCount(HloComputationPtr hlo_computation);
MLIR_CAPI_EXPORTED void
hloComputationGetInstructionsPostOrder(HloComputationPtr hlo_computation,
                                       int64_t num_instructions,
                                       HloInstructionPtr *hlo_instructions);

MLIR_CAPI_EXPORTED void freeHloInstruction(HloInstructionPtr hlo_instruction);
MLIR_CAPI_EXPORTED const char *
hloInstructionToString(HloInstructionPtr hlo_instruction,
                       int32_t print_options);
MLIR_CAPI_EXPORTED uint8_t
hloInstructionHasToApply(HloInstructionPtr hlo_instruction);
MLIR_CAPI_EXPORTED HloComputationPtr
hloInstructionGetToApply(HloInstructionPtr hlo_instruction);
MLIR_CAPI_EXPORTED uint8_t
hloInstructionGetOpcode(HloInstructionPtr hlo_instruction);
MLIR_CAPI_EXPORTED const char *hloOpcodeToString(uint8_t opcode);
MLIR_CAPI_EXPORTED uint8_t
hloInstructionIsFusion(HloInstructionPtr hlo_instruction);
MLIR_CAPI_EXPORTED uint8_t
hloInstructionGetFusionKind(HloInstructionPtr hlo_instruction);
MLIR_CAPI_EXPORTED const char *hloFusionKindToString(uint8_t kind);
MLIR_CAPI_EXPORTED HloComputationPtr
hloInstructionFusedInstructionsComputation(HloInstructionPtr hlo_instruction);

//===----------------------------------------------------------------------===//
// Cost Analysis
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void pjrt_hlo_module_cost_analysis_properties(
    PjRtClientPtr client, HeldHloModulePtr hlo_module,
    struct JLHloCostAnalysisProperties *jlproperties);
MLIR_CAPI_EXPORTED void ifrt_hlo_module_cost_analysis_properties(
    IfrtClientPtr client, HeldHloModulePtr hlo_module,
    struct JLHloCostAnalysisProperties *jlproperties);

//===----------------------------------------------------------------------===//
// GPU Performance Model
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED GPUPerformanceModelPtr CreateGPUPerformanceModel(
    MlirContext ctx, DeviceDescriptionPtr device_description);
MLIR_CAPI_EXPORTED void
RunAnalysisOnHloModule(GPUPerformanceModelPtr gpu_performance_model,
                       HeldHloModulePtr hlo_module);
MLIR_CAPI_EXPORTED void
EstimateRunTimeForInstruction(GPUPerformanceModelPtr gpu_performance_model,
                              HloInstructionPtr hlo_instruction,
                              struct JLEstimateRunTimeData *jldata);

//===----------------------------------------------------------------------===//
// XProf / Profiler
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void
InitializeXProfStubs(const char *cstr_worker_service_address);
MLIR_CAPI_EXPORTED void StartGrpcServer(int port);
MLIR_CAPI_EXPORTED int XSpaceToToolsData(
    const char **xspace_paths, int64_t num_paths, const char *tool_name,
    const char **bool_keys, const bool *bool_values, int64_t bool_count,
    const char **int_keys, const int *int_values, int64_t int_count,
    const char **str_keys, const char **str_values, int64_t str_count,
    char **result_data, int64_t *result_size, bool *is_binary, char **error);

//===----------------------------------------------------------------------===//
// Linkable Runtime  (used by the enzyme-jit path)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void reactantXLAThrow(const char *str);
MLIR_CAPI_EXPORTED void reactantXLAInit(LinkableRuntimePtr *lrtP,
                                        const char *backend);
MLIR_CAPI_EXPORTED void reactantXLADeInit(LinkableRuntimePtr *lrt);
MLIR_CAPI_EXPORTED void reactantXLAMemcpy(LinkableRuntimePtr *lrtP, void *dst,
                                          void *src, size_t size,
                                          int32_t direction);
MLIR_CAPI_EXPORTED void *reactantXLAMalloc(LinkableRuntimePtr *lrtP,
                                           uint64_t ptype, uint64_t shapeLen,
                                           uint64_t *shape);
MLIR_CAPI_EXPORTED void reactantXLAFree(LinkableRuntimePtr *lrtP,
                                        void *buffer0);
MLIR_CAPI_EXPORTED void reactantXLAExec(LinkableRuntimePtr *lrtP,
                                        const char *modstr, int64_t argcnt,
                                        void **args);

//===----------------------------------------------------------------------===//
// Compile / Debug options
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void *ReactantGetDebugOptions(size_t *size);
MLIR_CAPI_EXPORTED void *ReactantGetCompileOptions(size_t *size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // REACTANT_EXTRA_API_H
