#include <iostream>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Pass/PassManager.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Passes/Passes.h"

#include "mlir/CAPI/Support.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/python/profiler_utils.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"

#include "xla/python/ifrt/hlo/hlo_program.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

#include "llvm-c/TargetMachine.h"

// PJRT
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"

// CPU collectives
// #include "xla/backends/cpu/collectives/mpi_collectives.h"
#if defined(__linux__)
#include "gloo/transport/tcp/attr.h"
#include "gloo/transport/tcp/device.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#elif defined(__APPLE__)
#include "gloo/transport/uv/device.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#endif // defined(__linux__)

// shardy
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"
#include "shardy/integrations/c/attributes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"

// IFRT
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"

// IFRT - PJRT
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"

// IFRT - Proxy (RPC)
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"

// Cost Analysis
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_cost_analysis.h"

#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

// Triton did a dumb thing and their import is incompatible
// We don't use so disabling until upstream fix
// #include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/ExtensibleRTTI.h"
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace xla;

namespace mlir {
namespace enzyme {
void registerRemoveTransformPass();
void registerGenerateApplyPatternsPass();
} // namespace enzyme

namespace triton {
class TritonDialect;
}

} // namespace mlir

namespace reactant {

template <typename T> struct unwrap_type {
  typedef T type;
};
template <typename T> struct unwrap_type<std::shared_ptr<T>> {
  typedef T type;
};
template <typename T> struct unwrap_type<tsl::RCReference<T>> {
  typedef T type;
};

template <typename T> using unwrap_type_t = typename unwrap_type<T>::type;

template <typename T> struct HeldValue {
public:
  HeldValue(T &obj) : holded(obj) {}
  ~HeldValue() = default;

  unwrap_type_t<T> *ptr() const { return holded.get(); }

  T obj() const { return holded; }

  T value() const { return holded; }

  unwrap_type_t<T> *operator->() const { return ptr(); }

private:
  T holded;
};

template <typename T> HeldValue<T> *capture(T obj) {
  return new HeldValue<T>(obj);
}

} // namespace reactant

using reactant::HeldValue;
using HeldPjRtClient = HeldValue<std::shared_ptr<xla::PjRtClient>>;
using HeldPjRtBuffer = HeldValue<std::shared_ptr<xla::PjRtBuffer>>;
using HeldIfrtArray = HeldValue<tsl::RCReference<xla::ifrt::Array>>;
using HeldHloModule = HeldValue<std::shared_ptr<xla::HloModule>>;
using HeldIfrtSharding = HeldValue<std::shared_ptr<xla::ifrt::Sharding>>;

extern "C" void (*ReactantThrowError)(const char *) = nullptr;

// Utilities for `StatusOr`.
template <typename T> T MyValueOrThrow(absl::StatusOr<T> v) {
  if (!v.ok()) {
    ReactantThrowError(v.status().ToString().c_str());
  }
  return std::move(v).value();
}

extern "C" void ReactantHandleCuResult(uint32_t curesult) {
  if (curesult != 0) {
    std::string err = "Bad Cuda Result = " + std::to_string(curesult);
    if (ReactantThrowError) {
      ReactantThrowError(err.c_str());
    }
  }
}

// MLIR C-API extras
#pragma region MLIR Extra
extern "C" MlirAttribute mlirComplexAttrDoubleGet(MlirContext ctx,
                                                  MlirType type, double real,
                                                  double imag) {
  return wrap(
      complex::NumberAttr::get(cast<ComplexType>(unwrap(type)), real, imag));
}

extern "C" MlirAttribute mlirComplexAttrDoubleGetChecked(MlirLocation loc,
                                                         MlirType type,
                                                         double real,
                                                         double imag) {
  return wrap(complex::NumberAttr::getChecked(
      unwrap(loc), cast<ComplexType>(unwrap(type)), real, imag));
}

extern "C" bool mlirOperationInject(MlirContext ctx, MlirBlock block,
                                    MlirStringRef code, MlirLocation location,
                                    bool verify_after_parse) {
  ParserConfig config(unwrap(ctx), verify_after_parse);
  if (failed(parseSourceString(unwrap(code), unwrap(block), config)))
    return false;
  return true;
}

extern "C" MlirOperation mlirOperationParse(MlirContext ctx, MlirBlock block,
                                            MlirStringRef code,
                                            MlirLocation location,
                                            bool verify_after_parse) {
  ParserConfig config(unwrap(ctx), verify_after_parse);
  if (failed(parseSourceString(unwrap(code), unwrap(block), config)))
    return MlirOperation{nullptr};
  return MlirOperation{
      mlir::detail::constructContainerOpForParserIfNecessary<Operation *>(
          unwrap(block), config.getContext(), unwrap(location))
          .release()};
}

// TODO mlirComplexAttrGetnValue
// TODO extern "C" MlirTypeID mlirComplexAttrGetTypeID(void) { return
// wrap(complex::NumberAttr::getTypeID()); }

extern "C" void ReactantFuncSetResultAttr(MlirOperation op, intptr_t pos,
                                          MlirStringRef name,
                                          MlirAttribute attr) {
  llvm::cast<mlir::FunctionOpInterface>(unwrap(op))
      .setResultAttr(pos, unwrap(name), unwrap(attr));
}

extern "C" void ReactantFuncSetArgAttr(MlirOperation op, intptr_t pos,
                                       MlirStringRef name, MlirAttribute attr) {
  llvm::cast<mlir::FunctionOpInterface>(unwrap(op))
      .setArgAttr(pos, unwrap(name), unwrap(attr));
}

#pragma endregion

// auxiliar functions
#pragma region utils
template <typename T> const char *cstr_from_string(T text) {
  char *cstr = (char *)malloc(text.size() + 1);
  memcpy(cstr, text.data(), text.size());
  cstr[text.size()] = '\0';
  return cstr;
}

template <typename T>
T *unwrap_absl_statusor(absl::StatusOr<T> status, char **error_msg) {
  *error_msg = nullptr;
  if (!status.ok()) {
    auto str = status.message();
    char *err = (char *)malloc(str.size() + 1);
    memcpy(err, str.data(), str.size() + 1);
    *error_msg = err;
    return nullptr;
  }
  return status.value();
}
#pragma endregion

// int google::protobuf::io::CodedInputStream::default_recursion_limit_ = 100;
// int xla::_LayoutProto_default_instance_;

extern "C" void InitializeLogs() {
  const char *binary = "julia";
  int argc = 1;
  char *argv[] = {(char *)binary};
  char **argv2 = &argv[0];
  tsl::port::InitMain(binary, &argc, &argv2);
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();

  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
  LLVMInitializeAArch64AsmParser();
}

extern "C" void SetLogLevel(int level) {
  SetStderrThreshold((absl::LogSeverity)level);
  // absl::SetGlobalVLogLevel(level);
}

extern "C" void SetModuleLogLevel(const char *module_pattern, int level) {
  // absl::SetVLOGLevel(module_pattern, level);
}

extern "C" char *GetDefaultTargetTriple(void) {
  return LLVMGetDefaultTargetTriple();
}

extern "C" MLIR_CAPI_EXPORTED MlirAttribute
enzymeActivityAttrGet(MlirContext ctx, int32_t val) {
  return wrap(mlir::enzyme::ActivityAttr::get(unwrap(ctx),
                                              (mlir::enzyme::Activity)val));
}

// Create profiler session and start profiling
extern "C" tsl::ProfilerSession *
CreateProfilerSession(uint32_t device_tracer_level,
                      uint32_t host_tracer_level) {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_tracer_level(device_tracer_level);
  options.set_host_tracer_level(host_tracer_level);
  auto sess = tsl::ProfilerSession::Create(options);
  return sess.release();
}

extern "C" void ProfilerSessionCollectData(tsl::ProfilerSession *session,
                                           const char *path) {
  tensorflow::profiler::XSpace xspace;
  auto status = session->CollectData(&xspace);
  if (!status.ok())
    ReactantThrowError("cannot collect data for profiler");
  tsl::profiler::ExportToTensorBoard(xspace, path,
                                     /*also_export_trace_json*/ true);
}

extern "C" void ProfilerSessionDelete(tsl::ProfilerSession *session) {
  delete session;
}

extern "C" int64_t ProfilerActivityStart(const char *name, int level) {
  return tsl::profiler::TraceMe::ActivityStart(name, level);
}

extern "C" void ProfilerActivityEnd(int64_t id) {
  tsl::profiler::TraceMe::ActivityEnd(id);
}

extern "C" tsl::profiler::ProfilerServer *ProfilerServerStart(int32_t port) {
  auto server = new tsl::profiler::ProfilerServer();
  server->StartProfilerServer(port);
  return server;
}

extern "C" void ProfilerServerStop(tsl::profiler::ProfilerServer *server) {
  delete server;
}

PjRtClient *MakeCPUClientInternal(
    uint8_t asynchronous, int node_id,
    std::optional<std::shared_ptr<xla::cpu::CpuCollectives>> collectives) {
  CpuClientOptions options;

  options.process_id = node_id;
  options.asynchronous = asynchronous != 0;

  if (collectives.has_value())
    options.collectives = collectives.value();

  auto client = MyValueOrThrow(GetTfrtCpuClient(options));
  return client.release();
}

extern "C" PjRtClient *MakeCPUClient(uint8_t asynchronous, int node_id) {
  std::optional<std::shared_ptr<xla::cpu::CpuCollectives>> collectives;
  return MakeCPUClientInternal(asynchronous, node_id, collectives);
}

// xla/python/xla.cc 390
extern "C" PjRtClient *
MakeGPUClient(int node_id, int num_nodes, int64_t *allowed_devices,
              int64_t num_allowed_devices, double memory_fraction,
              bool preallocate, const char *platform_name, const char **error,
              void *distributed_runtime_client) {
  GpuClientOptions options;

  if (num_nodes > 1) {
    if (distributed_runtime_client == nullptr) {
      *error =
          "`distributed_runtime_client` must be non-null if `num_nodes` > 1";
      return nullptr;
    }
    auto typed_distributed_runtime_client = static_cast<
        HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *>(
        distributed_runtime_client);
    options.kv_store = GetDistributedKeyValueStore(
        typed_distributed_runtime_client->obj(), /*key_prefix=*/"");
  }

  // options.allocator_config =
  options.allocator_config.preallocate = preallocate;
  options.allocator_config.memory_fraction = memory_fraction;
  options.node_id = node_id;
  options.num_nodes = num_nodes;
  if (allowed_devices) {
    std::set<int> allowed_devices_set;
    for (int i = 0; i < num_allowed_devices; i++) {
      allowed_devices_set.insert(static_cast<int>(allowed_devices[i]));
    }
    options.allowed_devices = allowed_devices_set;
  } else {
    options.allowed_devices = std::optional<std::set<int>>();
  }
  options.platform_name =
      platform_name ? std::string(platform_name) : std::optional<std::string>();
  // options.collectives = num_nodes;
  auto clientErr = GetStreamExecutorGpuClient(options);

  if (!clientErr.ok()) {
    auto str = clientErr.status().message();
    char *err = (char *)malloc(str.size() + 1);
    memcpy(err, str.data(), str.size() + 1);
    *error = err;
    return nullptr;
  } else {
    auto client = std::move(clientErr).value();
    return client.release();
  }
}

const char *const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";

extern "C" const PJRT_Api *LoadPjrtPlugin(const char *device_type,
                                          const char *library_path,
                                          const char **error) {
  absl::StatusOr<const PJRT_Api *> pluginLoad =
      pjrt::LoadPjrtPlugin(std::string(device_type), std::string(library_path));
  if (!pluginLoad.ok()) {
    auto str = pluginLoad.status().message();
    char *err = (char *)malloc(str.size() + 1);
    memcpy(err, str.data(), str.size() + 1);
    *error = err;
    return nullptr;
  }
  return pluginLoad.value();
}

extern "C" int InitializePjrtPlugin(const char *device_type,
                                    const char **error) {
  absl::Status tpu_status = pjrt::InitializePjrtPlugin(device_type);
  if (!tpu_status.ok()) {
    auto str = tpu_status.message();
    char *err = (char *)malloc(str.size() + 1);
    memcpy(err, str.data(), str.size() + 1);
    *error = err;
    return 1;
  }
  return 0;
}

extern "C" PjRtClient *GetCApiClient(const char *device_type) {
  return xla::GetCApiClient(device_type).value().release();
}

extern "C" void pjrt_client_register_profiler(const PJRT_Api *api) {
  RegisterProfiler(api);
}

extern "C" PjRtClient *MakeClientUsingPluginAPI(const char *device_type,
                                                const char *library_path,
                                                const char *client_name,
                                                const char **error) {
  const PJRT_Api *pluginLoad = LoadPjrtPlugin(device_type, library_path, error);
  if (pluginLoad == nullptr)
    return nullptr;
  if (InitializePjrtPlugin(device_type, error) == 1)
    return nullptr;

  RegisterProfiler(pluginLoad);
  return GetCApiClient(client_name);
}

extern "C" PjRtClient *MakeTPUClient(const char *tpu_path, const char **error) {
  // Prefer $TPU_LIBRARY_PATH if set
  std::string tpu_library_path;
  if (auto path = llvm::sys::Process::GetEnv(kEnvTpuLibraryPath)) {
    tpu_library_path = *path;
  } else if (tpu_path) {
    tpu_library_path = std::string(tpu_path);
  } else {
    *error = "Could not find TPU path";
    return nullptr;
  }

  return MakeClientUsingPluginAPI("tpu", tpu_library_path.c_str(), "TPU",
                                  error);
}

extern "C" int ClientNumDevices(PjRtClient *client) {
  return client->device_count();
}

extern "C" int ClientNumAddressableDevices(PjRtClient *client) {
  return client->addressable_device_count();
}

extern "C" int ClientProcessIndex(PjRtClient *client) {
  return client->process_index();
}

extern "C" PjRtDevice *ClientGetDevice(PjRtClient *client, int device_id) {
  return MyValueOrThrow(client->LookupDevice(PjRtGlobalDeviceId(device_id)));
}

extern "C" PjRtDevice *ClientGetAddressableDevice(PjRtClient *client,
                                                  int device_id) {
  return MyValueOrThrow(
      client->LookupAddressableDevice(PjRtLocalDeviceId(device_id)));
}

extern "C" const char *ClientGetPlatformName(PjRtClient *client) {
  return cstr_from_string(client->platform_name());
}

extern "C" const char *DeviceGetKind(PjRtDevice *device) {
  return cstr_from_string(device->device_kind());
}

extern "C" void ClientGetDevices(PjRtClient *client, PjRtDevice **out_devices) {
  auto span = client->devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

extern "C" void ClientGetAddressableDevices(PjRtClient *client,
                                            PjRtDevice **out_devices) {
  auto span = client->addressable_devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

// To keep in sync with JLAllocatorStats in src/XLA.jl
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

extern "C" void PjRtDeviceGetAllocatorStats(PjRtDevice *device,
                                            JLAllocatorStats *jlstats) {
  auto stats = MyValueOrThrow(device->GetAllocatorStats());
  int64_t optnull = std::numeric_limits<int64_t>::min();

  jlstats->num_allocs = stats.num_allocs;
  jlstats->bytes_in_use = stats.bytes_in_use;
  jlstats->peak_bytes_in_use = stats.peak_bytes_in_use;
  jlstats->largest_alloc_size = stats.largest_alloc_size;
  jlstats->bytes_limit = stats.bytes_limit.value_or(optnull);
  jlstats->bytes_reserved = stats.bytes_reserved;
  jlstats->peak_bytes_reserved = stats.peak_bytes_reserved;
  jlstats->bytes_reservable_limit =
      stats.bytes_reservable_limit.value_or(optnull);
  jlstats->largest_free_block_bytes = stats.largest_free_block_bytes;
  jlstats->pool_bytes = stats.pool_bytes.value_or(optnull);
  jlstats->peak_pool_bytes = stats.peak_pool_bytes.value_or(optnull);
}

extern "C" void ifrt_device_get_allocator_stats(ifrt::Device *device,
                                                JLAllocatorStats *jlstats) {
  if (!llvm::isa<ifrt::PjRtDevice>(device)) {
    ReactantThrowError(
        "ifrt_device_get_allocator_stats: only supported for ifrt-pjrt.");
  }
  auto ifrt_pjrt_device = llvm::dyn_cast<ifrt::PjRtDevice>(device);
  PjRtDeviceGetAllocatorStats(ifrt_pjrt_device->pjrt_device(), jlstats);
}

extern "C" void ExecutableFree(xla::PjRtLoadedExecutable *exec) { delete exec; }

extern "C" PjRtDevice *BufferToDevice(PjRtBuffer *Buffer) {
  return Buffer->device();
}

extern "C" PjRtClient *BufferToClient(PjRtBuffer *Buffer) {
  return Buffer->client();
}

extern "C" const int64_t *BufferShape(PjRtBuffer *Buffer) {
  return Buffer->dimensions().data();
}

extern "C" int64_t BufferNDimensions(PjRtBuffer *Buffer) {
  return Buffer->dimensions().length();
}

extern "C" xla::PrimitiveType BufferPrimitiveType(PjRtBuffer *Buffer) {
  return Buffer->element_type();
}

extern "C" void PjRtBufferFree(PjRtBuffer *Buffer) { delete Buffer; }

extern "C" PjRtClient *DeviceToClient(PjRtDevice *Device) {
  return Device->client();
}

extern "C" PjRtClient *
PjRtLoadedExecutableGetClient(PjRtLoadedExecutable *exec) {
  return exec->client();
}

// https://openxla.org/xla/shapes
// This minor-to-major dimension order of 0 up to N-1 is akin to column-major
// (at rank 2). Assuming a monotonic ordering of dimensions, another way we may
// refer to this layout in the code is simply "dim 0 is minor".
std::vector<int64_t> col_major(int64_t dim) {
  std::vector<int64_t> minor_to_major;
  for (int i = 0; i < dim; i++) {
    minor_to_major.push_back(i); // dim-1-i);
    // minor_to_major.push_back(dim-1-i);
  }
  return minor_to_major;
}

extern "C" void ReactantLLVMParseCommandLineOptions(int argc,
                                                    const char *const *argv,
                                                    const char *Overview) {
  llvm::cl::ParseCommandLineOptions(argc, argv, StringRef(Overview),
                                    &llvm::nulls());
}

std::vector<int64_t> row_major(int64_t dim) {
  std::vector<int64_t> minor_to_major;
  for (int i = 0; i < dim; i++) {
    minor_to_major.push_back(dim - 1 - i);
  }
  return minor_to_major;
}
static void noop() {}

#ifdef REACTANT_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
extern "C" int32_t ReactantCudaDriverGetVersion() {
  int32_t data;
  ReactantHandleCuResult(cuDriverGetVersion(&data));
  return data;
}
extern "C" int32_t ReactantHermeticCudaGetVersion() { return CUDA_VERSION; }
#else
extern "C" int32_t ReactantCudaDriverGetVersion() { return 0; }
extern "C" int32_t ReactantHermeticCudaGetVersion() { return 0; }
#endif

extern "C" void *UnsafeBufferPointer(PjRtBuffer *buffer) {
  auto unsafe = MyValueOrThrow(buffer->client()->UnsafeBufferPointer(buffer));
  return (void *)unsafe;
}

extern "C" PjRtBuffer *ArrayFromHostBuffer(PjRtClient *client, void *data,
                                           uint64_t ptype, size_t dim,
                                           int64_t *cshape,
                                           PjRtDevice *device) {
  auto primtype = (xla::PrimitiveType)ptype;
  absl::Span<const int64_t> shape(cshape, dim);
  PjRtClient::HostBufferSemantics semantics =
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
  // xla::Layout layout(col_major(dim));
  // auto buffer = xla::MyValueOrThrow(client->BufferFromHostBuffer(data,
  // primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device,
  // &layout));
  llvm::errs() << " from host buffer\n";
  llvm::errs() << " dev: " << device->ToString() << "\n";
  llvm::errs() << " ms: " << *device->default_memory_space() << "\n";
  const xla::Layout *layout = nullptr;
  auto buffer = MyValueOrThrow(client->BufferFromHostBuffer(
      data, primtype, shape, /*byte_strides*/ {}, semantics, /*ondone*/ {},
      MyValueOrThrow(device->default_memory_space()), layout));
  auto bres = buffer.release();
  return bres;
}

extern "C" uint8_t BufferOnCPU(PjRtBuffer *buffer) { return buffer->IsOnCpu(); }

extern "C" PjRtBuffer *CopyBufferToDevice(PjRtBuffer *buffer,
                                          PjRtDevice *dst_device) {
  auto res = MyValueOrThrow(
      buffer->CopyToMemorySpace(*dst_device->default_memory_space()));
  return res.release();
}

extern "C" void BufferToHost(PjRtBuffer *buffer, void *data) {
  Shape shape(MyValueOrThrow(buffer->HostShape()));
  /// Grumpily the cpu copy code does not respect layout and does a raw copy
  /// For now, we assume a non-julia row major ordering
  /// If in the future it supports col_major we can swap to that.
  *shape.mutable_layout() = xla::Layout(row_major(shape.dimensions_size()));
  MutableBorrowingLiteral literal((const char *)data, shape);
  auto status = buffer->ToLiteralSync(&literal);
  if (!status.ok()) {
    printf("error copying to host: %s\n", status.ToString().c_str());
  }
}

extern "C" void FreeClient(PjRtClient *client) { delete client; }

extern "C" int64_t PjRtDeviceGetLocalDeviceId(PjRtDevice *device) {
  return device->local_device_id().value();
}

extern "C" int64_t PjRtDeviceGetGlobalDeviceId(PjRtDevice *device) {
  return device->global_device_id().value();
}

extern "C" int64_t PjRtDeviceGetLocalHardwareId(PjRtDevice *device) {
  return device->local_hardware_id().value();
}

#include "xla/service/custom_call_target_registry.h"
extern "C" void RegisterCustomCallTarget(const char *name, void *address,
                                         const char *platform) {
  CustomCallTargetRegistry::Global()->Register(std::string(name), address,
                                               std::string(platform));
}

#include "mlir/Target/LLVMIR/Import.h"
extern "C" MlirModule ConvertLLVMToMLIR(LLVMModuleRef lmod, MlirContext cctx) {
  auto llvmModule = std::unique_ptr<llvm::Module>(llvm::unwrap(lmod));
  mlir::MLIRContext &context = *unwrap(cctx);

  auto res = mlir::translateLLVMIRToModule(std::move(llvmModule), &context,
                                           /*emitExpensiveWarnings*/ false,
                                           /*dropDICompositeElements*/ false)
                 .release();
  return wrap(res);
}

#include "llvm/IRReader/IRReader.h"
extern "C" MlirModule ConvertLLVMStrToMLIR(const char *lmod, MlirContext cctx) {
  llvm::LLVMContext Context;
  llvm::SMDiagnostic Err;
  auto llvmModule =
      llvm::parseIR(llvm::MemoryBufferRef(lmod, "conversion"), Err, Context);
  if (!llvmModule) {
    std::string err_str;
    llvm::raw_string_ostream err_stream(err_str);
    Err.print(/*ProgName=*/"LLVMToMLIR", err_stream);
    err_stream.flush();
    if (ReactantThrowError) {
      llvm::errs() << lmod << "\n";
      ReactantThrowError(err_str.c_str());
      return wrap((mlir::ModuleOp) nullptr);
    }
  }
  mlir::MLIRContext &context = *unwrap(cctx);
  auto res = mlir::translateLLVMIRToModule(std::move(llvmModule), &context,
                                           /*emitExpensiveWarnings*/ false,
                                           /*dropDICompositeElements*/ false)
                 .release();
  if (!res) {
    llvm::errs() << lmod << "\n";
    ReactantThrowError("Could not translate LLVM IR to MLIR Module");
  }
  return wrap(res);
}

typedef PjRtFuture<> FutureType;
extern "C" void FreeFuture(FutureType *Future) { delete Future; }

extern "C" uint8_t FutureIsReady(FutureType *Future) {
  return Future->IsReady();
}

extern "C" void FutureAwait(FutureType *Future) { Future->Await(); }

xla::CompileOptions
GenerateCompileOptions(int64_t device_id, const int64_t *mesh_ids,
                       int64_t num_mesh_ids, const char *xla_gpu_cuda_data_dir,
                       bool use_shardy_partitioner, int64_t num_replicas,
                       int64_t num_partitions, bool use_spmd_partitioning) {
  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_cuda_data_dir(xla_gpu_cuda_data_dir);

  options.executable_build_options.set_num_replicas(num_replicas);
  options.executable_build_options.set_num_partitions(num_partitions);

  if (device_id < 0) {
    if (num_replicas == 1 && num_partitions == 1) {
      llvm::errs()
          << "[libReactantExtra] num_replicas & num_partitions are both 1, but "
             "device_id is negative. This can happen if you are sharding with "
             "a single device.\n";
    }

    assert(num_replicas * num_partitions == num_mesh_ids);

    options.executable_build_options.set_use_spmd_partitioning(
        use_spmd_partitioning);
    options.executable_build_options.set_use_shardy_partitioner(
        use_shardy_partitioner);

    // auto partitioning for GPUs is not available in open source version of XLA
    // options.executable_build_options.set_use_auto_spmd_partitioning(true);
    // std::vector<int64_t> mesh_shape_vec(mesh_shape, mesh_shape +
    // num_mesh_shape);
    // options.executable_build_options.set_auto_spmd_partitioning_mesh_shape(mesh_shape_vec);
    // std::vector<int64_t> mesh_ids_vec(mesh_ids, mesh_ids + num_mesh_ids);
    // options.executable_build_options.set_auto_spmd_partitioning_mesh_ids(mesh_ids_vec);

    xla::DeviceAssignment device_assignment(num_replicas, num_partitions);
    for (int64_t i = 0; i < num_replicas; ++i) {
      for (int64_t j = 0; j < num_partitions; ++j) {
        int64_t mesh_id = mesh_ids[i * num_partitions + j];
        assert(mesh_id >= 0);
        device_assignment(i, j) = mesh_id;
      }
    }
    options.executable_build_options.set_device_assignment(device_assignment);

    options.executable_build_options
        .set_allow_spmd_sharding_propagation_to_parameters({false});
    options.executable_build_options
        .set_allow_spmd_sharding_propagation_to_output({false});
  } else {
    assert(device_id >= 0);
    assert(num_replicas == 1);
    assert(num_partitions == 1);

    options.executable_build_options.set_device_ordinal(device_id);

    xla::DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = device_id;
    options.executable_build_options.set_device_assignment(device_assignment);
  }

  return options;
}

extern "C" xla::PjRtLoadedExecutable *
ClientCompile(PjRtClient *client, MlirModule cmod, int64_t device_id,
              const int64_t *mesh_ids, int64_t num_mesh_ids,
              const char *xla_gpu_cuda_data_dir, bool use_shardy_partitioner,
              int64_t num_replicas, int64_t num_partitions,
              bool use_spmd_partitioning) {
  CompileOptions options = GenerateCompileOptions(
      device_id, mesh_ids, num_mesh_ids, xla_gpu_cuda_data_dir,
      use_shardy_partitioner, num_replicas, num_partitions,
      use_spmd_partitioning);

  mlir::ModuleOp cmod_op = cast<ModuleOp>(*unwrap(cmod));

  if (use_spmd_partitioning && use_shardy_partitioner) {
    // https://github.com/openxla/xla/blob/b3c641b05692f3712fb3c272e38665fdfa28bdf8/xla/python/py_client.cc#L460
    auto status = xla::ExportShardyForHloRoundTrip(cmod_op);
    if (!status.ok()) {
      ReactantThrowError(status.ToString().c_str());
    }
  }

  auto exec_err = client->CompileAndLoad(cmod_op, options);

  if (!exec_err.ok()) {
    std::string err_str;
    llvm::raw_string_ostream err_stream(err_str);
    err_stream << cmod_op << "\n";
    err_stream << exec_err.status().ToString();
    ReactantThrowError(err_stream.str().c_str());
  }
  return std::move(exec_err).value().release();
}

extern "C" void
PjRtLoadedExecutableGetOuputShardings(xla::PjRtLoadedExecutable *exec,
                                      xla::OpSharding **op_shardings,
                                      int32_t num_op_shardings) {
  std::optional<std::vector<OpSharding>> shardings = exec->GetOutputShardings();
  if (!shardings.has_value()) {
    ReactantThrowError(
        "No sharding found for the output of the loaded executable");
  }

  std::vector<xla::OpSharding> hlo_op_shardings = shardings.value();
  if (num_op_shardings != hlo_op_shardings.size()) {
    ReactantThrowError(("Expected " + std::to_string(num_op_shardings) +
                        " shardings, got " +
                        std::to_string(hlo_op_shardings.size()))
                           .c_str());
  }

  for (int32_t i = 0; i < num_op_shardings; i++) {
    op_shardings[i] = new xla::OpSharding(hlo_op_shardings[i]);
  }
}

extern "C" void
PjRtLoadedExecutableGetParameterShardings(xla::PjRtLoadedExecutable *exec,
                                          xla::OpSharding **op_shardings,
                                          int32_t num_op_shardings) {
  std::optional<std::vector<OpSharding>> shardings =
      exec->GetParameterShardings();
  if (!shardings.has_value()) {
    ReactantThrowError(
        "No sharding found for the output of the loaded executable");
  }

  std::vector<xla::OpSharding> hlo_op_shardings = shardings.value();
  if (num_op_shardings != hlo_op_shardings.size()) {
    ReactantThrowError(("Expected " + std::to_string(num_op_shardings) +
                        " shardings, got " +
                        std::to_string(hlo_op_shardings.size()))
                           .c_str());
  }

  for (int32_t i = 0; i < num_op_shardings; i++) {
    op_shardings[i] = new xla::OpSharding(hlo_op_shardings[i]);
  }
}

extern "C" void XLAExecuteSharded(xla::PjRtLoadedExecutable *exec, int num_args,
                                  PjRtBuffer **op_args, PjRtDevice *device,
                                  uint8_t *is_arg_donatable, int num_results,
                                  PjRtBuffer **op_results, uint8_t *futures,
                                  FutureType **future_results) {
  // Create a vector of PjRtBuffer* from the input array.
  std::vector<PjRtBuffer *> argument_handles(op_args, op_args + num_args);

  // Set up execution options.
  ExecuteOptions options;
  for (size_t i = 0; i < num_args; i++) {
    if (!is_arg_donatable[i]) {
      options.non_donatable_input_indices.insert(static_cast<int>(i));
    }
  }
  options.untuple_result = true;

  // Optional future to hold asynchronous execution results.
  std::optional<PjRtFuture<>> returned_future;

  auto results = MyValueOrThrow(exec->ExecuteSharded(argument_handles, device,
                                                     options, returned_future,
                                                     /*fill_future=*/true));

  // Validate the number of results.
  if (results.size() != num_results) {
    ReactantThrowError(
        ("Error: results.size()=" + std::to_string(results.size()) +
         " does not match num_results=" + std::to_string(num_results) + "\n")
            .c_str());
  }

  // Handle futures if they are returned.
  auto future_val = returned_future.has_value();
  *futures = future_val;
  if (future_val) {
    for (size_t i = 0; i < num_results; i++) {
      future_results[i] = new FutureType(*returned_future);
    }
  }

  // Release the results into the output array.
  for (size_t i = 0; i < num_results; i++) {
    op_results[i] = results[i].release();
  }
}

// This isn't exposed to julia, but leaving it here since it is very useful for
// debugging sharding (and generally for the execute workflow)
void PrintPjRtBuffer(PjRtBuffer *buffer) {
  if (buffer) {
    xla::Shape shape = MyValueOrThrow(buffer->HostShape());
    auto dims = shape.dimensions();
    auto nelems = std::accumulate(dims.begin(), dims.end(), 1,
                                  std::multiplies<int64_t>());
    std::vector<float> host_data(nelems);
    BufferToHost(buffer, host_data.data());

    for (int i = 0; i < nelems; ++i) {
      std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "    Buffer is nullptr" << std::endl;
  }
  return;
}

extern "C" void XLAExecute(xla::PjRtLoadedExecutable *exec, int op_args_len,
                           PjRtBuffer **op_args, uint8_t *is_arg_donatable,
                           int num_results, PjRtBuffer **op_results,
                           uint8_t *futures, FutureType **future_results) {
  xla::DeviceAssignment device_assignment = exec->device_assignment();
  int num_devices = device_assignment.computation_count();

  // Ensure argument_handles is structured as num_devices x num_args
  std::vector<std::vector<PjRtBuffer *>> argument_handles(num_devices);
  int num_args = op_args_len / num_devices;

  // Distribute arguments across devices
  for (int device_idx = 0; device_idx < num_devices; ++device_idx) {
    argument_handles[device_idx].reserve(num_args);
    for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {
      argument_handles[device_idx].push_back(
          op_args[device_idx * num_args + arg_idx]);
    }
  }

  ExecuteOptions options;

  for (size_t i = 0; i < num_args; i++) {
    if (!is_arg_donatable[i])
      options.non_donatable_input_indices.insert((int)i);
  }
  options.untuple_result = true;

  std::optional<std::vector<FutureType>> returned_futures =
      std::vector<FutureType>();
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results =
      MyValueOrThrow(exec->Execute(
          static_cast<absl::Span<const std::vector<PjRtBuffer *>>>(
              argument_handles),
          options, returned_futures));

  if (results.size() != num_devices) {
    ReactantThrowError((" results.size()=" + std::to_string(results.size()) +
                        " num_devices=" + std::to_string(num_devices) + "\n")
                           .c_str());
  }

  for (int device_idx = 0; device_idx < num_devices; ++device_idx) {
    // Remove mesh_id lookup since we're using device_idx ordering
    if (results[device_idx].size() != num_results) {
      ReactantThrowError(
          (" results[" + std::to_string(device_idx) +
           "].size()=" + std::to_string(results[device_idx].size()) +
           " num_results=" + std::to_string(num_results) + "\n")
              .c_str());
    }
  }

  // Handle returned futures
  auto future_val = returned_futures.has_value();
  *futures = future_val;
  if (future_val) {
    if (returned_futures->size() != num_devices) {
      ReactantThrowError((" returned_futures->size()=" +
                          std::to_string(returned_futures->size()) +
                          " num_devices=" + std::to_string(num_devices) + "\n")
                             .c_str());
    }
  }

  // Copy results into the output buffers
  for (int device_idx = 0; device_idx < num_devices; ++device_idx) {
    for (int result_idx = 0; result_idx < num_results; ++result_idx) {
      int flat_index = device_idx * num_results + result_idx;
      op_results[flat_index] = results[device_idx][result_idx].release();
      if (future_val) {
        future_results[flat_index] =
            new FutureType((*returned_futures)[device_idx]);
      }
    }
  }
}

extern "C" int PjRtLoadedExecutableNumReplicas(PjRtLoadedExecutable *exec) {
  return exec->num_replicas();
}

extern "C" int PjRtLoadedExecutableNumPartitions(PjRtLoadedExecutable *exec) {
  return exec->num_partitions();
}

void prepareRegistry(mlir::DialectRegistry &registry);

extern "C" void RegisterDialects(MlirContext cctx) {
  mlir::MLIRContext &context = *unwrap(cctx);
  DialectRegistry registry;
  prepareRegistry(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::enzyme::EnzymeDialect>();
  context.loadDialect<mlir::enzymexla::EnzymeXLADialect>();
  // context.loadDialect<mlir::triton::TritonDialect>();
  context.loadDialect<mlir::tpu::TPUDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  context.loadDialect<mlir::sdy::SdyDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
}

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"

extern "C" void InitializePasses(MlirDialectRegistry creg) {
  mlir::registerenzymePasses();
  enzyme::registerenzymexlaPasses();

  // Register the standard passes we want.
  mlir::registerTransformsPasses();
  mlir::registerLowerAffinePass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerConvertSCFToOpenMPPass();
  mlir::affine::registerAffinePasses();
  mlir::registerReconcileUnrealizedCastsPass();

  /*
    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
      LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
      LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
      LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
      LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
      MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
      LLVM::LLVMStructType::attachInterface<
          PtrElementModel<LLVM::LLVMStructType>>(*ctx);
      LLVM::LLVMPointerType::attachInterface<
          PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
      LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
          *ctx);
    });
    */

  // Transform dialect and extensions.
  mlir::transform::registerInterpreterPass();
  mlir::enzyme::registerGenerateApplyPatternsPass();
  mlir::enzyme::registerRemoveTransformPass();

  // xla + shardy specific passes
  xla::sdy::registerSdyRoundTripExportPipeline();
  xla::sdy::registerSdyRoundTripImportPipeline();
  mlir::sdy::registerAllSdyPassesAndPipelines();
  xla::sdy::registerStablehloExportPipeline();
  xla::sdy::registerStablehloImportPipeline();
  xla::sdy::registerStablehloImportShardingsPass();
}

extern "C" void InitializeRegistry(MlirDialectRegistry creg) {
  mlir::DialectRegistry &registry = *unwrap(creg);
  prepareRegistry(registry);

  mlir::registerLLVMDialectImport(registry);
  mlir::registerNVVMDialectImport(registry);
  mlir::LLVM::registerInlinerInterface(registry);
}

/// Returns an unused symbol in `module` for `oldSymbolName` by trying numeric
/// suffix in `lastUsedID`.
static mlir::StringAttr renameSymbol(llvm::StringRef oldSymName,
                                     unsigned &lastUsedID,
                                     mlir::ModuleOp source,
                                     mlir::ModuleOp target) {
  using namespace llvm;
  using namespace mlir;
  SmallString<64> newSymName(oldSymName);
  newSymName.push_back('_');
  while (true) {
    auto possible = newSymName + Twine(++lastUsedID);
    if (!SymbolTable::lookupSymbolIn(source, possible.str()) &&
        !SymbolTable::lookupSymbolIn(target, possible.str())) {
      return StringAttr::get(target.getContext(), possible);
    }
  }
}

/// Checks if a symbol with the same name as `op` already exists in `source`.
/// If so, renames `op` and updates all its references in `target`.
static mlir::LogicalResult updateSymbolAndAllUses(mlir::SymbolOpInterface op,
                                                  mlir::ModuleOp source,
                                                  mlir::ModuleOp target,
                                                  unsigned &lastUsedID,
                                                  bool &shouldRemove) {
  using namespace llvm;
  using namespace mlir;

  auto opName = op.getName().str();

  if (!SymbolTable::lookupSymbolIn(target, opName)) {
    return success();
  }

  if (auto func = dyn_cast<FunctionOpInterface>(op.getOperation())) {
    if (func.isExternal()) {
      shouldRemove = true;
      return success();
    }
  }

  StringAttr newSymName = renameSymbol(opName, lastUsedID, source, target);

  if (failed(SymbolTable::replaceAllSymbolUses(op, newSymName, source)))
    return op.emitError("unable to update all symbol uses for ")
           << opName << " to " << newSymName;

  SymbolTable::setSymbolName(op, newSymName);
  return success();
}

extern "C" MlirOperation LinkInModule(MlirModule prevModC, MlirModule newModC,
                                      const char *entryfn) {
  auto prevMod = cast<ModuleOp>(*unwrap(prevModC));
  auto newMod = cast<ModuleOp>(*unwrap(newModC));

  Operation *entryFn = nullptr;

  unsigned lastUsedID = 0;

  for (auto &op : make_early_inc_range(*newMod.getBody())) {
    auto symbolOp = dyn_cast<SymbolOpInterface>(op);
    if (!symbolOp)
      continue;

    StringRef oldSymName = symbolOp.getName();

    if (oldSymName == entryfn) {
      entryFn = &op;
    }

    bool shouldRemove = false;
    if (failed(updateSymbolAndAllUses(symbolOp, newMod, prevMod, lastUsedID,
                                      shouldRemove))) {
      assert(0 && "failed to update all uses");
    }
    if (shouldRemove)
      op.erase();
    else
      SymbolTable::setSymbolVisibility(&op, SymbolTable::Visibility::Private);
  }
  prevMod.getBody()->getOperations().splice(
      prevMod.getBody()->getOperations().end(),
      newMod.getBody()->getOperations());
  return wrap(entryFn);
}

extern "C" void pjrt_client_dtor(HeldPjRtClient *client) { delete client; }

extern "C" int pjrt_client_num_devices(HeldPjRtClient *client) {
  return client->ptr()->device_count();
}

extern "C" int pjrt_client_num_addressable_devices(HeldPjRtClient *client) {
  return client->ptr()->addressable_device_count();
}

extern "C" int pjrt_client_pid(HeldPjRtClient *client) {
  return client->ptr()->process_index();
}

extern "C" PjRtDevice *pjrt_client_get_device(HeldPjRtClient *client,
                                              int device_id) {
  return ClientGetDevice(client->ptr(), device_id);
}

extern "C" PjRtDevice *
pjrt_client_get_addressable_device(HeldPjRtClient *client, int device_id) {
  return ClientGetAddressableDevice(client->ptr(), device_id);
}

extern "C" const char *pjrt_client_platform_name(HeldPjRtClient *client) {
  return ClientGetPlatformName(client->ptr());
}

// deprecated
// extern "C" HeldValue<std::shared_ptr<xla::PjRtBuffer>> *
// reactant_hold_pjrtbuffer(xla::PjRtBuffer *buffer) {
//   return reactant::capture(std::shared_ptr<xla::PjRtBuffer>(buffer));
// }

extern "C" HeldPjRtBuffer *pjrt_buffer_from_host(HeldPjRtClient *client,
                                                 void *data, uint64_t ptype,
                                                 size_t dim, int64_t *cshape,
                                                 PjRtDevice *device) {
  PjRtBuffer *buffer =
      ArrayFromHostBuffer(client->ptr(), data, ptype, dim, cshape, device);
  return reactant::capture(std::shared_ptr<PjRtBuffer>(buffer));
}

extern "C" void pjrt_buffer_dtor(HeldPjRtBuffer *buffer) { delete buffer; }

extern "C" void *pjrt_buffer_unsafe_buffer_pointer(HeldPjRtBuffer *buffer) {
  return UnsafeBufferPointer(buffer->ptr());
}

extern "C" bool pjrt_buffer_is_on_cpu(HeldPjRtBuffer *buffer) {
  return buffer->ptr()->IsOnCpu();
}

extern "C" HeldPjRtBuffer *pjrt_buffer_copy_to_device(HeldPjRtBuffer *buffer,
                                                      PjRtDevice *dst_device) {
  PjRtBuffer *ret = CopyBufferToDevice(buffer->ptr(), dst_device);
  return reactant::capture(std::shared_ptr<PjRtBuffer>(ret));
}

extern "C" void pjrt_buffer_to_host(HeldPjRtBuffer *buffer, void *data) {
  BufferToHost(buffer->ptr(), data);
}

extern "C" void pjrt_buffer_print(HeldPjRtBuffer *buffer) {
  PrintPjRtBuffer(buffer->ptr());
}

extern "C" PjRtDevice *pjrt_buffer_get_device(HeldPjRtBuffer *buffer) {
  return buffer->ptr()->device();
}

extern "C" HeldPjRtClient *pjrt_buffer_get_client(HeldPjRtBuffer *buffer) {
  return reactant::capture(
      std::shared_ptr<PjRtClient>(buffer->ptr()->client()));
}

extern "C" void ifrt_client_dtor(ifrt::Client *client) { delete client; }

// generic version, but IFRT-PjRt backend only supports SingleDeviceSharding
// and FullyReplicated. use `ifrt_pjrt_array_create` if using IFRT-PjRt.
extern "C" HeldIfrtArray *ifrt_client_make_array_from_host_buffer(
    ifrt::Client *client, void *data,
    int dtype_kind, // int
    int ndims, const int64_t *c_shape,
    HeldValue<std::shared_ptr<const ifrt::Sharding>> *sharding,
    int c_semantics) {
  auto dtype = ifrt::DType(static_cast<ifrt::DType::Kind>(dtype_kind));
  auto shape = ifrt::Shape(absl::Span<const int64_t>(c_shape, ndims));
  return reactant::capture(MyValueOrThrow(client->MakeArrayFromHostBuffer(
      data, dtype, shape,
      std::nullopt, // byte_strides
      sharding->obj(),
      static_cast<ifrt::Client::HostBufferSemantics>(c_semantics),
      [] {}, // on_done_with_host_buffer,
      client->CreateUserContext())));
}

extern "C" HeldIfrtArray *ifrt_client_make_single_shard_array_from_host_buffer(
    ifrt::Client *client, void *data,
    int dtype_kind, // int
    int ndims, const int64_t *c_shape, int c_semantics, ifrt::Device *device,
    const char *mem_kind) {
  auto memory_kind = ifrt::MemoryKind(std::string(mem_kind));
  auto sharding = reactant::capture(std::shared_ptr<const ifrt::Sharding>(
      ifrt::SingleDeviceSharding::Create(device, memory_kind).release()));
  return ifrt_client_make_array_from_host_buffer(
      client, data, dtype_kind, ndims, c_shape, sharding, c_semantics);
}

// all arrays are assumed to have same DType
// each process only provides arrays for its own addressable devices
extern "C" HeldIfrtArray *ifrt_client_assemble_array_from_single_shards(
    ifrt::Client *client, int32_t ndims, const int64_t *c_shape,
    HeldValue<std::shared_ptr<const ifrt::Sharding>> *sharding, int32_t narrays,
    HeldIfrtArray **c_arrays, int32_t c_semantics) {
  ifrt::Shape shape = ifrt::Shape(absl::Span<const int64_t>(c_shape, ndims));
  std::vector<tsl::RCReference<ifrt::Array>> arrays;
  for (int i = 0; i < narrays; i++) {
    arrays.emplace_back(c_arrays[i]->obj());
  }
  return reactant::capture(
      MyValueOrThrow(client->AssembleArrayFromSingleDeviceArrays(
          shape, sharding->obj(), absl::MakeSpan(arrays),
          static_cast<ifrt::ArrayCopySemantics>(c_semantics),
          ifrt::SingleDeviceShardSemantics::kAddressableShards)));
}

// we should deprecate this because is IFRT-PjRt specific
// try use `ifrt_client_make_single_shard_array_from_host_buffer` instead
extern "C" HeldIfrtArray *
ifrt_pjrt_array_create(ifrt::PjRtClient *client,
                       HeldValue<std::shared_ptr<xla::PjRtBuffer>> *buffer) {
  return reactant::capture(tsl::RCReference<ifrt::Array>(
      MyValueOrThrow(xla::ifrt::PjRtArray::Create(client, buffer->obj()))));
}


extern "C" void
ifrt_pjrt_loaded_executable_dtor(xla::ifrt::PjRtLoadedExecutable *exec) {
  delete exec;
}

extern "C" void ifrt_array_dtor(HeldIfrtArray *array) { delete array; }

// in principle, use ArrayCopySemantics::kAlwaysCopy (=0)
extern "C" FutureType *
ifrt_CopyArrayToHostBuffer(HeldIfrtArray *array, void *data,
                           ifrt::ArrayCopySemantics semantics) {
  return new FutureType(
      (*array)->CopyToHostBuffer(data, std::nullopt, semantics));
}

extern "C" void
PjRtLoadedExecutableGetHloModules(xla::PjRtLoadedExecutable *exec,
                                  void **hlo_modules, int32_t *nmodules) {
  auto hlo_modules_vec = MyValueOrThrow(exec->GetHloModules());
  *nmodules = hlo_modules_vec.size();
  for (int i = 0; i < *nmodules; i++) {
    hlo_modules[i] = reactant::capture(hlo_modules_vec[i]);
  }
}

extern "C" const char *HloModuleToString(HeldHloModule *hlo_module) {
  return cstr_from_string(hlo_module->obj()->ToString());
}

extern "C" void FreeHloModule(HeldHloModule *hlo_module) { delete hlo_module; }

#pragma region IfRtClient

// XXX: Bring back with the correct API
// extern "C" ifrt::proxy::GrpcServer *
// ifrt_proxy_grpc_server_create_from_ifrt_client_factory_cpu(
//     const char *c_address, uint8_t asynchronous, int node_id) {
//   std::string address = c_address;

//   return MyValueOrThrow(
//              ifrt::proxy::GrpcServer::CreateFromIfrtClientFactory(
//                  address,
//                  [asynchronous,
//                   node_id]() -> absl::StatusOr<std::shared_ptr<ifrt::Client>>
//                   {
//                    auto pjrt_client = std::shared_ptr<PjRtClient>(
//                        MakeCPUClient(asynchronous, node_id));
//                    return std::shared_ptr<ifrt::Client>(
//                        xla::ifrt::PjRtClient::Create(pjrt_client).release());
//                  }))
//       .release();
// }

// extern "C" ifrt::proxy::GrpcServer *
// ifrt_proxy_grpc_server_create_from_ifrt_client_factory_gpu(
//     int node_id, int num_nodes, int *allowed_devices, int
//     num_allowed_devices, double memory_fraction, bool preallocate, const char
//     *platform_name, const char **error) {
//   return MyValueOrThrow(
//              ifrt::proxy::GrpcServer::CreateFromIfrtClientFactory(
//                  std::string(),
//                  [node_id, num_nodes, allowed_devices, num_allowed_devices,
//                   memory_fraction, preallocate, platform_name,
//                   error]() -> absl::StatusOr<std::shared_ptr<ifrt::Client>> {
//                    auto pjrt_client =
//                    std::shared_ptr<PjRtClient>(MakeGPUClient(
//                        node_id, num_nodes, allowed_devices,
//                        num_allowed_devices, memory_fraction, preallocate,
//                        platform_name, error));
//                    return std::shared_ptr<ifrt::Client>(
//                        xla::ifrt::PjRtClient::Create(pjrt_client).release());
//                  }))
//       .release();
// }

// extern "C" ifrt::proxy::GrpcServer *
// ifrt_proxy_grpc_server_create_from_ifrt_client_factory_tpu(
//     const char *c_address, const char *tpu_path, const char **error) {
//   std::string address = c_address;
//
//   return MyValueOrThrow(
//              xla::ifrt::proxy::GrpcServer::CreateFromIfrtClientFactory(
//                  address,
//                  [](xla::ifrt::AttributeMap initialization_data) ->
//                  absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
//                    auto pjrt_client =
//                        std::shared_ptr<xla::PjRtClient>(GetCApiClient("TPU"));
//                    return
//                    xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
//                  }))
//       .release();
// }

extern "C" void ifrt_proxy_grpc_server_dtor(ifrt::proxy::GrpcServer *server) {
  delete server;
}

extern "C" const char *
ifrt_proxy_grpc_server_address(ifrt::proxy::GrpcServer *server) {
  return cstr_from_string(server->address());
}

extern "C" void ifrt_proxy_grpc_server_wait(ifrt::proxy::GrpcServer *server) {
  server->Wait();
}

// `c_proxy_server_address` must be of the form
// `<backend-transport>:<backend-address>`; e.g. "grpc:localhost"
// NOTE not sure if we must pass the port, but probably yes
// by default, set `connection_timeout_in_minutes` to 2
extern "C" ifrt::Client *
ifrt_proxy_create_client(const char *c_proxy_server_address,
                         int connection_timeout_in_minutes) {
  std::string proxy_server_address = c_proxy_server_address;
  ifrt::proxy::ClientConnectionOptions options = {
      absl::Minutes(connection_timeout_in_minutes),
      nullptr, // callback `on_disconnect`
      nullptr, // callback `on_connection_update`
  };
  return MyValueOrThrow(
             ifrt::proxy::CreateClient(proxy_server_address, options))
      .release();
}

extern "C" ifrt::Client *ifrt_pjrt_make_client(
    PjRtClient *pjrt_client, int node_id, int num_nodes,
    void *distributed_runtime_client, const char **error,
    std::string key_prefix,
    std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store) {
  ifrt::PjRtClient::CreateOptions options;
  options.pjrt_client = std::shared_ptr<PjRtClient>(pjrt_client);

  if (num_nodes > 1) {
    if (distributed_runtime_client == nullptr) {
      *error =
          "`distributed_runtime_client` must be non-null if `num_nodes` > 1";
      return nullptr;
    }
    if (kv_store.has_value()) {
      options.kv_store = kv_store.value();
    } else {
      auto typed_distributed_runtime_client = static_cast<
          HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *>(
          distributed_runtime_client);
      options.kv_store = GetDistributedKeyValueStore(
          typed_distributed_runtime_client->obj(), key_prefix);
    }
  }

  options.process_id = node_id;
  options.num_processes = num_nodes;

  return MyValueOrThrow(xla::ifrt::PjRtClient::Create(options)).release();
}

extern "C" ifrt::Client *ifrt_pjrt_make_client_with_default_kv_store(
    PjRtClient *pjrt_client, int node_id, int num_nodes,
    void *distributed_runtime_client, const char **error,
    std::string key_prefix) {
  std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store;
  return ifrt_pjrt_make_client(pjrt_client, node_id, num_nodes,
                               distributed_runtime_client, error, key_prefix,
                               kv_store);
}

const char *const kMpiTrampolineLibEnv = "MPITRAMPOLINE_LIB";

extern "C" ifrt::Client *
ifrt_make_pjrt_cpu_client(uint8_t asynchronous, int node_id, int num_nodes,
                          void *distributed_runtime_client,
                          const char **error) {
  std::optional<std::shared_ptr<xla::cpu::CpuCollectives>> collectives;
  std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store;

  if (distributed_runtime_client != nullptr) {
	  /*
    auto mpi_trampoline_path = llvm::sys::Process::GetEnv(kMpiTrampolineLibEnv);
    if (mpi_trampoline_path) {
      // Use MPI
      // TODO: How do we Finalize??
      auto mpi_collectives = std::make_shared<xla::cpu::MpiCollectives>();
      collectives = mpi_collectives;
      static_cast<xla::cpu::MpiCollectives *>(mpi_collectives.get())->Init();
    } else
	   */ 
    {
      // Use Gloo
      auto typed_distributed_runtime_client = static_cast<
          HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *>(
          distributed_runtime_client);
      kv_store =
          GetDistributedKeyValueStore(typed_distributed_runtime_client->obj(),
                                      /*key_prefix=*/"cpu:");
      auto gloo_kv_store =
          std::make_unique<xla::cpu::GlooKeyValueStore>(kv_store.value());
#if defined(__linux__)
      auto tcp_attrs = gloo::transport::tcp::attr();
      auto tcp_device = gloo::transport::tcp::CreateDevice(tcp_attrs);
      collectives = std::make_shared<xla::cpu::GlooCollectives>(
          std::move(gloo_kv_store), std::move(tcp_device));
#elif defined(__APPLE__)
      auto uv_attrs = gloo::transport::uv::attr();
      auto uv_device = gloo::transport::uv::CreateDevice(uv_attrs);
      collectives = std::make_shared<xla::cpu::GlooCollectives>(
          std::move(gloo_kv_store), std::move(uv_device));
#else
      ReactantThrowError(
          "Gloo TCP Collectives only implemented for linux and macos");
#endif
    }
  }

  PjRtClient *pjrt_client =
      MakeCPUClientInternal(asynchronous, node_id, collectives);
  if (pjrt_client == nullptr)
    return nullptr;
  return ifrt_pjrt_make_client(pjrt_client, node_id, num_nodes,
                               distributed_runtime_client, error, "cpu",
                               kv_store);
}

extern "C" ifrt::Client *
ifrt_make_pjrt_gpu_client(int node_id, int num_nodes, int64_t *allowed_devices,
                          int64_t num_allowed_devices, double memory_fraction,
                          bool preallocate, const char *platform_name,
                          const char **error,
                          void *distributed_runtime_client) {
  PjRtClient *pjrt_client = MakeGPUClient(
      node_id, num_nodes, allowed_devices, num_allowed_devices, memory_fraction,
      preallocate, platform_name, error, distributed_runtime_client);
  if (pjrt_client == nullptr)
    return nullptr;
  std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store;
  return ifrt_pjrt_make_client(pjrt_client, node_id, num_nodes,
                               distributed_runtime_client, error, "gpu",
                               kv_store);
}

extern "C" ifrt::Client *
ifrt_make_pjrt_tpu_client(const char *tpu_path, const char **error, int node_id,
                          int num_nodes, void *distributed_runtime_client) {
  PjRtClient *pjrt_client = MakeTPUClient(tpu_path, error);
  if (pjrt_client == nullptr)
    return nullptr;
  std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store;
  return ifrt_pjrt_make_client(pjrt_client, node_id, num_nodes,
                               distributed_runtime_client, error, "tpu",
                               kv_store);
}

extern "C" void ifrt_FreeClient(ifrt::Client *client) { delete client; }

extern "C" int ifrt_client_device_count(ifrt::Client *client) {
  return client->device_count();
}

extern "C" int ifrt_client_addressable_device_count(ifrt::Client *client) {
  return client->addressable_device_count();
}

extern "C" void ifrt_client_devices(ifrt::Client *client,
                                    ifrt::Device **out_devices) {
  auto span = client->devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

extern "C" void ifrt_client_addressable_devices(ifrt::Client *client,
                                                ifrt::Device **out_devices) {
  auto span = client->addressable_devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

extern "C" void ifrt_client_all_devices(ifrt::Client *client,
                                        ifrt::Device **out_devices) {
  auto span = client->GetAllDevices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

extern "C" ifrt::Device *ifrt_client_lookup_device(ifrt::Client *client,
                                                   int dev_id) {
  return MyValueOrThrow(
      client->LookupDevice(static_cast<ifrt::DeviceId>(dev_id)));
}

extern "C" ifrt::Device *
ifrt_client_lookup_addressable_device(ifrt::Client *client, int local_hw_id) {
  return MyValueOrThrow(client->LookupAddressableDevice(local_hw_id));
}

extern "C" int ifrt_ClientProcessIndex(ifrt::Client *client) {
  return client->process_index();
}

extern "C" const char *ifrt_ClientGetPlatformName(ifrt::Client *client) {
  return cstr_from_string(client->platform_name());
}

extern "C" ifrt::Device *ifrt_ClientGetDevice(ifrt::Client *client, int idx) {
  return MyValueOrThrow(client->LookupDevice(ifrt::DeviceId(idx)));
}

extern "C" ifrt::Device *ifrt_ClientGetAddressableDevice(ifrt::Client *client,
                                                         int idx) {
  return MyValueOrThrow(client->LookupAddressableDevice(idx));
}

#pragma endregion

#pragma region IfRtDevice

extern "C" int64_t ifrt_DeviceGetGlobalDeviceId(ifrt::Device *device) {
  return device->Id().value();
}

extern "C" const char *ifrt_DeviceGetKind(ifrt::Device *device) {
  return cstr_from_string(device->Kind());
}

extern "C" ifrt::Client *ifrt_DeviceToClient(ifrt::Device *device) {
  return device->client();
}

extern "C" bool ifrt_DeviceIsAddressable(ifrt::Device *device) {
  return device->IsAddressable();
}

static xla::ifrt::RCReferenceWrapper<ifrt::DeviceList>
ifrt_CreateDeviceListFromDevices(ifrt::Client *client,
                                 ifrt::Device **device_list,
                                 int32_t num_devices) {
  absl::Span<ifrt::Device *const> devices(device_list, num_devices);
  return client->MakeDeviceList(devices);
}

extern "C" ifrt::Memory *ifrt_DeviceGetDefaultMemory(ifrt::Device *device) {
  return MyValueOrThrow(device->DefaultMemory());
}

extern "C" ifrt::Memory **ifrt_DeviceGetMemories(ifrt::Device *device,
                                                 int32_t *size) {
  auto memory_list = device->Memories();
  *size = memory_list.size();
  return const_cast<ifrt::Memory **>(memory_list.data());
}

extern "C" ifrt::MemoryKind *ifrt_MemoryGetMemoryKind(ifrt::Memory *memory) {
  ifrt::MemoryKind *memory_kind = new ifrt::MemoryKind(memory->Kind());
  return memory_kind;
}

extern "C" const char *ifrt_MemoryToString(ifrt::Memory *memory) {
  return cstr_from_string(memory->ToString());
}

extern "C" const char *ifrt_MemoryKindToString(ifrt::MemoryKind *memory_kind) {
  auto memkind = memory_kind->memory_kind();
  if (!memkind.has_value())
    return "";
  return cstr_from_string(memkind.value());
}

extern "C" bool ifrt_MemoryKindsAreEqual(ifrt::MemoryKind *a,
                                         ifrt::MemoryKind *b) {
  return *a == *b;
}

#pragma endregion

#pragma region OpSharding

extern "C" void free_op_sharding(xla::OpSharding *op_sharding) {
  delete op_sharding;
}

extern "C" int32_t
op_sharding_to_op_sharding_type(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->type());
}

extern "C" int32_t
op_sharding_to_shard_group_type(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->shard_group_type());
}

extern "C" int32_t op_sharding_to_shard_group_id(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->shard_group_id());
}

extern "C" bool op_sharding_is_shard_group(xla::OpSharding *op_sharding) {
  return op_sharding->is_shard_group();
}

extern "C" bool
op_sharding_replicate_on_last_tile_dim(xla::OpSharding *op_sharding) {
  return op_sharding->replicate_on_last_tile_dim();
}

extern "C" bool op_sharding_has_last_tile_dims(xla::OpSharding *op_sharding) {
  return op_sharding->last_tile_dims_size() > 0;
}

extern "C" int32_t
op_sharding_last_tile_dims_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->last_tile_dims_size());
}

extern "C" void op_sharding_last_tile_dims(xla::OpSharding *op_sharding,
                                           int32_t *last_tile_dims) {
  std::vector<int32_t> last_tile_dims_vec(op_sharding->last_tile_dims().begin(),
                                          op_sharding->last_tile_dims().end());
  std::copy(last_tile_dims_vec.begin(), last_tile_dims_vec.end(),
            last_tile_dims);
  return;
}

extern "C" bool
op_sharding_has_iota_reshape_dims(xla::OpSharding *op_sharding) {
  return op_sharding->iota_reshape_dims_size() > 0;
}

extern "C" int32_t
op_sharding_iota_reshape_dims_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->iota_reshape_dims_size());
}

extern "C" void op_sharding_iota_reshape_dims(xla::OpSharding *op_sharding,
                                              int32_t *iota_reshape_dims) {
  std::vector<int32_t> iota_reshape_dims_vec(
      op_sharding->iota_reshape_dims().begin(),
      op_sharding->iota_reshape_dims().end());
  std::copy(iota_reshape_dims_vec.begin(), iota_reshape_dims_vec.end(),
            iota_reshape_dims);
  return;
}

extern "C" bool
op_sharding_has_iota_transpose_perm(xla::OpSharding *op_sharding) {
  return op_sharding->iota_transpose_perm_size() > 0;
}

extern "C" int32_t
op_sharding_iota_transpose_perm_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->iota_transpose_perm_size());
}

extern "C" void op_sharding_iota_transpose_perm(xla::OpSharding *op_sharding,
                                                int32_t *iota_transpose_perm) {
  std::vector<int32_t> iota_transpose_perm_vec(
      op_sharding->iota_transpose_perm().begin(),
      op_sharding->iota_transpose_perm().end());
  std::copy(iota_transpose_perm_vec.begin(), iota_transpose_perm_vec.end(),
            iota_transpose_perm);
  return;
}

extern "C" bool
op_sharding_has_tile_assignment_dimensions(xla::OpSharding *op_sharding) {
  return op_sharding->tile_assignment_dimensions_size() > 0;
}

extern "C" int32_t
op_sharding_tile_assignment_dimensions_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->tile_assignment_dimensions_size());
}

extern "C" void
op_sharding_tile_assignment_dimensions(xla::OpSharding *op_sharding,
                                       int32_t *tile_assignment_dimensions) {
  std::vector<int32_t> tile_assignment_dimensions_vec(
      op_sharding->tile_assignment_dimensions().begin(),
      op_sharding->tile_assignment_dimensions().end());
  std::copy(tile_assignment_dimensions_vec.begin(),
            tile_assignment_dimensions_vec.end(), tile_assignment_dimensions);
  return;
}

extern "C" bool
op_sharding_has_tile_assignment_devices(xla::OpSharding *op_sharding) {
  return op_sharding->tile_assignment_devices_size() > 0;
}

extern "C" int32_t
op_sharding_tile_assignment_devices_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->tile_assignment_devices_size());
}

extern "C" void
op_sharding_tile_assignment_devices(xla::OpSharding *op_sharding,
                                    int32_t *tile_assignment_devices) {
  std::vector<int32_t> tile_assignment_devices_vec(
      op_sharding->tile_assignment_devices().begin(),
      op_sharding->tile_assignment_devices().end());
  std::copy(tile_assignment_devices_vec.begin(),
            tile_assignment_devices_vec.end(), tile_assignment_devices);
  return;
}

#pragma endregion

#pragma region HloSharding

extern "C" void free_hlo_sharding(xla::HloSharding *hlo_sharding) {
  delete hlo_sharding;
}

extern "C" xla::HloSharding *
hlo_sharding_from_op_sharding(xla::OpSharding *op_sharding) {
  xla::HloSharding *hlo_sharding = new xla::HloSharding(
      MyValueOrThrow(xla::HloSharding::FromProto(*op_sharding)));
  return hlo_sharding;
}

extern "C" xla::OpSharding *
hlo_sharding_to_op_sharding(xla::HloSharding *hlo_sharding) {
  xla::OpSharding *op_sharding = new xla::OpSharding(hlo_sharding->ToProto());
  return op_sharding;
}

extern "C" const char *
hlo_sharding_to_string(const xla::HloSharding *hlo_sharding) {
  return cstr_from_string(hlo_sharding->ToString(true));
}

extern "C" ifrt::MemoryKind *ifrt_memory_kind_from_string(const char *c_str) {
  return new ifrt::MemoryKind(std::string(c_str));
}

extern "C" ifrt::MemoryKind *ifrt_memory_kind_with_optional_memory_space() {
  return new ifrt::MemoryKind(std::nullopt);
}

extern "C" bool ifrt_memory_kind_has_value(ifrt::MemoryKind *memory_kind) {
  return *memory_kind != ifrt::MemoryKind(std::nullopt);
}

extern "C" void free_ifrt_sharding(HeldIfrtSharding *sharding) {
  delete sharding;
}

extern "C" HeldIfrtSharding *ifrt_sharding_from_xla_hlo_sharding(
    ifrt::Client *client, ifrt::Device **device_list, int32_t num_devices,
    ifrt::MemoryKind *memory_kind, xla::HloSharding *xla_hlo_sharding) {
  // convert to ifrt::HloSharding
  auto hlo_sharding =
      ifrt::HloSharding::Create(
          ifrt_CreateDeviceListFromDevices(client, device_list, num_devices),
          *memory_kind, *xla_hlo_sharding)
          .release();
  // convert to ifrt::Sharding
  return reactant::capture(
      std::shared_ptr<ifrt::Sharding>(std::move(hlo_sharding)));
}

extern "C" xla::HloSharding *
ifrt_sharding_to_xla_hlo_sharding(HeldIfrtSharding *sharding) {
  const ifrt::Sharding *val = sharding->obj().get();
  if (!llvm::isa<ifrt::HloSharding>(val))
    ReactantThrowError("Expected a HloSharding");
  auto ifrt_hlo_sharding = llvm::dyn_cast<const ifrt::HloSharding>(val);
  xla::HloSharding *xla_hlo_sharding =
      new xla::HloSharding(ifrt_hlo_sharding->xla_hlo_sharding());
  return xla_hlo_sharding;
}

extern "C" bool
ifrt_sharding_is_single_device_sharding(HeldIfrtSharding *sharding) {
  return llvm::isa<const ifrt::SingleDeviceSharding>(sharding->obj().get());
}

extern "C" bool ifrt_sharding_is_fully_replicated(HeldIfrtSharding *sharding) {
  return sharding->obj()->IsFullyReplicated();
}

extern "C" const char *ifrt_sharding_to_string(HeldIfrtSharding *sharding) {
  return cstr_from_string(sharding->obj()->DebugString());
}

extern "C" int32_t ifrt_sharding_devices_size(HeldIfrtSharding *sharding) {
  return sharding->obj()->devices()->size();
}

extern "C" void ifrt_sharding_to_device_list(HeldIfrtSharding *sharding,
                                             ifrt::Device **devices) {
  auto device_list = sharding->obj()->devices()->devices();
  for (int i = 0; i < device_list.size(); i++) {
    devices[i] = device_list[i];
  }
}

extern "C" void ifrt_sharding_to_index_domains(HeldIfrtSharding *sharding,
                                               int64_t *array_size_list,
                                               int32_t array_size_len,
                                               int64_t *index_domain_origins,
                                               int64_t *index_domain_shapes) {
  std::vector<int64_t> array_size(array_size_len);
  for (int i = 0; i < array_size_len; i++) {
    array_size[i] = array_size_list[i];
  }
  auto array_size_span = absl::MakeSpan(array_size);
  auto array_shape = xla::ifrt::Shape(array_size_span);

  std::vector<ifrt::IndexDomain> index_domains =
      MyValueOrThrow(sharding->obj()->IndexDomains(array_shape));

  for (int i = 0; i < index_domains.size(); i++) {
    auto index_domain = index_domains[i];
    absl::Span<const int64_t> origin = index_domain.origin().elements();
    absl::Span<const int64_t> shape = index_domain.shape().dims();

    for (int j = 0; j < origin.size(); j++) {
      auto idx = i * origin.size() + j;
      index_domain_origins[idx] = origin[j];
      index_domain_shapes[idx] = shape[j];
    }
  }
}

extern "C" bool hlo_sharding_is_tuple(xla::HloSharding *hloSharding) {
  return hloSharding->IsTuple();
}

extern "C" bool hlo_sharding_is_replicated(xla::HloSharding *hloSharding) {
  return hloSharding->IsReplicated();
}

extern "C" bool hlo_sharding_is_manual(xla::HloSharding *hloSharding) {
  return hloSharding->IsManual();
}

extern "C" bool hlo_sharding_is_unknown(xla::HloSharding *hloSharding) {
  return hloSharding->IsUnknown();
}

extern "C" bool hlo_sharding_is_tiled(xla::HloSharding *hloSharding) {
  return hloSharding->IsTiled();
}

extern "C" bool hlo_sharding_is_maximal(xla::HloSharding *hloSharding) {
  return hloSharding->IsTileMaximal();
}

extern "C" bool
hlo_sharding_replicate_on_last_tile_dim(xla::HloSharding *hloSharding) {
  return hloSharding->ReplicateOnLastTileDim();
}

extern "C" int32_t
hlo_sharding_tile_assignment_dimensions_size(xla::HloSharding *hloSharding) {
  return static_cast<int32_t>(hloSharding->tile_assignment().num_dimensions());
}

extern "C" int32_t
hlo_sharding_tile_assignment_devices_size(xla::HloSharding *hloSharding) {
  return static_cast<int32_t>(hloSharding->tile_assignment().num_elements());
}

extern "C" void
hlo_sharding_tile_assignment_dimensions(xla::HloSharding *hloSharding,
                                        int64_t *dims, int32_t size) {
  auto tileAssignmentDims = hloSharding->tile_assignment().dimensions();
  for (int32_t i = 0; i < size; i++) {
    dims[i] = tileAssignmentDims[i];
  }
}

extern "C" void
hlo_sharding_tile_assignment_devices(xla::HloSharding *hloSharding,
                                     int64_t *devices, int32_t size) {
  auto tileAssignmentDevices = hloSharding->tile_assignment().array().data();
  for (int32_t i = 0; i < size; i++) {
    devices[i] = tileAssignmentDevices[i];
  }
}

extern "C" bool hlo_sharding_check_eq(xla::HloSharding *hloSharding,
                                      xla::HloSharding *other) {
  return *hloSharding == *other;
}

#pragma endregion

typedef ifrt::Future<> IfRtFutureType;

extern "C" void ifrt_free_future(IfRtFutureType *Future) { delete Future; }

extern "C" uint8_t ifrt_future_is_ready(IfRtFutureType *Future) {
  return Future->IsReady();
}

extern "C" void ifrt_future_await(IfRtFutureType *Future) { Future->Await(); }

#pragma region IfRtArray

extern "C" void ifrt_free_array(HeldIfrtArray *array) { delete array; }

extern "C" int64_t *ifrt_array_shape(HeldIfrtArray *array) {
  auto dims =
      static_cast<absl::Span<const int64_t>>(array->obj()->shape().dims());
  int64_t *dims_ptr = new int64_t[dims.size()];
  std::copy(dims.begin(), dims.end(), dims_ptr);
  return dims_ptr;
}

extern "C" int64_t ifrt_array_ndims(HeldIfrtArray *array) {
  return array->obj()->shape().dims().size();
}

extern "C" ifrt::DType ifrt_array_eltype(HeldIfrtArray *array) {
  return array->obj()->dtype();
}

extern "C" ifrt::Client *ifrt_array_to_client(HeldIfrtArray *array) {
  return array->obj()->client();
}

extern "C" HeldValue<std::shared_ptr<const ifrt::Sharding>> *
ifrt_array_to_sharding(HeldIfrtArray *array) {
  return reactant::capture(array->obj()->shared_ptr_sharding());
}

extern "C" void ifrt_array_copy_to_host_buffer(HeldIfrtArray *array,
                                               void *data) {
  std::optional<absl::Span<const int64_t>> byte_strides;
  auto future = array->obj()->CopyToHostBuffer(
      data, byte_strides, static_cast<ifrt::ArrayCopySemantics>(0));
  future.Await();
  return;
}

extern "C" HeldIfrtArray **ifrt_array_disassemble_into_single_device_arrays(
    HeldIfrtArray *array, int32_t c_semantics,
    int32_t c_single_device_shard_semantics, int32_t *narrays) {
  std::vector<tsl::RCReference<ifrt::Array>> single_device_arrays =
      MyValueOrThrow(array->obj()->DisassembleIntoSingleDeviceArrays(
          static_cast<ifrt::ArrayCopySemantics>(c_semantics),
          static_cast<ifrt::SingleDeviceShardSemantics>(
              c_single_device_shard_semantics)));

  *narrays = single_device_arrays.size();
  HeldIfrtArray **arrays = new HeldIfrtArray *[single_device_arrays.size()];
  for (int i = 0; i < single_device_arrays.size(); i++) {
    arrays[i] = reactant::capture(std::move(single_device_arrays[i]));
  }
  return arrays;
}

#pragma endregion

#pragma region xla::Distributed

extern "C" HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *
GetDistributedRuntimeClient(char *c_address, int32_t node_id,
                            int32_t rpc_timeout_in_seconds,
                            // int32_t init_timeout,
                            int32_t shutdown_timeout_in_minutes,
                            int32_t heartbeat_interval_in_seconds,
                            int max_missing_heartbeats, bool use_compression) {
  xla::DistributedRuntimeClient::Options options;
  options.node_id = node_id;
  options.rpc_timeout = absl::Seconds(rpc_timeout_in_seconds);
  // options.init_timeout = absl::Seconds(init_timeout);
  options.shutdown_timeout = absl::Minutes(shutdown_timeout_in_minutes);
  options.heartbeat_interval = absl::Seconds(heartbeat_interval_in_seconds);
  options.max_missing_heartbeats = max_missing_heartbeats;

  std::string address = c_address;

  return reactant::capture(
      xla::GetDistributedRuntimeClient(address, options, use_compression));
}

extern "C" void free_distributed_runtime_client(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *client) {
  delete client;
}

extern "C" void distributed_runtime_client_connect(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *client) {
  auto status = client->obj()->Connect();
  if (!status.ok())
    ReactantThrowError(status.ToString().c_str());
}

extern "C" void distributed_runtime_client_shutdown(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *client) {
  auto status = client->obj()->Shutdown();
  if (!status.ok())
    ReactantThrowError(status.ToString().c_str());
}

extern "C" xla::DistributedRuntimeService *GetDistributedRuntimeService(
    char *c_address, int num_nodes, int32_t heartbeat_interval_in_seconds,
    int max_missing_heartbeats, int32_t cluster_register_timeout_in_minutes,
    int32_t shutdown_timeout_in_minutes) {
  xla::CoordinationServiceImpl::Options options;
  options.num_nodes = num_nodes;
  options.heartbeat_interval = absl::Seconds(heartbeat_interval_in_seconds);
  options.max_missing_heartbeats = max_missing_heartbeats;
  options.cluster_register_timeout =
      absl::Minutes(cluster_register_timeout_in_minutes);
  options.shutdown_timeout = absl::Minutes(shutdown_timeout_in_minutes);

  std::string address = c_address;

  return MyValueOrThrow(xla::GetDistributedRuntimeService(address, options))
      .release();
}

extern "C" void free_distributed_runtime_service(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeService>> *service) {
  delete service;
}

extern "C" void distributed_runtime_service_shutdown(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeService>> *service) {
  service->obj()->Shutdown();
}

#pragma endregion

#pragma region Shardy

extern "C" xla::HloSharding *
hloShardingFromTensorShardingAttr(mlir::sdy::TensorShardingAttr attr,
                                  mlir::sdy::MeshAttr meshAttr) {
  mlir::ArrayRef<mlir::StringAttr> manual_axes = {};
  std::function<mlir::sdy::MeshAttr(mlir::sdy::TensorShardingAttr)>
      get_mesh_attr = [meshAttr](mlir::sdy::TensorShardingAttr local_attr) {
        return meshAttr;
      };

  return new xla::HloSharding(
      xla::sdy::convertToHloSharding(attr, get_mesh_attr, manual_axes));
}

// XXX: This is incorrect for multiple meshes. We need to use the current mesh
// to generate this instead of the global mesh Currently we are storing only a
// single mesh, so we can just use this.
extern "C" mlir::sdy::TensorShardingAttr hloShardingToTensorShardingAttr(
    mlir::MLIRContext *context, const xla::HloSharding *hloSharding,
    mlir::StringAttr meshName, mlir::sdy::MeshAttr meshAttr, int64_t rank,
    const bool *isClosed, const int64_t *priority) {
  const llvm::SmallDenseMap<int64_t, llvm::StringRef>
      deviceIdToMaximalMeshName =
          llvm::SmallDenseMap<int64_t, llvm::StringRef>();
  mlir::sdy::TensorShardingAttr tensorShardingAttr =
      xla::sdy::convertToSdySharding(*hloSharding, meshAttr,
                                     deviceIdToMaximalMeshName, rank,
                                     /*openDims=*/true);

  for (int64_t i = 0; i < rank; i++) {
    auto oldDimSharding = tensorShardingAttr.getDimSharding(i);

    std::optional<int64_t> dimPriority;
    if (priority[i] > 0)
      dimPriority = priority[i];

    tensorShardingAttr = tensorShardingAttr.replaceDimSharding(
        i, mlir::sdy::DimensionShardingAttr::get(oldDimSharding.getContext(),
                                                 oldDimSharding.getAxes(),
                                                 isClosed[i], dimPriority));
  }

}

#pragma endregion

#pragma region ifrt::LoadedExecutable

extern "C" void ifrt_loaded_executable_dtor(ifrt::LoadedExecutable *exec) {
  delete exec;
}

extern "C" void ifrt_loaded_executable_execute(
    ifrt::LoadedExecutable *exec, int num_args,
    HeldValue<tsl::RCReference<ifrt::Array>> **op_args,
    uint8_t *is_arg_donatable, int num_results,
    HeldValue<tsl::RCReference<ifrt::Array>> **op_results, uint8_t *futures,
    FutureType **status) {
  std::vector<tsl::RCReference<xla::ifrt::Array>> args;
  for (int i = 0; i < num_args; i++) {
    args.emplace_back(op_args[i]->obj());
  }

  ifrt::ExecuteOptions options;
  for (size_t i = 0; i < num_args; i++) {
    if (!is_arg_donatable[i]) {
      options.non_donatable_input_indices.insert(static_cast<int>(i));
    }
  }
  options.fill_status = true;

  auto result = MyValueOrThrow(exec->Execute(
      static_cast<absl::Span<tsl::RCReference<xla::ifrt::Array>>>(args),
      options, /* devices */ std::nullopt));

  if (result.outputs.size() != num_results) {
    llvm::errs() << "Error: results.size()=" << result.outputs.size()
                 << " does not match num_results=" << num_results << "\n";
    std::abort(); // Terminate if the number of results is incorrect.
  }

  // there is only 1 status and is valid because we set `options.fill_status =
  // true`
  *futures = true;
  *status = new FutureType(result.status);

  for (int i = 0; i < num_results; i++) {
    op_results[i] = reactant::capture(result.outputs[i]);
  }
}

extern "C" ifrt::Client *
ifrt_loaded_executable_client(ifrt::LoadedExecutable *exec) {
  return exec->client();
}

extern "C" void
ifrt_loaded_executable_get_parameter_shardings(ifrt::LoadedExecutable *exec,
                                               xla::OpSharding **op_shardings,
                                               int32_t num_op_shardings) {
  std::optional<std::vector<xla::OpSharding>> shardings =
      exec->GetParameterShardings();
  if (!shardings.has_value()) {
    ReactantThrowError(
        "No sharding found for the output of the loaded executable");
  }

  std::vector<xla::OpSharding> hlo_op_shardings = shardings.value();
  if (num_op_shardings != hlo_op_shardings.size()) {
    ReactantThrowError(("Expected " + std::to_string(num_op_shardings) +
                        " shardings, got " +
                        std::to_string(hlo_op_shardings.size()))
                           .c_str());
  }

  for (int32_t i = 0; i < num_op_shardings; i++) {
    op_shardings[i] = new xla::OpSharding(hlo_op_shardings[i]);
  }
}

extern "C" void
ifrt_loaded_executable_get_output_shardings(ifrt::LoadedExecutable *exec,
                                            xla::OpSharding **op_shardings,
                                            int32_t num_op_shardings) {
  std::optional<std::vector<xla::OpSharding>> shardings =
      exec->GetOutputShardings();
  if (!shardings.has_value()) {
    ReactantThrowError(
        "No sharding found for the output of the loaded executable");
  }

  std::vector<xla::OpSharding> hlo_op_shardings = shardings.value();
  if (num_op_shardings != hlo_op_shardings.size()) {
    ReactantThrowError(("Expected " + std::to_string(num_op_shardings) +
                        " shardings, got " +
                        std::to_string(hlo_op_shardings.size()))
                           .c_str());
  }

  for (int32_t i = 0; i < num_op_shardings; i++) {
    op_shardings[i] = new xla::OpSharding(hlo_op_shardings[i]);
  }
}

extern "C" void
ifrt_loaded_executable_get_hlo_modules(ifrt::LoadedExecutable *exec,
                                       void **hlo_modules, int32_t *nmodules) {
  auto hlo_modules_vec = MyValueOrThrow(exec->GetHloModules());
  *nmodules = hlo_modules_vec.size();
  for (int32_t i = 0; i < *nmodules; i++) {
    hlo_modules[i] = reactant::capture(hlo_modules_vec[i]);
  }
}

extern "C" int32_t
ifrt_loaded_executable_num_devices(ifrt::LoadedExecutable *exec) {
  return static_cast<int32_t>(exec->num_devices());
}

#pragma endregion

#pragma region CostAnalysis

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

extern "C" void pjrt_hlo_module_cost_analysis_properties(
    PjRtClient *client, HeldHloModule *hlo_module,
    JLHloCostAnalysisProperties *jlproperties) {
  auto analysis = MyValueOrThrow(client->GetHloCostAnalysis());
  auto err = hlo_module->obj()->entry_computation()->Accept(analysis.get());
  if (!err.ok()) {
    ReactantThrowError(err.ToString().c_str());
  }
  auto properties = analysis->properties();

  jlproperties->flops = properties["flops"];
  jlproperties->transcendentals = properties["transcendentals"];
  jlproperties->bytes_accessed = properties["bytes accessed"];
  jlproperties->optimal_seconds = properties["optimal seconds"];
  jlproperties->utilization = properties["utilization"];
  jlproperties->operand0_utilization = properties.operand_utilization(0);
  jlproperties->operand1_utilization = properties.operand_utilization(1);
  jlproperties->operand0_bytes_accessed = properties.operand_bytes_accessed(0);
  jlproperties->operand1_bytes_accessed = properties.operand_bytes_accessed(1);
  jlproperties->output_root_bytes_accessed = properties.output_bytes_accessed();
  jlproperties->reserved0 = 0.0;
  return;
}

extern "C" void ifrt_hlo_module_cost_analysis_properties(
    ifrt::Client *client, HeldHloModule *hlo_module,
    JLHloCostAnalysisProperties *jlproperties) {
  if (llvm::isa<ifrt::PjRtClient>(client)) {
    auto ifrt_pjrt_client = llvm::dyn_cast<ifrt::PjRtClient>(client);
    return pjrt_hlo_module_cost_analysis_properties(
        ifrt_pjrt_client->pjrt_client(), hlo_module, jlproperties);
  }
  ReactantThrowError(("Cost analysis not supported for this client: " +
                      std::string(client->runtime_type()))
                         .c_str());
}

#pragma endregion

extern "C" void dump_op(Operation *op) { llvm::errs() << *op << "\n"; }
extern "C" void dump_mval(mlir::Value v) { llvm::errs() << v << "\n"; }
extern "C" void dump_operation(Operation *op, const char *filename) {
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC, llvm::sys::fs::OF_Text);

  if (EC) {
    std::cerr << "Error opening file: " << EC.message() << std::endl;
    return;
  }

  op->print(file, mlir::OpPrintingFlags().enableDebugInfo(true, false));
}

extern "C" bool pjrt_device_is_addressable(PjRtDevice *device) {
  return device->IsAddressable();
}

extern "C" mlir::Operation *mlirGetParentOfTypeFunctionOp(mlir::Operation *op) {
  return op->getParentOfType<mlir::FunctionOpInterface>();
}

// batched copy
// https://github.com/jax-ml/jax/blob/2b86f38585a517ce50e8ddf964a4709040a1bd53/jaxlib/xla/py_array.cc#L1112

// xla::ifrt::CopyArrays
extern "C" HeldIfrtArray **ifrt_copy_arrays_to_device_with_sharding(
    ifrt::Client *client, HeldIfrtArray **arrays, int32_t num_arrays,
    HeldValue<std::shared_ptr<const ifrt::Sharding>> *dst_sharding,
    int32_t c_semantics) {
  std::vector<tsl::RCReference<ifrt::Array>> src_arrays_vec;
  for (int i = 0; i < num_arrays; i++) {
    src_arrays_vec.push_back(arrays[i]->obj());
  }

  auto dst_arrays = MyValueOrThrow(client->CopyArrays(
      absl::MakeSpan(src_arrays_vec), dst_sharding->obj()->devices(),
      dst_sharding->obj()->memory_kind(),
      static_cast<ifrt::ArrayCopySemantics>(c_semantics)));

  HeldIfrtArray **res_dst_arrays = new HeldIfrtArray *[num_arrays];
  for (int i = 0; i < num_arrays; i++) {
    arrays[i] = reactant::capture(std::move(dst_arrays[i]));
  }
  return res_dst_arrays;
}

ifrt::Client::MakeArraysFromHostBufferShardsSpec
ifrt_make_arrays_from_host_buffer_shards_spec(
    const void **host_buffers, int num_buffers,
    const int64_t **host_buffer_shapes,
    const int64_t **addressable_shard_indices,
    const int64_t *addressable_shard_indices_sizes, int dtype_kind, int ndims,
    const int64_t *final_buffer_shape,
    HeldValue<std::shared_ptr<const ifrt::Sharding>> *sharding) {
  ifrt::DType ifrt_dtype =
      ifrt::DType(static_cast<ifrt::DType::Kind>(dtype_kind));

  auto array_spec = ifrt::ArraySpec{
      /*dtype=*/ifrt_dtype,
      /*shape=*/
      ifrt::Shape(absl::Span<const int64_t>(final_buffer_shape, ndims)),
      /*sharding=*/sharding->obj()};

  absl::InlinedVector<
      std::pair<absl::InlinedVector<int64_t, 1>, ifrt::Client::HostBuffer>, 1>
      buffers;

  for (int i = 0; i < num_buffers; i++) {
    ifrt::Client::HostBuffer buffer = ifrt::Client::HostBuffer{
        /*data=*/host_buffers[i],
        /*dtype=*/ifrt_dtype,
        /*shape=*/
        ifrt::Shape(absl::Span<const int64_t>(host_buffer_shapes[i], ndims)),
    };

    absl::InlinedVector<int64_t, 1> indices;
    for (int j = 0; j < addressable_shard_indices_sizes[i]; j++) {
      indices.push_back(addressable_shard_indices[i][j]);
    }

    buffers.push_back(std::make_pair(indices, buffer));
  }

  return ifrt::Client::MakeArraysFromHostBufferShardsSpec{
      /*buffers=*/buffers,
      /*array_spec=*/array_spec,
  };
}

// TODO: We can batch the construction of multiple arrays into a single call.
extern "C" HeldIfrtArray *ifrt_make_array_from_host_buffer_shards(
    ifrt::Client *client, const void **host_buffers, int num_buffers,
    const int64_t **host_buffer_shapes,
    const int64_t **addressable_shard_indices,
    const int64_t *addressable_shard_indices_sizes, int dtype_kind, int ndims,
    const int64_t *final_buffer_shape,
    HeldValue<std::shared_ptr<const ifrt::Sharding>> *sharding,
    int32_t c_host_buffer_semantics) {
  auto spec = ifrt_make_arrays_from_host_buffer_shards_spec(
      host_buffers, num_buffers, host_buffer_shapes, addressable_shard_indices,
      addressable_shard_indices_sizes, dtype_kind, ndims, final_buffer_shape,
      sharding);
  auto arrays = MyValueOrThrow(client->MakeArraysFromHostBufferShards(
      absl::MakeSpan(&spec, 1),
      static_cast<ifrt::Client::HostBufferSemantics>(c_host_buffer_semantics),
      client->CreateUserContext()));
  return reactant::capture(arrays[0]);
}

extern "C" void addSdyPropagationPipeline(
    mlir::OpPassManager &pm, uint8_t keepShardingRules /*false*/,
    uint8_t conservativePropagation /*false*/,
    uint8_t debugShardingOrigins /*false*/,
    uint8_t debugPropagationEdgeSharding /*false*/,
    uint8_t skipConvertToReshard /*false*/, uint8_t skipInline /*false*/,
    uint8_t enableInsertExplicitCollectives /*false*/) {
  const mlir::sdy::PropagationOptions options{keepShardingRules != 0,
                                              "",
                                              conservativePropagation != 0,
                                              debugShardingOrigins != 0,
                                              debugPropagationEdgeSharding != 0,
                                              skipConvertToReshard != 0,
                                              skipInline != 0,
                                              enableInsertExplicitCollectives !=
                                                  0};
  mlir::sdy::addPropagationPipeline(pm, options);
}

struct RegistryHolder {
mlir::DialectRegistry registry;
  RegistryHolder() : registry() {
    InitializeRegistry(wrap(&registry));
    InitializePasses(wrap(&registry));
  }
};

RegistryHolder registry;

struct ClientHolder {
  xla::PjRtClient* client;
  int device;
  ClientHolder() {
	InitializeLogs();
	const char* error = NULL;
	auto mpi = getenv("OMPI_COMM_WORLD_RANK");
	device = 0;
	if (mpi) {
		llvm::errs() << " mpi : " << mpi << "\n";
		device = atoi(mpi);
	} else llvm::errs() << " mpi: null\n";
	client = nullptr;
	if (getenv("USE_TPU")) {
	if (device == 0) {
	client = MakeTPUClient(nullptr, &error);
	if (error) llvm::errs() << " error: " << error << "\n";
	}
	} else
	client = MakeCPUClient(1, 0);
	assert(client);
	if (client) {
	  device = min(device, client->device_count()-1);
	}
	llvm::errs() << " client: " << client << "\n";
	llvm::errs() <<" device: " << device << "\n";
  }
};

ClientHolder client;

#define TRUE_ true
#define FALSE_ false
#define doublereal double
#define integer int32_t
#define logical bool
int xerbla_(const char *srname, integer *info, int len)
{
    static char fmt_9999[] = "(\002 ** On entry to \002,a,\002 parameter num"
	    "ber \002,i2,\002 had \002,\002an illegal value\002)";

	printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		srname, *info);
	assert(0 &&" error");
	exit(1);
    return 0;
}
logical dlaisnan_(doublereal *din1, doublereal *din2)
{
    /* System generated locals */
    logical ret_val;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This routine is not for general use.  It exists solely to avoid */
/*  over-optimization in DISNAN. */

/*  DLAISNAN checks for NaNs by comparing its two arguments for */
/*  inequality.  NaN is the only floating-point value where NaN != NaN */
/*  returns .TRUE.  To check for NaNs, pass the same variable as both */
/*  arguments. */

/*  A compiler must assume that the two arguments are */
/*  not the same variable, and the test will not be optimized away. */
/*  Interprocedural or whole-program optimization may delete this */
/*  test.  The ISNAN functions will be replaced by the correct */
/*  Fortran 03 intrinsic once the intrinsic is widely available. */

/*  Arguments */
/*  ========= */

/*  DIN1     (input) DOUBLE PRECISION */
/*  DIN2     (input) DOUBLE PRECISION */
/*          Two numbers to compare for inequality. */

/*  ===================================================================== */

/*  .. Executable Statements .. */
    ret_val = *din1 != *din2;
    return ret_val;
} /* dlaisnan_ */

logical disnan_(doublereal *din)
{
    /* System generated locals */
    logical ret_val;

    /* Local variables */

/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DISNAN returns .TRUE. if its argument is NaN, and .FALSE. */
/*  otherwise.  To be replaced by the Fortran 2003 intrinsic in the */
/*  future. */

/*  Arguments */
/*  ========= */

/*  DIN      (input) DOUBLE PRECISION */
/*          Input to test for NaN. */

/*  ===================================================================== */

/*  .. External Functions .. */
/*  .. */
/*  .. Executable Statements .. */
    ret_val = dlaisnan_(din, din);
    return ret_val;
} /* disnan_ */

logical lsame_(char *ca, char *cb, int ca_size, int cb_size)
{
    /* System generated locals */
    logical ret_val;

    /* Local variables */
    integer inta, intb, zcode;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  LSAME returns .TRUE. if CA is the same letter as CB regardless of */
/*  case. */

/*  Arguments */
/*  ========= */

/*  CA      (input) CHARACTER*1 */

/*  CB      (input) CHARACTER*1 */
/*          CA and CB specify the single characters to be compared. */

/* ===================================================================== */

/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */

/*     Test if the characters are equal */

    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
	return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime */
/*     machines, on which ICHAR returns a value with bit 8 set. */
/*     ICHAR('A') on Prime machines returns 193 which is the same as */
/*     ICHAR('A') on an EBCDIC machine. */

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*        ASCII is assumed - ZCODE is the ASCII code of either lower or */
/*        upper case 'Z'. */

	if (inta >= 97 && inta <= 122) {
	    inta += -32;
	}
	if (intb >= 97 && intb <= 122) {
	    intb += -32;
	}

    } else if (zcode == 233 || zcode == 169) {

/*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or */
/*        upper case 'Z'. */

	if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta 
		>= 162 && inta <= 169) {
	    inta += 64;
	}
	if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb 
		>= 162 && intb <= 169) {
	    intb += 64;
	}

    } else if (zcode == 218 || zcode == 250) {

/*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code */
/*        plus 128 of either lower or upper case 'Z'. */

	if (inta >= 225 && inta <= 250) {
	    inta += -32;
	}
	if (intb >= 225 && intb <= 250) {
	    intb += -32;
	}
    }
    ret_val = inta == intb;

/*     RETURN */

/*     End of LSAME */

    return ret_val;
} /* lsame_ */

/* Subroutine */ void dlacpy_(char *uplo, integer *m, integer *n, const doublereal *
	a, integer *lda, doublereal *b, integer *ldb)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    integer i__, j;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLACPY copies all or part of a two-dimensional matrix A to another */
/*  matrix B. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies the part of the matrix A to be copied to B. */
/*          = 'U':      Upper triangular part */
/*          = 'L':      Lower triangular part */
/*          Otherwise:  All of the matrix A */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input) DOUBLE PRECISION array, dimension (LDA,N) */
/*          The m by n matrix A.  If UPLO = 'U', only the upper triangle */
/*          or trapezoid is accessed; if UPLO = 'L', only the lower */
/*          triangle or trapezoid is accessed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  B       (output) DOUBLE PRECISION array, dimension (LDB,N) */
/*          On exit, B = A in the locations specified by UPLO. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,M). */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    if (lsame_(uplo, (char*)"U", 1, 1)) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = min(j,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
/* L10: */
	    }
/* L20: */
	}
    } else if (lsame_(uplo, (char*)"L", 1, 1)) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
/* L30: */
	    }
/* L40: */
	}
    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
/* L50: */
	    }
/* L60: */
	}
    }
    return;

/*     End of DLACPY */

} /* dlacpy_ */

/* Subroutine */ int dgemmbase_(const char *transa_t, const char *transb_t, const integer *m, const integer *
	n, const integer *k, const doublereal *alpha, const doublereal *a, const integer *lda,
	const doublereal *b, const integer *ldb, const doublereal *beta, doublereal *c, const integer
	*ldc)
{

    char transa_v = *transa_t;
    char* transa = &transa_v;

    char transb_v = *transb_t;
    char* transb = &transb_v;

    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3;

    /* Local variables */
    integer info;
    logical nota, notb;
    doublereal temp;
    integer i, j, l, ncola;
    integer nrowa, nrowb;


/*  Purpose
    =======

    DGEMM  performs one of the matrix-matrix operations

       C := alpha*op( A )*op( B ) + beta*C,

    where  op( X ) is one of

       op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )

    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.


    Parameters
    ==========

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in

             the matrix multiplication as follows:

                TRANSA = 'N' or 'n',  op( A ) = A.

                TRANSA = 'T' or 't',  op( A ) = A'.

                TRANSA = 'C' or 'c',  op( A ) = A'.

             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in

             the matrix multiplication as follows:

                TRANSB = 'N' or 'n',  op( B ) = B.

                TRANSB = 'T' or 't',  op( B ) = B'.

                TRANSB = 'C' or 'c',  op( B ) = B'.

             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix

             op( A )  and of the  matrix  C.  M  must  be at least  zero.

             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix

             op( B ) and the number of columns of the matrix C. N must be

             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix

             op( A ) and the number of rows of the matrix op( B ). K must

             be at least  zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is

             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k

             part of the array  A  must contain the matrix  A,  otherwise

             the leading  k by m  part of the array  A  must contain  the

             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared

             in the calling (sub) program. When  TRANSA = 'N' or 'n' then

             LDA must be at least  max( 1, m ), otherwise  LDA must be at

             least  max( 1, k ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is

             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n

             part of the array  B  must contain the matrix  B,  otherwise

             the leading  n by k  part of the array  B  must contain  the

             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared

             in the calling (sub) program. When  TRANSB = 'N' or 'n' then

             LDB must be at least  max( 1, k ), otherwise  LDB must be at

             least  max( 1, n ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is

             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must

             contain the matrix  C,  except when  beta  is zero, in which

             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix

             ( alpha*op( A )*op( B ) + beta*C ).

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared

             in  the  calling  (sub)  program.   LDC  must  be  at  least

             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.



       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not

       transposed and set  NROWA, NCOLA and  NROWB  as the number of rows

       and  columns of  A  and the  number of  rows  of  B  respectively.



   Parameter adjustments
       Function Body */

#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]
#define B(I,J) b[(I)-1 + ((J)-1)* ( *ldb)]
#define C(I,J) c[(I)-1 + ((J)-1)* ( *ldc)]

    nota = lsame_((char*)transa, (char*)"N", 1, 1);
    notb = lsame_((char*)transb, (char*)"N", 1, 1);
    if (nota) {
	nrowa = *m;
	ncola = *k;
    } else {
	nrowa = *k;
	ncola = *m;
    }
    if (notb) {
	nrowb = *k;
    } else {
	nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! lsame_((char*)transa, (char*)"C", 1, 1) && ! lsame_((char*)transa, (char*)"T", 1, 1)) {
	info = 1;
    } else if (! notb && ! lsame_((char*)transb, (char*)"C", 1, 1) && ! lsame_((char*)transb,
	    (char*)"T", 1, 1)) {
	info = 2;
    } else if (*m < 0) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < max(1,nrowa)) {
	info = 8;
    } else if (*ldb < max(1,nrowb)) {
	info = 10;
    } else if (*ldc < max(1,*m)) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("DGEMM ", &info, 0);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
	return 0;
    }

/*     And if  alpha.eq.zero. */

    if (*alpha == 0.) {
	if (*beta == 0.) {
	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		i__2 = *m;
		for (i = 1; i <= *m; ++i) {
		    C(i,j) = 0.;
/* L10: */
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		i__2 = *m;
		for (i = 1; i <= *m; ++i) {
		    C(i,j) = *beta * C(i,j);
/* L30: */
		}
/* L40: */
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (notb) {
	if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		if (*beta == 0.) {
		    i__2 = *m;
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = 0.;
/* L50: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *m;
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = *beta * C(i,j);
/* L60: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= *k; ++l) {
		    if (B(l,j) != 0.) {
			temp = *alpha * B(l,j);
			i__3 = *m;
			for (i = 1; i <= *m; ++i) {
			    C(i,j) += temp * A(i,l);
/* L70: */
			}
		    }
/* L80: */
		}
/* L90: */
	    }
	} else {

/*           Form  C := alpha*A'*B + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		i__2 = *m;
		for (i = 1; i <= *m; ++i) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= *k; ++l) {
			temp += A(l,i) * B(l,j);
/* L100: */
		    }
		    if (*beta == 0.) {
			C(i,j) = *alpha * temp;
		    } else {
			C(i,j) = *alpha * temp + *beta * C(i,j);
		    }
/* L110: */
		}
/* L120: */
	    }
	}
    } else {
	if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		if (*beta == 0.) {
		    i__2 = *m;
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = 0.;
/* L130: */
		    }
		} else if (*beta != 1.) {
		    i__2 = *m;
		    for (i = 1; i <= *m; ++i) {
			C(i,j) = *beta * C(i,j);
/* L140: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= *k; ++l) {
		    if (B(j,l) != 0.) {
			temp = *alpha * B(j,l);
			i__3 = *m;
			for (i = 1; i <= *m; ++i) {
			    C(i,j) += temp * A(i,l);
/* L150: */
			}
		    }
/* L160: */
		}
/* L170: */
	    }
	} else {

/*           Form  C := alpha*A'*B' + beta*C */

	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		i__2 = *m;
		for (i = 1; i <= *m; ++i) {
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= *k; ++l) {
			temp += A(l,i) * B(j,l);
/* L180: */
		    }
		    if (*beta == 0.) {
			C(i,j) = *alpha * temp;
		    } else {
			C(i,j) = *alpha * temp + *beta * C(i,j);
		    }
/* L190: */
		}
/* L200: */
	    }
	}
    }

    return 0;
#undef A
#undef B
#undef C
/*     End of DGEMM . */

} /* dgemm_ */

template <typename T> 
static
Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

template <> 
Attribute makeAttr(mlir::Type elemType, APFloat val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<APFloat>(TT.getElementType(), val)));
  return FloatAttr::get(elemType, val);
}

extern "C" void dgemm_(char* transA, char* transB, int32_t* M, int32_t* N, int32_t* K, double* alpha,
		  double* a,
		  int32_t *lda,
		  double* b,
		  int32_t *ldb,
		  double* beta,
		  double* c,
		  int32_t* ldc) {
   if (client.client) {
     dgemmbase_(transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
     return;
   }

   bool transa = *transA == 'T' || *transA == 't';
   bool transb = *transB == 'T' || *transB == 't';
   std::tuple<bool, bool, int32_t, int32_t, int32_t> key { transa, transb, *M, *N, *K};
  static std::map<decltype(key), xla::PjRtLoadedExecutable *> executables;
  auto found = executables.find(key);
   xla::PjRtLoadedExecutable * exec;
     int64_t ATy[] = {(!transa) ? *M : *K, (!transa) ? *K : *M};
     std::swap(ATy[0], ATy[1]);
     int64_t BTy[] = {(!transb) ? *K : *N, (!transb) ? *N : *K};
     std::swap(BTy[0], BTy[1]);
     int64_t CTy[] = {*M, *N};
     std::swap(CTy[0], CTy[1]);
  if (found == executables.end()) {
     MLIRContext context(registry.registry);
     RegisterDialects(wrap(&context));

     mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::OpBuilder(&context).getUnknownLoc()));
     
  mlir::OpBuilder builder(module->getContext());
     auto dty = builder.getF64Type();
     mlir::Type types[] = { RankedTensorType::get(ATy, dty), RankedTensorType::get(BTy, dty), RankedTensorType::get(CTy, dty), RankedTensorType::get({}, dty), RankedTensorType::get({}, dty) };
     mlir::Type rettypes[] = { RankedTensorType::get(CTy, dty) };
     auto funcType = builder.getFunctionType(types, rettypes);

     auto loc = builder.getUnknownLoc();
      mlir::func::FuncOp function = mlir::func::FuncOp(mlir::func::FuncOp::create(loc, "main", funcType));
     module->push_back(function);
     builder.setInsertionPointToStart(function.addEntryBlock());
     auto ra = builder.create<stablehlo::TransposeOp>(loc, function.getArgument(0), std::vector<int64_t>{1, 0});
     auto rb = builder.create<stablehlo::TransposeOp>(loc, function.getArgument(1), std::vector<int64_t>{1, 0});
     auto rc = builder.create<stablehlo::TransposeOp>(loc, function.getArgument(2), std::vector<int64_t>{1, 0});

     auto res = builder.create<stablehlo::DotGeneralOp>(loc, rc.getType(), (mlir::Value)ra, (mlir::Value)rb,
	  stablehlo::DotDimensionNumbersAttr::get(
      		&context,
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/
		    std::vector<int64_t>{transa ? 0 : 1},
      /*rhsContractingDimensions=*/ 
		    std::vector<int64_t>{transb ? 1 : 0}), nullptr, nullptr);
     auto alphabc = builder.create<stablehlo::BroadcastInDimOp>(loc, res.getType(), function.getArgument(3), std::vector<int64_t>{});
     auto betabc = builder.create<stablehlo::BroadcastInDimOp>(loc, res.getType(), function.getArgument(3), std::vector<int64_t>{});
     auto res2 = builder.create<stablehlo::MulOp>(loc, res, alphabc);
     auto res3 = builder.create<stablehlo::MulOp>(loc, rc, betabc);
     auto res4 = builder.create<stablehlo::AddOp>(loc, res2, res3);
     auto fres = builder.create<stablehlo::TransposeOp>(loc, res4, std::vector<int64_t>{1, 0});
     Value resv[] = {fres};
     builder.create<stablehlo::ReturnOp>(loc, resv);

     int device_id = client.device;
     int64_t* mesh_ids = nullptr;
     int num_mesh_ids = 0;
     const char *xla_gpu_cuda_data_dir = "";
     bool use_shardy_partitioner = false;
     int64_t num_replicas = 1;
     int64_t num_partitions = 1;
     bool use_spmd_partitioning = false;
     llvm::errs() <<  " client: " << client.client << "\n";
     exec = ClientCompile(client.client, wrap(module.get()), device_id, mesh_ids, num_mesh_ids, xla_gpu_cuda_data_dir, use_shardy_partitioner, num_replicas, num_partitions, use_spmd_partitioning);
     executables[key] = exec;
  } else {
     exec = found->second;
  }

  // TODO avoid copy if contiguous
  double *Abuf = (double*)malloc(sizeof(double)*(*M)*(*K));
  char layout = '\0';
  dlacpy_(&layout, (!transa) ? M : K, (!transa) ? K : M, a, lda, Abuf, (!transa) ? M : K);
  
  double *Bbuf = (double*)malloc(sizeof(double)*(*K)*(*N));
  dlacpy_(&layout, (!transb) ? K : N, (!transb) ? N : K, b, ldb, Bbuf, (!transb) ? K : N);
  
  double *Cbuf = (double*)malloc(sizeof(double)*(*K)*(*N));
  dlacpy_(&layout, M, N, c, ldc, Cbuf, M);
  
  int device_id = client.device;
  
  PjRtDevice *device = ClientGetDevice(client.client, device_id);

  llvm::errs() << "device: " << device->ToString() << "\n";
  int num_args = 5;
  int dtype = 12;
  PjRtBuffer *args[] = { 
	  ArrayFromHostBuffer(client.client, Abuf, dtype, 2, ATy, device),
	  ArrayFromHostBuffer(client.client, Bbuf, dtype, 2, BTy, device),
	  ArrayFromHostBuffer(client.client, Cbuf, dtype, 2, CTy, device),
	  ArrayFromHostBuffer(client.client, alpha, dtype, 0, CTy, device),
	  ArrayFromHostBuffer(client.client, beta, dtype, 0, CTy, device)
  };
  
  uint8_t is_arg_donatable[5] = {true, true, true, true, true};
  int num_results = 1;
  PjRtBuffer *results[1];
  uint8_t futures[1] = { 0 };
  FutureType * future_results[1] = { 0 };
  XLAExecuteSharded(exec, num_args, args, device, is_arg_donatable, num_results, results, futures, future_results);

  if (futures[0]) {
    FutureAwait(future_results[0]);
    FreeFuture(future_results[0]);
  }
  BufferToHost(results[0], Cbuf);
  PjRtBufferFree(results[0]);

  dlacpy_(&layout, M, N, Cbuf, M, c, ldc);

  free(Abuf);
  free(Bbuf);
  free(Cbuf);
}

