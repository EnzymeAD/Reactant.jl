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
#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

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
#if defined(REACTANT_CUDA) || defined(REACTANT_ROCM)
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#endif
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/stablehlo.h"

// CPU collectives
#if defined(__linux__)
#include "gloo/transport/tcp/attr.h"
#include "gloo/transport/tcp/device.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#include "xla/backends/cpu/collectives/mpi_collectives.h"
#elif defined(__APPLE__)
#include "gloo/transport/uv/device.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#include "xla/backends/cpu/collectives/mpi_collectives.h"
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
#include "xla/xla.pb.h"

#if defined(REACTANT_CUDA) || defined(REACTANT_ROCM)
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#endif

// Broken upstream x/ref https://github.com/jax-ml/jax/issues/33344
// #include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

#include "llvm/Support/ExtensibleRTTI.h"
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include "plugin/xprof/worker/stub_factory.h"
#include "xprof/convert/tool_options.h"
#include "xprof/pywrap/profiler_plugin_impl.h"

#include "absl/status/statusor.h"
#include "xla/pjrt/proto/compile_options.pb.h"

using namespace mlir;
using namespace xla;
using ::tensorflow::profiler::ToolOptions;

namespace mlir {
namespace enzyme {
void registerRemoveTransformPass();
void registerGenerateApplyPatternsPass();
} // namespace enzyme

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

#define REACTANT_ABI extern "C" MLIR_CAPI_EXPORTED

using reactant::HeldValue;
using HeldPjRtClient = HeldValue<std::shared_ptr<xla::PjRtClient>>;
using HeldPjRtBuffer = HeldValue<std::shared_ptr<xla::PjRtBuffer>>;
using HeldIfrtArray = HeldValue<tsl::RCReference<xla::ifrt::Array>>;
using HeldHloModule = HeldValue<std::shared_ptr<xla::HloModule>>;
using HeldIfrtSharding = HeldValue<std::shared_ptr<xla::ifrt::Sharding>>;
using HeldIfrtLoadedExecutable =
    HeldValue<std::shared_ptr<xla::ifrt::LoadedExecutable>>;

REACTANT_ABI void (*ReactantThrowError)(const char *) = nullptr;

// Utilities for `StatusOr`.
template <typename T> T MyValueOrThrow(absl::StatusOr<T> v) {
  if (!v.ok()) {
    ReactantThrowError(v.status().ToString().c_str());
  }
  return std::move(v).value();
}

REACTANT_ABI void ReactantHandleCuResult(uint32_t curesult) {
  if (curesult != 0) {
    std::string err = "Bad Cuda Result = " + std::to_string(curesult);
    if (ReactantThrowError) {
      ReactantThrowError(err.c_str());
    }
  }
}

// MLIR C-API extras
#pragma region MLIR Extra
REACTANT_ABI MlirAttribute mlirComplexAttrDoubleGet(MlirContext ctx,
                                                    MlirType type, double real,
                                                    double imag) {
  return wrap(
      complex::NumberAttr::get(cast<ComplexType>(unwrap(type)), real, imag));
}

REACTANT_ABI MlirAttribute mlirComplexAttrDoubleGetChecked(MlirLocation loc,
                                                           MlirType type,
                                                           double real,
                                                           double imag) {
  return wrap(complex::NumberAttr::getChecked(
      unwrap(loc), cast<ComplexType>(unwrap(type)), real, imag));
}

REACTANT_ABI bool mlirOperationInject(MlirContext ctx, MlirBlock block,
                                      MlirStringRef code, MlirLocation location,
                                      bool verify_after_parse) {
  ParserConfig config(unwrap(ctx), verify_after_parse);
  if (failed(parseSourceString(unwrap(code), unwrap(block), config)))
    return false;
  return true;
}

REACTANT_ABI MlirOperation mlirOperationParse(MlirContext ctx, MlirBlock block,
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

REACTANT_ABI MlirType mlirGetFunctionTypeFromOperation(MlirOperation op) {
  if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(unwrap(op))) {
    return wrap(funcOp.getFunctionType());
  }
  ReactantThrowError("Not a function op");
}

REACTANT_ABI bool mlirIsFunctionOpInterface(MlirOperation op) {
  return llvm::isa<mlir::FunctionOpInterface>(unwrap(op));
}

// TODO mlirComplexAttrGetnValue
// TODO REACTANT_ABI MlirTypeID mlirComplexAttrGetTypeID(void) { return
// wrap(complex::NumberAttr::getTypeID()); }

REACTANT_ABI void ReactantFuncSetResultAttr(MlirOperation op, intptr_t pos,
                                            MlirStringRef name,
                                            MlirAttribute attr) {
  llvm::cast<mlir::FunctionOpInterface>(unwrap(op))
      .setResultAttr(pos, unwrap(name), unwrap(attr));
}

REACTANT_ABI void ReactantFuncSetArgAttr(MlirOperation op, intptr_t pos,
                                         MlirStringRef name,
                                         MlirAttribute attr) {
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

REACTANT_ABI void InitializeLogs() {
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

REACTANT_ABI void SetLogLevel(int level) {
  SetStderrThreshold((absl::LogSeverity)level);
  // absl::SetGlobalVLogLevel(level);
}

REACTANT_ABI void SetModuleLogLevel(const char *module_pattern, int level) {
  // absl::SetVLOGLevel(module_pattern, level);
}

REACTANT_ABI char *GetDefaultTargetTriple(void) {
  return LLVMGetDefaultTargetTriple();
}

REACTANT_ABI MLIR_CAPI_EXPORTED MlirAttribute
enzymeActivityAttrGet(MlirContext ctx, int32_t val) {
  return wrap(mlir::enzyme::ActivityAttr::get(unwrap(ctx),
                                              (mlir::enzyme::Activity)val));
}

REACTANT_ABI MLIR_CAPI_EXPORTED MlirType enzymeTraceTypeGet(MlirContext ctx) {
  return wrap(mlir::enzyme::TraceType::get(unwrap(ctx)));
}

REACTANT_ABI MLIR_CAPI_EXPORTED MlirType
enzymeConstraintTypeGet(MlirContext ctx) {
  return wrap(mlir::enzyme::ConstraintType::get(unwrap(ctx)));
}

REACTANT_ABI MLIR_CAPI_EXPORTED MlirAttribute
enzymeSymbolAttrGet(MlirContext ctx, uint64_t symbol) {
  mlir::Attribute attr = mlir::enzyme::SymbolAttr::get(unwrap(ctx), symbol);
  return wrap(attr);
}

REACTANT_ABI MLIR_CAPI_EXPORTED MlirAttribute
enzymeRngDistributionAttrGet(MlirContext ctx, int32_t val) {
  return wrap(mlir::enzyme::RngDistributionAttr::get(
      unwrap(ctx), (mlir::enzyme::RngDistribution)val));
}

REACTANT_ABI MLIR_CAPI_EXPORTED MlirAttribute
enzymeMCMCAlgorithmAttrGet(MlirContext ctx, int32_t val) {
  return wrap(mlir::enzyme::MCMCAlgorithmAttr::get(
      unwrap(ctx), (mlir::enzyme::MCMCAlgorithm)val));
}

// Create profiler session and start profiling
REACTANT_ABI tsl::ProfilerSession *
CreateProfilerSession(uint32_t device_tracer_level,
                      uint32_t host_tracer_level) {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_tracer_level(device_tracer_level);
  options.set_host_tracer_level(host_tracer_level);
  auto sess = tsl::ProfilerSession::Create(options);
  return sess.release();
}

REACTANT_ABI void ProfilerSessionCollectData(tsl::ProfilerSession *session,
                                             const char *path) {
  tensorflow::profiler::XSpace xspace;
  auto status = session->CollectData(&xspace);
  if (!status.ok())
    ReactantThrowError("cannot collect data for profiler");
  tsl::profiler::ExportToTensorBoard(xspace, path,
                                     /*also_export_trace_json*/ true);
}

REACTANT_ABI void ProfilerSessionDelete(tsl::ProfilerSession *session) {
  delete session;
}

REACTANT_ABI int64_t ProfilerActivityStart(const char *name, int level) {
  return tsl::profiler::TraceMe::ActivityStart(name, level);
}

REACTANT_ABI void ProfilerActivityEnd(int64_t id) {
  tsl::profiler::TraceMe::ActivityEnd(id);
}

REACTANT_ABI tsl::profiler::ProfilerServer *ProfilerServerStart(int32_t port) {
  auto server = new tsl::profiler::ProfilerServer();
  server->StartProfilerServer(port);
  return server;
}

REACTANT_ABI void ProfilerServerStop(tsl::profiler::ProfilerServer *server) {
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

  return MyValueOrThrow(GetPjRtCpuClient(options)).release();
}

REACTANT_ABI PjRtClient *MakeCPUClient(uint8_t asynchronous, int node_id) {
  std::optional<std::shared_ptr<xla::cpu::CpuCollectives>> collectives;
  return MakeCPUClientInternal(asynchronous, node_id, collectives);
}

// xla/python/xla.cc 390
REACTANT_ABI PjRtClient *
MakeGPUClient(int node_id, int num_nodes, int64_t *allowed_devices,
              int64_t num_allowed_devices, double memory_fraction,
              bool preallocate, const char *platform_name, const char **error,
              void *distributed_runtime_client) {
#if defined(REACTANT_CUDA) || defined(REACTANT_ROCM)
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
#else
  *error = "ReactantExtra was not built with GPU support";
  return nullptr;
#endif
}

const char *const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";

REACTANT_ABI const PJRT_Api *LoadPjrtPlugin(const char *device_type,
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

REACTANT_ABI int InitializePjrtPlugin(const char *device_type,
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

REACTANT_ABI PjRtClient *GetCApiClient(const char *device_type) {
  return xla::GetCApiClient(device_type).value().release();
}

REACTANT_ABI void pjrt_client_register_profiler(const PJRT_Api *api) {
  RegisterProfiler(api);
}

REACTANT_ABI PjRtClient *MakeClientUsingPluginAPI(const char *device_type,
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

REACTANT_ABI PjRtClient *MakeTPUClient(const char *tpu_path,
                                       const char **error) {
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

REACTANT_ABI int ClientNumDevices(PjRtClient *client) {
  return client->device_count();
}

REACTANT_ABI int ClientNumAddressableDevices(PjRtClient *client) {
  return client->addressable_device_count();
}

REACTANT_ABI int ClientProcessIndex(PjRtClient *client) {
  return client->process_index();
}

REACTANT_ABI PjRtDevice *ClientGetDevice(PjRtClient *client, int device_id) {
  return MyValueOrThrow(client->LookupDevice(PjRtGlobalDeviceId(device_id)));
}

REACTANT_ABI PjRtDevice *ClientGetAddressableDevice(PjRtClient *client,
                                                    int device_id) {
  return MyValueOrThrow(
      client->LookupAddressableDevice(PjRtLocalDeviceId(device_id)));
}

REACTANT_ABI const char *ClientGetPlatformName(PjRtClient *client) {
  return cstr_from_string(client->platform_name());
}

REACTANT_ABI const char *DeviceGetKind(PjRtDevice *device) {
  return cstr_from_string(device->device_kind());
}

REACTANT_ABI void ClientGetDevices(PjRtClient *client,
                                   PjRtDevice **out_devices) {
  auto span = client->devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

REACTANT_ABI void ClientGetAddressableDevices(PjRtClient *client,
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

REACTANT_ABI void PjRtDeviceGetAllocatorStats(PjRtDevice *device,
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

REACTANT_ABI void ifrt_device_get_allocator_stats(ifrt::Device *device,
                                                  JLAllocatorStats *jlstats) {
  if (!llvm::isa<ifrt::PjRtDevice>(device)) {
    ReactantThrowError(
        "ifrt_device_get_allocator_stats: only supported for ifrt-pjrt.");
  }
  auto ifrt_pjrt_device = llvm::dyn_cast<ifrt::PjRtDevice>(device);
  PjRtDeviceGetAllocatorStats(ifrt_pjrt_device->pjrt_device(), jlstats);
}

REACTANT_ABI void ExecutableFree(xla::PjRtLoadedExecutable *exec) {
  delete exec;
}

REACTANT_ABI PjRtDevice *BufferToDevice(PjRtBuffer *Buffer) {
  return Buffer->device();
}

REACTANT_ABI PjRtClient *BufferToClient(PjRtBuffer *Buffer) {
  return Buffer->client();
}

REACTANT_ABI const int64_t *BufferShape(PjRtBuffer *Buffer) {
  return Buffer->dimensions().data();
}

REACTANT_ABI int64_t BufferNDimensions(PjRtBuffer *Buffer) {
  return Buffer->dimensions().length();
}

REACTANT_ABI xla::PrimitiveType BufferPrimitiveType(PjRtBuffer *Buffer) {
  return Buffer->element_type();
}

REACTANT_ABI void PjRtBufferFree(PjRtBuffer *Buffer) { delete Buffer; }

REACTANT_ABI PjRtClient *DeviceToClient(PjRtDevice *Device) {
  return Device->client();
}

REACTANT_ABI PjRtClient *
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

REACTANT_ABI void ReactantLLVMParseCommandLineOptions(int argc,
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

#ifdef REACTANT_CUDA

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

REACTANT_ABI int32_t ReactantCudaDriverGetVersion() {
  int32_t data;
  ReactantHandleCuResult(cuDriverGetVersion(&data));
  return data;
}

REACTANT_ABI int32_t ReactantHermeticCudaGetVersion() { return CUDA_VERSION; }

REACTANT_ABI int32_t ReactantCudaDeviceGetComputeCapalilityMajor() {
  CUdevice cuDevice;
  ReactantHandleCuResult(cuDeviceGet(&cuDevice, 0));
  int major;
  ReactantHandleCuResult(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  return major;
}

REACTANT_ABI int32_t ReactantCudaDeviceGetComputeCapalilityMinor() {
  CUdevice cuDevice;
  ReactantHandleCuResult(cuDeviceGet(&cuDevice, 0));
  int minor;
  ReactantHandleCuResult(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  return minor;
}

REACTANT_ABI int32_t ReactantCudaDeviceGetWarpSizeInThreads() {
  CUdevice cuDevice;
  ReactantHandleCuResult(cuDeviceGet(&cuDevice, 0));
  int warpSize;
  ReactantHandleCuResult(
      cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice));
  return warpSize;
}

REACTANT_ABI void ReactantCudaDeviceGetProperties(DeviceProperties *jlprops,
                                                  int32_t device_id) {
  cudaDeviceProp props;
  ReactantHandleCuResult(cudaGetDeviceProperties(&props, device_id));

  jlprops->totalGlobalMem = props.totalGlobalMem;
  jlprops->sharedMemPerBlock = props.sharedMemPerBlock;
  jlprops->regsPerBlock = props.regsPerBlock;
  jlprops->warpSize = props.warpSize;
  jlprops->maxThreadsPerBlock = props.maxThreadsPerBlock;
  jlprops->maxThreadsDim[0] = props.maxThreadsDim[0];
  jlprops->maxThreadsDim[1] = props.maxThreadsDim[1];
  jlprops->maxThreadsDim[2] = props.maxThreadsDim[2];
  jlprops->maxGridSize[0] = props.maxGridSize[0];
  jlprops->maxGridSize[1] = props.maxGridSize[1];
  jlprops->maxGridSize[2] = props.maxGridSize[2];
  jlprops->totalConstMem = props.totalConstMem;
  jlprops->major = props.major;
  jlprops->minor = props.minor;
  jlprops->multiProcessorCount = props.multiProcessorCount;
  jlprops->canMapHostMemory = props.canMapHostMemory;
  jlprops->l2CacheSize = props.l2CacheSize;
  jlprops->maxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
}

REACTANT_ABI void ReactantCudaGetRegsSpillsMaxThreadsFromBinary(
    const char *binary, const char *fnname, int32_t *regs, int32_t *spills,
    int32_t *maxThreads) {
  CUfunction fun;
  CUmodule mod;

  ReactantHandleCuResult(cuModuleLoadData(&mod, binary));
  ReactantHandleCuResult(cuModuleGetFunction(&fun, mod, fnname));

  ReactantHandleCuResult(
      cuFuncGetAttribute(regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  ReactantHandleCuResult(
      cuFuncGetAttribute(spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  *spills /= 4;
  ReactantHandleCuResult(cuFuncGetAttribute(
      maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));

  return;
}

inline stream_executor::SemanticVersion
GetStreamExecutorVersion(int32_t version) {
  return stream_executor::SemanticVersion(version / 1000, (version % 1000) / 10,
                                          version % 10);
}

inline int32_t GetCudaIntegerAttribute(cudaDeviceAttr attribute,
                                       int32_t device_id) {
  int32_t value;
  ReactantHandleCuResult(cudaDeviceGetAttribute(&value, attribute, device_id));
  return value;
}

static int32_t CUDACoresPerSM(int32_t major, int32_t minor) {
  switch (major) {
  case 2:
    return 32;
  case 3:
    return 192;
  case 7:
    return 64;
  case 8:
    return minor == 0 ? 64 : 128;
  default:
    return 128;
  }
}

REACTANT_ABI stream_executor::DeviceDescription *
CudaGetStreamExecutorDeviceDescription(int32_t device_id) {
  stream_executor::DeviceDescription *device_description =
      new stream_executor::DeviceDescription();

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);

  device_description->set_gpu_compute_capability(
      stream_executor::CudaComputeCapability(props.major, props.minor));

  device_description->set_threads_per_block_limit(props.maxThreadsPerBlock);
  device_description->set_threads_per_warp(props.warpSize);
  device_description->set_shared_memory_per_block(props.sharedMemPerBlock);
  device_description->set_shared_memory_per_block_optin(GetCudaIntegerAttribute(
      cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
  device_description->set_shared_memory_per_core(GetCudaIntegerAttribute(
      cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));
  device_description->set_threads_per_core_limit(GetCudaIntegerAttribute(
      cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
  device_description->set_core_count(props.multiProcessorCount);
  device_description->set_fpus_per_core(
      CUDACoresPerSM(props.major, props.minor));
  device_description->set_block_dim_limit_x(props.maxGridSize[0]);
  device_description->set_block_dim_limit_y(props.maxGridSize[1]);
  device_description->set_block_dim_limit_z(props.maxGridSize[2]);

  // Memory bandwidth (bytes/sec) ≈ 2 * memClock(Hz) * busWidth(bytes)
  // props.memoryClockRate is in kHz; bus width is in bits.
  const double mem_clock_hz = static_cast<double>(GetCudaIntegerAttribute(
                                  cudaDevAttrMemoryClockRate, device_id)) *
                              1000.0;
  const double bus_bytes = static_cast<double>(props.memoryBusWidth) / 8.0;
  const double bandwidth_Bps = 2.0 * mem_clock_hz * bus_bytes; // DDR assumption
  device_description->set_memory_bandwidth(
      static_cast<uint64_t>(bandwidth_Bps));

  device_description->set_l2_cache_size(
      GetCudaIntegerAttribute(cudaDevAttrL2CacheSize, device_id));

  // SM clock (GHz). props.clockRate is kHz.
  device_description->set_clock_rate_ghz(
      static_cast<double>(
          GetCudaIntegerAttribute(cudaDevAttrClockRate, device_id)) /
      1.0e6);
  device_description->set_device_memory_size(props.totalGlobalMem);

  // Registers
  device_description->set_registers_per_core_limit(GetCudaIntegerAttribute(
      cudaDevAttrMaxRegistersPerMultiprocessor, device_id));
  device_description->set_registers_per_block_limit(
      GetCudaIntegerAttribute(cudaDevAttrMaxRegistersPerBlock, device_id));

  // CUDA versions
  int drv = 0, rtm = 0;
  cudaRuntimeGetVersion(&rtm);
  device_description->set_runtime_version(GetStreamExecutorVersion(rtm));
  cudaDriverGetVersion(&drv);
  device_description->set_driver_version(GetStreamExecutorVersion(drv));

  return device_description;
}

#else

REACTANT_ABI int32_t ReactantCudaDriverGetVersion() { return 0; }

REACTANT_ABI int32_t ReactantHermeticCudaGetVersion() { return 0; }

REACTANT_ABI int32_t ReactantCudaDeviceGetComputeCapalilityMajor() { return 0; }

REACTANT_ABI int32_t ReactantCudaDeviceGetComputeCapalilityMinor() { return 0; }

REACTANT_ABI int32_t ReactantCudaDeviceGetWarpSizeInThreads() { return 0; }

REACTANT_ABI void ReactantCudaDeviceGetProperties(DeviceProperties *jlprops,
                                                  int32_t device_id) {}

REACTANT_ABI void ReactantCudaGetRegsSpillsMaxThreadsFromBinary(
    const char *binary, const char *fnname, int32_t *regs, int32_t *spills,
    int32_t *maxThreads) {}

REACTANT_ABI stream_executor::DeviceDescription *
CudaGetStreamExecutorDeviceDescription(int32_t device_id) {
  return nullptr;
}

#endif

REACTANT_ABI const char *
deviceDescriptionToString(stream_executor::DeviceDescription *device) {
  return cstr_from_string(device->ToString());
}

REACTANT_ABI void *UnsafeBufferPointer(PjRtBuffer *buffer) {
  auto unsafe = MyValueOrThrow(buffer->client()->UnsafeBufferPointer(buffer));
  return (void *)unsafe;
}

REACTANT_ABI PjRtBuffer *ArrayFromHostBuffer(PjRtClient *client, void *data,
                                             uint64_t ptype, size_t dim,
                                             const int64_t *cshape,
                                             PjRtDevice *device) {
  auto primtype = (xla::PrimitiveType)ptype;
  absl::Span<const int64_t> shape(cshape, dim);
  PjRtClient::HostBufferSemantics semantics =
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
  // xla::Layout layout(col_major(dim));
  // auto buffer = xla::MyValueOrThrow(client->BufferFromHostBuffer(data,
  // primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device,
  // &layout));
  const xla::Layout *layout = nullptr;
  auto buffer = MyValueOrThrow(client->BufferFromHostBuffer(
      data, primtype, shape, /*byte_strides*/ {}, semantics, /*ondone*/ {},
      *device->default_memory_space(), layout));
  auto bres = buffer.release();
  return bres;
}

REACTANT_ABI void CopyToBuffer(PjRtClient *client, PjRtBuffer *buffer,
                               void *data, size_t offset, size_t size,
                               PjRtBuffer **bufferP) {
  if (buffer->IsOnCpu()) {
    auto unsafe =
        (char *)MyValueOrThrow(buffer->client()->UnsafeBufferPointer(buffer));
    memcpy(unsafe + offset, data, size);
    // memcpy((char*) ((AbstractCpuBuffer*)buffer)->untyped_data() + offset,
    // data, size);
    return;
  }

  auto pid = client->platform_id();
  if (pid == xla::TpuId()) {
    auto dims = buffer->on_device_shape().dimensions();
    // TODO: note this assume that we want to copy the entire buffer size.
    auto buf2 = ArrayFromHostBuffer(client, data, buffer->element_type(),
                                    dims.size(), dims.data(), buffer->device());
    *bufferP = buf2;
    PjRtBufferFree((PjRtBuffer *)buffer);
    return;
  }

  auto raw_buffer =
      MyValueOrThrow(PjRtRawBuffer::CreateRawAliasOfBuffer(buffer));
  auto future = raw_buffer->CopyRawHostToDevice(data, offset, size);
  future.Await();
#if 0
  if (buffer->IsOnCpu()) {
    memcpy((char*)client->UnsafeBufferPointer(buffer) + offset, data, size);
    return;
  }

  if (pid == xla::CudaId()) {
    auto stream_client = (xla::PjRtStreamExecutorClient*)lrt->client;


    auto memory_space = *memory_device->default_memory_space();
    auto shape = MyValueOrThrow(cpu_client->MakeDefaultShapeForMemorySpace(memory_space));
    auto on_device_bytes = MyValueOrThrow(cpu_client->GetOnDeviceBytesCount(memory_space, shape));
    auto buf = MyValueOrThrow(cpu_client->AllocateRawBuffer(memory_space, on_device_bytes_count, /*retry_on_oom*/true, /*allocateafter*/{}));
    lrt->allocations.try_emplace(buf, on_device_bytes);

    cpu_client->LinearizeHostBufferInfo(data, 
    return buf;
  }

  } else if (pid == xla::CudaId()) {
  } else if (pid == xla::TpuId()) {
  }
  switch(client->platform_id()) {
    case xla::CpuId();
  }
  absl::
  auto atm = MyValueOrThrow(client->CreateBuffersForAsyncHostToDevice({buffer->on_device_shape()}, buffer->memory_space()));
#endif
}

REACTANT_ABI void BufferToHost(PjRtBuffer *buffer, void *data) {
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

REACTANT_ABI void CopyFromBuffer(PjRtClient *client, PjRtBuffer *buffer,
                                 void *data, size_t offset, size_t size,
                                 PjRtBuffer **bufferP) {

  auto pid = client->platform_id();
  if (pid == xla::TpuId()) {
    // TODO: note this assume that we want to copy the entire buffer size.
    BufferToHost(buffer, data);
    return;
  }

  auto future = buffer->CopyRawToHost(data, offset, size);
  future.Await();
#if 0
  if (buffer->IsOnCpu()) {
    memcpy((char*)client->UnsafeBufferPointer(buffer) + offset, data, size);
    return;
  }

  auto pid = client->platform_id();
  if (pid == xla::CudaId()) {
    auto stream_client = (xla::PjRtStreamExecutorClient*)lrt->client;


    auto memory_space = *memory_device->default_memory_space();
    auto shape = MyValueOrThrow(cpu_client->MakeDefaultShapeForMemorySpace(memory_space));
    auto on_device_bytes = MyValueOrThrow(cpu_client->GetOnDeviceBytesCount(memory_space, shape));
    auto buf = MyValueOrThrow(cpu_client->AllocateRawBuffer(memory_space, on_device_bytes_count, /*retry_on_oom*/true, /*allocateafter*/{}));
    lrt->allocations.try_emplace(buf, on_device_bytes);

    cpu_client->LinearizeHostBufferInfo(data, 
    return buf;
  }

  } else if (pid == xla::CudaId()) {
  } else if (pid == xla::TpuId()) {
  }
  switch(client->platform_id()) {
    case xla::CpuId();
  }
  absl::
  auto atm = MyValueOrThrow(client->CreateBuffersForAsyncHostToDevice({buffer->on_device_shape()}, buffer->memory_space()));
#endif
}

REACTANT_ABI PjRtBuffer *UninitPJRTBuffer(PjRtClient *client,
                                          PjRtDevice *device, uint64_t ptype,
                                          uint64_t shapeLen,
                                          uint64_t *__restrict__ shape) {
  auto memory_space = *device->default_memory_space();
  xla::Shape xlashape(
      (xla::PrimitiveType)ptype,
      absl::Span<const int64_t>((const int64_t *)shape, shapeLen));
  auto xbuffer =
      MyValueOrThrow(client->CreateUninitializedBuffer(xlashape, memory_space));
  return xbuffer.release();
}

REACTANT_ABI uint8_t BufferOnCPU(PjRtBuffer *buffer) {
  return buffer->IsOnCpu();
}

REACTANT_ABI PjRtBuffer *CopyBufferToDevice(PjRtBuffer *buffer,
                                            PjRtDevice *dst_device) {
  auto res = MyValueOrThrow(
      buffer->CopyToMemorySpace(*dst_device->default_memory_space()));
  return res.release();
}

REACTANT_ABI void FreeClient(PjRtClient *client) { delete client; }

REACTANT_ABI int64_t PjRtDeviceGetLocalDeviceId(PjRtDevice *device) {
  return device->local_device_id().value();
}

REACTANT_ABI int64_t PjRtDeviceGetGlobalDeviceId(PjRtDevice *device) {
  return device->global_device_id().value();
}

REACTANT_ABI int64_t PjRtDeviceGetLocalHardwareId(PjRtDevice *device) {
  return device->local_hardware_id().value();
}

#include "xla/service/custom_call_target_registry.h"
REACTANT_ABI void RegisterCustomCallTarget(const char *name, void *address,
                                           const char *platform) {
  CustomCallTargetRegistry::Global()->Register(std::string(name), address,
                                               std::string(platform));
}

#include "mlir/Target/LLVMIR/Import.h"
REACTANT_ABI MlirModule ConvertLLVMToMLIR(LLVMModuleRef lmod,
                                          MlirContext cctx) {
  auto llvmModule = std::unique_ptr<llvm::Module>(llvm::unwrap(lmod));
  mlir::MLIRContext &context = *unwrap(cctx);

  auto res = mlir::translateLLVMIRToModule(std::move(llvmModule), &context,
                                           /*emitExpensiveWarnings*/ false,
                                           /*dropDICompositeElements*/ false)
                 .release();
  return wrap(res);
}

#include "llvm/IRReader/IRReader.h"
REACTANT_ABI MlirModule ConvertLLVMStrToMLIR(const char *lmod,
                                             MlirContext cctx) {
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

typedef xla::Future<> FutureType;
REACTANT_ABI void FreeFuture(FutureType *Future) { delete Future; }

REACTANT_ABI uint8_t FutureIsReady(FutureType *Future) {
  return Future->IsReady();
}

REACTANT_ABI void FutureAwait(FutureType *Future) { Future->Await(); }

xla::CompileOptions
GenerateCompileOptions(int64_t device_id, const int64_t *mesh_ids,
                       int64_t num_mesh_ids, const char *xla_gpu_cuda_data_dir,
                       bool use_shardy_partitioner, int64_t num_replicas,
                       int64_t num_partitions, bool use_spmd_partitioning,
                       bool kernel_cache_enabled, const char *kernel_cache_path,
                       bool autotune_cache_enabled,
                       const char *autotune_cache_path, int process_id) {
  xla::CompileOptions options;
  auto debug_options = options.executable_build_options.mutable_debug_options();

  debug_options->set_xla_gpu_cuda_data_dir(xla_gpu_cuda_data_dir);
  debug_options->set_xla_enable_enzyme_comms_opt(true);
  debug_options->set_xla_gpu_experimental_use_raft_select_k(true);

  if (kernel_cache_enabled) {
    debug_options->set_xla_gpu_kernel_cache_file(kernel_cache_path);
    debug_options->set_xla_gpu_enable_llvm_module_compilation_parallelism(true);
  }

  if (autotune_cache_enabled) {
    debug_options->set_xla_gpu_per_fusion_autotune_cache_dir(
        autotune_cache_path);

    if (process_id <= 0) {
      debug_options->set_xla_gpu_experimental_autotune_cache_mode(
          xla::DebugOptions::AutotuneCacheMode::
              DebugOptions_AutotuneCacheMode_AUTOTUNE_CACHE_MODE_UPDATE);
    } else {
      debug_options->set_xla_gpu_experimental_autotune_cache_mode(
          xla::DebugOptions::AutotuneCacheMode::
              DebugOptions_AutotuneCacheMode_AUTOTUNE_CACHE_MODE_READ);
    }
  }

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

xla::CompileOptions GenerateCompileOptions(const char *compile_options_proto,
                                           size_t compile_options_proto_size) {
  if (compile_options_proto == nullptr || compile_options_proto_size == 0) {
    return xla::CompileOptions();
  }

  xla::CompileOptionsProto proto;
  if (!proto.ParseFromArray(compile_options_proto,
                            compile_options_proto_size)) {
    ReactantThrowError("Failed to parse CompileOptionsProto protobuf");
  }

  auto options_or = xla::CompileOptions::FromProto(proto);
  if (!options_or.ok()) {
    ReactantThrowError(options_or.status().ToString().c_str());
  }

  return std::move(options_or).value();
}

xla::PjRtLoadedExecutable *ClientCompileInternal(PjRtClient *client,
                                                 MlirModule cmod,
                                                 xla::CompileOptions options) {
  mlir::ModuleOp cmod_op = cast<ModuleOp>(*unwrap(cmod));

  if (options.executable_build_options.use_spmd_partitioning() &&
      options.executable_build_options.use_shardy_partitioner()) {
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

REACTANT_ABI xla::PjRtLoadedExecutable *
ClientCompile(PjRtClient *client, MlirModule cmod, int64_t device_id,
              const int64_t *mesh_ids, int64_t num_mesh_ids,
              const char *xla_gpu_cuda_data_dir, bool use_shardy_partitioner,
              int64_t num_replicas, int64_t num_partitions,
              bool use_spmd_partitioning, bool kernel_cache_enabled,
              const char *kernel_cache_path, bool autotune_cache_enabled,
              const char *autotune_cache_path, int process_id) {
  return ClientCompileInternal(
      client, cmod,
      GenerateCompileOptions(
          device_id, mesh_ids, num_mesh_ids, xla_gpu_cuda_data_dir,
          use_shardy_partitioner, num_replicas, num_partitions,
          use_spmd_partitioning, kernel_cache_enabled, kernel_cache_path,
          autotune_cache_enabled, autotune_cache_path, process_id));
}

REACTANT_ABI xla::PjRtLoadedExecutable *
ClientCompileWithProto(PjRtClient *client, MlirModule cmod,
                       const char *compile_options_proto,
                       size_t compile_options_proto_size) {
  return ClientCompileInternal(
      client, cmod,
      GenerateCompileOptions(compile_options_proto,
                             compile_options_proto_size));
}

REACTANT_ABI void
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

REACTANT_ABI void
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

REACTANT_ABI void XLAExecuteSharded(xla::PjRtLoadedExecutable *exec,
                                    int num_args, PjRtBuffer **op_args,
                                    PjRtDevice *device,
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

  // Optional future to hold asynchronous execution results.
  std::optional<xla::Future<>> returned_future;

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

REACTANT_ABI void XLAExecute(xla::PjRtLoadedExecutable *exec, int op_args_len,
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

REACTANT_ABI int PjRtLoadedExecutableNumReplicas(PjRtLoadedExecutable *exec) {
  return exec->num_replicas();
}

REACTANT_ABI int PjRtLoadedExecutableNumPartitions(PjRtLoadedExecutable *exec) {
  return exec->num_partitions();
}

REACTANT_ABI void RegisterDialects(MlirContext cctx) {
  mlir::MLIRContext &context = *unwrap(cctx);
  DialectRegistry registry;
  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);
  mlir::enzyme::registerInterfaces(registry);

  context.appendDialectRegistry(registry);
  mlir::enzyme::loadAllRegisteredDialects(context);
}

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"

REACTANT_ABI void InitializePasses(MlirDialectRegistry creg) {
  mlir::enzyme::initializePasses();
}

REACTANT_ABI void InitializeRegistry(MlirDialectRegistry creg) {
  mlir::DialectRegistry &registry = *unwrap(creg);
  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);
  mlir::enzyme::registerInterfaces(registry);

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

REACTANT_ABI MlirOperation LinkInModule(MlirModule prevModC, MlirModule newModC,
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

REACTANT_ABI void pjrt_client_dtor(HeldPjRtClient *client) { delete client; }

REACTANT_ABI int pjrt_client_num_devices(HeldPjRtClient *client) {
  return client->ptr()->device_count();
}

REACTANT_ABI int pjrt_client_num_addressable_devices(HeldPjRtClient *client) {
  return client->ptr()->addressable_device_count();
}

REACTANT_ABI int pjrt_client_pid(HeldPjRtClient *client) {
  return client->ptr()->process_index();
}

REACTANT_ABI PjRtDevice *pjrt_client_get_device(HeldPjRtClient *client,
                                                int device_id) {
  return ClientGetDevice(client->ptr(), device_id);
}

REACTANT_ABI PjRtDevice *
pjrt_client_get_addressable_device(HeldPjRtClient *client, int device_id) {
  return ClientGetAddressableDevice(client->ptr(), device_id);
}

REACTANT_ABI const char *pjrt_client_platform_name(HeldPjRtClient *client) {
  return ClientGetPlatformName(client->ptr());
}

// deprecated
// REACTANT_ABI HeldValue<std::shared_ptr<xla::PjRtBuffer>> *
// reactant_hold_pjrtbuffer(xla::PjRtBuffer *buffer) {
//   return reactant::capture(std::shared_ptr<xla::PjRtBuffer>(buffer));
// }

REACTANT_ABI HeldPjRtBuffer *pjrt_buffer_from_host(HeldPjRtClient *client,
                                                   void *data, uint64_t ptype,
                                                   size_t dim, int64_t *cshape,
                                                   PjRtDevice *device) {
  PjRtBuffer *buffer =
      ArrayFromHostBuffer(client->ptr(), data, ptype, dim, cshape, device);
  return reactant::capture(std::shared_ptr<PjRtBuffer>(buffer));
}

REACTANT_ABI void pjrt_buffer_dtor(HeldPjRtBuffer *buffer) { delete buffer; }

REACTANT_ABI void *pjrt_buffer_unsafe_buffer_pointer(HeldPjRtBuffer *buffer) {
  return UnsafeBufferPointer(buffer->ptr());
}

REACTANT_ABI bool pjrt_buffer_is_on_cpu(HeldPjRtBuffer *buffer) {
  return buffer->ptr()->IsOnCpu();
}

REACTANT_ABI HeldPjRtBuffer *
pjrt_buffer_copy_to_device(HeldPjRtBuffer *buffer, PjRtDevice *dst_device) {
  PjRtBuffer *ret = CopyBufferToDevice(buffer->ptr(), dst_device);
  return reactant::capture(std::shared_ptr<PjRtBuffer>(ret));
}

REACTANT_ABI void pjrt_buffer_to_host(HeldPjRtBuffer *buffer, void *data) {
  BufferToHost(buffer->ptr(), data);
}

REACTANT_ABI void pjrt_buffer_print(HeldPjRtBuffer *buffer) {
  PrintPjRtBuffer(buffer->ptr());
}

REACTANT_ABI PjRtDevice *pjrt_buffer_get_device(HeldPjRtBuffer *buffer) {
  return buffer->ptr()->device();
}

REACTANT_ABI HeldPjRtClient *pjrt_buffer_get_client(HeldPjRtBuffer *buffer) {
  return reactant::capture(
      std::shared_ptr<PjRtClient>(buffer->ptr()->client()));
}

REACTANT_ABI void ifrt_client_dtor(ifrt::Client *client) { delete client; }

// generic version, but IFRT-PjRt backend only supports SingleDeviceSharding
// and FullyReplicated. use `ifrt_pjrt_array_create` if using IFRT-PjRt.
REACTANT_ABI HeldIfrtArray *ifrt_client_make_array_from_host_buffer(
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
      static_cast<ifrt::Client::HostBufferSemantics>(c_semantics), [] {})));
}

REACTANT_ABI HeldIfrtArray *
ifrt_client_make_single_shard_array_from_host_buffer(
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
REACTANT_ABI HeldIfrtArray *ifrt_client_assemble_array_from_single_shards(
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
REACTANT_ABI HeldIfrtArray *
ifrt_pjrt_array_create(ifrt::PjRtClient *client,
                       HeldValue<std::shared_ptr<xla::PjRtBuffer>> *buffer) {
  return reactant::capture(
      tsl::RCReference<ifrt::Array>(MyValueOrThrow(xla::ifrt::PjRtArray::Create(
          client, buffer->obj(), /*has_custom_layout*/ false))));
}

HeldIfrtLoadedExecutable *
ifrt_compile_internal(ifrt::Client *client, MlirModule cmod,
                      xla::CompileOptions compile_options) {

  xla::ifrt::DeviceListRef devices = MyValueOrThrow(
      xla::ifrt::GetDeviceListFromXlaCompileOptions(client, compile_options));
  auto options = std::make_unique<xla::ifrt::XlaCompileOptions>(
      compile_options, std::move(devices));

  mlir::ModuleOp cmod_op = cast<ModuleOp>(*unwrap(cmod));
  if (compile_options.executable_build_options.use_spmd_partitioning() &&
      compile_options.executable_build_options.use_shardy_partitioner()) {
    // https://github.com/openxla/xla/blob/b3c641b05692f3712fb3c272e38665fdfa28bdf8/xla/python/py_client.cc#L460
    auto status = xla::ExportShardyForHloRoundTrip(cmod_op);
    if (!status.ok()) {
      ReactantThrowError(status.ToString().c_str());
    }
  }

  auto program =
      std::make_unique<xla::ifrt::HloProgram>(xla::ifrt::HloProgram(cmod_op));
  auto compiler = client->GetDefaultCompiler();

  return reactant::capture(MyValueOrThrow(
      compiler->CompileAndLoad(std::move(program), std::move(options))));
}

// we might me interested in the `Compiler::Compile` method variant that accepts
// `Topology`
REACTANT_ABI HeldIfrtLoadedExecutable *
ifrt_compile(ifrt::Client *client, MlirModule cmod, int64_t device_id,
             const int64_t *mesh_ids, int64_t num_mesh_ids,
             const char *xla_gpu_cuda_data_dir, bool use_shardy_partitioner,
             int64_t num_replicas, int64_t num_partitions,
             bool use_spmd_partitioning, bool kernel_cache_enabled,
             const char *kernel_cache_path, bool autotune_cache_enabled,
             const char *autotune_cache_path, int process_id) {
  return ifrt_compile_internal(
      client, cmod,
      GenerateCompileOptions(
          device_id, mesh_ids, num_mesh_ids, xla_gpu_cuda_data_dir,
          use_shardy_partitioner, num_replicas, num_partitions,
          use_spmd_partitioning, kernel_cache_enabled, kernel_cache_path,
          autotune_cache_enabled, autotune_cache_path, process_id));
}

REACTANT_ABI HeldIfrtLoadedExecutable *
ifrt_compile_with_proto(ifrt::Client *client, MlirModule cmod,
                        const char *compile_options_proto,
                        size_t compile_options_proto_size) {
  return ifrt_compile_internal(
      client, cmod,
      GenerateCompileOptions(compile_options_proto,
                             compile_options_proto_size));
}

REACTANT_ABI void
ifrt_pjrt_loaded_executable_dtor(xla::ifrt::PjRtLoadedExecutable *exec) {
  delete exec;
}

REACTANT_ABI void ifrt_array_dtor(HeldIfrtArray *array) { delete array; }

// in principle, use ArrayCopySemantics::kAlwaysCopy (=0)
REACTANT_ABI FutureType *
ifrt_CopyArrayToHostBuffer(HeldIfrtArray *array, void *data,
                           ifrt::ArrayCopySemantics semantics) {
  return new FutureType(
      (*array)->CopyToHostBuffer(data, std::nullopt, semantics));
}

REACTANT_ABI void
PjRtLoadedExecutableGetHloModules(xla::PjRtLoadedExecutable *exec,
                                  void **hlo_modules, int32_t *nmodules) {
  auto hlo_modules_vec = MyValueOrThrow(exec->GetHloModules());
  *nmodules = hlo_modules_vec.size();
  for (int i = 0; i < *nmodules; i++) {
    hlo_modules[i] = reactant::capture(hlo_modules_vec[i]);
  }
}

HloPrintOptions getHloPrintOptions(int32_t print_options) {
  switch (print_options) {
  case 0:
    return HloPrintOptions::Default();
  case 1:
    return HloPrintOptions::ShortParsable();
  case 2:
    return HloPrintOptions::Canonical();
  case 3:
    return HloPrintOptions::Fingerprint();
  case 4:
    return HloPrintOptions::ModuleFingerprint();
  default:
    ReactantThrowError("Invalid print_options");
  }
}

REACTANT_ABI const char *HloModuleToString(HeldHloModule *hlo_module,
                                           int32_t print_options) {
  return cstr_from_string(
      hlo_module->obj()->ToString(getHloPrintOptions(print_options)));
}

REACTANT_ABI void FreeHloModule(HeldHloModule *hlo_module) {
  delete hlo_module;
}

#pragma region IfRtClient

// XXX: Bring back with the correct API
// REACTANT_ABI ifrt::proxy::GrpcServer *
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

// REACTANT_ABI ifrt::proxy::GrpcServer *
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

// REACTANT_ABI ifrt::proxy::GrpcServer *
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

REACTANT_ABI void ifrt_proxy_grpc_server_dtor(ifrt::proxy::GrpcServer *server) {
  delete server;
}

REACTANT_ABI const char *
ifrt_proxy_grpc_server_address(ifrt::proxy::GrpcServer *server) {
  return cstr_from_string(server->address());
}

REACTANT_ABI void ifrt_proxy_grpc_server_wait(ifrt::proxy::GrpcServer *server) {
  server->Wait();
}

// `c_proxy_server_address` must be of the form
// `<backend-transport>:<backend-address>`; e.g. "grpc:localhost"
// NOTE not sure if we must pass the port, but probably yes
// by default, set `connection_timeout_in_minutes` to 2
REACTANT_ABI ifrt::Client *
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

REACTANT_ABI ifrt::Client *ifrt_pjrt_make_client(
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

REACTANT_ABI ifrt::Client *ifrt_pjrt_make_client_with_default_kv_store(
    PjRtClient *pjrt_client, int node_id, int num_nodes,
    void *distributed_runtime_client, const char **error,
    const char *key_prefix) {
  std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store;
  return ifrt_pjrt_make_client(pjrt_client, node_id, num_nodes,
                               distributed_runtime_client, error, key_prefix,
                               kv_store);
}

const char *const kMpiTrampolineLibEnv = "MPITRAMPOLINE_LIB";

REACTANT_ABI ifrt::Client *
ifrt_make_pjrt_cpu_client(uint8_t asynchronous, int node_id, int num_nodes,
                          void *distributed_runtime_client,
                          const char **error) {
  std::optional<std::shared_ptr<xla::cpu::CpuCollectives>> collectives;
  std::optional<std::shared_ptr<KeyValueStoreInterface>> kv_store;

  if (distributed_runtime_client != nullptr) {
    auto mpi_trampoline_path = llvm::sys::Process::GetEnv(kMpiTrampolineLibEnv);
    if (mpi_trampoline_path) {
#if defined(__linux__) || defined(__APPLE__)
      // Use MPI
      // TODO: How do we Finalize??
      auto mpi_collectives = std::make_shared<xla::cpu::MpiCollectives>();
      collectives = mpi_collectives;
      static_cast<xla::cpu::MpiCollectives *>(mpi_collectives.get())->Init();
#else
      ReactantThrowError(
          "MPI TCP Collectives only implemented for linux and macos");
#endif
    } else {
      // Use Gloo
      auto typed_distributed_runtime_client = static_cast<
          HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *>(
          distributed_runtime_client);
      kv_store =
          GetDistributedKeyValueStore(typed_distributed_runtime_client->obj(),
                                      /*key_prefix=*/"cpu:");
#if defined(__linux__)
      auto gloo_kv_store =
          std::make_unique<xla::cpu::GlooKeyValueStore>(kv_store.value());
      auto tcp_attrs = gloo::transport::tcp::attr();
      auto tcp_device = gloo::transport::tcp::CreateDevice(tcp_attrs);
      collectives = std::make_shared<xla::cpu::GlooCollectives>(
          std::move(gloo_kv_store), std::move(tcp_device));
#elif defined(__APPLE__)
      auto gloo_kv_store =
          std::make_unique<xla::cpu::GlooKeyValueStore>(kv_store.value());
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

REACTANT_ABI ifrt::Client *
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

REACTANT_ABI ifrt::Client *
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

REACTANT_ABI void ifrt_FreeClient(ifrt::Client *client) { delete client; }

REACTANT_ABI int ifrt_client_device_count(ifrt::Client *client) {
  return client->device_count();
}

REACTANT_ABI int ifrt_client_addressable_device_count(ifrt::Client *client) {
  return client->addressable_device_count();
}

REACTANT_ABI void ifrt_client_devices(ifrt::Client *client,
                                      ifrt::Device **out_devices) {
  auto span = client->devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

REACTANT_ABI void ifrt_client_addressable_devices(ifrt::Client *client,
                                                  ifrt::Device **out_devices) {
  auto span = client->addressable_devices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

REACTANT_ABI void ifrt_client_all_devices(ifrt::Client *client,
                                          ifrt::Device **out_devices) {
  auto span = client->GetAllDevices();
  for (int i = 0; i < span.size(); i++) {
    out_devices[i] = span[i];
  }
}

REACTANT_ABI ifrt::Device *ifrt_client_lookup_device(ifrt::Client *client,
                                                     int dev_id) {
  return MyValueOrThrow(
      client->LookupDevice(static_cast<ifrt::DeviceId>(dev_id)));
}

REACTANT_ABI ifrt::Device *
ifrt_client_lookup_addressable_device(ifrt::Client *client, int local_hw_id) {
  return MyValueOrThrow(client->LookupAddressableDevice(local_hw_id));
}

REACTANT_ABI int ifrt_ClientProcessIndex(ifrt::Client *client) {
  return client->process_index();
}

REACTANT_ABI const char *ifrt_ClientGetPlatformName(ifrt::Client *client) {
  return cstr_from_string(client->platform_name());
}

REACTANT_ABI ifrt::Device *ifrt_ClientGetDevice(ifrt::Client *client, int idx) {
  return MyValueOrThrow(client->LookupDevice(ifrt::DeviceId(idx)));
}

REACTANT_ABI ifrt::Device *ifrt_ClientGetAddressableDevice(ifrt::Client *client,
                                                           int idx) {
  return MyValueOrThrow(client->LookupAddressableDevice(idx));
}

#pragma endregion

#pragma region IfRtDevice

REACTANT_ABI int64_t ifrt_DeviceGetGlobalDeviceId(ifrt::Device *device) {
  return device->Id().value();
}

REACTANT_ABI const char *ifrt_DeviceGetKind(ifrt::Device *device) {
  return cstr_from_string(device->Kind());
}

REACTANT_ABI ifrt::Client *ifrt_DeviceToClient(ifrt::Device *device) {
  return device->client();
}

REACTANT_ABI bool ifrt_DeviceIsAddressable(ifrt::Device *device) {
  return device->IsAddressable();
}

REACTANT_ABI int64_t ifrt_DeviceGetLocalHardwareId(ifrt::Device *device) {
  if (!llvm::isa<ifrt::PjRtDevice>(device)) {
    ReactantThrowError(
        "ifrt_DeviceGetLocalHardwareId: only supported for ifrt-pjrt.");
  }
  auto ifrt_pjrt_device = llvm::dyn_cast<ifrt::PjRtDevice>(device);
  return ifrt_pjrt_device->pjrt_device()->local_hardware_id().value();
}

static xla::ifrt::RCReferenceWrapper<ifrt::DeviceList>
ifrt_CreateDeviceListFromDevices(ifrt::Client *client,
                                 ifrt::Device **device_list,
                                 int32_t num_devices) {
  absl::Span<ifrt::Device *const> devices(device_list, num_devices);
  return MyValueOrThrow(client->MakeDeviceList(devices));
}

REACTANT_ABI ifrt::Memory *ifrt_DeviceGetDefaultMemory(ifrt::Device *device) {
  return MyValueOrThrow(device->DefaultMemory());
}

REACTANT_ABI ifrt::Memory **ifrt_DeviceGetMemories(ifrt::Device *device,
                                                   int32_t *size) {
  auto memory_list = device->Memories();
  *size = memory_list.size();
  return const_cast<ifrt::Memory **>(memory_list.data());
}

REACTANT_ABI ifrt::MemoryKind *ifrt_MemoryGetMemoryKind(ifrt::Memory *memory) {
  ifrt::MemoryKind *memory_kind = new ifrt::MemoryKind(memory->Kind());
  return memory_kind;
}

REACTANT_ABI const char *ifrt_MemoryToString(ifrt::Memory *memory) {
  return cstr_from_string(memory->ToString());
}

REACTANT_ABI const char *
ifrt_MemoryKindToString(ifrt::MemoryKind *memory_kind) {
  auto memkind = memory_kind->memory_kind();
  if (!memkind.has_value())
    return "";
  return cstr_from_string(memkind.value());
}

REACTANT_ABI bool ifrt_MemoryKindsAreEqual(ifrt::MemoryKind *a,
                                           ifrt::MemoryKind *b) {
  return *a == *b;
}

#pragma endregion

#pragma region OpSharding

REACTANT_ABI void free_op_sharding(xla::OpSharding *op_sharding) {
  delete op_sharding;
}

REACTANT_ABI int32_t
op_sharding_to_op_sharding_type(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->type());
}

REACTANT_ABI int32_t
op_sharding_to_shard_group_type(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->shard_group_type());
}

REACTANT_ABI int32_t
op_sharding_to_shard_group_id(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->shard_group_id());
}

REACTANT_ABI bool op_sharding_is_shard_group(xla::OpSharding *op_sharding) {
  return op_sharding->is_shard_group();
}

REACTANT_ABI bool
op_sharding_replicate_on_last_tile_dim(xla::OpSharding *op_sharding) {
  return op_sharding->replicate_on_last_tile_dim();
}

REACTANT_ABI bool op_sharding_has_last_tile_dims(xla::OpSharding *op_sharding) {
  return op_sharding->last_tile_dims_size() > 0;
}

REACTANT_ABI int32_t
op_sharding_last_tile_dims_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->last_tile_dims_size());
}

REACTANT_ABI void op_sharding_last_tile_dims(xla::OpSharding *op_sharding,
                                             int32_t *last_tile_dims) {
  std::vector<int32_t> last_tile_dims_vec(op_sharding->last_tile_dims().begin(),
                                          op_sharding->last_tile_dims().end());
  std::copy(last_tile_dims_vec.begin(), last_tile_dims_vec.end(),
            last_tile_dims);
  return;
}

REACTANT_ABI bool
op_sharding_has_iota_reshape_dims(xla::OpSharding *op_sharding) {
  return op_sharding->iota_reshape_dims_size() > 0;
}

REACTANT_ABI int32_t
op_sharding_iota_reshape_dims_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->iota_reshape_dims_size());
}

REACTANT_ABI void op_sharding_iota_reshape_dims(xla::OpSharding *op_sharding,
                                                int32_t *iota_reshape_dims) {
  std::vector<int32_t> iota_reshape_dims_vec(
      op_sharding->iota_reshape_dims().begin(),
      op_sharding->iota_reshape_dims().end());
  std::copy(iota_reshape_dims_vec.begin(), iota_reshape_dims_vec.end(),
            iota_reshape_dims);
  return;
}

REACTANT_ABI bool
op_sharding_has_iota_transpose_perm(xla::OpSharding *op_sharding) {
  return op_sharding->iota_transpose_perm_size() > 0;
}

REACTANT_ABI int32_t
op_sharding_iota_transpose_perm_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->iota_transpose_perm_size());
}

REACTANT_ABI void
op_sharding_iota_transpose_perm(xla::OpSharding *op_sharding,
                                int32_t *iota_transpose_perm) {
  std::vector<int32_t> iota_transpose_perm_vec(
      op_sharding->iota_transpose_perm().begin(),
      op_sharding->iota_transpose_perm().end());
  std::copy(iota_transpose_perm_vec.begin(), iota_transpose_perm_vec.end(),
            iota_transpose_perm);
  return;
}

REACTANT_ABI bool
op_sharding_has_tile_assignment_dimensions(xla::OpSharding *op_sharding) {
  return op_sharding->tile_assignment_dimensions_size() > 0;
}

REACTANT_ABI int32_t
op_sharding_tile_assignment_dimensions_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->tile_assignment_dimensions_size());
}

REACTANT_ABI void
op_sharding_tile_assignment_dimensions(xla::OpSharding *op_sharding,
                                       int32_t *tile_assignment_dimensions) {
  std::vector<int32_t> tile_assignment_dimensions_vec(
      op_sharding->tile_assignment_dimensions().begin(),
      op_sharding->tile_assignment_dimensions().end());
  std::copy(tile_assignment_dimensions_vec.begin(),
            tile_assignment_dimensions_vec.end(), tile_assignment_dimensions);
  return;
}

REACTANT_ABI bool
op_sharding_has_tile_assignment_devices(xla::OpSharding *op_sharding) {
  return op_sharding->tile_assignment_devices_size() > 0;
}

REACTANT_ABI int32_t
op_sharding_tile_assignment_devices_size(xla::OpSharding *op_sharding) {
  return static_cast<int32_t>(op_sharding->tile_assignment_devices_size());
}

REACTANT_ABI void
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

REACTANT_ABI void free_hlo_sharding(xla::HloSharding *hlo_sharding) {
  delete hlo_sharding;
}

REACTANT_ABI xla::HloSharding *
hlo_sharding_from_op_sharding(xla::OpSharding *op_sharding) {
  xla::HloSharding *hlo_sharding = new xla::HloSharding(
      MyValueOrThrow(xla::HloSharding::FromProto(*op_sharding)));
  return hlo_sharding;
}

REACTANT_ABI xla::OpSharding *
hlo_sharding_to_op_sharding(xla::HloSharding *hlo_sharding) {
  xla::OpSharding *op_sharding = new xla::OpSharding(hlo_sharding->ToProto());
  return op_sharding;
}

REACTANT_ABI const char *
hlo_sharding_to_string(const xla::HloSharding *hlo_sharding) {
  return cstr_from_string(hlo_sharding->ToString(true));
}

REACTANT_ABI ifrt::MemoryKind *ifrt_memory_kind_from_string(const char *c_str) {
  return new ifrt::MemoryKind(std::string(c_str));
}

REACTANT_ABI ifrt::MemoryKind *ifrt_memory_kind_with_optional_memory_space() {
  return new ifrt::MemoryKind(std::nullopt);
}

REACTANT_ABI bool ifrt_memory_kind_has_value(ifrt::MemoryKind *memory_kind) {
  return *memory_kind != ifrt::MemoryKind(std::nullopt);
}

REACTANT_ABI void free_ifrt_sharding(HeldIfrtSharding *sharding) {
  delete sharding;
}

REACTANT_ABI HeldIfrtSharding *ifrt_sharding_from_xla_hlo_sharding(
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

REACTANT_ABI xla::HloSharding *
ifrt_sharding_to_xla_hlo_sharding(HeldIfrtSharding *sharding) {
  const ifrt::Sharding *val = sharding->obj().get();
  if (!llvm::isa<ifrt::HloSharding>(val))
    ReactantThrowError("Expected a HloSharding");
  auto ifrt_hlo_sharding = llvm::dyn_cast<const ifrt::HloSharding>(val);
  xla::HloSharding *xla_hlo_sharding =
      new xla::HloSharding(ifrt_hlo_sharding->xla_hlo_sharding());
  return xla_hlo_sharding;
}

REACTANT_ABI bool
ifrt_sharding_is_single_device_sharding(HeldIfrtSharding *sharding) {
  return llvm::isa<const ifrt::SingleDeviceSharding>(sharding->obj().get());
}

REACTANT_ABI bool
ifrt_sharding_is_fully_replicated(HeldIfrtSharding *sharding) {
  return sharding->obj()->IsFullyReplicated();
}

REACTANT_ABI const char *ifrt_sharding_to_string(HeldIfrtSharding *sharding) {
  return cstr_from_string(sharding->obj()->DebugString());
}

REACTANT_ABI int32_t ifrt_sharding_devices_size(HeldIfrtSharding *sharding) {
  return sharding->obj()->devices()->size();
}

REACTANT_ABI void ifrt_sharding_to_device_list(HeldIfrtSharding *sharding,
                                               ifrt::Device **devices) {
  auto device_list = sharding->obj()->devices()->devices();
  for (int i = 0; i < device_list.size(); i++) {
    devices[i] = device_list[i];
  }
}

REACTANT_ABI void ifrt_sharding_to_index_domains(HeldIfrtSharding *sharding,
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

REACTANT_ABI bool hlo_sharding_is_tuple(xla::HloSharding *hloSharding) {
  return hloSharding->IsTuple();
}

REACTANT_ABI bool hlo_sharding_is_replicated(xla::HloSharding *hloSharding) {
  return hloSharding->IsReplicated();
}

REACTANT_ABI bool hlo_sharding_is_manual(xla::HloSharding *hloSharding) {
  return hloSharding->IsManual();
}

REACTANT_ABI bool hlo_sharding_is_unknown(xla::HloSharding *hloSharding) {
  return hloSharding->IsUnknown();
}

REACTANT_ABI bool hlo_sharding_is_tiled(xla::HloSharding *hloSharding) {
  return hloSharding->IsTiled();
}

REACTANT_ABI bool hlo_sharding_is_maximal(xla::HloSharding *hloSharding) {
  return hloSharding->IsTileMaximal();
}

REACTANT_ABI bool
hlo_sharding_replicate_on_last_tile_dim(xla::HloSharding *hloSharding) {
  return hloSharding->ReplicateOnLastTileDim();
}

REACTANT_ABI int32_t
hlo_sharding_tile_assignment_dimensions_size(xla::HloSharding *hloSharding) {
  return static_cast<int32_t>(hloSharding->tile_assignment().num_dimensions());
}

REACTANT_ABI int32_t
hlo_sharding_tile_assignment_devices_size(xla::HloSharding *hloSharding) {
  return static_cast<int32_t>(hloSharding->tile_assignment().num_elements());
}

REACTANT_ABI void
hlo_sharding_tile_assignment_dimensions(xla::HloSharding *hloSharding,
                                        int64_t *dims, int32_t size) {
  auto tileAssignmentDims = hloSharding->tile_assignment().dimensions();
  for (int32_t i = 0; i < size; i++) {
    dims[i] = tileAssignmentDims[i];
  }
}

REACTANT_ABI void
hlo_sharding_tile_assignment_devices(xla::HloSharding *hloSharding,
                                     int64_t *devices, int32_t size) {
  auto tileAssignmentDevices = hloSharding->tile_assignment().array().data();
  for (int32_t i = 0; i < size; i++) {
    devices[i] = tileAssignmentDevices[i];
  }
}

REACTANT_ABI bool hlo_sharding_check_eq(xla::HloSharding *hloSharding,
                                        xla::HloSharding *other) {
  return *hloSharding == *other;
}

#pragma endregion

typedef tsl::Future<> IfRtFutureType;

REACTANT_ABI void ifrt_free_future(IfRtFutureType *Future) { delete Future; }

REACTANT_ABI uint8_t ifrt_future_is_ready(IfRtFutureType *Future) {
  return Future->IsReady();
}

REACTANT_ABI void ifrt_future_await(IfRtFutureType *Future) { Future->Await(); }

#pragma region IfRtArray

REACTANT_ABI void ifrt_free_array(HeldIfrtArray *array) { delete array; }

REACTANT_ABI int64_t *ifrt_array_shape(HeldIfrtArray *array) {
  auto dims =
      static_cast<absl::Span<const int64_t>>(array->obj()->shape().dims());
  int64_t *dims_ptr = new int64_t[dims.size()];
  std::copy(dims.begin(), dims.end(), dims_ptr);
  return dims_ptr;
}

REACTANT_ABI int64_t ifrt_array_ndims(HeldIfrtArray *array) {
  return array->obj()->shape().dims().size();
}

REACTANT_ABI ifrt::DType ifrt_array_eltype(HeldIfrtArray *array) {
  return array->obj()->dtype();
}

REACTANT_ABI ifrt::Client *ifrt_array_to_client(HeldIfrtArray *array) {
  return array->obj()->client();
}

REACTANT_ABI HeldValue<std::shared_ptr<const ifrt::Sharding>> *
ifrt_array_to_sharding(HeldIfrtArray *array) {
  return reactant::capture(array->obj()->shared_ptr_sharding());
}

REACTANT_ABI void ifrt_array_copy_to_host_buffer(HeldIfrtArray *array,
                                                 void *data) {
  std::optional<absl::Span<const int64_t>> byte_strides;
  auto future = array->obj()->CopyToHostBuffer(
      data, byte_strides, static_cast<ifrt::ArrayCopySemantics>(0));
  future.Await();
  return;
}

REACTANT_ABI HeldIfrtArray **ifrt_array_disassemble_into_single_device_arrays(
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

REACTANT_ABI HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *
GetDistributedRuntimeClient(char *c_address, int32_t node_id,
                            int32_t rpc_timeout_in_seconds,
                            int32_t init_timeout,
                            int32_t shutdown_timeout_in_minutes,
                            int32_t heartbeat_timeout_in_seconds,
                            bool use_compression) {
  xla::DistributedRuntimeClient::Options options;
  options.node_id = node_id;
  options.rpc_timeout = absl::Seconds(rpc_timeout_in_seconds);
  options.init_timeout = absl::Seconds(init_timeout);
  options.shutdown_timeout = absl::Minutes(shutdown_timeout_in_minutes);
  options.heartbeat_timeout = absl::Seconds(heartbeat_timeout_in_seconds);

  std::string address = c_address;

  return reactant::capture(
      xla::GetDistributedRuntimeClient(address, options, use_compression));
}

REACTANT_ABI void free_distributed_runtime_client(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *client) {
  delete client;
}

REACTANT_ABI void distributed_runtime_client_connect(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *client) {
  auto status = client->obj()->Connect();
  if (!status.ok())
    ReactantThrowError(status.ToString().c_str());
}

REACTANT_ABI void distributed_runtime_client_shutdown(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeClient>> *client) {
  auto status = client->obj()->Shutdown();
  if (!status.ok())
    ReactantThrowError(status.ToString().c_str());
}

REACTANT_ABI xla::DistributedRuntimeService *
GetDistributedRuntimeService(char *c_address, int num_nodes,
                             int32_t heartbeat_timeout_in_seconds,
                             int32_t cluster_register_timeout_in_minutes,
                             int32_t shutdown_timeout_in_minutes) {
  xla::CoordinationServiceImpl::Options options;
  options.num_nodes = num_nodes;
  options.heartbeat_timeout = absl::Seconds(heartbeat_timeout_in_seconds);
  options.cluster_register_timeout =
      absl::Minutes(cluster_register_timeout_in_minutes);
  options.shutdown_timeout = absl::Minutes(shutdown_timeout_in_minutes);

  std::string address = c_address;

  return MyValueOrThrow(xla::GetDistributedRuntimeService(address, options))
      .release();
}

REACTANT_ABI void free_distributed_runtime_service(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeService>> *service) {
  delete service;
}

REACTANT_ABI void distributed_runtime_service_shutdown(
    HeldValue<std::shared_ptr<xla::DistributedRuntimeService>> *service) {
  service->obj()->Shutdown();
}

#pragma endregion

#pragma region Shardy

REACTANT_ABI xla::HloSharding *
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
REACTANT_ABI mlir::sdy::TensorShardingAttr hloShardingToTensorShardingAttr(
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

  return mlir::sdy::TensorShardingAttr::get(
      context, meshName, tensorShardingAttr.getDimShardings(),
      tensorShardingAttr.getReplicatedAxes(),
      tensorShardingAttr.getUnreducedAxes());
}

#pragma endregion

#pragma region ifrt::LoadedExecutable

REACTANT_ABI void ifrt_loaded_executable_dtor(HeldIfrtLoadedExecutable *exec) {
  delete exec;
}

REACTANT_ABI void ifrt_loaded_executable_execute(
    HeldIfrtLoadedExecutable *exec, int num_args,
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

  auto result = MyValueOrThrow(exec->obj()->Execute(
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
  *status = new IfRtFutureType(result.status);

  for (int i = 0; i < num_results; i++) {
    op_results[i] = reactant::capture(result.outputs[i]);
  }
}

REACTANT_ABI ifrt::Client *
ifrt_loaded_executable_client(HeldIfrtLoadedExecutable *exec) {
  return exec->obj()->client();
}

REACTANT_ABI void
ifrt_loaded_executable_get_parameter_shardings(HeldIfrtLoadedExecutable *exec,
                                               xla::OpSharding **op_shardings,
                                               int32_t num_op_shardings) {
  std::optional<std::vector<xla::OpSharding>> shardings =
      exec->obj()->GetParameterShardings();
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

REACTANT_ABI void
ifrt_loaded_executable_get_output_shardings(HeldIfrtLoadedExecutable *exec,
                                            xla::OpSharding **op_shardings,
                                            int32_t num_op_shardings) {
  std::optional<std::vector<xla::OpSharding>> shardings =
      exec->obj()->GetOutputShardings();
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

REACTANT_ABI void
ifrt_loaded_executable_get_hlo_modules(HeldIfrtLoadedExecutable *exec,
                                       void **hlo_modules, int32_t *nmodules) {
  auto hlo_modules_vec = MyValueOrThrow(exec->obj()->GetHloModules());
  *nmodules = hlo_modules_vec.size();
  for (int32_t i = 0; i < *nmodules; i++) {
    hlo_modules[i] = reactant::capture(hlo_modules_vec[i]);
  }
}

REACTANT_ABI int32_t
ifrt_loaded_executable_num_devices(HeldIfrtLoadedExecutable *exec) {
  return static_cast<int32_t>(exec->obj()->num_devices());
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

REACTANT_ABI void pjrt_hlo_module_cost_analysis_properties(
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

REACTANT_ABI void ifrt_hlo_module_cost_analysis_properties(
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

REACTANT_ABI void dump_op(Operation *op) { llvm::errs() << *op << "\n"; }
REACTANT_ABI void dump_mval(mlir::Value v) { llvm::errs() << v << "\n"; }
REACTANT_ABI void dump_operation(Operation *op, const char *filename) {
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC, llvm::sys::fs::OF_Text);

  if (EC) {
    std::cerr << "Error opening file: " << EC.message() << std::endl;
    return;
  }

  op->print(file, mlir::OpPrintingFlags().enableDebugInfo(true, false));
}

REACTANT_ABI bool pjrt_device_is_addressable(PjRtDevice *device) {
  return device->IsAddressable();
}

REACTANT_ABI mlir::Operation *
mlirGetParentOfTypeFunctionOp(mlir::Operation *op) {
  return op->getParentOfType<mlir::FunctionOpInterface>();
}

// batched copy
// https://github.com/jax-ml/jax/blob/2b86f38585a517ce50e8ddf964a4709040a1bd53/jaxlib/xla/py_array.cc#L1112

// xla::ifrt::CopyArrays
REACTANT_ABI HeldIfrtArray **ifrt_copy_arrays_to_device_with_sharding(
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
REACTANT_ABI HeldIfrtArray *ifrt_make_array_from_host_buffer_shards(
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
      static_cast<ifrt::Client::HostBufferSemantics>(c_host_buffer_semantics)));
  return reactant::capture(arrays[0]);
}

REACTANT_ABI void addSdyPropagationPipeline(
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

REACTANT_ABI HeldIfrtArray *ifrt_copy_array(HeldIfrtArray *array) {
  auto pjrtArray = dyn_cast<ifrt::PjRtArray>(array->obj().get());
  if (pjrtArray) {
    std::optional<ifrt::DeviceListRef> devices;
    std::optional<ifrt::MemoryKind> memory_kind;
    auto res = MyValueOrThrow(pjrtArray->Copy(
        devices, memory_kind, static_cast<ifrt::ArrayCopySemantics>(0)));
    return reactant::capture(res);
  }
  ReactantThrowError("Only ifrt-pjrt arrays are supported for now");
}

struct LinkableRuntime {
  mlir::DialectRegistry registry;
  xla::PjRtClient *client;
  int device;
  bool shouldFreeClient;
  DenseMap<const char *, std::map<std::vector<std::vector<int64_t>>,
                                  xla::PjRtLoadedExecutable *>>
      executables;

  // Set of allocated pointers to size
  std::set<void *, std::greater<void *>> allocations;

  LinkableRuntime(const std::string &backend) : registry() {
    InitializeRegistry(wrap(&registry));
    InitializePasses(wrap(&registry));
    InitializeLogs();
    shouldFreeClient = true;
    const char *error = NULL;
    auto mpi = getenv("OMPI_COMM_WORLD_RANK");
    device = 0;
    if (mpi) {
      device = atoi(mpi);
    }

    client = nullptr;

    if (backend == "xla-tpu") {
      if (device == 0) {
        client = MakeTPUClient(nullptr, &error);
        if (error) {
          llvm::errs() << " error: " << error << "\n";
          exit(1);
        }
      }
    } else if (backend == "xla-gpu") {
      int node_id = 0;
      int num_nodes = 1;
      int64_t *allowed_devices = NULL;
      int num_allowed_devices = 0;
      double mem_fraction = 0.75;
      bool gpu_preallocate = true;
      const char *refstr;
      const char *platform = "gpu";
      void *distributed_runtime_client = NULL;
      client = MakeGPUClient(node_id, num_nodes, allowed_devices,
                             num_allowed_devices, mem_fraction, gpu_preallocate,
                             platform, &refstr, distributed_runtime_client);
      if (!client) {
        llvm::errs() << " error: " << refstr << "\n";
        exit(1);
      }
      assert(client);
      // Weird stream issue in freeing cuda client.
      shouldFreeClient = false;
    } else {
      client = MakeCPUClient(1, 0);
      assert(client);
    }

    if (client) {
      device = min(device, client->device_count() - 1);
    }
  }

  ~LinkableRuntime() {
    if (client && shouldFreeClient) {
      delete client;
    }
  }
};

static std::tuple<PjRtBuffer *, /*offset*/ size_t, PjRtBuffer **>
bufferAndOffset(LinkableRuntime *__restrict__ lrt, void *ptr) {
  auto found = lrt->allocations.lower_bound(ptr);
  assert(found != lrt->allocations.end());
  auto start = (PjRtBuffer **)(*found);
  return std::tuple<PjRtBuffer *, /*offset*/ size_t, PjRtBuffer **>(
      *start, (size_t)ptr - (size_t)start, start);
}

REACTANT_ABI void reactantXLAThrow(const char *str) {
  printf("Error: %s\n", str);
  exit(1);
}

REACTANT_ABI void reactantXLAInit(LinkableRuntime **__restrict__ lrtP,
                                  const char *__restrict__ backend) {
  *lrtP = new LinkableRuntime(backend);
  ReactantThrowError = reactantXLAThrow;
}

REACTANT_ABI void reactantXLADeInit(LinkableRuntime **__restrict__ lrt) {
  delete *lrt;
}

REACTANT_ABI void reactantXLAMemcpy(LinkableRuntime **__restrict__ lrtP,
                                    void *__restrict__ dst,
                                    void *__restrict__ src, size_t size,
                                    int32_t direction) {
  auto lrt = *lrtP;
  switch (direction) {
  case 0: // cudaMemcpyHostToHost = 0
    llvm_unreachable("host to host copy unsupported");
    break;
  case 1: // cudaMemcpyHostToDevice
  {
    auto &&[dstB, dstO, start] = bufferAndOffset(lrt, dst);
    CopyToBuffer(lrt->client, dstB, src, dstO, size, start);
    break;
  }
  case 2: // cudaMemcpyDeviceToHost
  {
    auto &&[srcB, srcO, start] = bufferAndOffset(lrt, src);
    CopyFromBuffer(lrt->client, srcB, dst, srcO, size, start);
    break;
  }
  case 3: // cudaMemcpyDeviceToDevice
    llvm_unreachable("device to device copy unsupported");
    break;
  default: // cudaMemcpyDeviceToDevice
    llvm_unreachable("unknown copy unsupported");
    break;
  }
}

REACTANT_ABI void *reactantXLAMalloc(LinkableRuntime **__restrict__ lrtP,
                                     uint64_t ptype, uint64_t shapeLen,
                                     uint64_t *__restrict__ shape) {
  auto lrt = *lrtP;
  PjRtDevice *device = ClientGetDevice(lrt->client, lrt->device);

  auto xbuffer0 = UninitPJRTBuffer(lrt->client, device, ptype, shapeLen, shape);
  void **xbuffer = (void **)malloc(sizeof(void *));
  xbuffer[0] = xbuffer0;
  lrt->allocations.insert((void *)xbuffer);
  return xbuffer;
}

REACTANT_ABI void reactantXLAFree(LinkableRuntime **__restrict__ lrtP,
                                  void *__restrict__ buffer0) {
  void *buffer = *(void **)buffer0;
  free(buffer0);
  PjRtBufferFree((PjRtBuffer *)buffer);
}

REACTANT_ABI void reactantXLAExec(LinkableRuntime **__restrict__ lrtP,
                                  const char *modstr, int64_t argcnt,
                                  void **args) {
  auto lrt = *lrtP;
  auto &cache = lrt->executables[modstr];
  std::vector<PjRtBuffer *> baseArrays(argcnt);
  std::vector<PjRtBuffer **> basePtrs(argcnt);

  std::vector<std::vector<int64_t>> sizeKey;
  sizeKey.reserve(argcnt);
  for (int64_t i = 0; i < argcnt; i++) {
    auto &&[argB, argO, argP] = bufferAndOffset(lrt, args[i]);
    if (argO != 0) {
      llvm::errs() << "only zero-offset execution supported, argument " << i
                   << " had byte offset of " << argO << "\n";
      exit(1);
    }
    baseArrays[i] = argB;
    basePtrs[i] = argP;
    auto dims = argB->on_device_shape().dimensions();
    sizeKey.emplace_back(dims.begin(), dims.end());
  }

  auto iter = cache.find(sizeKey);

  if (iter == cache.end()) {
    MLIRContext context(lrt->registry);
    RegisterDialects(wrap(&context));

    mlir::OwningOpRef<mlir::ModuleOp> module(
        mlir::ModuleOp::create(mlir::OpBuilder(&context).getUnknownLoc()));

    ParserConfig config(&context, /*verify_after_parse*/ true);
    if (failed(parseSourceString(modstr, module->getBody(), config))) {
      llvm::errs() << " failed to parse module:\n";
      exit(1);
    }

    auto funcOp = cast<func::FuncOp>(&module->getBody()->back());

    mlir::OpBuilder builder(module->getContext());
    funcOp.setSymName(builder.getStringAttr("main"));
    funcOp.setVisibility(SymbolTable::Visibility::Public);

    for (int64_t i = 0; i < argcnt; i++) {
      funcOp.setArgAttr(i, "tf.aliasing_output", builder.getI64IntegerAttr(i));
    }

    PassManager pm(module->getContext());

    SmallVector<mlir::Type> types;
    for (int64_t i = 0; i < argcnt; i++) {
      auto RTT = MyValueOrThrow(xla::ConvertShapeToType<mlir::RankedTensorType>(
          baseArrays[i]->on_device_shape(), builder));
      types.push_back(RTT);
    }
    pm.addPass(mlir::stablehlo::createStablehloRefineArgumentsPass(types));
    pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        stablehlo::createStablehloCanonicalizeDynamismPass());
    pm.addPass(mlir::enzyme::createEnzymeHLOOptPass());

    if (!mlir::succeeded(pm.run(*module))) {
      llvm::errs() << " failed to run passes\n";
      exit(1);
    }

    auto exec =
        ClientCompileWithProto(lrt->client, wrap(module.get()), nullptr, 0);

    iter = cache.try_emplace(sizeKey, exec).first;
  }

  auto exec = iter->second;

  uint8_t *is_arg_donatable = (uint8_t *)malloc(argcnt);
  for (int i = 0; i < argcnt; i++)
    is_arg_donatable[i] = 1;
  int num_results = argcnt;
  std::vector<PjRtBuffer *> results(argcnt);
  std::vector<uint8_t> futures(argcnt, 0);
  std::vector<FutureType *> future_results(argcnt, nullptr);
  PjRtDevice *device = ClientGetDevice(lrt->client, lrt->device);
  XLAExecuteSharded(exec, argcnt, baseArrays.data(), device, is_arg_donatable,
                    num_results, results.data(), futures.data(),
                    future_results.data());
  free(is_arg_donatable);
  for (int64_t i = 0; i < argcnt; i++) {
    *basePtrs[i] = results[i];
    if (futures[i]) {
      FutureAwait(future_results[i]);
      FreeFuture(future_results[i]);
    }
  }
}

REACTANT_ABI HeldHloModule *convertMlirModuleToHloModule(MlirModule mod) {
  mlir::ModuleOp cmod_op = cast<ModuleOp>(*unwrap(mod));
  std::shared_ptr<xla::HloModule> hlo_module =
      std::move(MyValueOrThrow(xla::ConvertStablehloToHlo(cmod_op)));
  return reactant::capture(hlo_module);
}

REACTANT_ABI HeldHloModule *
parseAndReturnUnverifiedHloModule(const char *cstr) {
  absl::string_view str(cstr);
  auto hlo_module_status = xla::ParseAndReturnUnverifiedModule(str);
  if (!hlo_module_status.ok()) {
    ReactantThrowError(hlo_module_status.status().ToString().c_str());
  }
  std::shared_ptr<xla::HloModule> hlo_module =
      std::move(hlo_module_status.value());
  return reactant::capture(hlo_module);
}

REACTANT_ABI xla::HloComputation *
hloModuleGetEntryComputation(HeldHloModule *hlo_module) {
  return hlo_module->obj()->entry_computation();
}

REACTANT_ABI void freeHloComputation(HloComputation *hlo_computation) {
  delete hlo_computation;
}

REACTANT_ABI const char *hloComputationToString(HloComputation *hlo_computation,
                                                int32_t print_options) {
  return cstr_from_string(
      hlo_computation->ToString(getHloPrintOptions(print_options)));
}

REACTANT_ABI int64_t
hloComputationInstructionCount(HloComputation *hlo_computation) {
  return hlo_computation->instruction_count();
}

REACTANT_ABI void
hloComputationGetInstructionsPostOrder(HloComputation *hlo_computation,
                                       int64_t num_instructions,
                                       HloInstruction **hlo_instructions) {
  std::vector<HloInstruction *> instructions =
      hlo_computation->MakeInstructionPostOrder();
  assert(instructions.size() == num_instructions);
  for (int i = 0; i < num_instructions; i++) {
    hlo_instructions[i] = instructions[i];
  }
}

REACTANT_ABI void freeHloInstruction(HloInstruction *hlo_instruction) {
  delete hlo_instruction;
}

REACTANT_ABI const char *hloInstructionToString(HloInstruction *hlo_instruction,
                                                int32_t print_options) {
  return cstr_from_string(
      hlo_instruction->ToString(getHloPrintOptions(print_options)));
}

REACTANT_ABI uint8_t hloInstructionHasToApply(HloInstruction *hlo_instruction) {
  return hlo_instruction->has_to_apply();
}

REACTANT_ABI HloComputation *
hloInstructionGetToApply(HloInstruction *hlo_instruction) {
  return hlo_instruction->to_apply();
}

REACTANT_ABI uint8_t hloInstructionGetOpcode(HloInstruction *hlo_instruction) {
  return static_cast<uint8_t>(hlo_instruction->opcode());
}

REACTANT_ABI const char *hloOpcodeToString(uint8_t opcode) {
  return cstr_from_string(xla::HloOpcodeString(static_cast<HloOpcode>(opcode)));
}

REACTANT_ABI uint8_t hloInstructionIsFusion(HloInstruction *hlo_instruction) {
  if (dynamic_cast<HloFusionInstruction *>(hlo_instruction)) {
    return 1;
  }
  return 0;
}

REACTANT_ABI uint8_t
hloInstructionGetFusionKind(HloInstruction *hlo_instruction) {
  if (auto hlo_instruction_fusion =
          dynamic_cast<HloFusionInstruction *>(hlo_instruction)) {
    return static_cast<uint8_t>(hlo_instruction_fusion->fusion_kind());
  }
  ReactantThrowError("hloInstructionGetFusionKind: not a fusion instruction");
}

REACTANT_ABI const char *hloFusionKindToString(uint8_t kind) {
  return cstr_from_string(
      xla::ToString(static_cast<HloInstruction::FusionKind>(kind)));
}

REACTANT_ABI HloComputation *
hloInstructionFusedInstructionsComputation(HloInstruction *hlo_instruction) {
  if (auto hlo_instruction_fusion =
          dynamic_cast<HloFusionInstruction *>(hlo_instruction)) {
    return hlo_instruction_fusion->fused_instructions_computation();
  }
  ReactantThrowError("hloInstructionFusedInstructionsComputation: not a fusion "
                     "instruction");
}

struct JLEstimateRunTimeData {
  int64_t flops;
  int64_t bytes_read;
  int64_t bytes_written;
  int64_t read_time_ns;
  int64_t write_time_ns;
  int64_t compute_time_ns;
  int64_t execution_time_ns;
};

#if defined(REACTANT_CUDA) || defined(REACTANT_ROCM)
namespace details {

// Cost analysis for individual instructions.
class GPUPerformanceModel {
public:
  GPUPerformanceModel(mlir::MLIRContext *mlir_context,
                      stream_executor::DeviceDescription *device_description)
      : mlir_context_(std::move(mlir_context)),
        device_description_(*device_description),
        hlo_cost_analysis_options_{.count_multiple_input_accesses = true},
        fusion_analysis_cache_(device_description_),
        gpu_hlo_cost_analysis_(hlo_cost_analysis_options_, device_description_),
        gpu_performance_model_(device_description_, fusion_analysis_cache_,
                               gpu_performance_model_cache_, mlir_context_) {}

  void RunAnalysisOnHloModule(std::shared_ptr<xla::HloModule> hlo_module) {
    hlo_module->entry_computation()->Accept(&gpu_hlo_cost_analysis_);
    ran_analysis_ = true;
  }

  xla::gpu::EstimateRunTimeData
  EstimateRunTimeForInstruction(HloInstruction *hlo_instruction) {
    if (!ran_analysis_) {
      ReactantThrowError("Must call RunAnalysisOnHloModule before calling "
                         "EstimateRunTimeForInstruction");
    }
    return gpu_performance_model_.EstimateRunTimeForInstruction(
        hlo_instruction, &gpu_hlo_cost_analysis_);
  }

private:
  mlir::MLIRContext *mlir_context_;
  xla::gpu::GpuHloCostAnalysis::Options hlo_cost_analysis_options_;
  stream_executor::DeviceDescription device_description_;
  xla::gpu::HloFusionAnalysisCache fusion_analysis_cache_;
  xla::gpu::GpuHloCostAnalysis gpu_hlo_cost_analysis_;
  xla::gpu::GpuPerformanceModelCache gpu_performance_model_cache_;
  xla::gpu::GpuPerformanceModel gpu_performance_model_;
  bool ran_analysis_ = false;
};

} // namespace details

REACTANT_ABI details::GPUPerformanceModel *CreateGPUPerformanceModel(
    MlirContext ctx, stream_executor::DeviceDescription *device_description) {
  return new details::GPUPerformanceModel(unwrap(ctx), device_description);
}

REACTANT_ABI void
RunAnalysisOnHloModule(details::GPUPerformanceModel *gpu_performance_model,
                       HeldHloModule *hlo_module) {
  gpu_performance_model->RunAnalysisOnHloModule(hlo_module->obj());
}

REACTANT_ABI void EstimateRunTimeForInstruction(
    details::GPUPerformanceModel *gpu_performance_model,
    HloInstruction *hlo_instruction, JLEstimateRunTimeData *jldata) {
  auto data =
      gpu_performance_model->EstimateRunTimeForInstruction(hlo_instruction);
  jldata->flops = data.flops;
  jldata->bytes_read = data.bytes_read;
  jldata->bytes_written = data.bytes_written;
  jldata->read_time_ns = absl::ToInt64Nanoseconds(data.read_time);
  jldata->write_time_ns = absl::ToInt64Nanoseconds(data.write_time);
  jldata->compute_time_ns = absl::ToInt64Nanoseconds(data.compute_time);
  jldata->execution_time_ns = absl::ToInt64Nanoseconds(data.exec_time);
}

#else

REACTANT_ABI void *CreateGPUPerformanceModel(
    MlirContext ctx, stream_executor::DeviceDescription *device_description) {
  return nullptr;
}

REACTANT_ABI void RunAnalysisOnHloModule(void *gpu_performance_model,
                                         HloModule *hlo_module) {
  ReactantThrowError("RunAnalysisOnHloModule is only supported if Reactant "
                     "was compiled with CUDA or ROCM support.");
}

REACTANT_ABI void EstimateRunTimeForInstruction(void *gpu_performance_model,
                                                HloInstruction *hlo_instruction,
                                                JLEstimateRunTimeData *jldata) {
  ReactantThrowError(
      "EstimateRunTimeForInstruction is only supported if Reactant "
      "was compiled with CUDA or ROCM support.");
}

#endif

REACTANT_ABI void
InitializeXProfStubs(const char *cstr_worker_service_address) {
  std::string worker_service_address = std::string(cstr_worker_service_address);
  xprof::profiler::InitializeStubs(worker_service_address);
}

REACTANT_ABI void StartGrpcServer(int port) {
  xprof::pywrap::StartGrpcServer(port);
}

// Creates a ToolOptions map from Julia arrays.
// Takes 6 arrays: 3 pairs of (keys, values) for bool, int, and char* options.
// Each array pair has a corresponding count parameter.
ToolOptions ToolOptionsFromJuliaArrays(
    const char **bool_keys, const bool *bool_values, int64_t bool_count,
    const char **int_keys, const int *int_values, int64_t int_count,
    const char **str_keys, const char **str_values, int64_t str_count) {
  ToolOptions map;

  // Add bool options
  for (int64_t i = 0; i < bool_count; ++i) {
    if (bool_keys[i] != nullptr) {
      map.emplace(std::string(bool_keys[i]),
                  std::variant<bool, int, std::string>(bool_values[i]));
    }
  }

  // Add int options
  for (int64_t i = 0; i < int_count; ++i) {
    if (int_keys[i] != nullptr) {
      map.emplace(std::string(int_keys[i]),
                  std::variant<bool, int, std::string>(int_values[i]));
    }
  }

  // Add string options
  for (int64_t i = 0; i < str_count; ++i) {
    if (str_keys[i] != nullptr && str_values[i] != nullptr) {
      map.emplace(
          std::string(str_keys[i]),
          std::variant<bool, int, std::string>(std::string(str_values[i])));
    }
  }

  return map;
}

// C API wrapper for xprof::pywrap::XSpaceToToolsData
// Returns:
//   - result_data: pointer to the result data (caller must free with free())
//   - result_size: size of the result data
//   - is_binary: whether the result is binary data
//   - error: error message if failed (caller must free with free())
// Returns 0 on success, non-zero on failure
REACTANT_ABI int XSpaceToToolsData(
    const char **xspace_paths, int64_t num_paths, const char *tool_name,
    const char **bool_keys, const bool *bool_values, int64_t bool_count,
    const char **int_keys, const int *int_values, int64_t int_count,
    const char **str_keys, const char **str_values, int64_t str_count,
    char **result_data, int64_t *result_size, bool *is_binary, char **error) {
  *error = nullptr;
  *result_data = nullptr;
  *result_size = 0;
  *is_binary = false;

  // Convert xspace paths to vector
  std::vector<std::string> xspace_paths_vec;
  xspace_paths_vec.reserve(num_paths);
  for (int64_t i = 0; i < num_paths; ++i) {
    if (xspace_paths[i] != nullptr) {
      xspace_paths_vec.push_back(std::string(xspace_paths[i]));
    }
  }

  // Build tool options
  ToolOptions tool_options = ToolOptionsFromJuliaArrays(
      bool_keys, bool_values, bool_count, int_keys, int_values, int_count,
      str_keys, str_values, str_count);

  // Call XSpaceToToolsData
  absl::StatusOr<std::pair<std::string, bool>> result =
      xprof::pywrap::XSpaceToToolsData(xspace_paths_vec, std::string(tool_name),
                                       tool_options);

  if (!result.ok()) {
    auto str = result.status().message();
    char *err = (char *)malloc(str.size() + 1);
    memcpy(err, str.data(), str.size() + 1);
    *error = err;
    return 1;
  }

  // Copy result data
  const std::string &data = result->first;
  *result_size = static_cast<int64_t>(data.size());
  *result_data = (char *)malloc(data.size());
  memcpy(*result_data, data.data(), data.size());
  *is_binary = result->second;

  return 0;
}
