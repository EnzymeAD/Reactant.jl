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
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"

#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "xla/python/profiler_utils.h"
#include "tsl/platform/init_main.h"

#include "xla/python/ifrt/hlo/hlo_program.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

#include "llvm-c/TargetMachine.h"

// shardy
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/integrations/c/attributes.h"
#include "xla/pjrt/mlir_to_hlo.h"

// IFRT
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
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

#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/ExtensibleRTTI.h"

using namespace mlir;
using namespace llvm;
using namespace xla;

namespace mlir {
namespace enzyme {
void registerRemoveTransformPass();
void registerGenerateApplyPatternsPass();
} // namespace enzyme
} // namespace mlir

extern "C" void (*ReactantThrowError)(const char *) = nullptr;

// Utilities for `StatusOr`.
template <typename T> T MyValueOrThrow(absl::StatusOr<T> v) {
  if (ReactantThrowError) {
    if (!v.ok()) {
      ReactantThrowError(v.status().ToString().c_str());
      throw xla::XlaRuntimeError(v.status().ToString().c_str());
    }
    return std::move(v).value();
  } else {
    return xla::ValueOrThrow(std::move(v));
  }
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

// TODO mlirComplexAttrGetnValue
// TODO extern "C" MlirTypeID mlirComplexAttrGetTypeID(void) { return
// wrap(complex::NumberAttr::getTypeID()); }

extern "C" void ReactantFuncSetResultAttr(MlirOperation op, intptr_t pos,
                                          MlirStringRef name, MlirAttribute attr) {
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
  const char* binary = "julia";
  int argc = 1;
  const char* argv[] = {binary};
  tensorflow::port::InitMain(binary, &argc, &argv);
  absl::InitializeLog();
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

extern "C" PjRtClient *MakeCPUClient(uint8_t asynchronous, int node_id,
                                     int num_nodes) {
  CpuClientOptions options;
  // options.kv_store = "etcd";
  options.process_id = node_id;
  // options.num_nodes = num_nodes;
  // options.collectives = num_nodes;
  options.asynchronous = asynchronous != 0;
  auto client = MyValueOrThrow(GetTfrtCpuClient(options));
  return client.release();
}

// xla/python/xla.cc 390
extern "C" PjRtClient *
MakeGPUClient(int node_id, int num_nodes, int *allowed_devices,
              int num_allowed_devices, double memory_fraction, bool preallocate,
              const char *platform_name, const char **error) {
  GpuClientOptions options;
  // options.kv_store = "etcd";
  // options.allocator_config =
  options.allocator_config.preallocate = preallocate;
  options.allocator_config.memory_fraction = memory_fraction;
  options.node_id = node_id;
  options.num_nodes = num_nodes;
  options.allowed_devices =
      allowed_devices ? std::set<int>(allowed_devices,
                                      allowed_devices + num_allowed_devices)
                      : std::optional<std::set<int>>();
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

  const PJRT_Api *pluginLoad =
      LoadPjrtPlugin("tpu", tpu_library_path.c_str(), error);
  if (pluginLoad == nullptr)
    return nullptr;
  auto tpu_status = InitializePjrtPlugin("tpu", error);
  if (tpu_status)
    return nullptr;

  RegisterProfiler(pluginLoad);
  return GetCApiClient("TPU");
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

extern "C" void ExecutableFree(xla::PjRtLoadedExecutable *exec) { delete exec; }

extern "C" PjRtDevice *BufferToDevice(PjRtBuffer *Buffer) {
  return Buffer->device();
}

extern "C" PjRtClient *BufferToClient(PjRtBuffer *Buffer) {
  return Buffer->client();
}

extern "C" const int64_t* BufferShape(PjRtBuffer *Buffer) {
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

extern "C" PjRtClient *PjRtLoadedExecutableGetClient(PjRtLoadedExecutable *exec) {
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
  const xla::Layout* layout = nullptr;
  auto buffer = MyValueOrThrow(
      client->BufferFromHostBuffer(data, primtype, shape, /*byte_strides*/ {},
                                   semantics, /*ondone*/ {}, *device->default_memory_space(), layout));
  auto bres = buffer.release();
  return bres;
}

extern "C" uint8_t BufferOnCPU(PjRtBuffer *buffer) { return buffer->IsOnCpu(); }

extern "C" PjRtBuffer *CopyBufferToDevice(PjRtBuffer *buffer,
                                          PjRtDevice *dst_device) {
  auto res = MyValueOrThrow(buffer->CopyToMemorySpace(*dst_device->default_memory_space()));
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
  auto llvmModule = std::unique_ptr<llvm::Module>(unwrap(lmod));
  mlir::MLIRContext &context = *unwrap(cctx);

  auto res = mlir::translateLLVMIRToModule(std::move(llvmModule), &context,
                                           /*emitExpensiveWarnings*/ false,
                                           /*dropDICompositeElements*/ false)
                 .release();
  return wrap(res);
}

#include "llvm/IRReader/IRReader.h"
extern "C" MlirModule ConvertLLVMStrToMLIR(const char *lmod, MlirContext cctx) {
  LLVMContext Context;
  SMDiagnostic Err;
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

extern "C" xla::PjRtLoadedExecutable *ClientCompile(PjRtClient *client,
                                                    MlirModule cmod,
                                                    int64_t device_id,
                                                    bool is_sharded,
                                                    // const int64_t *mesh_shape,
                                                    // int64_t num_mesh_shape,
                                                    const int64_t *mesh_ids,
                                                    int64_t num_mesh_ids,
                                                    const char* xla_gpu_cuda_data_dir) {
  auto program =
      std::make_unique<xla::ifrt::HloProgram>(cast<ModuleOp>(*unwrap(cmod)));

  CompileOptions options;
  options.executable_build_options.mutable_debug_options()->set_xla_gpu_cuda_data_dir(xla_gpu_cuda_data_dir);

  auto cmodop = cast<ModuleOp>(*unwrap(cmod));

  if (is_sharded) {
    assert(device_id < 0);

    options.executable_build_options.set_num_replicas(1);
    options.executable_build_options.set_num_partitions(num_mesh_ids);

    options.executable_build_options.set_use_spmd_partitioning(true);
    options.executable_build_options.set_use_shardy_partitioner(true);

    // auto partitioning for GPUs is not available in open source version of XLA
    // options.executable_build_options.set_use_auto_spmd_partitioning(true);
    // std::vector<int64_t> mesh_shape_vec(mesh_shape, mesh_shape + num_mesh_shape);
    // options.executable_build_options.set_auto_spmd_partitioning_mesh_shape(mesh_shape_vec);
    // std::vector<int64_t> mesh_ids_vec(mesh_ids, mesh_ids + num_mesh_ids);
    // options.executable_build_options.set_auto_spmd_partitioning_mesh_ids(mesh_ids_vec);

    xla::DeviceAssignment device_assignment(1, num_mesh_ids);
    for (int64_t i = 0; i < num_mesh_ids; ++i) {
      int64_t mesh_id = mesh_ids[i];
      assert(mesh_id >= 0);
      device_assignment(0, mesh_id) = i;
    }
    options.executable_build_options.set_device_assignment(device_assignment);

    // https://github.com/openxla/xla/blob/b3c641b05692f3712fb3c272e38665fdfa28bdf8/xla/python/py_client.cc#L460
    auto status = xla::ExportShardyForHloRoundTrip(cmodop);
    if (!status.ok())
      ReactantThrowError(status.ToString().c_str());
  } else {
    assert(device_id >= 0);

    options.executable_build_options.set_num_replicas(1);
    options.executable_build_options.set_num_partitions(1);
    options.executable_build_options.set_device_ordinal(device_id);

    xla::DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = device_id;
    options.executable_build_options.set_device_assignment(device_assignment);
  }

  auto addressable_devices = client->addressable_devices();
  if (!addressable_devices.empty()) {
    int device_ordinal = options.executable_build_options.device_ordinal();
    if (device_ordinal < 0) {
      device_ordinal = 0;
    }
    assert(device_ordinal < addressable_devices.size());
    auto stats = addressable_devices[device_ordinal]->GetAllocatorStats();
    if (stats.ok() && stats->bytes_limit) {
      options.executable_build_options.set_device_memory_size(*stats->bytes_limit);
    }
  }
  auto exec = MyValueOrThrow(client->Compile(cmodop, options));
  return exec.release();
}

struct JLOpSharding {
  int32_t type;
  int32_t n_tile_dimensions;
  int64_t *tile_dimensions;
  int32_t n_layout_minor_to_major;
  int64_t *layout_minor_to_major;
  bool replicate_on_last_tile_dim;
  int32_t n_last_tile_dims;
  int32_t *last_tile_dims;
  int32_t n_tile_assignment_dimensions;
  int64_t *tile_assignment_dimensions;
  int32_t n_tile_assignment_devices;
  int64_t *tile_assignment_devices;
  int32_t n_iota_reshape_dims;
  int64_t *iota_reshape_dims;
  int32_t n_iota_transpose_perm;
  int32_t *iota_transpose_perm;
  bool is_shard_group;
  int64_t shard_group_id;
  int32_t shard_group_type;
};

extern "C" void PjRtLoadedExecutableGetOuputShardings(xla::PjRtLoadedExecutable *exec,
                                                      JLOpSharding **jl_op_shardings,
                                                      int32_t num_op_shardings) {
  std::optional<std::vector<OpSharding>> shardings = exec->GetOutputShardings();
  if (!shardings.has_value()) {
    ReactantThrowError("No sharding found for the output of the loaded executable");
  }

  std::vector<xla::OpSharding> hlo_op_shardings = shardings.value();
  if (num_op_shardings != hlo_op_shardings.size()) {
    ReactantThrowError(("Expected " + std::to_string(num_op_shardings) +
                        " shardings, got " +
                        std::to_string(hlo_op_shardings.size())).c_str());
  }

  for (int32_t i = 0; i < num_op_shardings; i++) {
    auto &op_sharding = hlo_op_shardings[i];
    auto &jl_op_sharding = jl_op_shardings[i];

    jl_op_sharding->type = op_sharding.type();
    jl_op_sharding->replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim();

    auto &shape = op_sharding.tile_shape();
    std::vector<int64_t> dimensions(
        shape.dimensions().begin(), shape.dimensions().end());
    jl_op_sharding->n_tile_dimensions = dimensions.size();
    jl_op_sharding->tile_dimensions = new int64_t[dimensions.size()];
    std::copy(dimensions.begin(), dimensions.end(), jl_op_sharding->tile_dimensions);

    if (shape.has_layout()) {
      auto &layout = shape.layout();
      std::vector<int64_t> minor_to_major(
          layout.minor_to_major().begin(), layout.minor_to_major().end());
      jl_op_sharding->n_layout_minor_to_major = minor_to_major.size();
      jl_op_sharding->layout_minor_to_major = new int64_t[minor_to_major.size()];
      std::copy(
          minor_to_major.begin(), minor_to_major.end(),
          jl_op_sharding->layout_minor_to_major);
    } else {
      jl_op_sharding->n_layout_minor_to_major = 0;
      jl_op_sharding->layout_minor_to_major = nullptr;
    }

    std::vector<int> last_tile_dims(op_sharding.last_tile_dims().begin(),
        op_sharding.last_tile_dims().end());
    jl_op_sharding->n_last_tile_dims = last_tile_dims.size();
    jl_op_sharding->last_tile_dims = new int[last_tile_dims.size()];
    std::copy(last_tile_dims.begin(), last_tile_dims.end(), jl_op_sharding->last_tile_dims);

    std::vector<int64_t> tile_assignment_dimensions(
        op_sharding.tile_assignment_dimensions().begin(),
        op_sharding.tile_assignment_dimensions().end());
    jl_op_sharding->n_tile_assignment_dimensions = tile_assignment_dimensions.size();
    jl_op_sharding->tile_assignment_dimensions = new int64_t[
        tile_assignment_dimensions.size()];
    std::copy(tile_assignment_dimensions.begin(), tile_assignment_dimensions.end(),
        jl_op_sharding->tile_assignment_dimensions);

    std::vector<int64_t> tile_assignment_devices(
        op_sharding.tile_assignment_devices().begin(),
        op_sharding.tile_assignment_devices().end());
    jl_op_sharding->n_tile_assignment_devices = tile_assignment_devices.size();
    jl_op_sharding->tile_assignment_devices = new int64_t[tile_assignment_devices.size()];
    std::copy(tile_assignment_devices.begin(), tile_assignment_devices.end(),
        jl_op_sharding->tile_assignment_devices);

    std::vector<int64_t> iota_reshape_dims(op_sharding.iota_reshape_dims().begin(),
        op_sharding.iota_reshape_dims().end());
    jl_op_sharding->n_iota_reshape_dims = iota_reshape_dims.size();
    jl_op_sharding->iota_reshape_dims = new int64_t[iota_reshape_dims.size()];
    std::copy(iota_reshape_dims.begin(), iota_reshape_dims.end(),
        jl_op_sharding->iota_reshape_dims);

    std::vector<int> iota_transpose_perm(op_sharding.iota_transpose_perm().begin(),
        op_sharding.iota_transpose_perm().end());
    jl_op_sharding->n_iota_transpose_perm = iota_transpose_perm.size();
    jl_op_sharding->iota_transpose_perm = new int[iota_transpose_perm.size()];
    std::copy(iota_transpose_perm.begin(), iota_transpose_perm.end(),
        jl_op_sharding->iota_transpose_perm);

    jl_op_sharding->is_shard_group = op_sharding.is_shard_group();
    jl_op_sharding->shard_group_id = op_sharding.shard_group_id();
    jl_op_sharding->shard_group_type = op_sharding.shard_group_type();
  }
}

typedef PjRtFuture<> FutureType;
extern "C" void FreeFuture(FutureType *Future) { delete Future; }

extern "C" uint8_t FutureIsReady(FutureType *Future) {
  return Future->IsReady();
}

extern "C" void FutureAwait(FutureType *Future) { Future->Await(); }

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

  auto results = MyValueOrThrow(
      exec->ExecuteSharded(argument_handles,
          device, options, returned_future, /*fill_future=*/true));

  // Validate the number of results.
  if (results.size() != num_results) {
    llvm::errs() << "Error: results.size()=" << results.size()
                 << " does not match num_results=" << num_results << "\n";
    std::abort(); // Terminate if the number of results is incorrect.
  }

  // Handle futures if they are returned.
  if (returned_future.has_value()) {
    *futures = true;
    for (size_t i = 0; i < num_results; i++) {
      future_results[i] = new FutureType(*returned_future);
    }
  } else {
    *futures = false;
  }

  // Release the results into the output array.
  for (size_t i = 0; i < num_results; i++) {
    op_results[i] = results[i].release();
  }
}

extern "C" void XLAExecute(xla::PjRtLoadedExecutable *exec, int op_args_len,
                           PjRtBuffer **op_args,
                           const int64_t *mesh_ids, int64_t num_mesh_ids,
                           uint8_t *is_arg_donatable,
                           int num_results, PjRtBuffer **op_results,
                           uint8_t *futures, FutureType **future_results) {
  // Ensure argument_handles is structured as num_mesh_ids x num_args
  std::vector<std::vector<PjRtBuffer *>> argument_handles(num_mesh_ids);
  int num_args = op_args_len / num_mesh_ids;

  // Distribute arguments across devices
  for (int device_idx = 0; device_idx < num_mesh_ids; ++device_idx) {
    int64_t mesh_id = mesh_ids[device_idx];

    // Validate mesh_id
    if (mesh_id < 0 || mesh_id >= num_mesh_ids) {
      ReactantThrowError(("Invalid mesh_id " + std::to_string(mesh_id) + " at device_idx " +
                          std::to_string(device_idx)).c_str());
    }

    argument_handles[mesh_id].reserve(num_args);
    for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {
      // Assuming op_args is a flat array of size num_devices * num_args
      // where arguments for each device are contiguous
      argument_handles[mesh_id].push_back(op_args[mesh_id * num_args + arg_idx]);
    }
  }

  ExecuteOptions options;

  for (size_t i = 0; i < num_args; i++) {
    if (!is_arg_donatable[i])
      options.non_donatable_input_indices.insert((int)i);
  }
  options.untuple_result = true;

  std::optional<std::vector<FutureType>> returned_futures = std::vector<FutureType>();
  auto results = MyValueOrThrow(
      exec->Execute(static_cast<absl::Span<const std::vector<PjRtBuffer *>>>(
                        argument_handles),
                    options, returned_futures));

  assert(results.size() == num_mesh_ids);

  for (int device_idx = 0; device_idx < num_mesh_ids; ++device_idx) {
    int64_t mesh_id = mesh_ids[device_idx];
    if (results[mesh_id].size() != num_results) {
      llvm::errs() << " results[" << mesh_id << "].size()=" << results[mesh_id].size()
                   << " num_results=" << num_results << "\n";
    }
    assert(results[mesh_id].size() == num_results);
  }

  // Handle returned futures
  if (returned_futures.has_value()) {
    *futures = true;
    assert(returned_futures->size() == num_mesh_ids);
  } else {
    *futures = false;
  }

  // Copy results into the output buffers
  for (int device_idx = 0; device_idx < num_mesh_ids; ++device_idx) {
    int64_t mesh_id = mesh_ids[device_idx];
    for (int result_idx = 0; result_idx < num_results; ++result_idx) {
      int flat_index = mesh_id * num_results + result_idx;
      op_results[flat_index] = results[mesh_id][result_idx].release();
      if (returned_futures.has_value()) {
        future_results[flat_index] = new FutureType((*returned_futures)[mesh_id]);
      }
    }
  }
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
  context.loadDialect<mlir::triton::TritonDialect>();
  context.loadDialect<mlir::tpu::TPUDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  context.loadDialect<mlir::sdy::SdyDialect>();
}

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"

extern "C" void InitializePasses(MlirDialectRegistry creg) {
  mlir::registerenzymePasses();
  enzyme::registerenzymexlaPasses();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerConvertAffineToStandardPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerConvertSCFToOpenMPPass();
  mlir::affine::registerAffinePasses();
  mlir::registerReconcileUnrealizedCasts();

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

namespace reactant {

template <typename T> struct unwrap_type { typedef T type; };
template <typename T> struct unwrap_type<std::shared_ptr<T>> { typedef T type; };
template <typename T> struct unwrap_type<tsl::RCReference<T>> { typedef T type; };

template <typename T> using unwrap_type_t = typename unwrap_type<T>::type;

template<typename T>
struct HeldValue {
 public:
    HeldValue(T& obj) : holded(obj) {}
    ~HeldValue() = default;

    unwrap_type_t<T>* ptr() const {
        return holded.get();
    }

    T obj() const {
        return holded;
    }

    T value() const {
        return holded;
    }

    unwrap_type_t<T>* operator->() const {
        return ptr();
    }

 private:
    T holded;
};

template <typename T>
HeldValue<T>* capture(T obj) {
    return new HeldValue<T>(obj);
}

} // namespace reactant

using reactant::HeldValue;

extern "C" HeldValue<std::shared_ptr<PjRtClient>>* reactant_hold_pjrtclient(xla::PjRtClient* client) {
  return reactant::capture(std::shared_ptr<PjRtClient>(client));
}

extern "C" void reactant_release_pjrtclient(HeldValue<std::shared_ptr<PjRtClient>>* client) { delete client; }

extern "C" HeldValue<std::shared_ptr<xla::PjRtBuffer>>* reactant_hold_pjrtbuffer(xla::PjRtBuffer* buffer) {
  return reactant::capture(std::shared_ptr<xla::PjRtBuffer>(buffer));
}

extern "C" void reactant_release_pjrtbuffer(HeldValue<std::shared_ptr<PjRtBuffer>>* buffer) { delete buffer; }

extern "C" ifrt::Client* ifrt_pjrt_MakeClient(HeldValue<std::shared_ptr<PjRtClient>>* pjrt_client) {
  xla::ifrt::PjRtClient::CreateOptions options = {pjrt_client->obj()};
  return MyValueOrThrow(xla::ifrt::PjRtClient::Create(options)).release();
}

extern "C" void ifrt_FreeClient(ifrt::Client* client) { delete client; }

extern "C" xla::ifrt::LoadedExecutable* ifrt_ClientCompile(ifrt::PjRtClient* client, MlirModule mlir_mod) {
  mlir::ModuleOp mlir_mod_op = cast<ModuleOp>(*unwrap(mlir_mod));
  // TODO import sharding config from `ClientCompile`?
  xla::CompileOptions compile_options;
  // TODO can't create LoadedExecutable from mlir::ModuleOp on IFRT-proxy backend
  return MyValueOrThrow(xla::ifrt::PjRtLoadedExecutable::Create(client, mlir_mod_op, compile_options, std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>())).release();
}

extern "C" void ifrt_pjrt_FreeLoadedExecutable(xla::ifrt::PjRtLoadedExecutable* exec) { delete exec; }

// TODO replace with `Client::MakeArrayFromHostBuffer` and generalize to `ifrt::Client`
extern "C" HeldValue<tsl::RCReference<xla::ifrt::Array>>* ifrt_pjrt_ArrayFromHostBuffer(ifrt::PjRtClient* client, HeldValue<std::shared_ptr<xla::PjRtBuffer>>* buffer) {
  return reactant::capture(tsl::RCReference<ifrt::Array>(MyValueOrThrow(xla::ifrt::PjRtArray::Create(client, buffer->obj()))));
}

extern "C" void reactant_release_ifrt_array(HeldValue<tsl::RCReference<xla::ifrt::Array>>* array) { delete array; }

extern "C" void ifrt_Execute(ifrt::LoadedExecutable* exec, int num_args, HeldValue<tsl::RCReference<ifrt::Array>>** op_args, uint8_t* is_arg_donatable, int num_results, HeldValue<tsl::RCReference<ifrt::Array>>** op_results, uint8_t *futures, FutureType** status) {
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
  
  auto result = MyValueOrThrow(exec->Execute(static_cast<absl::Span<tsl::RCReference<xla::ifrt::Array>>>(args), options, /* devices */ std::nullopt));

  if (result.outputs.size() != num_results) {
    llvm::errs() << "Error: results.size()=" << result.outputs.size()
                 << " does not match num_results=" << num_results << "\n";
    std::abort(); // Terminate if the number of results is incorrect.
  }

  // there is only 1 status and is valid because we set `options.fill_status = true`
  *futures = true;
  *status = new FutureType(result.status);

  for (int i = 0; i < num_results; i++) {
    op_results[i] = reactant::capture(result.outputs[i]);
  }
}

// in principle, use ArrayCopySemantics::kAlwaysCopy (=0)
extern "C" FutureType* ifrt_CopyArrayToHostBuffer(HeldValue<tsl::RCReference<xla::ifrt::Array>>* array, void* data, ifrt::ArrayCopySemantics semantics) {
  return new FutureType((*array)->CopyToHostBuffer(data, std::nullopt, semantics));
}
