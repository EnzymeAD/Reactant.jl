#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Pass/PassManager.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Passes/Passes.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
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
#include "llvm/Support/TargetSelect.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "absl/log/initialize.h"
#include "absl/log/globals.h"

#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/service/cpu/simple_orc_jit.h"

#include "xla/python/ifrt/hlo/hlo_program.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Process.h"

using namespace mlir;
using namespace llvm;
using namespace xla;

// int google::protobuf::io::CodedInputStream::default_recursion_limit_ = 100;
// int xla::_LayoutProto_default_instance_;

extern "C" void InitializeLogs() {
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
    //absl::SetGlobalVLogLevel(level);
}

extern "C" void SetModuleLogLevel(const char* module_pattern, int level) {
    //absl::SetVLOGLevel(module_pattern, level);
}

extern "C"
MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(
    MlirContext ctx, int32_t val) {
    return wrap(mlir::enzyme::ActivityAttr::get(unwrap(ctx), (mlir::enzyme::Activity)val));
}

extern "C" PjRtClient* MakeCPUClient(uint8_t asynchronous, int node_id, int num_nodes) {
    CpuClientOptions options;
    // options.kv_store = "etcd";
    options.process_id = node_id;
    // options.num_nodes = num_nodes;
    // options.collectives = num_nodes;
    options.asynchronous = asynchronous != 0;
    auto client = xla::ValueOrThrow(GetTfrtCpuClient(options));
    return client.release();
}

// xla/python/xla.cc 390
extern "C" PjRtClient* MakeGPUClient(int node_id, int num_nodes, int* allowed_devices, int num_allowed_devices, const char* platform_name, const char** error) {
    GpuClientOptions options;
    // options.kv_store = "etcd";
    // options.allocator_config = 
    options.node_id = node_id;
    options.num_nodes = num_nodes;
    options.allowed_devices = allowed_devices ? std::set<int>(allowed_devices, allowed_devices + num_allowed_devices) : std::optional<std::set<int>>();
    options.platform_name = platform_name ? std::string(platform_name) : std::optional<std::string>();
    // options.collectives = num_nodes;
    auto clientErr = GetStreamExecutorGpuClient(options);

    if (!clientErr.ok()) {
      auto str = clientErr.status().message();
      char* err = (char*)malloc(str.size()+1);
      memcpy(err, str.data(), str.size()+1);
      *error = err;
      return nullptr;
    } else {
      auto client = std::move(clientErr).value();
      return client.release();
    }
}

const char* const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";

extern "C" PJRT_Api* LoadPjrtPlugin(const char* device_type, const char* library_path, const char** error) {
    absl::StatusOr<const PJRT_Api*> pluginLoad = pjrt::LoadPjrtPlugin(std::string(device_type), std::string(library_path));
    if (!pluginLoad.ok()) {
        auto str = pluginLoad.status().message();
        char* err = (char*)malloc(str.size()+1);
        memcpy(err, str.data(), str.size()+1);
        *error = err;
        return nullptr;
    }
    return pluginLoad.value();
}

extern "C" int InitializePjrtPlugin(const char* device_type, const char** error) {
    absl::Status tpu_status = pjrt::InitializePjrtPlugin(device_type);
    if (!tpu_status.ok()) {
      auto str = tpu_status.message();
      char* err = (char*)malloc(str.size()+1);
      memcpy(err, str.data(), str.size()+1);
      *error = err;
      return 1;
    }
    return 0;
}

extern "C" PjRtClient* GetCApiClient(const char* device_type) {
    return xla::GetCApiClient(device_type).value().release();
}

extern "C" PjRtClient* MakeTPUClient(const char* tpu_path , const char** error) {
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

    const PJRT_Api* pluginLoad = LoadPjrtPlugin("tpu", tpu_library_path.c_str(), error);
    if (pluginLoad == nullptr)
        return nullptr;


    auto tpu_status = InitializePjrtPlugin("tpu", error);
    if (tpu_status)
        return nullptr;

    return GetCApiClient("TPU");
}

extern "C" int ClientNumDevices(PjRtClient* client) {
    return client->device_count();
}

extern "C" int ClientNumAddressableDevices(PjRtClient* client) {
    return client->addressable_device_count();
}

extern "C" int ClientProcessIndex(PjRtClient* client) {
    return client->process_index();
}

extern "C" PjRtDevice* ClientGetDevice(PjRtClient* client, int device_id) {
    return xla::ValueOrThrow(client->LookupDevice(PjRtGlobalDeviceId(device_id)));
}

extern "C" PjRtDevice* ClientGetAddressableDevice(PjRtClient* client, int device_id) {
    return xla::ValueOrThrow(client->LookupAddressableDevice(PjRtLocalDeviceId(device_id)));
}

extern "C" void ExecutableFree(xla::PjRtLoadedExecutable* exec) {
    delete exec;
}

extern "C" PjRtDevice* BufferToDevice(PjRtBuffer* Buffer) {
    return Buffer->device();
}

extern "C" PjRtClient* BufferToClient(PjRtBuffer* Buffer) {
    return Buffer->client();
}

extern "C" PjRtClient* DeviceToClient(PjRtDevice* Device) {
    return Device->client();
}

extern "C" void PjRtBufferFree(PjRtBuffer* Buffer) {
    delete Buffer;
}

// https://openxla.org/xla/shapes
// This minor-to-major dimension order of 0 up to N-1 is akin to column-major (at rank 2). Assuming a monotonic ordering of dimensions, another way we may refer to this layout in the code is simply "dim 0 is minor".
std::vector<int64_t> col_major(int64_t dim) {
    std::vector<int64_t> minor_to_major;
    for (int i=0; i<dim; i++) {
        minor_to_major.push_back(i);//dim-1-i);
        // minor_to_major.push_back(dim-1-i);
    }
    return minor_to_major;
}
static void noop() {}

extern "C" void* UnsafeBufferPointer(PjRtBuffer* buffer) {
    auto unsafe = xla::ValueOrThrow(buffer->client()->UnsafeBufferPointer(buffer));
    return (void*)unsafe;
}

extern "C" PjRtBuffer* ArrayFromHostBuffer(PjRtClient* client, void* data, MlirType mtype, size_t dim, int64_t* cshape, PjRtDevice* device) {
    auto primtype = ConvertMlirTypeToPrimitiveType(unwrap(mtype));
    absl::Span<const int64_t> shape(cshape, dim);
    PjRtClient::HostBufferSemantics semantics = PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
    //xla::Layout layout(col_major(dim));
    //auto buffer = xla::ValueOrThrow(client->BufferFromHostBuffer(data, primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device, &layout));
    auto buffer = xla::ValueOrThrow(client->BufferFromHostBuffer(data, primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device));
    auto bres = buffer.release();
    return bres;
}

extern "C" uint8_t BufferOnCPU(PjRtBuffer* buffer) {
    return buffer->IsOnCpu();
}


extern "C" PjRtBuffer* CopyBufferToDevice(PjRtBuffer* buffer, PjRtDevice* dst_device) {
    auto res = xla::ValueOrThrow(buffer->CopyToDevice(dst_device));
    return res.release();
}

extern "C" void BufferToHost(PjRtBuffer* buffer, void* data) {
    Shape shape(xla::ValueOrThrow(buffer->HostShape()));
    /// Grumpily the cpu copy code does not respect layout and does a raw copy
    /*
    auto &layout = *shape.mutable_layout();
    layout.clear_minor_to_major();
    for (auto index : col_major(shape.dimensions_size())) {
        layout.add_minor_to_major(index);
    }
    */
    MutableBorrowingLiteral literal((const char*)data, shape);
    auto status = buffer->ToLiteralSync(&literal);
    if (!status.ok()) {
        printf("error copying to host: %s\n", status.ToString().c_str());
    }
}

extern "C" void FreeClient(PjRtClient * client) {
    delete client;
}

/* Note that this */
extern "C" xla::PjRtLoadedExecutable* ClientCompile(PjRtClient * client, MlirModule cmod) {
    auto program = std::make_unique<xla::ifrt::HloProgram>(cast<ModuleOp>(*unwrap(cmod)));

    CompileOptions options;
    // options.argument_layouts;
    // options.executable_build_options.set_device_ordinal();
    // options.executable_build_options.set_result_layout();
    
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
    auto exec = xla::ValueOrThrow(client->Compile(cast<ModuleOp>(*unwrap(cmod)), options));
    return exec.release();
}

typedef PjRtFuture<> FutureType;
extern "C" void FreeFuture(FutureType* Future) {
    delete Future;
}

extern "C" uint8_t FutureIsReady(FutureType* Future) {
    return Future->IsReady();
}

extern "C" void FutureAwait(FutureType* Future) {
    Future->Await();
}

extern "C" void XLAExecute(xla::PjRtLoadedExecutable* exec, int num_args, PjRtBuffer** op_args, uint8_t* is_arg_donatable, int num_results, PjRtBuffer** op_results, uint8_t *futures, FutureType** future_results) {
    std::vector<std::vector<PjRtBuffer*>> argument_handles;
    argument_handles.emplace_back(op_args, op_args + num_args);

    ExecuteOptions options;

    for (size_t i=0; i<num_args; i++) {
        if (!is_arg_donatable[i])
            options.non_donatable_input_indices.insert((int)i);
    }
    options.untuple_result = true;
    std::optional<std::vector<FutureType>> returned_futures;
    auto results = xla::ValueOrThrow(exec->Execute(static_cast<absl::Span<const std::vector<PjRtBuffer*>>>(argument_handles), options, returned_futures));

    assert(results.size() == 1);

    if (results[0].size() != num_results) {
        llvm::errs() <<" results.size()=" << results.size() << " num_results=" << num_results << "\n";
    }
    assert(results[0].size() == num_results);
    if (returned_futures) {
        *futures = true;
        assert(returned_futures->size() == num_results);
        for (size_t i=0; i<num_results; i++) {
            future_results[i] = new FutureType((*returned_futures)[i]);
        }
    } else {
        *futures = false;
    }

    for (size_t i=0; i<num_results; i++) {
        op_results[i] = results[0][i].release();
    }
}

extern "C" void RegisterDialects(MlirContext cctx) {
  mlir::MLIRContext &context = *unwrap(cctx);
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::enzyme::EnzymeDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
}
extern "C" void InitializeRegistryAndPasses(MlirDialectRegistry creg) {
  mlir::DialectRegistry &registry = *unwrap(creg);

  // Register MLIR stuff
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::NVVM::NVVMDialect>();
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<DLTIDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect>();
  registry.insert<mlir::chlo::ChloDialect>();

  registry.insert<mlir::enzyme::EnzymeDialect>();

  mlir::registerenzymePasses();
  regsiterenzymeXLAPasses();
  mlir::enzyme::registerXLAAutoDiffInterfaces(registry);

  mlir::func::registerInlinerExtension(registry);

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

  // Register the autodiff interface implementations for upstream dialects.
  enzyme::registerCoreDialectAutodiffInterfaces(registry);

  // Transform dialect and extensions.
  mlir::transform::registerInterpreterPass();
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::enzyme::registerGenerateApplyPatternsPass();
  mlir::enzyme::registerRemoveTransformPass();
  mlir::enzyme::registerEnzymeJaxTransformExtension(registry);
}
