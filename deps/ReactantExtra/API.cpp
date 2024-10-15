#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
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
#include "xla/service/cpu/simple_orc_jit.h"

#include "xla/python/ifrt/hlo/hlo_program.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Process.h"

#include "llvm-c/TargetMachine.h"

// IFRT
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/compiler.h"

using namespace mlir;
using namespace llvm;
using namespace xla;

// MLIR C-API extras
#pragma region MLIR Extra
#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirComplexType, const void);
DEFINE_C_API_METHODS(MlirComplexType, mlir::ComplexType)

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGet(MlirContext ctx, MlirComplexType type, double real, double imag) {
    return wrap(complex::NumberAttr::get(unwrap(type), real, imag));
}

MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGetChecked(MlirLocation loc, MlirComplexType type, double real, double imag) {
    return wrap(complex::NumberAttr::getChecked(unwrap(loc), unwrap(type), real, imag));
}

// TODO mlirComplexAttrGetValue
// TODO MLIR_CAPI_EXPORTED MlirTypeID mlirComplexAttrGetTypeID(void) { return wrap(complex::NumberAttr::getTypeID()); }
#pragma endregion

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

extern "C" char *GetDefaultTargetTriple(void) {
  return LLVMGetDefaultTargetTriple();
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

extern "C" const PJRT_Api* LoadPjrtPlugin(const char* device_type, const char* library_path, const char** error) {
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

std::vector<int64_t> row_major(int64_t dim) {
    std::vector<int64_t> minor_to_major;
    for (int i=0; i<dim; i++) {
        minor_to_major.push_back(dim-1-i);
    }
    return minor_to_major;
}
static void noop() {}

extern "C" void* UnsafeBufferPointer(PjRtBuffer* buffer) {
    auto unsafe = xla::ValueOrThrow(buffer->client()->UnsafeBufferPointer(buffer));
    return (void*)unsafe;
}

extern "C" PjRtBuffer* ArrayFromHostBuffer(PjRtClient* client, void* data, uint64_t ptype, size_t dim, int64_t* cshape, PjRtDevice* device) {
    auto primtype = (xla::PrimitiveType)ptype;
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
    /// For now, we assume a non-julia row major ordering
    /// If in the future it supports col_major we can swap to that.
    *shape.mutable_layout() = xla::Layout(row_major(shape.dimensions_size()));
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

#pragma mark xla::ifrt::DType
extern "C" ifrt::DType* ifrt_dtype_ctor(ifrt::DType::Kind kind) {
    return new ifrt::DType(kind);
}

extern "C" void ifrt_dtype_free(ifrt::DType* dtype) {
    delete dtype;
}

extern "C" ifrt::DType::Kind ifrt_dtype_kind(ifrt::DType* dtype) {
    return dtype->kind();
}

extern "C" bool ifrt_dtype_eq(ifrt::DType* dtype1, ifrt::DType* dtype2) {
    return *dtype1 == *dtype2;
}

extern "C" bool ifrt_dtype_ne(ifrt::DType* dtype1, ifrt::DType* dtype2) {
    return *dtype1 != *dtype2;
}

// Returns -1 if not aligned to a byte boundary or there is no fixed size
extern "C" int ifrt_dtype_byte_size(ifrt::DType* dtype) {
    auto byte_size = dtype->byte_size();
    if (byte_size.has_value()) {
        return byte_size.value();
    }
    return -1;
}

// Returns -1 if there is no fixed size
extern "C" int ifrt_dtype_bit_size(ifrt::DType* dtype) {
    auto bit_size = dtype->bit_size();
    if (bit_size.has_value()) {
        return bit_size.value();
    }
    return -1;
}

extern "C" const char* ifrt_dtype_debug_string(ifrt::DType* dtype) {
    return cstr_from_string(dtype->DebugString());
}

#pragma mark xla::ifrt::Shape
extern "C" ifrt::Shape* ifrt_shape_ctor(const int64_t* dims, size_t dims_size) {
    return new ifrt::Shape(absl::Span<const int64_t>(dims, dims_size));
}

extern "C" void ifrt_shape_free(ifrt::Shape* shape) {
    delete shape;
}

extern "C" const int64_t* ifrt_shape_dims(ifrt::Shape* shape) {
    return shape->dims().data();
}

extern "C" int64_t ifrt_shape_dims_num_elements(ifrt::Shape* shape) {
    return shape->num_elements();
}

extern "C" const char* ifrt_shape_debug_string(ifrt::Shape* shape) {
    return cstr_from_string(shape->DebugString());
}

#pragma mark xla::ifrt::DynamicShape
extern "C" ifrt::DynamicShape* ifrt_dynamicshape_ctor(ifrt::Shape* shape, bool dynamic_dims_mask) {
    std::vector<bool> bool_vector(dynamic_dims_mask, dynamic_dims_mask + shape->dims().size());
    auto tag = ifrt::BoundedDynamicShapeTag(absl::Span<const bool>(bool_vector));
    return new ifrt::DynamicShape(*shape, tag);
}

extern "C" void ifrt_dynamicshape_free(ifrt::DynamicShape* shape) {
    delete shape;
}

// TODO ifrt::DynamicShape::GetTag

extern "C" bool ifrt_dynamicshape_eq(ifrt::DynamicShape* shape1, ifrt::DynamicShape* shape2) {
    return *shape1 == *shape2;
}

extern "C" bool ifrt_dynamicshape_ne(ifrt::DynamicShape* shape1, ifrt::DynamicShape* shape2) {
    return *shape1 != *shape2;
}

extern "C" ifrt::Shape* ifrt_dynamicshape_get_padded_shape(ifrt::DynamicShape* shape) {
    return xla::ValueOrThrow(shape->GetPaddedShape()).release();
}

extern "C" bool ifrt_dynamicshape_is_dynamic_dim(ifrt::DynamicShape* shape, int dimension) {
    return shape->IsDynamicDim(dimension);
}

extern "C" const char* ifrt_dynamicshape_debug_string(ifrt::DynamicShape* shape) {
    return cstr_from_string(shape->DebugString());
}

#pragma mark xla::ifrt::Index
extern "C" ifrt::Index* ifrt_index_ctor(const int64_t* elements, size_t elements_size) {
    return new ifrt::Index(absl::Span<const int64_t>(elements, elements_size));
}

extern "C" ifrt::Index* ifrt_index_zeros(int num_elements) {
    return new ifrt::Index(ifrt::Index::Zeros(num_elements));
}

extern "C" void ifrt_index_free(ifrt::Index* index) {
    delete index;
}

extern "C" const int64_t* ifrt_index_elements(ifrt::Index* index) {
    return index->elements().data();
}

extern "C" int ifrt_index_count(ifrt::Index* index) {
    return index->elements().size();
}

extern "C" bool ifrt_index_eq(ifrt::Index* index1, ifrt::Index* index2) {
    return *index1 == *index2;
}

extern "C" bool ifrt_index_ne(ifrt::Index* index1, ifrt::Index* index2) {
    return *index1 != *index2;
}

extern "C" ifrt::Index* ifrt_index_add(ifrt::Index* index, ifrt::Index* offset) {
    return new ifrt::Index(*index + *offset);
}

extern "C" ifrt::Index* ifrt_index_sub(ifrt::Index* index, ifrt::Index* offset) {
    return new ifrt::Index(*index - *offset);
}

// WARN we're not checking if the multiplier has the same size as the index
extern "C" ifrt::Index* ifrt_index_mul(ifrt::Index* index, const int64_t* multiplier) {
    return new ifrt::Index(*index * absl::Span<const int64_t>(multiplier, ifrt_index_count(index)));
}

extern "C" void ifrt_index_add_inplace(ifrt::Index* index, ifrt::Index* offset) {
    *index += *offset;
}

extern "C" void ifrt_index_sub_inplace(ifrt::Index* index, ifrt::Index* offset) {
    *index -= *offset;
}

extern "C" void ifrt_index_mul_inplace(ifrt::Index* index, const int64_t* multiplier) {
    *index *= absl::Span<const int64_t>(multiplier, ifrt_index_count(index));
}

extern "C" const char* ifrt_index_debug_string(ifrt::Index* index) {
    return cstr_from_string(index->DebugString());
}

#pragma mark xla::ifrt::IndexDomain
extern "C" ifrt::IndexDomain* ifrt_indexdomain_ctor(ifrt::Shape* shape) {
    return new ifrt::IndexDomain(*shape);
}

extern "C" ifrt::IndexDomain* ifrt_indexdomain_ctor_with_origin(ifrt::Index* origin, ifrt::Shape* shape) {
    return new ifrt::IndexDomain(*origin, *shape);
}

extern "C" void ifrt_indexdomain_free(ifrt::IndexDomain* index_domain) {
    delete index_domain;
}

extern "C" const ifrt::Index* ifrt_indexdomain_origin(ifrt::IndexDomain* index_domain) {
    return &index_domain->origin();
}

extern "C" const ifrt::Shape* ifrt_indexdomain_shape(ifrt::IndexDomain* index_domain) {
    return &index_domain->shape();
}

extern "C" bool ifrt_indexdomain_eq(ifrt::IndexDomain* index_domain1, ifrt::IndexDomain* index_domain2) {
    return *index_domain1 == *index_domain2;
}

extern "C" bool ifrt_indexdomain_ne(ifrt::IndexDomain* index_domain1, ifrt::IndexDomain* index_domain2) {
    return *index_domain1 != *index_domain2;
}

extern "C" ifrt::IndexDomain* ifrt_indexdomain_add(ifrt::IndexDomain* index_domain, ifrt::Index* offset) {
    return new ifrt::IndexDomain(*index_domain + *offset);
}

extern "C" ifrt::IndexDomain* ifrt_indexdomain_sub(ifrt::IndexDomain* index_domain, ifrt::Index* offset) {
    return new ifrt::IndexDomain(*index_domain - *offset);
}

extern "C" void ifrt_indexdomain_add_inplace(ifrt::IndexDomain* index_domain, ifrt::Index* offset) {
    *index_domain += *offset;
}

extern "C" void ifrt_indexdomain_sub_inplace(ifrt::IndexDomain* index_domain, ifrt::Index* offset) {
    *index_domain -= *offset;
}

extern "C" const char* ifrt_indexdomain_debug_string(ifrt::IndexDomain* index_domain) {
    return cstr_from_string(index_domain->DebugString());
}

#pragma mark xla::ifrt::MemoryKind
// extern "C" ifrt::MemoryKind* ifrt_memorykind() {
//     return new ifrt::MemoryKind();
// }

// extern "C" ifrt::MemoryKind* ifrt_memorykind_from_string(const char* kind) {
//     return new ifrt::MemoryKind(std::optional<std::string>{kind});
// }

// extern "C" void ifrt_memorykind_free(ifrt::MemoryKind* memory_kind) {
//     delete memory_kind;
// }

#pragma mark xla::ifrt::Memory
extern "C" ifrt::Memory* ifrt_memory_ctor() {
    return new ifrt::Memory();
}

extern "C" void ifrt_memory_free(ifrt::Memory* memory) {
    delete memory;
}

// MemoryId is a struct with a single int32_t field --> check out xla/python/ifrt/memory.h
extern "C" ifrt::MemoryId ifrt_memory_id(ifrt::Memory* memory) {
    return memory->Id();
}

// TODO ifrt_memory_kind

extern "C" const char* ifrt_memory_to_string(ifrt::Memory* memory) {
    return cstr_from_string(memory->ToString());
}

extern "C" const char* ifrt_memory_debug_string(ifrt::Memory* memory) {
    return cstr_from_string(memory->DebugString());
}

// TODO ifrt_memory_devices

#pragma mark xla::ifrt::Device
extern "C" ifrt::Device* ifrt_device_ctor() {
    return new ifrt::Device();
}

extern "C" void ifrt_device_free(ifrt::Device* device) {
    delete device;
}

extern "C" ifrt::Client* ifrt_device_client(ifrt::Device* device) {
    return device->client();
}

// DeviceId is a struct with a single int32_t field --> check out xla/pjrt/pjrt_common.h
extern "C" ifrt::DeviceId ifrt_device_id(ifrt::Device* device) {
    return device->id();
}

// TODO ifrt_device_attributes

extern "C" const char* ifrt_device_kind(ifrt::Device* device) {
    return cstr_from_string(device->kind());
}

extern "C" const char* ifrt_device_to_string(ifrt::Device* device) {
    return cstr_from_string(device->ToString());
}

extern "C" const char* ifrt_device_from_string(ifrt::Client* client) {
    return cstr_from_string(client->DebugString());
}

extern "C" ifrt::Memory* ifrt_device_default_memory(ifrt::Device* device, char** error) {
    return unwrap_absl_statusor(device->DefaultMemory(), error);
}

// TODO ifrt_device_memories

extern "C" bool ifrt_device_is_addressable(ifrt::Device* device) {
    return device->IsAddressable();
}

extern "C" int ifrt_device_process_index(ifrt::Device* device) {
    return device->process_index();
}

#pragma mark xla::ifrt::Sharding
// TODO ifrt_sharding_devices
// TODO ifrt_sharding_memory_kind

// extern "C" void ifrt_sharding_disassemble(ifrt::Sharding* sharding, ifrt::Shape* shape, char** error) {
//     auto status = sharding->Disassemble(*shape);
//     if (!status.ok()) {
//         auto str = status.message();
//         char* err = (char*)malloc(str.size()+1);
//         memcpy(err, str.data(), str.size()+1);
//         *error = err;
//     }
// }

// TODO ifrt_sharding_disassemble_dynamic_shape
// TODO ifrt_sharding_index_domains

extern "C" const char* ifrt_sharding_debug_string(ifrt::Sharding* sharding) {
    return cstr_from_string(sharding->DebugString());
}

#pragma mark xla::ifrt::Array
extern "C" ifrt::Array* ifrt_array_ctor() {
    return new ifrt::Array();
}

// extern "C" void ifrt_array_free(ifrt::Array* array) {
//     delete array;
// }

extern "C" ifrt::DType ifrt_array_dtype(ifrt::Array* array) {
    return array->dtype();
}

// ...

#pragma mark xla::ifrt::Client
extern "C" int ifrt_client_device_count(ifrt::Client* client) {
    return client->device_count();
}

extern "C" int ifrt_client_addressable_device_count(ifrt::Client* client) {
    return client->addressable_device_count();
}

extern "C" ifrt::Device* ifrt_client_devices(ifrt::Client* client) {
    return client->devices().data();
}

extern "C" ifrt::Device* ifrt_client_addressable_devices(ifrt::Client* client) {
    return client->addressable_devices().data();
}

extern "C" int ifrt_client_process_index(ifrt::Client* client) {
    return client->process_index();
}

// TODO xla::ifrt::Client::GetDefaultDeviceAssignment

extern "C" ifrt::Device* ifrt_client_lookup_device(ifrt::Client* client, int device_id, **) {
    return xla::ValueOrThrow(client->LookupDevice(ifrt::DeviceId(device_id)));
}

extern "C" ifrt::Device* ifrt_client_lookup_addressable_device(ifrt::Client* client, int device_id, **) {
    return xla::ValueOrThrow(client->LookupAddressableDevice(ifrt::DeviceId(device_id)));
}

extern "C" ifrt::Compiler* ifrt_client_default_compiler(ifrt::Client* client) {
    return client->GetDefaultCompiler();
}

// TODO ifrt_client_topology_for_devices
// TODO ifrt_client_default_layout_for_device

#pragma mark xla::ifrt::Executable
extern "C" const char* ifrt_executable_name(ifrt::Executable* executable) {
    return cstr_from_string(executable->name());
}

extern "C" const char* ifrt_executable_fingerprint(ifrt::Executable* executable) {
    auto result = xla::ValueOrThrow(executable->fingerprint());
    if (!result.has_value()) return "";
    return cstr_from_string(result.value());
}

extern "C" const char* ifrt_executable_serialize(ifrt::Executable* executable) {
    return cstr_from_string(xla::ValueOrThrow(executable->Serialize()));
}

extern "C" int ifrt_executable_num_devices(ifrt::Executable* executable) {
    return executable->num_devices();
}

extern "C" int64_t ifrt_executable_size(ifrt::Executable* executable) {
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::ifrt::Executable::GetCompiledMemoryStats
// TODO xla::ifrt::Executable::GetParameterShardings
// TODO xla::ifrt::Executable::GetOutputShardings
// TODO xla::ifrt::Executable::GetParameterLayouts
// TODO xla::ifrt::Executable::GetOutputLayouts
// TODO xla::ifrt::Executable::GetHloModules
// TODO xla::ifrt::Executable::GetCostAnalysis

#pragma mark xla::ifrt::LoadedExecutable
extern "C" ifrt::Client* ifrt_loadedexecutable_client(ifrt::LoadedExecutable* executable) {
    return executable->client();
}

extern "C" const char* ifrt_loadedexecutable_name(ifrt::LoadedExecutable* executable) {
    return cstr_from_string(executable->name());
}

extern "C" const char* ifrt_loadedexecutable_fingerprint(ifrt::LoadedExecutable* executable) {
    auto result = xla::ValueOrThrow(executable->fingerprint());
    if (!result.has_value()) return "";
    return cstr_from_string(result.value());
}

extern "C" const char* ifrt_loadedexecutable_serialize(ifrt::LoadedExecutable* executable) {
    return cstr_from_string(xla::ValueOrThrow(executable->Serialize()));
}

extern "C" ifrt::Future<>* ifrt_loadedexecutable_get_ready_future(ifrt::LoadedExecutable* executable) {
    return executable->GetReadyFuture();
}

extern "C" int ifrt_loadedexecutable_num_devices(ifrt::LoadedExecutable* executable) {
    return executable->num_devices();
}

extern "C" int64_t ifrt_loadedexecutable_size(ifrt::LoadedExecutable* executable) {
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::ifrt::GetCompiledMemoryStats
// TODO xla::ifrt::GetParameterShardings
// TODO xla::ifrt::GetOutputShardings
// TODO xla::ifrt::GetParameterLayouts
// TODO xla::ifrt::GetOutputLayouts
// TODO xla::ifrt::GetHloModules
// TODO xla::ifrt::GetOutputMemoryKinds
// TODO xla::ifrt::GetCostAnalysis

// extern "C" ifrt::LoadedExecutable::ExecuteResult* ifrt_loadedexecutable_execute(ifrt::LoadedExecutable* executable, ifrt::Array** args, size_t args_size, ifrt::Array** results, size_t results_size, ifrt::Future<*>** futures, size_t futures_size) {
//     std::vector<ifrt::Array*> arguments(args, args + args_size);
//     std::vector<ifrt::Array*> result(results, results + results_size);
//     std::vector<ifrt::Future<*>*> future(futures, futures + futures_size);
//     return xla::ValueOrThrow(executable->Execute(arguments, result, future));
// }

extern "C" ifrt::Future<> ifrt_loadedexecutable_delete(ifrt::LoadedExecutable* executable) {
    return executable->Delete();
}

extern "C" bool ifrt_loadedexecutable_is_deleted(ifrt::LoadedExecutable* executable) {
    return executable->IsDeleted();
}

// TODO xla::ifrt::LoadedExecutable::addressable_device_logical_ids
// TODO xla::ifrt::LoadedExecutable::addressable_devices

// TODO auxiliary functions for xla::ifrt::LoadedExecutable::ExecuteResult

#pragma mark xla::ifrt::CustomCallProgram

// auxiliar functions
#pragma mark -
template<typename T>
const char* cstr_from_string(T text) {
    char* cstr = (char*)malloc(text.size() + 1);
    memcpy(cstr, text.data(), text.size());
    cstr[text.size()] = '\0';
    return cstr;
}

template<typename T>
T* unwrap_absl_statusor(absl::StatusOr<T> status, char** error_msg) {
    *error_msg = nullptr;
    if (!status.ok()) {
        auto str = pluginLoad.status().message();
        char* err = (char*)malloc(str.size()+1);
        memcpy(err, str.data(), str.size()+1);
        *error_msg = err;
        return nullptr;
    }
    return status.value();
}
