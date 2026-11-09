// Dynamic library (dlopen/dlsym) C ABI contract test for libReactantExtra.so.

#include "absl/debugging/leak_check.h"
#include <gtest/gtest.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Opaque C++ runtime handle passed across dynamically resolved C ABI calls.
struct LinkableRuntime;

// Minimal opaque C-API handles matching the MLIR C-API layout so the test can
// invoke libReactantExtra.so without linking against MLIR or ReactantExtra.
struct MlirContext {
  void *ptr;
};
struct MlirDialectRegistry {
  void *ptr;
};
struct MlirModule {
  void *ptr;
};
struct MlirStringRef {
  const char *data;
  size_t length;
};

} // namespace

// Locates libReactantExtra.so in the test runfiles tree across Google3 and OSS
// Bazel layouts. Returns an empty string if the shared library is not found.
static std::string findLibReactantExtraPath() {
  std::vector<std::string> candidates = {
      "third_party/reactant/jl/deps/ReactantExtra/libReactantExtra.so",
      "deps/ReactantExtra/libReactantExtra.so",
      "./libReactantExtra.so",
      "../libReactantExtra.so",
  };
  const char *test_srcdir = std::getenv("TEST_SRCDIR");
  const char *test_workspace = std::getenv("TEST_WORKSPACE");
  if (test_srcdir != nullptr && test_workspace != nullptr) {
    std::string base =
        std::string(test_srcdir) + "/" + std::string(test_workspace) + "/";
    candidates.push_back(
        base +
        "third_party/reactant/jl/deps/ReactantExtra/libReactantExtra.so");
    candidates.push_back(base + "deps/ReactantExtra/libReactantExtra.so");
    candidates.push_back(base + "libReactantExtra.so");
  }
  for (const std::string &path : candidates) {
    if (access(path.c_str(), R_OK) == 0)
      return path;
  }
  return "";
}

namespace {

// Test fixture that loads libReactantExtra.so once via dlopen and provides
// typed symbol resolution via dlsym.
class DlopenAbiTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    shared_library_path_ = findLibReactantExtraPath();
    ASSERT_FALSE(shared_library_path_.empty())
        << "Could not locate libReactantExtra.so in runfiles.";
    dlerror();
    handle_ = dlopen(shared_library_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    const char *error = dlerror();
    ASSERT_NE(handle_, nullptr) << "dlopen(" << shared_library_path_
                                << ") failed: " << (error ? error : "unknown");

    // InitializeLogs performs one-time process/logging setup inside the
    // dynamically loaded shared library whose internal static state outlives
    // individual tests.
    absl::LeakCheckDisabler disabler;
    auto initialize_logs =
        reinterpret_cast<void (*)()>(dlsym(handle_, "InitializeLogs"));
    ASSERT_NE(initialize_logs, nullptr);
    initialize_logs();
  }

  // Resolves `name` in `handle_` and casts it to `FunctionPointer`.
  template <typename FunctionPointer>
  FunctionPointer resolveSymbol(const char *name) {
    dlerror();
    void *symbol = dlsym(handle_, name);
    const char *error = dlerror();
    EXPECT_EQ(error, nullptr)
        << "dlsym(" << name << ") failed: " << (error ? error : "");
    EXPECT_NE(symbol, nullptr) << "Symbol " << name << " resolved to nullptr";
    return reinterpret_cast<FunctionPointer>(symbol);
  }

  // Resolved path to libReactantExtra.so in runfiles.
  static std::string shared_library_path_;
  // Process-wide dlopen handle for libReactantExtra.so.
  static void *handle_;
};

std::string DlopenAbiTest::shared_library_path_;
void *DlopenAbiTest::handle_ = nullptr;

} // namespace

TEST_F(DlopenAbiTest, ExportedCSymbolsAreUnmangledAndResolvable) {
  // clang-format off
  static constexpr const char *kRequiredSymbols[] = {
      "ArrayFromHostBuffer",
      "BufferNDimensions",
      "BufferOnCPU",
      "BufferPrimitiveType",
      "BufferShape",
      "BufferToClient",
      "BufferToDevice",
      "BufferToHost",
      "ClientCompile",
      "ClientCompileWithProto",
      "ClientGetAddressableDevice",
      "ClientGetAddressableDevices",
      "ClientGetDevice",
      "ClientGetDevices",
      "ClientGetPlatformName",
      "ClientNumAddressableDevices",
      "ClientNumDevices",
      "ClientProcessIndex",
      "ConvertLLVMBCToMLIR",
      "ConvertLLVMStrToMLIR",
      "CopyBufferToDevice",
      "CopyFromBuffer",
      "CopyToBuffer",
      "DeviceGetKind",
      "DeviceToClient",
      "ExecutableFree",
      "FreeClient",
      "FreeFuture",
      "FreeHloModule",
      "FutureAwait",
      "FutureIsReady",
      "GetCpuPjrtApi",
      "GetDefaultTargetTriple",
      "HloModuleToString",
      "InitializeLogs",
      "InitializePasses",
      "InitializeRegistry",
      "LinkInModule",
      "MakeCPUClient",
      "MakeClientFromApi",
      "PjRtBufferFree",
      "PjRtBufferGetReadyFuture",
      "PjRtClientLoadSerializedExecutable",
      "PjRtDeviceGetGlobalDeviceId",
      "PjRtDeviceGetLocalDeviceId",
      "PjRtDeviceGetLocalHardwareId",
      "PjRtLoadedExecutableGetClient",
      "PjRtLoadedExecutableGetCompiledMemoryStats",
      "PjRtLoadedExecutableGetHloModules",
      "PjRtLoadedExecutableGetOuputShardings",
      "PjRtLoadedExecutableGetParameterShardings",
      "PjRtLoadedExecutableNumPartitions",
      "PjRtLoadedExecutableNumReplicas",
      "PjRtLoadedExecutableSerialize",
      "ReactantFuncSetArgAttr",
      "ReactantFuncSetResultAttr",
      "ReactantGetCompileOptions",
      "ReactantGetDebugOptions",
      "ReactantLLVMParseCommandLineOptions",
      "ReactantLexMLIR",
      "ReactantThrowError",
      "RegisterCustomCallTarget",
      "RegisterDialects",
      "RegisterEnzymeXLACPUHandler",
      "RegisterEnzymeXLAGPUHandler",
      "SetLogLevel",
      "SetModuleLogLevel",
      "UninitPJRTBuffer",
      "UnsafeBufferPointer",
      "XLAExecute",
      "XLAExecuteSharded",
      "addSdyPropagationPipeline",
      "convertMlirModuleToHloModule",
      "dump_mval",
      "dump_op",
      "dump_string",
      "enzymeActivityAttrGet",
      "enzymeRngDistributionAttrGet",
      "enzymeSupportAttrGet",
      "enzymeSymbolAttrGet",
      "free_hlo_sharding",
      "free_ifrt_memory_kind",
      "free_ifrt_sharding",
      "free_op_sharding",
      "hloComputationGetInstructionsPostOrder",
      "hloComputationInstructionCount",
      "hloComputationToString",
      "hloFusionKindToString",
      "hloInstructionFusedInstructionsComputation",
      "hloInstructionGetFusionKind",
      "hloInstructionGetOpcode",
      "hloInstructionGetToApply",
      "hloInstructionHasToApply",
      "hloInstructionIsFusion",
      "hloInstructionToString",
      "hloModuleGetEntryComputation",
      "hloOpcodeToString",
      "hloShardingFromTensorShardingAttr",
      "hloShardingToTensorShardingAttr",
      "hlo_sharding_check_eq",
      "hlo_sharding_check_eq_ignoring_metadata",
      "hlo_sharding_from_op_sharding",
      "hlo_sharding_is_manual",
      "hlo_sharding_is_maximal",
      "hlo_sharding_is_replicated",
      "hlo_sharding_is_tiled",
      "hlo_sharding_is_tuple",
      "hlo_sharding_is_unknown",
      "hlo_sharding_replicate_on_last_tile_dim",
      "hlo_sharding_tile_assignment_devices",
      "hlo_sharding_tile_assignment_devices_size",
      "hlo_sharding_tile_assignment_dimensions",
      "hlo_sharding_tile_assignment_dimensions_size",
      "hlo_sharding_to_op_sharding",
      "hlo_sharding_to_string",
      "ifrt_ClientGetAddressableDevice",
      "ifrt_ClientGetDevice",
      "ifrt_ClientGetPlatformName",
      "ifrt_ClientProcessIndex",
      "ifrt_CopyArrayToHostBuffer",
      "ifrt_DeviceGetDefaultMemory",
      "ifrt_DeviceGetGlobalDeviceId",
      "ifrt_DeviceGetKind",
      "ifrt_DeviceGetLocalHardwareId",
      "ifrt_DeviceGetMemories",
      "ifrt_DeviceIsAddressable",
      "ifrt_DeviceToClient",
      "ifrt_FreeClient",
      "ifrt_MemoryGetMemoryKind",
      "ifrt_MemoryKindToString",
      "ifrt_MemoryKindsAreEqual",
      "ifrt_MemoryToString",
      "ifrt_array_copy_to_host_buffer",
      "ifrt_array_disassemble_into_single_device_arrays",
      "ifrt_array_ndims",
      "ifrt_array_shape",
      "ifrt_array_to_client",
      "ifrt_array_to_sharding",
      "ifrt_client_addressable_device_count",
      "ifrt_client_addressable_devices",
      "ifrt_client_all_devices",
      "ifrt_client_device_count",
      "ifrt_client_devices",
      "ifrt_client_load_serialized_executable",
      "ifrt_client_lookup_addressable_device",
      "ifrt_client_lookup_device",
      "ifrt_client_make_single_shard_array_from_host_buffer",
      "ifrt_compile",
      "ifrt_compile_with_proto",
      "ifrt_copy_array",
      "ifrt_free_array",
      "ifrt_free_future",
      "ifrt_future_await",
      "ifrt_future_is_ready",
      "ifrt_hlo_module_cost_analysis_properties",
      "ifrt_loaded_executable_client",
      "ifrt_loaded_executable_dtor",
      "ifrt_loaded_executable_execute",
      "ifrt_loaded_executable_get_compiled_memory_stats",
      "ifrt_loaded_executable_get_hlo_modules",
      "ifrt_loaded_executable_num_devices",
      "ifrt_loaded_executable_serialize",
      "ifrt_make_pjrt_cpu_client",
      "ifrt_memory_kind_from_string",
      "ifrt_memory_kind_has_value",
      "ifrt_memory_kind_with_optional_memory_space",
      "ifrt_sharding_devices_size",
      "ifrt_sharding_from_xla_hlo_sharding",
      "ifrt_sharding_is_fully_replicated",
      "ifrt_sharding_is_single_device_sharding",
      "ifrt_sharding_to_device_list",
      "ifrt_sharding_to_index_domains",
      "ifrt_sharding_to_string",
      "ifrt_sharding_to_xla_hlo_sharding",
      "mlirContextAppendDialectRegistry",
      "mlirContextCreate",
      "mlirContextDestroy",
      "mlirDialectRegistryCreate",
      "mlirDialectRegistryDestroy",
      "mlirGetFunctionTypeFromOperation",
      "mlirGetParentOfTypeFunctionOp",
      "mlirIsFunctionOpInterface",
      "mlirModuleCreateParse",
      "mlirModuleDestroy",
      "mlirModuleGetOperation",
      "mlirOperationInject",
      "mlirOperationParse",
      "mlirPassManagerCreate",
      "mlirPassManagerDestroy",
      "mlirPassManagerEnableXLATraceTiming",
      "mlirPassManagerRunOnOp",
      "op_sharding_has_iota_reshape_dims",
      "op_sharding_has_iota_transpose_perm",
      "op_sharding_has_last_tile_dims",
      "op_sharding_has_tile_assignment_devices",
      "op_sharding_has_tile_assignment_dimensions",
      "op_sharding_iota_reshape_dims",
      "op_sharding_iota_reshape_dims_size",
      "op_sharding_iota_transpose_perm",
      "op_sharding_iota_transpose_perm_size",
      "op_sharding_is_shard_group",
      "op_sharding_replicate_on_last_tile_dim",
      "op_sharding_tile_assignment_devices",
      "op_sharding_tile_assignment_devices_size",
      "op_sharding_tile_assignment_dimensions",
      "op_sharding_tile_assignment_dimensions_size",
      "op_sharding_to_op_sharding_type",
      "parseAndReturnUnverifiedHloModule",
      "pjrt_device_is_addressable",
      "pjrt_hlo_module_cost_analysis_properties",
      "reactantXLADeInit",
      "reactantXLAExec",
      "reactantXLAFree",
      "reactantXLAInit",
      "reactantXLAMalloc",
      "reactantXLAMemcpy",
      "reactantXLAThrow",
      "registerEnzymeJaXXLAFFI",
      "registerReactantXLAFFI",
      "sdyMeshAttrGet",
  };
  // clang-format on

  for (const char *symbol_name : kRequiredSymbols) {
    dlerror();
    void *symbol = dlsym(handle_, symbol_name);
    const char *error = dlerror();
    EXPECT_EQ(error, nullptr)
        << "dlsym(" << symbol_name << ") error: " << (error ? error : "");
    EXPECT_NE(symbol, nullptr) << "Symbol " << symbol_name << " is null";
  }
}

TEST_F(DlopenAbiTest, DynamicInvokeLinkableRuntimeAndMlirHlo) {
  auto initialize_logs = resolveSymbol<void (*)()>("InitializeLogs");
  auto xla_init = resolveSymbol<void (*)(LinkableRuntime **, const char *)>(
      "reactantXLAInit");
  auto xla_deinit =
      resolveSymbol<void (*)(LinkableRuntime **)>("reactantXLADeInit");
  auto xla_malloc =
      resolveSymbol<void *(*)(LinkableRuntime **, uint64_t, uint64_t,
                              uint64_t *)>("reactantXLAMalloc");
  auto xla_free =
      resolveSymbol<void (*)(LinkableRuntime **, void *)>("reactantXLAFree");
  auto xla_memcpy =
      resolveSymbol<void (*)(LinkableRuntime **, void *, void *, size_t,
                             int32_t)>("reactantXLAMemcpy");
  auto xla_exec =
      resolveSymbol<void (*)(LinkableRuntime **, const char *, int64_t, void **,
                             int64_t, const int64_t *)>("reactantXLAExec");

  initialize_logs();
  LinkableRuntime *runtime = nullptr;
  xla_init(&runtime, "cpu");
  ASSERT_NE(runtime, nullptr);

  constexpr uint64_t kPrimitiveTypeF32 = 11;
  constexpr int32_t kMemcpyHostToDevice = 1;
  constexpr int32_t kMemcpyDeviceToHost = 2;
  uint64_t num_elements = 4;
  constexpr size_t kSizeInBytes = 4 * sizeof(float);
  void *device_output =
      xla_malloc(&runtime, kPrimitiveTypeF32, /*rank=*/1, &num_elements);
  void *device_x =
      xla_malloc(&runtime, kPrimitiveTypeF32, /*rank=*/1, &num_elements);
  void *device_y =
      xla_malloc(&runtime, kPrimitiveTypeF32, /*rank=*/1, &num_elements);
  ASSERT_NE(device_output, nullptr);
  ASSERT_NE(device_x, nullptr);
  ASSERT_NE(device_y, nullptr);

  float host_x[4] = {1.5f, 2.5f, 3.5f, 4.5f};
  float host_y[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  float host_output[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  xla_memcpy(&runtime, device_output, host_output, kSizeInBytes,
             kMemcpyHostToDevice);
  xla_memcpy(&runtime, device_x, host_x, kSizeInBytes, kMemcpyHostToDevice);
  xla_memcpy(&runtime, device_y, host_y, kSizeInBytes, kMemcpyHostToDevice);

  static constexpr const char *kLinearCombinationMlir = R"mlir(
    func.func @main(%out: tensor<?xf32>, %x: tensor<?xf32>, %y: tensor<?xf32>)
        -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      %doubled = stablehlo.add %x, %x : tensor<?xf32>
      %sum = stablehlo.add %doubled, %y : tensor<?xf32>
      return %sum, %x, %y : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
    }
  )mlir";

  void *arguments[3] = {device_output, device_x, device_y};
  xla_exec(&runtime, kLinearCombinationMlir, /*num_arguments=*/3, arguments,
           /*num_constants=*/0, /*constants=*/nullptr);

  xla_memcpy(&runtime, host_output, device_output, kSizeInBytes,
             kMemcpyDeviceToHost);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(host_output[i], 2.0f * host_x[i] + host_y[i]);
  }

  xla_free(&runtime, device_output);
  xla_free(&runtime, device_x);
  xla_free(&runtime, device_y);
  xla_deinit(&runtime);
  EXPECT_EQ(runtime, nullptr);

  auto context_create = resolveSymbol<MlirContext (*)()>("mlirContextCreate");
  auto context_destroy =
      resolveSymbol<void (*)(MlirContext)>("mlirContextDestroy");
  auto registry_create =
      resolveSymbol<MlirDialectRegistry (*)()>("mlirDialectRegistryCreate");
  auto registry_destroy = resolveSymbol<void (*)(MlirDialectRegistry)>(
      "mlirDialectRegistryDestroy");
  auto context_append_registry =
      resolveSymbol<void (*)(MlirContext, MlirDialectRegistry)>(
          "mlirContextAppendDialectRegistry");
  auto initialize_registry =
      resolveSymbol<void (*)(MlirDialectRegistry)>("InitializeRegistry");
  auto register_dialects =
      resolveSymbol<void (*)(MlirContext)>("RegisterDialects");
  auto initialize_passes =
      resolveSymbol<void (*)(MlirDialectRegistry)>("InitializePasses");
  auto module_parse = resolveSymbol<MlirModule (*)(MlirContext, MlirStringRef)>(
      "mlirModuleCreateParse");
  auto module_destroy =
      resolveSymbol<void (*)(MlirModule)>("mlirModuleDestroy");
  auto convert_mlir_to_hlo =
      resolveSymbol<void *(*)(MlirModule)>("convertMlirModuleToHloModule");
  auto hlo_get_entry_computation =
      resolveSymbol<void *(*)(void *)>("hloModuleGetEntryComputation");
  auto hlo_instruction_count =
      resolveSymbol<int64_t (*)(void *)>("hloComputationInstructionCount");
  auto free_hlo_module = resolveSymbol<void (*)(void *)>("FreeHloModule");

  MlirDialectRegistry registry = registry_create();
  initialize_registry(registry);
  initialize_passes(registry);

  MlirContext context = context_create();
  context_append_registry(context, registry);
  register_dialects(context);

  static constexpr const char *kExponentialModule = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.exponential %arg0 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
  )mlir";
  MlirStringRef module_ref{kExponentialModule, std::strlen(kExponentialModule)};
  MlirModule module = module_parse(context, module_ref);
  ASSERT_NE(module.ptr, nullptr);

  void *hlo_module = convert_mlir_to_hlo(module);
  ASSERT_NE(hlo_module, nullptr);
  void *entry_computation = hlo_get_entry_computation(hlo_module);
  ASSERT_NE(entry_computation, nullptr);
  EXPECT_GE(hlo_instruction_count(entry_computation), 2);

  free_hlo_module(hlo_module);
  module_destroy(module);
  context_destroy(context);
  registry_destroy(registry);
}
