// Unit tests for XLA Typed FFI host callback registration and execution in
// xla_ffi.cpp.

#include "test/reactant_extra_test.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using ::reactant_test::makeStringRef;
using ::reactant_test::ScopedErrorHandler;
using ::reactant_test::XlaMemcpyDirection;
using ::reactant_test::XlaPrimitiveType;
using ::testing::ElementsAre;

static constexpr int32_t kHostBackendId = 1;

static int32_t g_last_backend = 0;
static void *g_last_stream = reinterpret_cast<void *>(0x1);
static int g_callback_invocations = 0;

// Host callback computing `output0 = 2 * input0 + input1` and
// `output1 = input0 - input1` across four f32 elements while recording the
// received backend ID and stream pointer.
extern "C" bool multiInputOutputCallback(void **inputs, void **outputs,
                                         int32_t backend, void *stream) {
  g_last_backend = backend;
  g_last_stream = stream;
  ++g_callback_invocations;

  const float *input0 = static_cast<const float *>(inputs[0]);
  const float *input1 = static_cast<const float *>(inputs[1]);
  float *output0 = static_cast<float *>(outputs[0]);
  float *output1 = static_cast<float *>(outputs[1]);
  for (int i = 0; i < 4; ++i) {
    output0[i] = 2.0f * input0[i] + input1[i];
    output1[i] = input0[i] - input1[i];
  }
  return true;
}

// Host callback taking zero inputs and populating a 3-element i32 output.
extern "C" bool zeroInputFillCallback(void ** /*inputs*/, void **outputs,
                                      int32_t backend, void *stream) {
  if (backend != kHostBackendId || stream != nullptr)
    return false;
  int32_t *output0 = static_cast<int32_t *>(outputs[0]);
  output0[0] = 42;
  output0[1] = 43;
  output0[2] = 44;
  return true;
}

// Host callback that unconditionally reports failure.
extern "C" bool failingCallback(void ** /*inputs*/, void ** /*outputs*/,
                                int32_t /*backend*/, void * /*stream*/) {
  return false;
}

// Legacy XLA CPU custom call computing `destination = 3 * source + 1`.
extern "C" void legacyCpuCustomCall(void *destination_buffer,
                                    const void **source_buffers) {
  const float *source = static_cast<const float *>(source_buffers[0]);
  float *destination = static_cast<float *>(destination_buffer);
  for (int i = 0; i < 4; ++i) {
    destination[i] = source[i] * 3.0f + 1.0f;
  }
}

// Builds a StableHLO module invoking `@reactant_julia_callback` with two
// `tensor<4xf32>` inputs and two `tensor<4xf32>` outputs.
static std::string makeTwoInputCallbackModule(int64_t callback_address) {
  return R"mlir(
module @ffi_two_in_two_out {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = stablehlo.custom_call @reactant_julia_callback(%arg0, %arg1) {
      api_version = 4 : i32,
      backend_config = {callback_ptr = )mlir" +
         std::to_string(callback_address) + R"mlir( : i64}
    } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
}
)mlir";
}

// Builds a StableHLO module invoking `@reactant_julia_callback` with a single
// `tensor<4xf32>` input and a single `tensor<4xf32>` output.
static std::string makeSingleInputCallbackModule(int64_t callback_address) {
  return R"mlir(
module @ffi_one_in_one_out {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.custom_call @reactant_julia_callback(%arg0) {
      api_version = 4 : i32,
      backend_config = {callback_ptr = )mlir" +
         std::to_string(callback_address) + R"mlir( : i64}
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
}

TEST(XlaFfiTest, HostJuliaCallbackMultiInputMultiOutput) {
  ScopedErrorHandler error_guard;
  InitializeLogs();
  registerReactantXLAFFI();
  registerReactantXLAFFI();

  void *client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  ASSERT_NE(client, nullptr);
  void *device = ClientGetAddressableDevice(client, 0);
  ASSERT_NE(device, nullptr);

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);

  int64_t callback_address =
      reinterpret_cast<int64_t>(&multiInputOutputCallback);
  std::string mlir_source = makeTwoInputCallbackModule(callback_address);
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(mlir_source.c_str()));
  ASSERT_NE(module.ptr, nullptr);

  void *executable = ClientCompile(
      client, module, /*device_id=*/0, /*mesh_ids=*/nullptr,
      /*num_mesh_ids=*/0, /*xla_gpu_cuda_data_dir=*/"",
      /*use_shardy_partitioner=*/false, /*num_replicas=*/1,
      /*num_partitions=*/1, /*use_spmd_partitioning=*/false,
      /*kernel_cache_enabled=*/false, /*kernel_cache_path=*/"",
      /*autotune_cache_enabled=*/false, /*autotune_cache_path=*/"",
      /*process_id=*/0, /*enable_enzyme_comms=*/false);
  ASSERT_NE(executable, nullptr);

  float host_input0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_input1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t shape[1] = {4};
  void *buffer0 =
      ArrayFromHostBuffer(client, host_input0, XlaPrimitiveType::kF32,
                          /*num_dimensions=*/1, shape, device);
  void *buffer1 =
      ArrayFromHostBuffer(client, host_input1, XlaPrimitiveType::kF32,
                          /*num_dimensions=*/1, shape, device);

  g_last_backend = 0;
  g_last_stream = reinterpret_cast<void *>(0x1);
  g_callback_invocations = 0;

  void *arguments[2] = {buffer0, buffer1};
  uint8_t is_donatable[2] = {0, 0};
  void *results[2] = {nullptr, nullptr};
  uint8_t has_futures = 0;
  void *future_handles[2] = {nullptr, nullptr};
  XLAExecuteSharded(executable, /*num_arguments=*/2, arguments, device,
                    is_donatable, /*num_results=*/2, results, &has_futures,
                    future_handles);
  if (has_futures) {
    for (int i = 0; i < 2; ++i) {
      if (future_handles[i] != nullptr) {
        FutureAwait(future_handles[i]);
        FreeFuture(future_handles[i]);
      }
    }
  }
  EXPECT_TRUE(ScopedErrorHandler::error().empty())
      << ScopedErrorHandler::error();
  EXPECT_EQ(g_callback_invocations, 1);
  EXPECT_EQ(g_last_backend, kHostBackendId);
  EXPECT_EQ(g_last_stream, nullptr);

  float host_output0[4] = {0.0f};
  float host_output1[4] = {0.0f};
  BufferToHost(results[0], host_output0);
  BufferToHost(results[1], host_output1);
  EXPECT_THAT(host_output0, ElementsAre(12.0f, 24.0f, 36.0f, 48.0f));
  EXPECT_THAT(host_output1, ElementsAre(-9.0f, -18.0f, -27.0f, -36.0f));

  PjRtBufferFree(results[0]);
  PjRtBufferFree(results[1]);
  PjRtBufferFree(buffer0);
  PjRtBufferFree(buffer1);
  ExecutableFree(executable);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  FreeClient(client);
}

TEST(XlaFfiTest, HostJuliaCallbackZeroInputsSingleOutput) {
  ScopedErrorHandler error_guard;
  InitializeLogs();
  registerReactantXLAFFI();

  void *client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  ASSERT_NE(client, nullptr);
  void *device = ClientGetAddressableDevice(client, 0);
  ASSERT_NE(device, nullptr);

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);

  int64_t callback_address = reinterpret_cast<int64_t>(&zeroInputFillCallback);
  std::string mlir_source = R"mlir(
module @ffi_zero_in_one_out {
  func.func @main() -> tensor<3xi32> {
    %0 = stablehlo.custom_call @reactant_julia_callback() {
      api_version = 4 : i32,
      backend_config = {callback_ptr = )mlir" +
                            std::to_string(callback_address) + R"mlir( : i64}
    } : () -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}
)mlir";
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(mlir_source.c_str()));
  ASSERT_NE(module.ptr, nullptr);

  void *executable = ClientCompile(
      client, module, /*device_id=*/0, /*mesh_ids=*/nullptr,
      /*num_mesh_ids=*/0, /*xla_gpu_cuda_data_dir=*/"",
      /*use_shardy_partitioner=*/false, /*num_replicas=*/1,
      /*num_partitions=*/1, /*use_spmd_partitioning=*/false,
      /*kernel_cache_enabled=*/false, /*kernel_cache_path=*/"",
      /*autotune_cache_enabled=*/false, /*autotune_cache_path=*/"",
      /*process_id=*/0, /*enable_enzyme_comms=*/false);
  ASSERT_NE(executable, nullptr);

  void *results[1] = {nullptr};
  uint8_t has_futures = 0;
  void *future_handles[1] = {nullptr};
  XLAExecuteSharded(executable, /*num_arguments=*/0, /*arguments=*/nullptr,
                    device, /*is_argument_donatable=*/nullptr,
                    /*num_results=*/1, results, &has_futures, future_handles);
  if (has_futures && future_handles[0] != nullptr) {
    FutureAwait(future_handles[0]);
    FreeFuture(future_handles[0]);
  }
  EXPECT_TRUE(ScopedErrorHandler::error().empty())
      << ScopedErrorHandler::error();

  int32_t host_output[3] = {0, 0, 0};
  BufferToHost(results[0], host_output);
  EXPECT_THAT(host_output, ElementsAre(42, 43, 44));

  PjRtBufferFree(results[0]);
  ExecutableFree(executable);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  FreeClient(client);
}

// Prints `message` to stderr and aborts so death tests can match the error.
extern "C" void abortOnReactantError(const char *message) {
  std::fprintf(stderr, "ReactantError: %s\n", message ? message : "(null)");
  std::fflush(stderr);
  std::abort();
}

// Compiles and executes a single-input `@reactant_julia_callback` with
// `callback_address` while `ReactantThrowError` is configured to abort.
static void runSingleInputCallbackExpectingError(int64_t callback_address) {
  InitializeLogs();
  registerReactantXLAFFI();
  ReactantThrowError = &abortOnReactantError;

  void *client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  void *device = ClientGetAddressableDevice(client, 0);

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);

  std::string mlir_source = makeSingleInputCallbackModule(callback_address);
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(mlir_source.c_str()));

  void *executable = ClientCompile(
      client, module, /*device_id=*/0, /*mesh_ids=*/nullptr,
      /*num_mesh_ids=*/0, /*xla_gpu_cuda_data_dir=*/"",
      /*use_shardy_partitioner=*/false, /*num_replicas=*/1,
      /*num_partitions=*/1, /*use_spmd_partitioning=*/false,
      /*kernel_cache_enabled=*/false, /*kernel_cache_path=*/"",
      /*autotune_cache_enabled=*/false, /*autotune_cache_path=*/"",
      /*process_id=*/0, /*enable_enzyme_comms=*/false);

  float host_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[1] = {4};
  void *buffer = ArrayFromHostBuffer(client, host_input, XlaPrimitiveType::kF32,
                                     /*num_dimensions=*/1, shape, device);

  void *arguments[1] = {buffer};
  uint8_t is_donatable[1] = {0};
  void *results[1] = {nullptr};
  uint8_t futures[1] = {0};
  void *future_handles[1] = {nullptr};
  XLAExecuteSharded(executable, /*num_arguments=*/1, arguments, device,
                    is_donatable, /*num_results=*/1, results, futures,
                    future_handles);
  if (futures[0] && future_handles[0] != nullptr) {
    FutureAwait(future_handles[0]);
    FreeFuture(future_handles[0]);
  }
}

TEST(XlaFfiTest, NullCallbackPointerPropagatesError) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(runSingleInputCallbackExpectingError(0),
               "reactant_julia_callback: null callback pointer");
}

TEST(XlaFfiTest, CallbackReturningFalsePropagatesError) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  int64_t callback_address = reinterpret_cast<int64_t>(&failingCallback);
  EXPECT_DEATH(runSingleInputCallbackExpectingError(callback_address),
               "reactant_julia_callback: callback returned false");
}

TEST(XlaFfiTest, HostJuliaCallbackViaLinkableRuntimeAndLegacyRegistration) {
  ScopedErrorHandler error_guard;
  InitializeLogs();
  registerReactantXLAFFI();
  RegisterCustomCallTarget("reactant_test_legacy_target",
                           reinterpret_cast<void *>(&legacyCpuCustomCall),
                           "Host");

  LinkableRuntime *runtime = nullptr;
  reactantXLAInit(&runtime, "cpu");
  ASSERT_NE(runtime, nullptr);

  uint64_t shape[1] = {4};
  void *buffer0 =
      reactantXLAMalloc(&runtime, XlaPrimitiveType::kF32, /*rank=*/1, shape);
  void *buffer1 =
      reactantXLAMalloc(&runtime, XlaPrimitiveType::kF32, /*rank=*/1, shape);
  void *output_buffer0 =
      reactantXLAMalloc(&runtime, XlaPrimitiveType::kF32, /*rank=*/1, shape);
  void *output_buffer1 =
      reactantXLAMalloc(&runtime, XlaPrimitiveType::kF32, /*rank=*/1, shape);

  float host_input0[4] = {2.0f, 4.0f, 6.0f, 8.0f};
  float host_input1[4] = {1.0f, 3.0f, 5.0f, 7.0f};
  reactantXLAMemcpy(&runtime, buffer0, host_input0, sizeof(host_input0),
                    XlaMemcpyDirection::kMemcpyHostToDevice);
  reactantXLAMemcpy(&runtime, buffer1, host_input1, sizeof(host_input1),
                    XlaMemcpyDirection::kMemcpyHostToDevice);

  int64_t callback_address =
      reinterpret_cast<int64_t>(&multiInputOutputCallback);
  std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>,
                %arg2: tensor<?xf32>, %arg3: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %in0 = stablehlo.dynamic_slice %arg0, %c0, sizes = [4]
      : (tensor<?xf32>, tensor<i64>) -> tensor<4xf32>
  %in1 = stablehlo.dynamic_slice %arg1, %c0, sizes = [4]
      : (tensor<?xf32>, tensor<i64>) -> tensor<4xf32>
  %res:2 = stablehlo.custom_call @reactant_julia_callback(%in0, %in1) {
    api_version = 4 : i32,
    backend_config = {callback_ptr = )mlir" +
                            std::to_string(callback_address) + R"mlir( : i64}
  } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %upd0 = stablehlo.dynamic_update_slice %arg2, %res#0, %c0
      : (tensor<?xf32>, tensor<4xf32>, tensor<i64>) -> tensor<?xf32>
  %upd1 = stablehlo.dynamic_update_slice %arg3, %res#1, %c0
      : (tensor<?xf32>, tensor<4xf32>, tensor<i64>) -> tensor<?xf32>
  return %arg0, %arg1, %upd0, %upd1
      : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}
)mlir";

  void *arguments[4] = {buffer0, buffer1, output_buffer0, output_buffer1};
  reactantXLAExec(&runtime, mlir_source.c_str(), /*num_arguments=*/4, arguments,
                  /*num_constants=*/0, /*constants=*/nullptr);
  EXPECT_TRUE(ScopedErrorHandler::error().empty())
      << ScopedErrorHandler::error();

  float host_output0[4] = {0.0f};
  float host_output1[4] = {0.0f};
  reactantXLAMemcpy(&runtime, host_output0, output_buffer0,
                    sizeof(host_output0),
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  reactantXLAMemcpy(&runtime, host_output1, output_buffer1,
                    sizeof(host_output1),
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  EXPECT_THAT(host_output0, ElementsAre(5.0f, 11.0f, 17.0f, 23.0f));
  EXPECT_THAT(host_output1, ElementsAre(1.0f, 1.0f, 1.0f, 1.0f));

  reactantXLAFree(&runtime, output_buffer1);
  reactantXLAFree(&runtime, output_buffer0);
  reactantXLAFree(&runtime, buffer1);
  reactantXLAFree(&runtime, buffer0);
  reactantXLADeInit(&runtime);
}
