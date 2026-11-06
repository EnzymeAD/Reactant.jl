// Unit tests for PJRT and IFRT CPU client, buffer, compilation, and execution
// C ABI entry points in API.cpp.

#include "test/reactant_extra_test.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using ::reactant_test::makeStringRef;
using ::reactant_test::ScopedErrorHandler;
using ::reactant_test::XlaPrimitiveType;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

static constexpr const char *kAddMultiplyModule = R"mlir(
module @reactant_pjrt_ifrt_test {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %sum = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %product = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
    return %sum, %product : tensor<4xf32>, tensor<4xf32>
  }
}
)mlir";

TEST(PjRtCpuTest, ClientAndDeviceQueries) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  void *client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(ScopedErrorHandler::error().empty());

  int num_devices = ClientNumDevices(client);
  int num_addressable_devices = ClientNumAddressableDevices(client);
  EXPECT_GT(num_devices, 0);
  EXPECT_EQ(num_devices, num_addressable_devices);
  EXPECT_EQ(ClientProcessIndex(client), 0);

  const char *platform_name = ClientGetPlatformName(client);
  ASSERT_NE(platform_name, nullptr);
  EXPECT_GT(std::string(platform_name).size(), 0u);
  free(const_cast<char *>(platform_name));

  std::vector<void *> devices(num_devices, nullptr);
  std::vector<void *> addressable_devices(num_addressable_devices, nullptr);
  ClientGetDevices(client, devices.data());
  ClientGetAddressableDevices(client, addressable_devices.data());
  ASSERT_NE(devices[0], nullptr);
  EXPECT_EQ(devices[0], addressable_devices[0]);

  void *device = ClientGetAddressableDevice(client, 0);
  ASSERT_EQ(device, devices[0]);
  int64_t global_id = PjRtDeviceGetGlobalDeviceId(device);
  EXPECT_EQ(ClientGetDevice(client, static_cast<int>(global_id)), device);
  EXPECT_EQ(DeviceToClient(device), client);
  EXPECT_TRUE(pjrt_device_is_addressable(device));
  EXPECT_GE(PjRtDeviceGetLocalDeviceId(device), 0);
  EXPECT_GE(PjRtDeviceGetLocalHardwareId(device), 0);

  const char *device_kind = DeviceGetKind(device);
  ASSERT_NE(device_kind, nullptr);
  EXPECT_GT(std::string(device_kind).size(), 0u);
  free(const_cast<char *>(device_kind));

  char *target_triple = GetDefaultTargetTriple();
  ASSERT_NE(target_triple, nullptr);
  EXPECT_GT(std::string(target_triple).size(), 0u);
  free(target_triple);

  FreeClient(client);
}

TEST(PjRtCpuTest, CpuPluginCApiClient) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  const PJRT_Api *api = GetCpuPjrtApi();
  ASSERT_NE(api, nullptr);

  const char *error = nullptr;
  void *client = MakeClientFromApi(api, "cpu_c_api", "cpu_c_api", &error);
  if (error != nullptr) {
    std::string error_message(error);
    free(const_cast<char *>(error));
    FAIL() << "MakeClientFromApi failed: " << error_message;
  }
  ASSERT_NE(client, nullptr);
  EXPECT_GT(ClientNumDevices(client), 0);

  FreeClient(client);
}

TEST(PjRtCpuTest, BufferTransfersAndProperties) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  void *client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  ASSERT_NE(client, nullptr);
  void *device = ClientGetAddressableDevice(client, 0);
  ASSERT_NE(device, nullptr);

  float host_input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t shape[2] = {2, 3};
  void *buffer = ArrayFromHostBuffer(client, host_input, XlaPrimitiveType::kF32,
                                     /*num_dimensions=*/2, shape, device);
  ASSERT_NE(buffer, nullptr);

  EXPECT_EQ(BufferNDimensions(buffer), 2);
  const int64_t *buffer_shape = BufferShape(buffer);
  ASSERT_NE(buffer_shape, nullptr);
  EXPECT_EQ(buffer_shape[0], 2);
  EXPECT_EQ(buffer_shape[1], 3);
  EXPECT_EQ(BufferPrimitiveType(buffer),
            static_cast<int32_t>(XlaPrimitiveType::kF32));
  EXPECT_NE(BufferOnCPU(buffer), 0u);
  EXPECT_EQ(BufferToDevice(buffer), device);
  EXPECT_EQ(BufferToClient(buffer), client);

  void *ready_future = PjRtBufferGetReadyFuture(buffer);
  ASSERT_NE(ready_future, nullptr);
  FutureAwait(ready_future);
  EXPECT_NE(FutureIsReady(ready_future), 0u);
  FreeFuture(ready_future);

  float host_output[6] = {0.0f};
  BufferToHost(buffer, host_output);
  EXPECT_THAT(host_output, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f));

  const float *unsafe_pointer =
      static_cast<const float *>(UnsafeBufferPointer(buffer));
  ASSERT_NE(unsafe_pointer, nullptr);
  EXPECT_FLOAT_EQ(unsafe_pointer[0], 1.0f);
  EXPECT_FLOAT_EQ(unsafe_pointer[5], 6.0f);

  uint64_t uninitialized_shape[1] = {4};
  void *uninitialized_buffer = UninitPJRTBuffer(
      client, device, XlaPrimitiveType::kF32, /*rank=*/1, uninitialized_shape);
  ASSERT_NE(uninitialized_buffer, nullptr);

  float patch_full[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  CopyToBuffer(client, uninitialized_buffer, patch_full,
               /*offset_in_bytes=*/0, sizeof(patch_full),
               &uninitialized_buffer);
  float patch_middle[2] = {200.0f, 300.0f};
  CopyToBuffer(client, uninitialized_buffer, patch_middle,
               /*offset_in_bytes=*/sizeof(float), sizeof(patch_middle),
               &uninitialized_buffer);

  float read_full[4] = {0.0f};
  CopyFromBuffer(client, uninitialized_buffer, read_full,
                 /*offset_in_bytes=*/0, sizeof(read_full),
                 &uninitialized_buffer);
  EXPECT_THAT(read_full, ElementsAre(10.0f, 200.0f, 300.0f, 40.0f));

  float read_slice[2] = {0.0f};
  CopyFromBuffer(client, uninitialized_buffer, read_slice,
                 /*offset_in_bytes=*/sizeof(float), sizeof(read_slice),
                 &uninitialized_buffer);
  EXPECT_THAT(read_slice, ElementsAre(200.0f, 300.0f));

  void *copied_buffer = CopyBufferToDevice(uninitialized_buffer, device);
  ASSERT_NE(copied_buffer, nullptr);
  float copied_output[4] = {0.0f};
  BufferToHost(copied_buffer, copied_output);
  EXPECT_THAT(copied_output, ElementsAre(10.0f, 200.0f, 300.0f, 40.0f));

  PjRtBufferFree(copied_buffer);
  PjRtBufferFree(uninitialized_buffer);
  PjRtBufferFree(buffer);
  FreeClient(client);
}

TEST(PjRtCpuTest, CompileExecuteInspectAndSerialize) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  void *client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  ASSERT_NE(client, nullptr);
  void *device = ClientGetAddressableDevice(client, 0);
  ASSERT_NE(device, nullptr);

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(kAddMultiplyModule));
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
  EXPECT_TRUE(ScopedErrorHandler::error().empty());

  EXPECT_EQ(PjRtLoadedExecutableGetClient(executable), client);
  EXPECT_EQ(PjRtLoadedExecutableNumReplicas(executable), 1);
  EXPECT_EQ(PjRtLoadedExecutableNumPartitions(executable), 1);

  JLCompiledMemoryStats memory_statistics{};
  PjRtLoadedExecutableGetCompiledMemoryStats(executable, &memory_statistics);
  EXPECT_GE(memory_statistics.output_size_in_bytes +
                memory_statistics.host_output_size_in_bytes,
            0);

  void *hlo_modules[4] = {nullptr};
  int32_t num_modules = 0;
  PjRtLoadedExecutableGetHloModules(executable, hlo_modules, &num_modules);
  ASSERT_EQ(num_modules, 1);
  ASSERT_NE(hlo_modules[0], nullptr);
  const char *hlo_text = HloModuleToString(hlo_modules[0], /*print_options=*/0);
  ASSERT_NE(hlo_text, nullptr);
  EXPECT_THAT(std::string(hlo_text), HasSubstr("add"));
  EXPECT_THAT(std::string(hlo_text), HasSubstr("multiply"));
  free(const_cast<char *>(hlo_text));
  FreeHloModule(hlo_modules[0]);

  float input_a[4] = {2.0f, 3.0f, 4.0f, 5.0f};
  float input_b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t shape[1] = {4};
  void *buffer_a = ArrayFromHostBuffer(client, input_a, XlaPrimitiveType::kF32,
                                       /*num_dimensions=*/1, shape, device);
  void *buffer_b = ArrayFromHostBuffer(client, input_b, XlaPrimitiveType::kF32,
                                       /*num_dimensions=*/1, shape, device);

  void *arguments[2] = {buffer_a, buffer_b};
  uint8_t is_donatable[2] = {0, 0};
  void *sharded_results[2] = {nullptr, nullptr};
  uint8_t has_sharded_futures = 0;
  void *sharded_future_handles[2] = {nullptr, nullptr};
  XLAExecuteSharded(executable, /*num_arguments=*/2, arguments, device,
                    is_donatable, /*num_results=*/2, sharded_results,
                    &has_sharded_futures, sharded_future_handles);
  if (has_sharded_futures) {
    for (int i = 0; i < 2; ++i) {
      if (sharded_future_handles[i] != nullptr) {
        FutureAwait(sharded_future_handles[i]);
        FreeFuture(sharded_future_handles[i]);
      }
    }
  }
  float sum_output[4] = {0.0f};
  float product_output[4] = {0.0f};
  BufferToHost(sharded_results[0], sum_output);
  BufferToHost(sharded_results[1], product_output);
  EXPECT_THAT(sum_output, ElementsAre(12.0f, 23.0f, 34.0f, 45.0f));
  EXPECT_THAT(product_output, ElementsAre(20.0f, 60.0f, 120.0f, 200.0f));
  PjRtBufferFree(sharded_results[0]);
  PjRtBufferFree(sharded_results[1]);

  void *execution_results[2] = {nullptr, nullptr};
  uint8_t has_futures = 0;
  void *execution_future_handles[2] = {nullptr, nullptr};
  XLAExecute(executable, /*num_arguments=*/2, arguments, is_donatable,
             /*num_results=*/2, execution_results, &has_futures,
             execution_future_handles);
  if (has_futures) {
    for (int i = 0; i < 2; ++i) {
      if (execution_future_handles[i] != nullptr) {
        FutureAwait(execution_future_handles[i]);
        FreeFuture(execution_future_handles[i]);
      }
    }
  }
  BufferToHost(execution_results[0], sum_output);
  BufferToHost(execution_results[1], product_output);
  EXPECT_THAT(sum_output, ElementsAre(12.0f, 23.0f, 34.0f, 45.0f));
  EXPECT_THAT(product_output, ElementsAre(20.0f, 60.0f, 120.0f, 200.0f));
  PjRtBufferFree(execution_results[0]);
  PjRtBufferFree(execution_results[1]);

  size_t serialized_size_in_bytes = 0;
  uint8_t *serialized =
      PjRtLoadedExecutableSerialize(executable, &serialized_size_in_bytes);
  ASSERT_NE(serialized, nullptr);
  EXPECT_GT(serialized_size_in_bytes, 0u);

  void *reloaded_executable = PjRtClientLoadSerializedExecutable(
      client, serialized, serialized_size_in_bytes,
      /*compile_options_proto=*/nullptr, /*compile_options_proto_size=*/0);
  free(serialized);
  ASSERT_NE(reloaded_executable, nullptr);

  void *reloaded_results[2] = {nullptr, nullptr};
  uint8_t has_reloaded_futures = 0;
  void *reloaded_future_handles[2] = {nullptr, nullptr};
  XLAExecuteSharded(reloaded_executable, /*num_arguments=*/2, arguments, device,
                    is_donatable, /*num_results=*/2, reloaded_results,
                    &has_reloaded_futures, reloaded_future_handles);
  if (has_reloaded_futures) {
    for (int i = 0; i < 2; ++i) {
      if (reloaded_future_handles[i] != nullptr) {
        FutureAwait(reloaded_future_handles[i]);
        FreeFuture(reloaded_future_handles[i]);
      }
    }
  }
  BufferToHost(reloaded_results[0], sum_output);
  BufferToHost(reloaded_results[1], product_output);
  EXPECT_THAT(sum_output, ElementsAre(12.0f, 23.0f, 34.0f, 45.0f));
  EXPECT_THAT(product_output, ElementsAre(20.0f, 60.0f, 120.0f, 200.0f));
  PjRtBufferFree(reloaded_results[0]);
  PjRtBufferFree(reloaded_results[1]);

  MlirModule second_module =
      mlirModuleCreateParse(context, makeStringRef(kAddMultiplyModule));
  ASSERT_NE(second_module.ptr, nullptr);
  void *proto_executable = ClientCompileWithProto(
      client, second_module, /*compile_options_proto=*/nullptr,
      /*compile_options_proto_size=*/0);
  ASSERT_NE(proto_executable, nullptr);

  ExecutableFree(proto_executable);
  ExecutableFree(reloaded_executable);
  ExecutableFree(executable);
  PjRtBufferFree(buffer_a);
  PjRtBufferFree(buffer_b);
  mlirModuleDestroy(second_module);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  FreeClient(client);
}

TEST(IfrtCpuTest, ClientDeviceAndMemoryQueries) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  const char *error = nullptr;
  void *client = ifrt_make_pjrt_cpu_client(
      /*asynchronous=*/1, /*node_id=*/0, /*num_nodes=*/1,
      /*distributed_runtime_client=*/nullptr, &error);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(client, nullptr);

  int num_devices = ifrt_client_device_count(client);
  int num_addressable_devices = ifrt_client_addressable_device_count(client);
  EXPECT_GT(num_devices, 0);
  EXPECT_EQ(num_devices, num_addressable_devices);
  EXPECT_EQ(ifrt_ClientProcessIndex(client), 0);

  const char *platform_name = ifrt_ClientGetPlatformName(client);
  ASSERT_NE(platform_name, nullptr);
  EXPECT_GT(std::string(platform_name).size(), 0u);
  free(const_cast<char *>(platform_name));

  std::vector<void *> devices(num_devices, nullptr);
  std::vector<void *> addressable_devices(num_addressable_devices, nullptr);
  std::vector<void *> all_devices(num_devices, nullptr);
  ifrt_client_devices(client, devices.data());
  ifrt_client_addressable_devices(client, addressable_devices.data());
  ifrt_client_all_devices(client, all_devices.data());
  ASSERT_NE(devices[0], nullptr);
  EXPECT_EQ(devices[0], addressable_devices[0]);
  EXPECT_EQ(devices[0], all_devices[0]);

  void *device = ifrt_ClientGetAddressableDevice(client, 0);
  ASSERT_EQ(device, devices[0]);
  int64_t global_id = ifrt_DeviceGetGlobalDeviceId(device);
  int64_t local_hardware_id = ifrt_DeviceGetLocalHardwareId(device);
  EXPECT_EQ(ifrt_ClientGetDevice(client, static_cast<int>(global_id)), device);
  EXPECT_EQ(ifrt_client_lookup_device(client, static_cast<int>(global_id)),
            device);
  EXPECT_EQ(ifrt_client_lookup_addressable_device(
                client, static_cast<int>(local_hardware_id)),
            device);
  EXPECT_TRUE(ifrt_DeviceIsAddressable(device));
  EXPECT_EQ(ifrt_DeviceToClient(device), client);

  const char *device_kind = ifrt_DeviceGetKind(device);
  ASSERT_NE(device_kind, nullptr);
  EXPECT_GT(std::string(device_kind).size(), 0u);
  free(const_cast<char *>(device_kind));

  void *default_memory = ifrt_DeviceGetDefaultMemory(device);
  ASSERT_NE(default_memory, nullptr);
  int32_t num_memories = 0;
  void **memories = ifrt_DeviceGetMemories(device, &num_memories);
  ASSERT_NE(memories, nullptr);
  EXPECT_GT(num_memories, 0);

  const char *memory_string = ifrt_MemoryToString(default_memory);
  ASSERT_NE(memory_string, nullptr);
  EXPECT_GT(std::string(memory_string).size(), 0u);
  free(const_cast<char *>(memory_string));

  void *memory_kind = ifrt_MemoryGetMemoryKind(default_memory);
  ASSERT_NE(memory_kind, nullptr);
  const char *memory_kind_string = ifrt_MemoryKindToString(memory_kind);
  ASSERT_NE(memory_kind_string, nullptr);
  EXPECT_STREQ(memory_kind_string, "device");

  void *memory_kind_from_string =
      ifrt_memory_kind_from_string(memory_kind_string);
  EXPECT_TRUE(ifrt_MemoryKindsAreEqual(memory_kind, memory_kind_from_string));
  free(const_cast<char *>(memory_kind_string));

  void *pinned_memory_kind = ifrt_memory_kind_from_string("pinned_host");
  ASSERT_NE(pinned_memory_kind, nullptr);
  EXPECT_TRUE(ifrt_memory_kind_has_value(pinned_memory_kind));
  EXPECT_FALSE(ifrt_MemoryKindsAreEqual(memory_kind, pinned_memory_kind));

  void *optional_memory_kind = ifrt_memory_kind_with_optional_memory_space();
  ASSERT_NE(optional_memory_kind, nullptr);
  EXPECT_FALSE(ifrt_memory_kind_has_value(optional_memory_kind));
  EXPECT_TRUE(ifrt_MemoryKindsAreEqual(memory_kind, optional_memory_kind));
  const char *optional_memory_kind_string =
      ifrt_MemoryKindToString(optional_memory_kind);
  ASSERT_NE(optional_memory_kind_string, nullptr);
  EXPECT_STREQ(optional_memory_kind_string, "device");
  free(const_cast<char *>(optional_memory_kind_string));

  free_ifrt_memory_kind(optional_memory_kind);
  free_ifrt_memory_kind(pinned_memory_kind);
  free_ifrt_memory_kind(memory_kind_from_string);
  free_ifrt_memory_kind(memory_kind);
  ifrt_FreeClient(client);
}

TEST(IfrtCpuTest, ArrayCreationShardingCompileExecuteAndSerialize) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  const char *error = nullptr;
  void *client = ifrt_make_pjrt_cpu_client(
      /*asynchronous=*/1, /*node_id=*/0, /*num_nodes=*/1,
      /*distributed_runtime_client=*/nullptr, &error);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(client, nullptr);

  void *device = ifrt_ClientGetAddressableDevice(client, 0);
  ASSERT_NE(device, nullptr);
  void *default_memory = ifrt_DeviceGetDefaultMemory(device);
  void *memory_kind = ifrt_MemoryGetMemoryKind(default_memory);

  float input_a[4] = {1.5f, 2.5f, 3.5f, 4.5f};
  float input_b[4] = {2.0f, 4.0f, 6.0f, 8.0f};
  int64_t shape[1] = {4};
  void *array_a = ifrt_client_make_single_shard_array_from_host_buffer(
      client, input_a, static_cast<int>(XlaPrimitiveType::kF32),
      /*num_dimensions=*/1, shape, /*semantics=*/0, device, memory_kind);
  void *array_b = ifrt_client_make_single_shard_array_from_host_buffer(
      client, input_b, static_cast<int>(XlaPrimitiveType::kF32),
      /*num_dimensions=*/1, shape, /*semantics=*/0, device, memory_kind);
  ASSERT_NE(array_a, nullptr);
  ASSERT_NE(array_b, nullptr);

  EXPECT_EQ(ifrt_array_ndims(array_a), 1);
  int64_t *array_shape = ifrt_array_shape(array_a);
  ASSERT_NE(array_shape, nullptr);
  EXPECT_EQ(array_shape[0], 4);
  delete[] array_shape;
  EXPECT_EQ(ifrt_array_to_client(array_a), client);

  void *sharding = ifrt_array_to_sharding(array_a);
  ASSERT_NE(sharding, nullptr);
  EXPECT_TRUE(ifrt_sharding_is_single_device_sharding(sharding));
  EXPECT_TRUE(ifrt_sharding_is_fully_replicated(sharding));
  EXPECT_EQ(ifrt_sharding_devices_size(sharding), 1);
  void *sharding_devices[1] = {nullptr};
  ifrt_sharding_to_device_list(sharding, sharding_devices);
  EXPECT_EQ(sharding_devices[0], device);
  const char *sharding_string = ifrt_sharding_to_string(sharding);
  ASSERT_NE(sharding_string, nullptr);
  EXPECT_GT(std::string(sharding_string).size(), 0u);
  free(const_cast<char *>(sharding_string));
  free_ifrt_sharding(sharding);

  void *copied_array = ifrt_copy_array(array_a);
  ASSERT_NE(copied_array, nullptr);
  int32_t num_shards = 0;
  void **shards = ifrt_array_disassemble_into_single_device_arrays(
      copied_array, /*semantics=*/0, /*single_device_shard_semantics=*/0,
      &num_shards);
  ASSERT_EQ(num_shards, 1);
  float shard_output[4] = {0.0f};
  ifrt_array_copy_to_host_buffer(shards[0], shard_output);
  EXPECT_THAT(shard_output, ElementsAre(1.5f, 2.5f, 3.5f, 4.5f));
  ifrt_free_array(shards[0]);
  delete[] shards;
  ifrt_free_array(copied_array);

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(kAddMultiplyModule));
  ASSERT_NE(module.ptr, nullptr);

  void *executable =
      ifrt_compile(client, module, /*device_id=*/0, /*mesh_ids=*/nullptr,
                   /*num_mesh_ids=*/0, /*xla_gpu_cuda_data_dir=*/"",
                   /*use_shardy_partitioner=*/false, /*num_replicas=*/1,
                   /*num_partitions=*/1, /*use_spmd_partitioning=*/false,
                   /*kernel_cache_enabled=*/false, /*kernel_cache_path=*/"",
                   /*autotune_cache_enabled=*/false, /*autotune_cache_path=*/"",
                   /*process_id=*/0, /*xla_enable_enzyme_comms_opt=*/false);
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(ifrt_loaded_executable_client(executable), client);
  EXPECT_EQ(ifrt_loaded_executable_num_devices(executable), 1);

  JLCompiledMemoryStats memory_statistics{};
  ifrt_loaded_executable_get_compiled_memory_stats(executable,
                                                   &memory_statistics);
  void *hlo_modules[4] = {nullptr};
  int32_t num_modules = 0;
  ifrt_loaded_executable_get_hlo_modules(executable, hlo_modules, &num_modules);
  ASSERT_EQ(num_modules, 1);
  FreeHloModule(hlo_modules[0]);

  void *arguments[2] = {array_a, array_b};
  uint8_t is_donatable[2] = {0, 0};
  void *results[2] = {nullptr, nullptr};
  uint8_t has_future = 0;
  void *status_future = nullptr;
  ifrt_loaded_executable_execute(executable, /*num_arguments=*/2, arguments,
                                 is_donatable, /*num_results=*/2, results,
                                 &has_future, &status_future);
  ASSERT_NE(has_future, 0u);
  ASSERT_NE(status_future, nullptr);
  ifrt_future_await(status_future);
  EXPECT_NE(ifrt_future_is_ready(status_future), 0u);
  ifrt_free_future(status_future);

  float sum_output[4] = {0.0f};
  float product_output[4] = {0.0f};
  ifrt_array_copy_to_host_buffer(results[0], sum_output);
  void *copy_future =
      ifrt_CopyArrayToHostBuffer(results[1], product_output, /*semantics=*/0);
  ifrt_future_await(copy_future);
  ifrt_free_future(copy_future);

  EXPECT_THAT(sum_output, ElementsAre(3.5f, 6.5f, 9.5f, 12.5f));
  EXPECT_THAT(product_output, ElementsAre(3.0f, 10.0f, 21.0f, 36.0f));
  ifrt_free_array(results[0]);
  ifrt_free_array(results[1]);

  size_t serialized_size_in_bytes = 0;
  uint8_t *serialized =
      ifrt_loaded_executable_serialize(executable, &serialized_size_in_bytes);
  ASSERT_NE(serialized, nullptr);
  EXPECT_GT(serialized_size_in_bytes, 0u);

  void *reloaded_executable = ifrt_client_load_serialized_executable(
      client, serialized, serialized_size_in_bytes,
      /*compile_options_proto=*/nullptr, /*compile_options_proto_size=*/0);
  free(serialized);
  ASSERT_NE(reloaded_executable, nullptr);

  void *reloaded_results[2] = {nullptr, nullptr};
  uint8_t reloaded_has_future = 0;
  void *reloaded_status = nullptr;
  ifrt_loaded_executable_execute(reloaded_executable, /*num_arguments=*/2,
                                 arguments, is_donatable, /*num_results=*/2,
                                 reloaded_results, &reloaded_has_future,
                                 &reloaded_status);
  ifrt_future_await(reloaded_status);
  ifrt_free_future(reloaded_status);
  ifrt_array_copy_to_host_buffer(reloaded_results[0], sum_output);
  ifrt_array_copy_to_host_buffer(reloaded_results[1], product_output);
  EXPECT_THAT(sum_output, ElementsAre(3.5f, 6.5f, 9.5f, 12.5f));
  EXPECT_THAT(product_output, ElementsAre(3.0f, 10.0f, 21.0f, 36.0f));
  ifrt_free_array(reloaded_results[0]);
  ifrt_free_array(reloaded_results[1]);

  ifrt_loaded_executable_dtor(reloaded_executable);
  ifrt_loaded_executable_dtor(executable);
  ifrt_free_array(array_a);
  ifrt_free_array(array_b);
  free_ifrt_memory_kind(memory_kind);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  ifrt_FreeClient(client);
}
