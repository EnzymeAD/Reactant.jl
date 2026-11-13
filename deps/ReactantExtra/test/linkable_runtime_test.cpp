// Unit tests for the C++ linkable XLA runtime entry points in API.cpp.

#include "test/reactant_extra_test.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

using ::reactant_test::XlaMemcpyDirection;
using ::reactant_test::XlaPrimitiveType;

namespace {

// Test fixture managing a CPU LinkableRuntime instance per test.
class LinkableRuntimeTest : public ::testing::Test {
protected:
  void SetUp() override {
    reactantXLAInit(&runtime_, "cpu");
    ASSERT_NE(runtime_, nullptr);
  }

  void TearDown() override {
    if (runtime_ != nullptr) {
      reactantXLADeInit(&runtime_);
      EXPECT_EQ(runtime_, nullptr);
    }
  }

  // Owned runtime instance under test.
  LinkableRuntime *runtime_ = nullptr;
};

} // namespace

TEST_F(LinkableRuntimeTest, InitIsIdempotentAndDeInitClearsSlot) {
  LinkableRuntime *initial_runtime = runtime_;
  reactantXLAInit(&runtime_, "cpu");
  EXPECT_EQ(runtime_, initial_runtime);

  reactantXLADeInit(&runtime_);
  EXPECT_EQ(runtime_, nullptr);

  reactantXLAInit(&runtime_, "cpu:0");
  EXPECT_NE(runtime_, nullptr);
}

TEST_F(LinkableRuntimeTest, FreeNullIsNoOp) {
  reactantXLAFree(&runtime_, nullptr);
}

TEST_F(LinkableRuntimeTest, MallocMemcpyHostDeviceAndDeviceToDevice) {
  uint64_t size_in_bytes = 64;
  void *device_a = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kS8,
                                     /*rank=*/1, &size_in_bytes);
  void *device_b = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kS8,
                                     /*rank=*/1, &size_in_bytes);
  ASSERT_NE(device_a, nullptr);
  ASSERT_NE(device_b, nullptr);

  std::vector<uint8_t> host_input(size_in_bytes);
  for (size_t i = 0; i < size_in_bytes; ++i) {
    host_input[i] = static_cast<uint8_t>(i * 3 + 7);
  }

  reactantXLAMemcpy(&runtime_, device_a, host_input.data(), size_in_bytes,
                    XlaMemcpyDirection::kMemcpyHostToDevice);
  reactantXLAMemcpy(&runtime_, device_b, device_a, /*size_in_bytes=*/0,
                    XlaMemcpyDirection::kMemcpyDeviceToDevice);
  reactantXLAMemcpy(&runtime_, device_b, device_a, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToDevice);

  std::vector<uint8_t> host_output(size_in_bytes, 0);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_b, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  EXPECT_EQ(host_output, host_input);

  std::vector<uint8_t> patch = {99, 100, 101, 102};
  constexpr ptrdiff_t kPatchOffsetInBytes = 16;
  void *device_b_offset = static_cast<char *>(device_b) + kPatchOffsetInBytes;
  reactantXLAMemcpy(&runtime_, device_b_offset, patch.data(), patch.size(),
                    XlaMemcpyDirection::kMemcpyHostToDevice);

  std::vector<uint8_t> patch_output(patch.size(), 0);
  reactantXLAMemcpy(&runtime_, patch_output.data(), device_b_offset,
                    patch.size(), XlaMemcpyDirection::kMemcpyDeviceToHost);
  EXPECT_EQ(patch_output, patch);

  reactantXLAFree(&runtime_, device_a);
  reactantXLAFree(&runtime_, device_b);
}

TEST_F(LinkableRuntimeTest, ExecMutatesOutputAndPreservesReadOnlyInputs) {
  uint64_t num_elements = 4;
  size_t size_in_bytes = num_elements * sizeof(float);
  void *device_output = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kF32,
                                          /*rank=*/1, &num_elements);
  void *device_x = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kF32,
                                     /*rank=*/1, &num_elements);
  void *device_y = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kF32,
                                     /*rank=*/1, &num_elements);

  std::vector<float> host_x = {1.5f, 2.5f, -3.0f, 4.25f};
  std::vector<float> host_y = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> zeros(num_elements, 0.0f);
  reactantXLAMemcpy(&runtime_, device_output, zeros.data(), size_in_bytes,
                    XlaMemcpyDirection::kMemcpyHostToDevice);
  reactantXLAMemcpy(&runtime_, device_x, host_x.data(), size_in_bytes,
                    XlaMemcpyDirection::kMemcpyHostToDevice);
  reactantXLAMemcpy(&runtime_, device_y, host_y.data(), size_in_bytes,
                    XlaMemcpyDirection::kMemcpyHostToDevice);

  static constexpr char kAddModule[] = R"mlir(
    func.func @main(%out: tensor<?xf32>, %x: tensor<?xf32>, %y: tensor<?xf32>)
        -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      %sum = stablehlo.add %x, %y : tensor<?xf32>
      return %sum, %x, %y : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
    }
  )mlir";

  void *arguments[3] = {device_output, device_x, device_y};
  reactantXLAExec(&runtime_, kAddModule, /*num_arguments=*/3, arguments,
                  /*num_constants=*/0, /*constants=*/nullptr);

  std::vector<float> host_output(num_elements, 0.0f);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_output, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<float> expected_first = {11.5f, 22.5f, 27.0f, 44.25f};
  EXPECT_EQ(host_output, expected_first);

  // Re-running the same module exercises the compiled executable cache.
  arguments[0] = device_output;
  arguments[1] = device_output;
  arguments[2] = device_y;
  reactantXLAExec(&runtime_, kAddModule, /*num_arguments=*/3, arguments,
                  /*num_constants=*/0, /*constants=*/nullptr);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_output, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<float> expected_second = {21.5f, 42.5f, 57.0f, 84.25f};
  EXPECT_EQ(host_output, expected_second);

  std::vector<float> host_x_after(num_elements, 0.0f);
  reactantXLAMemcpy(&runtime_, host_x_after.data(), device_x, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  EXPECT_EQ(host_x_after, host_x);

  reactantXLAFree(&runtime_, device_output);
  reactantXLAFree(&runtime_, device_x);
  reactantXLAFree(&runtime_, device_y);
}

TEST_F(LinkableRuntimeTest, ExecSupportsSubBufferOffsetViews) {
  uint64_t num_elements = 4;
  size_t size_in_bytes = num_elements * sizeof(float);
  void *device_buffer = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kF32,
                                          /*rank=*/1, &num_elements);
  std::vector<float> initial_values = {1.0f, 2.0f, 3.0f, 4.0f};
  reactantXLAMemcpy(&runtime_, device_buffer, initial_values.data(),
                    size_in_bytes, XlaMemcpyDirection::kMemcpyHostToDevice);

  static constexpr char kScaleTailModule[] = R"mlir(
    func.func @main(%v: tensor<?xf32>) -> tensor<?xf32> {
      %c0 = stablehlo.constant dense<0> : tensor<i64>
      %slice = stablehlo.dynamic_slice %v, %c0, sizes = [2]
          : (tensor<?xf32>, tensor<i64>) -> tensor<2xf32>
      %c10 = stablehlo.constant dense<10.0> : tensor<2xf32>
      %scaled = stablehlo.multiply %slice, %c10 : tensor<2xf32>
      %updated = stablehlo.dynamic_update_slice %v, %scaled, %c0
          : (tensor<?xf32>, tensor<2xf32>, tensor<i64>) -> tensor<?xf32>
      return %updated : tensor<?xf32>
    }
  )mlir";

  void *tail_view = static_cast<char *>(device_buffer) + 2 * sizeof(float);
  void *arguments[1] = {tail_view};
  reactantXLAExec(&runtime_, kScaleTailModule, /*num_arguments=*/1, arguments,
                  /*num_constants=*/0, /*constants=*/nullptr);

  std::vector<float> host_output(num_elements, 0.0f);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_buffer, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<float> expected = {1.0f, 2.0f, 30.0f, 40.0f};
  EXPECT_EQ(host_output, expected);

  reactantXLAFree(&runtime_, device_buffer);
}

TEST_F(LinkableRuntimeTest, ExecSupportsNullAndSpecializedScalarConstants) {
  uint64_t num_elements = 4;
  size_t size_in_bytes = num_elements * sizeof(int32_t);
  void *device_x = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kS32,
                                     /*rank=*/1, &num_elements);
  std::vector<int32_t> initial_values = {2, 4, 6, 8};
  reactantXLAMemcpy(&runtime_, device_x, initial_values.data(), size_in_bytes,
                    XlaMemcpyDirection::kMemcpyHostToDevice);

  static constexpr char kSpecializeAndNullModule[] = R"mlir(
    func.func @main(%x: tensor<?xi32>, %opt: tensor<?xi8>, %scale: tensor<i32>)
        -> (tensor<?xi32>, tensor<?xi8>) {
      %c0 = stablehlo.constant dense<0> : tensor<i64>
      %slice = stablehlo.dynamic_slice %x, %c0, sizes = [4]
          : (tensor<?xi32>, tensor<i64>) -> tensor<4xi32>
      %broadcast_scale = stablehlo.broadcast_in_dim %scale, dims = []
          : (tensor<i32>) -> tensor<4xi32>
      %product = stablehlo.multiply %slice, %broadcast_scale : tensor<4xi32>
      %updated = stablehlo.dynamic_update_slice %x, %product, %c0
          : (tensor<?xi32>, tensor<4xi32>, tensor<i64>) -> tensor<?xi32>
      return %updated, %opt : tensor<?xi32>, tensor<?xi8>
    }
  )mlir";

  void *arguments[2] = {device_x, nullptr};
  int64_t scale_three = 3;
  reactantXLAExec(&runtime_, kSpecializeAndNullModule, /*num_arguments=*/2,
                  arguments, /*num_constants=*/1, &scale_three);

  std::vector<int32_t> host_output(num_elements, 0);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_x, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<int32_t> expected_after_three = {6, 12, 18, 24};
  EXPECT_EQ(host_output, expected_after_three);

  // Specializing with a distinct constant must not reuse the cached scale=3
  // executable.
  int64_t scale_five = 5;
  reactantXLAExec(&runtime_, kSpecializeAndNullModule, /*num_arguments=*/2,
                  arguments, /*num_constants=*/1, &scale_five);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_x, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<int32_t> expected_after_five = {30, 60, 90, 120};
  EXPECT_EQ(host_output, expected_after_five);

  reactantXLAFree(&runtime_, device_x);
}

TEST_F(LinkableRuntimeTest, ExecFoldsAliasedViewsOfSameBaseAllocation) {
  uint64_t num_elements = 4;
  size_t size_in_bytes = num_elements * sizeof(float);
  void *device_buffer = reactantXLAMalloc(&runtime_, XlaPrimitiveType::kF32,
                                          /*rank=*/1, &num_elements);
  std::vector<float> initial_values = {1.0f, 2.0f, 10.0f, 20.0f};
  reactantXLAMemcpy(&runtime_, device_buffer, initial_values.data(),
                    size_in_bytes, XlaMemcpyDirection::kMemcpyHostToDevice);

  static constexpr char kAddTailIntoHeadModule[] = R"mlir(
    func.func @main(%full: tensor<?xf32>, %tail: tensor<?xf32>)
        -> (tensor<?xf32>, tensor<?xf32>) {
      %c0 = stablehlo.constant dense<0> : tensor<i64>
      %head = stablehlo.dynamic_slice %full, %c0, sizes = [2]
          : (tensor<?xf32>, tensor<i64>) -> tensor<2xf32>
      %tail_slice = stablehlo.dynamic_slice %tail, %c0, sizes = [2]
          : (tensor<?xf32>, tensor<i64>) -> tensor<2xf32>
      %sum = stablehlo.add %head, %tail_slice : tensor<2xf32>
      %updated = stablehlo.dynamic_update_slice %full, %sum, %c0
          : (tensor<?xf32>, tensor<2xf32>, tensor<i64>) -> tensor<?xf32>
      return %updated, %tail : tensor<?xf32>, tensor<?xf32>
    }
  )mlir";

  void *tail_view = static_cast<char *>(device_buffer) + 2 * sizeof(float);
  void *arguments[2] = {device_buffer, tail_view};
  reactantXLAExec(&runtime_, kAddTailIntoHeadModule, /*num_arguments=*/2,
                  arguments, /*num_constants=*/0, /*constants=*/nullptr);

  std::vector<float> host_output(num_elements, 0.0f);
  reactantXLAMemcpy(&runtime_, host_output.data(), device_buffer, size_in_bytes,
                    XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<float> expected = {11.0f, 22.0f, 10.0f, 20.0f};
  EXPECT_EQ(host_output, expected);

  reactantXLAFree(&runtime_, device_buffer);
}

TEST_F(LinkableRuntimeTest, UnregisteredPointerAndThrowExitWithError) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  int stack_variable = 42;
  EXPECT_DEATH(reactantXLAFree(&runtime_, &stack_variable),
               "not a reactant allocation");
  EXPECT_DEATH(reactantXLAMemcpy(&runtime_, &stack_variable, &stack_variable,
                                 sizeof(stack_variable),
                                 XlaMemcpyDirection::kMemcpyHostToDevice),
               "does not belong to any reactant allocation");
  EXPECT_DEATH(reactantXLAThrow("deliberate test error"), "");
}
