// Unit tests for MLIR, Shardy, HLO inspection, sharding conversion, and cost
// analysis C ABI utilities in API.cpp.

#include "test/reactant_extra_test.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ::reactant_test::makeStringRef;
using ::reactant_test::printAttributeToString;
using ::reactant_test::printOperationToString;
using ::reactant_test::ScopedErrorHandler;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// Token kind constants returned by ReactantLexMLIR in API.cpp.
enum MlirLexTokenKind : int32_t {
  kLexTokenSsaIdentifier = 1,
  kLexTokenSymbolRef = 2,
  kLexTokenPunctuation = 5,
  kLexTokenBareIdentifier = 7,
};

TEST(MlirHloUtilsTest, RegistryDialectsAndSdyPropagationPipeline) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  InitializeRegistry(registry);
  InitializePasses(registry);

  MlirContext context = mlirContextCreate();
  mlirContextAppendDialectRegistry(context, registry);
  RegisterDialects(context);

  static constexpr const char *kSdyModule = R"mlir(
module @sdy_propagation_test {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(
      %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> tensor<4x4xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
)mlir";
  MlirModule module = mlirModuleCreateParse(context, makeStringRef(kSdyModule));
  ASSERT_NE(module.ptr, nullptr);

  MlirPassManager pass_manager = mlirPassManagerCreate(context);
  mlirPassManagerEnableXLATraceTiming(pass_manager);
  MlirOpPassManager operation_pass_manager =
      mlirPassManagerGetAsOpPassManager(pass_manager);
  addSdyPropagationPipeline(
      operation_pass_manager, /*keep_sharding_rules=*/0,
      /*conservative_propagation=*/0, /*debug_sharding_origins=*/0,
      /*debug_propagation_edge_sharding=*/0, /*skip_convert_to_reshard=*/0,
      /*skip_inline=*/0, /*enable_insert_explicit_collectives=*/0);

  MlirLogicalResult result =
      mlirPassManagerRunOnOp(pass_manager, mlirModuleGetOperation(module));
  EXPECT_NE(result.value, 0);

  std::string propagated_ir =
      printOperationToString(mlirModuleGetOperation(module));
  EXPECT_THAT(propagated_ir, HasSubstr("stablehlo.add"));
  EXPECT_THAT(propagated_ir, HasSubstr("sdy.sharding"));

  mlirPassManagerDestroy(pass_manager);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  mlirDialectRegistryDestroy(registry);
}

TEST(MlirHloUtilsTest, OperationParseInjectFuncAttrsAndLinkInModule) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);
  MlirLocation location = mlirLocationUnknownGet(context);

  MlirModule target_module = mlirModuleCreateEmpty(location);
  MlirBlock target_body = mlirModuleGetBody(target_module);
  static constexpr const char *kInjectedFunction = R"mlir(
func.func @helper(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<2xf32>
  return %0 : tensor<2xf32>
}
)mlir";
  ASSERT_TRUE(mlirOperationInject(context, target_body,
                                  makeStringRef(kInjectedFunction), location,
                                  /*verify_after_parse=*/true));

  MlirOperation function_operation = mlirBlockGetFirstOperation(target_body);
  ASSERT_NE(function_operation.ptr, nullptr);
  EXPECT_TRUE(mlirIsFunctionOpInterface(function_operation));
  EXPECT_FALSE(
      mlirIsFunctionOpInterface(mlirModuleGetOperation(target_module)));

  MlirType function_type = mlirGetFunctionTypeFromOperation(function_operation);
  EXPECT_NE(function_type.ptr, nullptr);

  MlirRegion function_region = mlirOperationGetFirstRegion(function_operation);
  MlirBlock function_block = mlirRegionGetFirstBlock(function_region);
  MlirOperation inner_add = mlirBlockGetFirstOperation(function_block);
  ASSERT_NE(inner_add.ptr, nullptr);
  EXPECT_EQ(mlirGetParentOfTypeFunctionOp(inner_add.ptr),
            function_operation.ptr);

  MlirAttribute activity_attribute =
      enzymeActivityAttrGet(context, /*value=*/0);
  ASSERT_NE(activity_attribute.ptr, nullptr);
  ReactantFuncSetArgAttr(function_operation, /*position=*/0,
                         makeStringRef("reactant.arg_test"),
                         activity_attribute);
  ReactantFuncSetResultAttr(function_operation, /*position=*/0,
                            makeStringRef("reactant.res_test"),
                            activity_attribute);
  std::string function_ir = printOperationToString(function_operation);
  EXPECT_THAT(function_ir, HasSubstr("reactant.arg_test"));
  EXPECT_THAT(function_ir, HasSubstr("reactant.res_test"));

  MlirModule parse_module = mlirModuleCreateEmpty(location);
  MlirBlock parse_block = mlirModuleGetBody(parse_module);
  MlirOperation parsed_operation = mlirOperationParse(
      context, parse_block,
      makeStringRef("func.func @single(%x: f32) -> f32 { return %x : f32 }"),
      location, /*verify_after_parse=*/true);
  ASSERT_NE(parsed_operation.ptr, nullptr);
  EXPECT_TRUE(mlirOperationVerify(parsed_operation));
  mlirOperationDestroy(parsed_operation);
  mlirModuleDestroy(parse_module);

  // LinkInModule renames colliding private symbols from the source module.
  static constexpr const char *kSourceModule = R"mlir(
module {
  func.func private @helper(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
  func.func @entry(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = func.call @helper(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
)mlir";
  MlirModule source_module =
      mlirModuleCreateParse(context, makeStringRef(kSourceModule));
  ASSERT_NE(source_module.ptr, nullptr);

  MlirOperation entry_operation =
      LinkInModule(target_module, source_module, "entry");
  ASSERT_NE(entry_operation.ptr, nullptr);
  EXPECT_TRUE(mlirOperationVerify(mlirModuleGetOperation(target_module)));

  std::string linked_ir =
      printOperationToString(mlirModuleGetOperation(target_module));
  EXPECT_THAT(linked_ir, HasSubstr("@helper_1"));
  EXPECT_THAT(linked_ir, HasSubstr("@entry"));

  mlirModuleDestroy(source_module);
  mlirModuleDestroy(target_module);
  mlirContextDestroy(context);
}

TEST(MlirHloUtilsTest, LexerAndLLVMIRTranslation) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  InitializeRegistry(registry);
  MlirContext context = mlirContextCreate();
  mlirContextAppendDialectRegistry(context, registry);
  RegisterDialects(context);

  static constexpr const char *kLexInput =
      "func.func @foo(%arg0: tensor<2xf32>) -> tensor<2xf32>";
  int32_t input_length = static_cast<int32_t>(std::strlen(kLexInput));
  constexpr int32_t kMaxTokens = 32;
  std::vector<int32_t> token_kinds(kMaxTokens, 0);
  std::vector<int32_t> token_offsets(kMaxTokens, 0);
  std::vector<int32_t> token_lengths(kMaxTokens, 0);

  int32_t num_tokens =
      ReactantLexMLIR(context, kLexInput, input_length, token_kinds.data(),
                      token_offsets.data(), token_lengths.data(), kMaxTokens);
  ASSERT_GT(num_tokens, 5);
  EXPECT_EQ(token_kinds[0], MlirLexTokenKind::kLexTokenBareIdentifier);
  EXPECT_EQ(std::string(kLexInput + token_offsets[0], token_lengths[0]),
            "func.func");
  EXPECT_EQ(token_kinds[1], MlirLexTokenKind::kLexTokenSymbolRef);
  EXPECT_EQ(std::string(kLexInput + token_offsets[1], token_lengths[1]),
            "@foo");
  EXPECT_EQ(token_kinds[2], MlirLexTokenKind::kLexTokenPunctuation);
  EXPECT_EQ(token_kinds[3], MlirLexTokenKind::kLexTokenSsaIdentifier);
  EXPECT_EQ(std::string(kLexInput + token_offsets[3], token_lengths[3]),
            "%arg0");

  static constexpr const char *kValidLlvmIr = R"llvm(
define i32 @add_i32(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
)llvm";
  MlirModule llvm_module = ConvertLLVMStrToMLIR(kValidLlvmIr, context);
  ASSERT_NE(llvm_module.ptr, nullptr);
  std::string translated_ir =
      printOperationToString(mlirModuleGetOperation(llvm_module));
  EXPECT_THAT(translated_ir, HasSubstr("@add_i32"));
  mlirModuleDestroy(llvm_module);

  ScopedErrorHandler::clear();
  MlirModule invalid_module =
      ConvertLLVMStrToMLIR("this is not valid llvm ir", context);
  EXPECT_EQ(invalid_module.ptr, nullptr);
  EXPECT_FALSE(ScopedErrorHandler::error().empty());

  mlirContextDestroy(context);
  mlirDialectRegistryDestroy(registry);
}

TEST(MlirHloUtilsTest, HloConversionParsingAndInstructionInspection) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);

  static constexpr const char *kStablehloModule = R"mlir(
module @hlo_conv_test {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(kStablehloModule));
  ASSERT_NE(module.ptr, nullptr);

  void *converted_hlo = convertMlirModuleToHloModule(module);
  ASSERT_NE(converted_hlo, nullptr);
  const char *converted_hlo_text =
      HloModuleToString(converted_hlo, /*print_options=*/0);
  ASSERT_NE(converted_hlo_text, nullptr);
  EXPECT_THAT(std::string(converted_hlo_text), HasSubstr("add"));
  free(const_cast<char *>(converted_hlo_text));
  FreeHloModule(converted_hlo);
  mlirModuleDestroy(module);
  mlirContextDestroy(context);

  static constexpr const char *kHloText = R"hlo(
HloModule fusion_and_reduce_test

%add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(f32[] %lhs, f32[] %rhs)
}

%fused_comp (p0: f32[4]) -> f32[4] {
  %p0 = f32[4] parameter(0)
  ROOT %mul = f32[4] multiply(f32[4] %p0, f32[4] %p0)
}

ENTRY %main (x: f32[4]) -> f32[] {
  %x = f32[4] parameter(0)
  %fusion = f32[4] fusion(f32[4] %x), kind=kLoop, calls=%fused_comp
  %zero = f32[] constant(0)
  ROOT %red = f32[] reduce(f32[4] %fusion, f32[] %zero), dimensions={0},
      to_apply=%add_f32
}
)hlo";

  void *hlo_module = parseAndReturnUnverifiedHloModule(kHloText);
  ASSERT_NE(hlo_module, nullptr);
  EXPECT_TRUE(ScopedErrorHandler::error().empty());

  void *entry_computation = hloModuleGetEntryComputation(hlo_module);
  ASSERT_NE(entry_computation, nullptr);
  const char *computation_text =
      hloComputationToString(entry_computation, /*print_options=*/0);
  ASSERT_NE(computation_text, nullptr);
  EXPECT_THAT(std::string(computation_text), HasSubstr("fusion"));
  EXPECT_THAT(std::string(computation_text), HasSubstr("reduce"));
  free(const_cast<char *>(computation_text));

  int64_t num_instructions = hloComputationInstructionCount(entry_computation);
  ASSERT_EQ(num_instructions, 4);
  std::vector<void *> instructions(num_instructions, nullptr);
  hloComputationGetInstructionsPostOrder(entry_computation, num_instructions,
                                         instructions.data());

  bool saw_fusion = false;
  bool saw_reduce = false;
  for (void *instruction : instructions) {
    ASSERT_NE(instruction, nullptr);
    const char *instruction_text =
        hloInstructionToString(instruction, /*print_options=*/0);
    ASSERT_NE(instruction_text, nullptr);
    free(const_cast<char *>(instruction_text));

    uint8_t opcode = hloInstructionGetOpcode(instruction);
    const char *opcode_string = hloOpcodeToString(opcode);
    ASSERT_NE(opcode_string, nullptr);
    std::string opcode_name(opcode_string);
    free(const_cast<char *>(opcode_string));

    if (hloInstructionIsFusion(instruction)) {
      saw_fusion = true;
      EXPECT_EQ(opcode_name, "fusion");
      uint8_t fusion_kind = hloInstructionGetFusionKind(instruction);
      const char *fusion_kind_string = hloFusionKindToString(fusion_kind);
      ASSERT_NE(fusion_kind_string, nullptr);
      EXPECT_STREQ(fusion_kind_string, "kLoop");
      free(const_cast<char *>(fusion_kind_string));

      void *fused_computation =
          hloInstructionFusedInstructionsComputation(instruction);
      ASSERT_NE(fused_computation, nullptr);
      EXPECT_EQ(hloComputationInstructionCount(fused_computation), 2);
    }

    if (hloInstructionHasToApply(instruction)) {
      saw_reduce = true;
      EXPECT_EQ(opcode_name, "reduce");
      void *to_apply_computation = hloInstructionGetToApply(instruction);
      ASSERT_NE(to_apply_computation, nullptr);
      EXPECT_EQ(hloComputationInstructionCount(to_apply_computation), 3);
    }
  }
  EXPECT_TRUE(saw_fusion);
  EXPECT_TRUE(saw_reduce);

  FreeHloModule(hlo_module);
}

TEST(MlirHloUtilsTest, ShardingConversionsAndRoundTrips) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);

  MlirAttribute mesh_attribute = mlirAttributeParseGet(
      context, makeStringRef("#sdy.mesh<[\"x\"=2, \"y\"=2]>"));
  MlirAttribute sharding_attribute = mlirAttributeParseGet(
      context, makeStringRef("#sdy.sharding<@mesh, [{\"x\"}, {\"y\"}]>"));
  ASSERT_NE(mesh_attribute.ptr, nullptr);
  ASSERT_NE(sharding_attribute.ptr, nullptr);

  void *hlo_sharding =
      hloShardingFromTensorShardingAttr(sharding_attribute, mesh_attribute);
  ASSERT_NE(hlo_sharding, nullptr);

  EXPECT_TRUE(hlo_sharding_is_tiled(hlo_sharding));
  EXPECT_FALSE(hlo_sharding_is_replicated(hlo_sharding));
  EXPECT_FALSE(hlo_sharding_is_tuple(hlo_sharding));
  EXPECT_FALSE(hlo_sharding_is_manual(hlo_sharding));
  EXPECT_FALSE(hlo_sharding_is_unknown(hlo_sharding));
  EXPECT_FALSE(hlo_sharding_is_maximal(hlo_sharding));
  EXPECT_FALSE(hlo_sharding_replicate_on_last_tile_dim(hlo_sharding));

  const char *hlo_sharding_string = hlo_sharding_to_string(hlo_sharding);
  ASSERT_NE(hlo_sharding_string, nullptr);
  EXPECT_THAT(std::string(hlo_sharding_string), HasSubstr("devices=[2,2]"));
  free(const_cast<char *>(hlo_sharding_string));

  int32_t num_dimensions =
      hlo_sharding_tile_assignment_dimensions_size(hlo_sharding);
  ASSERT_EQ(num_dimensions, 2);
  int64_t dimensions[2] = {0, 0};
  hlo_sharding_tile_assignment_dimensions(hlo_sharding, dimensions,
                                          num_dimensions);
  EXPECT_THAT(dimensions, ElementsAre(2, 2));

  int32_t num_devices = hlo_sharding_tile_assignment_devices_size(hlo_sharding);
  ASSERT_EQ(num_devices, 4);
  int64_t devices[4] = {0, 0, 0, 0};
  hlo_sharding_tile_assignment_devices(hlo_sharding, devices, num_devices);
  EXPECT_THAT(devices, ElementsAre(0, 1, 2, 3));

  void *op_sharding = hlo_sharding_to_op_sharding(hlo_sharding);
  ASSERT_NE(op_sharding, nullptr);
  EXPECT_GT(op_sharding_to_op_sharding_type(op_sharding), 0);
  EXPECT_FALSE(op_sharding_is_shard_group(op_sharding));
  EXPECT_FALSE(op_sharding_replicate_on_last_tile_dim(op_sharding));
  EXPECT_TRUE(op_sharding_has_tile_assignment_dimensions(op_sharding));
  ASSERT_EQ(op_sharding_tile_assignment_dimensions_size(op_sharding), 2);
  int32_t op_sharding_dimensions[2] = {0, 0};
  op_sharding_tile_assignment_dimensions(op_sharding, op_sharding_dimensions);
  EXPECT_THAT(op_sharding_dimensions, ElementsAre(2, 2));

  void *roundtrip_hlo_sharding = hlo_sharding_from_op_sharding(op_sharding);
  ASSERT_NE(roundtrip_hlo_sharding, nullptr);
  EXPECT_TRUE(hlo_sharding_check_eq(hlo_sharding, roundtrip_hlo_sharding));
  EXPECT_TRUE(hlo_sharding_check_eq_ignoring_metadata(hlo_sharding,
                                                      roundtrip_hlo_sharding));

  MlirAttribute mesh_name =
      mlirAttributeParseGet(context, makeStringRef("\"mesh\""));
  bool is_closed[2] = {true, false};
  int64_t priority[2] = {2, 0};
  MlirAttribute roundtrip_shardy_attribute = hloShardingToTensorShardingAttr(
      context, hlo_sharding, mesh_name, mesh_attribute, /*rank=*/2, is_closed,
      priority);
  ASSERT_NE(roundtrip_shardy_attribute.ptr, nullptr);
  std::string shardy_string =
      printAttributeToString(roundtrip_shardy_attribute);
  EXPECT_THAT(shardy_string, HasSubstr("#sdy.sharding<@mesh"));
  EXPECT_THAT(shardy_string, HasSubstr("\"x\""));
  EXPECT_THAT(shardy_string, HasSubstr("\"y\", ?"));
  EXPECT_THAT(shardy_string, HasSubstr("p2"));

  MlirAttribute single_device_mesh =
      mlirAttributeParseGet(context, makeStringRef("#sdy.mesh<[\"x\"=1]>"));
  MlirAttribute single_device_sharding = mlirAttributeParseGet(
      context, makeStringRef("#sdy.sharding<@mesh, [{\"x\"}, {}]>"));
  void *single_device_hlo_sharding = hloShardingFromTensorShardingAttr(
      single_device_sharding, single_device_mesh);

  const char *error = nullptr;
  void *ifrt_client = ifrt_make_pjrt_cpu_client(
      /*asynchronous=*/1, /*node_id=*/0, /*num_nodes=*/1,
      /*distributed_runtime_client=*/nullptr, &error);
  ASSERT_NE(ifrt_client, nullptr);
  void *ifrt_device = ifrt_ClientGetAddressableDevice(ifrt_client, 0);
  void *default_memory = ifrt_DeviceGetDefaultMemory(ifrt_device);
  void *memory_kind = ifrt_MemoryGetMemoryKind(default_memory);

  void *device_list[1] = {ifrt_device};
  void *ifrt_sharding = ifrt_sharding_from_xla_hlo_sharding(
      ifrt_client, device_list, /*num_devices=*/1, memory_kind,
      single_device_hlo_sharding);
  ASSERT_NE(ifrt_sharding, nullptr);
  void *extracted_hlo_sharding =
      ifrt_sharding_to_xla_hlo_sharding(ifrt_sharding);
  ASSERT_NE(extracted_hlo_sharding, nullptr);
  EXPECT_TRUE(hlo_sharding_check_eq(single_device_hlo_sharding,
                                    extracted_hlo_sharding));

  int64_t array_size[2] = {4, 8};
  int64_t origins[2] = {-1, -1};
  int64_t shapes[2] = {-1, -1};
  ifrt_sharding_to_index_domains(ifrt_sharding, array_size, /*array_rank=*/2,
                                 origins, shapes);
  EXPECT_THAT(origins, ElementsAre(0, 0));
  EXPECT_THAT(shapes, ElementsAre(4, 8));

  free_hlo_sharding(extracted_hlo_sharding);
  free_ifrt_sharding(ifrt_sharding);
  free_ifrt_memory_kind(memory_kind);
  ifrt_FreeClient(ifrt_client);
  free_hlo_sharding(single_device_hlo_sharding);
  free_hlo_sharding(roundtrip_hlo_sharding);
  free_op_sharding(op_sharding);
  free_hlo_sharding(hlo_sharding);
  mlirContextDestroy(context);
}

TEST(MlirHloUtilsTest, CostAnalysisAndOptionsSerialization) {
  ScopedErrorHandler error_guard;
  InitializeLogs();

  void *pjrt_client = MakeCPUClient(/*asynchronous=*/1, /*node_id=*/0);
  ASSERT_NE(pjrt_client, nullptr);

  const char *error = nullptr;
  void *ifrt_client = ifrt_make_pjrt_cpu_client(
      /*asynchronous=*/1, /*node_id=*/0, /*num_nodes=*/1,
      /*distributed_runtime_client=*/nullptr, &error);
  ASSERT_NE(ifrt_client, nullptr);

  static constexpr const char *kDotExpHlo = R"hlo(
HloModule cost_analysis_test

ENTRY %main (a: f32[8,8], b: f32[8,8]) -> f32[8,8] {
  %a = f32[8,8] parameter(0)
  %b = f32[8,8] parameter(1)
  %dot = f32[8,8] dot(f32[8,8] %a, f32[8,8] %b),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %exp = f32[8,8] exponential(f32[8,8] %dot)
}
)hlo";

  void *hlo_module = parseAndReturnUnverifiedHloModule(kDotExpHlo);
  ASSERT_NE(hlo_module, nullptr);

  JLHloCostAnalysisProperties pjrt_properties{};
  pjrt_hlo_module_cost_analysis_properties(pjrt_client, hlo_module,
                                           &pjrt_properties);
  EXPECT_GT(pjrt_properties.flops, 0.0f);
  EXPECT_GT(pjrt_properties.transcendentals, 0.0f);
  EXPECT_GT(pjrt_properties.bytes_accessed, 0.0f);
  EXPECT_GT(pjrt_properties.output_root_bytes_accessed, 0.0f);

  JLHloCostAnalysisProperties ifrt_properties{};
  ifrt_hlo_module_cost_analysis_properties(ifrt_client, hlo_module,
                                           &ifrt_properties);
  EXPECT_FLOAT_EQ(ifrt_properties.flops, pjrt_properties.flops);
  EXPECT_FLOAT_EQ(ifrt_properties.transcendentals,
                  pjrt_properties.transcendentals);
  EXPECT_FLOAT_EQ(ifrt_properties.bytes_accessed,
                  pjrt_properties.bytes_accessed);

  size_t debug_size_in_bytes = 0;
  void *debug_bytes = ReactantGetDebugOptions(&debug_size_in_bytes);
  ASSERT_NE(debug_bytes, nullptr);
  EXPECT_GT(debug_size_in_bytes, 0u);
  free(debug_bytes);

  size_t compile_size_in_bytes = 0;
  void *compile_bytes = ReactantGetCompileOptions(&compile_size_in_bytes);
  ASSERT_NE(compile_bytes, nullptr);
  EXPECT_GT(compile_size_in_bytes, 0u);

  MlirContext context = mlirContextCreate();
  RegisterDialects(context);
  static constexpr const char *kSimpleModule = R"mlir(
module @proto_compile_test {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
  MlirModule module =
      mlirModuleCreateParse(context, makeStringRef(kSimpleModule));
  ASSERT_NE(module.ptr, nullptr);

  void *pjrt_executable = ClientCompileWithProto(
      pjrt_client, module, static_cast<const uint8_t *>(compile_bytes),
      compile_size_in_bytes);
  ASSERT_NE(pjrt_executable, nullptr);
  ExecutableFree(pjrt_executable);
  free(compile_bytes);

  mlirModuleDestroy(module);
  mlirContextDestroy(context);
  FreeHloModule(hlo_module);
  ifrt_FreeClient(ifrt_client);
  FreeClient(pjrt_client);
}
