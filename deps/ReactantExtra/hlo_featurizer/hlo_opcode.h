#pragma once

#include <string>

#include "absl/strings/string_view.h"

#include "xla/hlo/ir/hlo_opcode.h"

namespace reactant {
namespace ml_lib {

using namespace xla;

#define FEATURE_WINDOW 1 << 0
#define FEATURE_OP_NON_ZERO 1 << 1
#define FEATURE_OP_ZERO 1 << 2
#define FEATURE_MODULE_NON_ZERO 1 << 3

// Node features.
const constexpr uint16_t kMinimalNodeFeatureCount = 113;
const constexpr uint16_t kOpLevelNonZeroNodeFeatureCount = 27;
const constexpr uint16_t kModuleLevelNonZeroNodeFeatureCount = 29;

// Module features.
const constexpr uint16_t kWindowConfigFeatureCount = 24;

// Config features.
const constexpr uint16_t kFusionConfigFeatureCount = 1;
const constexpr uint16_t kLayoutConfigFeatureCount = 18;
const constexpr uint16_t kDotConfigFeatureCount = 3;
// make sure to compute feature ranges after module features are finalized

inline uint8_t GetIncludeFeatureBits(absl::string_view task) {
  if (task == "op_window_cost") {
    return FEATURE_OP_NON_ZERO | FEATURE_WINDOW;
  }
  if (task == "module_fusion_cost" || task == "module_layout_cost" ||
      task == "module_dot_cost") {
    return FEATURE_OP_NON_ZERO;
  }
  return 0;
}

inline uint16_t GetNodeFeatureCount(absl::string_view task) {
  if (task == "op_window_cost") {
    return kMinimalNodeFeatureCount + kOpLevelNonZeroNodeFeatureCount;
  }
  if (task == "module_fusion_cost" || task == "module_layout_cost" ||
      task == "module_dot_cost") {
    return kMinimalNodeFeatureCount + kOpLevelNonZeroNodeFeatureCount;
  }
  return kMinimalNodeFeatureCount;
}

inline uint16_t GetModuleFeatureCount(absl::string_view task) {
  if (task == "op_window_cost") {
    return kWindowConfigFeatureCount;
  }
  return 0;
}

inline uint16_t GetConfigFeatureCount(absl::string_view task) {
  if (task == "module_fusion_cost") {
    return kFusionConfigFeatureCount;
  }
  if (task == "module_layout_cost") {
    return kLayoutConfigFeatureCount;
  }
  if (task == "module_dot_cost") {
    return kDotConfigFeatureCount;
  }
  return 0;
}

} // namespace ml_lib
} // namespace reactant
