#include "absl/strings/str_format.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "mlir/CAPI/IR.h"

#include <cstdint>
#include <string_view>
#include <vector>

#define REACTANT_ABI extern "C" MLIR_CAPI_EXPORTED

namespace reactant {
namespace reactant_ffi {

namespace ffi = xla::ffi;

// ============================================================================
// Generic Julia callback handler for custom calls.
//
// The Julia side emits a stablehlo.custom_call targeting
// "reactant_julia_callback" with api_version = 4 (TYPED_FFI). The
// backend_config dict carries a single i64 attribute "callback_ptr" that
// encodes the address of a C-callable Julia function:
//
//   bool callback(void** inputs, void** outputs, int32_t backend);
//
// Backend Values:
//   1: Host
//   2: CUDA
// ============================================================================
using JuliaCallbackFn = bool (*)(void ** /*inputs*/, void ** /*outputs*/,
                                 int32_t /*backend*/);

template <int32_t Backend>
xla::ffi::Error juliaCallback(ffi::RemainingArgs args, ffi::RemainingRets rets,
                              int64_t callback_ptr) {
  auto fn = reinterpret_cast<JuliaCallbackFn>(callback_ptr);
  if (!fn) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "reactant_julia_callback: null callback pointer");
  }

  int64_t num_inputs = static_cast<int64_t>(args.size());
  int64_t num_outputs = static_cast<int64_t>(rets.size());

  std::vector<void *> input_ptrs(num_inputs);
  for (int64_t i = 0; i < num_inputs; ++i) {
    auto buf = args.get<ffi::AnyBuffer>(i);
    if (!buf.has_value()) {
      return ffi::Error(
          ffi::ErrorCode::kInternal,
          absl::StrFormat(
              "reactant_julia_callback: failed to get input buffer %d", i));
    }
    input_ptrs[i] = buf->untyped_data();
  }

  std::vector<void *> output_ptrs(num_outputs);
  for (int64_t i = 0; i < num_outputs; ++i) {
    auto buf = rets.get<ffi::AnyBuffer>(i);
    if (!buf.has_value()) {
      return ffi::Error(
          ffi::ErrorCode::kInternal,
          absl::StrFormat(
              "reactant_julia_callback: failed to get output buffer %d", i));
    }
    output_ptrs[i] = (*buf)->untyped_data();
  }

  bool ok = fn(input_ptrs.data(), output_ptrs.data(), Backend);
  if (!ok) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "reactant_julia_callback: callback returned false");
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    juliaCallbackHandlerHost, juliaCallback<1>,
    xla::ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attr<int64_t>(
        "callback_ptr"));

#if defined(REACTANT_CUDA)
XLA_FFI_DEFINE_HANDLER(
    juliaCallbackHandlerCUDA, juliaCallback<2>,
    xla::ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attr<int64_t>(
        "callback_ptr"));
#endif

void registerReactantXLAInternalFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "Host", juliaCallbackHandlerHost);
#if defined(REACTANT_CUDA)
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "CUDA", juliaCallbackHandlerCUDA);
#endif
}

} // namespace reactant_ffi
} // namespace reactant

REACTANT_ABI void registerReactantXLAFFI() {
  reactant::reactant_ffi::registerReactantXLAInternalFFI();
}
