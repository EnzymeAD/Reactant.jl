#include "absl/strings/str_format.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#if defined(REACTANT_CUDA)
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif

#if defined(REACTANT_ROCM)
#ifndef JAX_GPU_HIP
#define JAX_GPU_HIP 1
#endif
#include "jaxlib/gpu/digamma_kernels.h"
#include "jaxlib/gpu/lgamma_kernels.h"
#include "jaxlib/gpu/linalg_kernels.h"
#include "jaxlib/gpu/prng_kernels.h"
#include "jaxlib/gpu/rnn_kernels.h"
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/sparse_kernels.h"
#include "jaxlib/gpu/vendor.h"
#endif

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
//   bool callback(void** inputs, void** outputs, int32_t backend,
//                 void* stream);
//
// Backend Values:
//   1: Host (stream is null)
//   2: CUDA (stream is XLA's cudaStream_t; the Julia side is responsible for
//      ordering its work with respect to it)
//   3: ROCM (stream is XLA's hipStream_t; the Julia side is responsible for
//      ordering its work with respect to it)
// ============================================================================
using JuliaCallbackFn = bool (*)(void ** /*inputs*/, void ** /*outputs*/,
                                 int32_t /*backend*/, void * /*stream*/);

template <int32_t Backend>
xla::ffi::Error juliaCallback(void *stream, ffi::RemainingArgs args,
                              ffi::RemainingRets rets, int64_t callback_ptr) {
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

  bool ok = fn(input_ptrs.data(), output_ptrs.data(), Backend, stream);
  if (!ok) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "reactant_julia_callback: callback returned false");
  }

  return ffi::Error::Success();
}

xla::ffi::Error juliaHostCallback(ffi::RemainingArgs args,
                                  ffi::RemainingRets rets,
                                  int64_t callback_ptr) {
  return juliaCallback<1>(nullptr, args, rets, callback_ptr);
}

XLA_FFI_DEFINE_HANDLER(
    juliaCallbackHandlerHost, juliaHostCallback,
    xla::ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attr<int64_t>(
        "callback_ptr"));

#if defined(REACTANT_CUDA)

xla::ffi::Error juliaCUDACallback(cudaStream_t stream, ffi::RemainingArgs args,
                                  ffi::RemainingRets rets,
                                  int64_t callback_ptr) {
  return juliaCallback<2>(stream, args, rets, callback_ptr);
}

XLA_FFI_DEFINE_HANDLER(juliaCallbackHandlerCUDA, juliaCUDACallback,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<int64_t>("callback_ptr"));

#endif

#if defined(REACTANT_ROCM)

xla::ffi::Error juliaROCMCallback(hipStream_t stream, ffi::RemainingArgs args,
                                  ffi::RemainingRets rets,
                                  int64_t callback_ptr) {
  return juliaCallback<3>(stream, args, rets, callback_ptr);
}

XLA_FFI_DEFINE_HANDLER(juliaCallbackHandlerROCM, juliaROCMCallback,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<hipStream_t>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<int64_t>("callback_ptr"));

#endif

void registerReactantXLAInternalFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "Host", juliaCallbackHandlerHost);
#if defined(REACTANT_CUDA)
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "CUDA", juliaCallbackHandlerCUDA);
#endif
#if defined(REACTANT_ROCM)
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "ROCM", juliaCallbackHandlerROCM);

  // Upstream JAX provides @jax//jaxlib/cuda:cuda_gpu_kernels for static CUDA
  // FFI registration, but has no equivalent rocm_gpu_kernels target. Register
  // all JAX ROCm FFI handlers on "ROCM" directly.
  static const auto *api = xla::ffi::GetXlaFfiApi();
  XLA_FFI_REGISTER_HANDLER(api, "hipdnn_rnn_ffi_v2", "ROCM",
                           ::jax::hip::kRNNForward);
  XLA_FFI_REGISTER_HANDLER(api, "hipdnn_rnn_bwd_ffi_v2", "ROCM",
                           ::jax::hip::kRNNBackward);

  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_getrf_ffi", "ROCM",
                           ::jax::hip::GetrfFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_syrk_ffi", "ROCM",
                           ::jax::hip::SyrkFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_geqrf_ffi", "ROCM",
                           ::jax::hip::GeqrfFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_orgqr_ffi", "ROCM",
                           ::jax::hip::OrgqrFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_ormqr_ffi", "ROCM",
                           ::jax::hip::OrmqrFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_potrf_ffi", "ROCM",
                           ::jax::hip::PotrfFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_syevd_ffi", "ROCM",
                           ::jax::hip::SyevdFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_sytrd_ffi", "ROCM",
                           ::jax::hip::SytrdFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_gesvd_ffi", "ROCM",
                           ::jax::hip::GesvdFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_gesvdj_ffi", "ROCM",
                           ::jax::hip::GesvdjFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hipsolver_gesdd_ffi", "ROCM",
                           ::jax::hip::GesddFfi);

  XLA_FFI_REGISTER_HANDLER(api, "hip_cholesky_update_ffi", "ROCM",
                           ::jax::hip::CholeskyUpdateFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hip_digamma_ffi", "ROCM",
                           ::jax::hip::DigammaFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hip_lgamma_ffi", "ROCM",
                           ::jax::hip::LgammaFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hip_lu_pivots_to_permutation", "ROCM",
                           ::jax::hip::LuPivotsToPermutation);
  XLA_FFI_REGISTER_HANDLER(api, "hip_tridiagonal_solve_perturbed", "ROCM",
                           ::jax::hip::TridiagonalSolvePerturbedFfi);
  XLA_FFI_REGISTER_HANDLER(api, "hip_threefry2x32_ffi", "ROCM",
                           ::jax::hip::ThreeFry2x32Ffi);

  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_csr_todense_ffi_v2", "ROCM",
                           ::jax::hip::kCsrToDense);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_csr_fromdense_ffi_v2", "ROCM",
                           ::jax::hip::kCsrFromDense);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_csr_matvec_ffi_v2", "ROCM",
                           ::jax::hip::kCsrMatvec);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_csr_matmat_ffi_v2", "ROCM",
                           ::jax::hip::kCsrMatmat);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_coo_todense_ffi_v2", "ROCM",
                           ::jax::hip::kCooToDense);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_coo_fromdense_ffi_v2", "ROCM",
                           ::jax::hip::kCooFromDense);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_coo_matvec_ffi_v2", "ROCM",
                           ::jax::hip::kCooMatvec);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_coo_matmat_ffi_v2", "ROCM",
                           ::jax::hip::kCooMatmat);
  XLA_FFI_REGISTER_HANDLER(api, "hipsparse_gtsv2_ffi", "ROCM",
                           ::jax::hip::kGtsv2);
#endif
}

} // namespace reactant_ffi
} // namespace reactant

REACTANT_ABI void registerReactantXLAFFI() {
  reactant::reactant_ffi::registerReactantXLAInternalFFI();
}
