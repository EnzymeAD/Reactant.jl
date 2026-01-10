#include "absl/strings/str_format.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "mlir/CAPI/IR.h"

#if defined(REACTANT_CUDA)
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/blas_handle_pool.h"
#endif

#define REACTANT_ABI extern "C" MLIR_CAPI_EXPORTED

using namespace xla;

namespace reactant {
namespace cuda {

#if defined(REACTANT_CUDA)

#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp8.h"
#include "third_party/gpus/cuda/include/cufft.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "third_party/gpus/cuda/include/cusolver_common.h"

using namespace jax;

#define SOLVER_BLAS_DISPATCH_IMPL(impl, ...)                                   \
  switch (dataType) {                                                          \
  case ffi::F32:                                                               \
    return impl<float>(__VA_ARGS__);                                           \
  case ffi::F64:                                                               \
    return impl<double>(__VA_ARGS__);                                          \
  case ffi::C64:                                                               \
    return impl<cuComplex>(__VA_ARGS__);                                       \
  case ffi::C128:                                                              \
    return impl<cuDoubleComplex>(__VA_ARGS__);                                 \
  default:                                                                     \
    break;                                                                     \
  }

template <typename T>
ffi::Error GetHostScalar(double value_real, double value_imag, T *host_value) {
  if constexpr (std::is_same<T, float>::value) {
    *host_value = static_cast<float>(value_real);
  } else if constexpr (std::is_same<T, double>::value) {
    *host_value = value_real;
  } else if constexpr (std::is_same<T, cuComplex>::value) {
    *host_value = cuComplex{static_cast<float>(value_real),
                            static_cast<float>(value_imag)};
  } else if constexpr (std::is_same<T, cuDoubleComplex>::value) {
    *host_value = cuDoubleComplex{value_real, value_imag};
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error GetHostScalar(CUstream stream, ffi::AnyBuffer buffer,
                         T *host_value) {
  // Ensure buffer has exactly 1 element
  if (buffer.element_count() != 1) {
    return ffi::Error::InvalidArgument(
        absl::StrFormat("Expected scalar buffer with 1 element, got %d",
                        buffer.element_count()));
  }
  // memcpy to host
  cudaMemcpyAsync(host_value, buffer.untyped_data(), sizeof(T),
                  cudaMemcpyDeviceToHost, stream);
  return ffi::Error::Success();
}

template <typename T>
ffi::Error GetHostScalar(CUstream stream, bool use_attribute, double value_real,
                         double value_imag, ffi::AnyBuffer buffer,
                         T *host_value) {
  if (use_attribute) {
    return GetHostScalar<T>(value_real, value_imag, host_value);
  }
  return GetHostScalar<T>(stream, buffer, host_value);
}

inline ffi::Error CublasStatusToError(cublasStatus_t status,
                                      const char *op_name) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return ffi::Error::Success();
  }
  const char *error_name;
  switch (status) {
  case CUBLAS_STATUS_NOT_INITIALIZED:
    error_name = "CUBLAS_STATUS_NOT_INITIALIZED";
    break;
  case CUBLAS_STATUS_ALLOC_FAILED:
    error_name = "CUBLAS_STATUS_ALLOC_FAILED";
    break;
  case CUBLAS_STATUS_INVALID_VALUE:
    error_name = "CUBLAS_STATUS_INVALID_VALUE";
    break;
  case CUBLAS_STATUS_ARCH_MISMATCH:
    error_name = "CUBLAS_STATUS_ARCH_MISMATCH";
    break;
  case CUBLAS_STATUS_MAPPING_ERROR:
    error_name = "CUBLAS_STATUS_MAPPING_ERROR";
    break;
  case CUBLAS_STATUS_EXECUTION_FAILED:
    error_name = "CUBLAS_STATUS_EXECUTION_FAILED";
    break;
  case CUBLAS_STATUS_INTERNAL_ERROR:
    error_name = "CUBLAS_STATUS_INTERNAL_ERROR";
    break;
  case CUBLAS_STATUS_NOT_SUPPORTED:
    error_name = "CUBLAS_STATUS_NOT_SUPPORTED";
    break;
  default:
    error_name = "UNKNOWN";
    break;
  }
  return ffi::Error::InvalidArgument(
      absl::StrFormat("%s failed with status %s", op_name, error_name));
}

namespace blas {

template <typename T>
ffi::Error Syrk(cublasHandle_t handle, cublasFillMode_t uplo,
                cublasOperation_t trans, int n, int k, const T *alpha,
                const T *a, int lda, const T *beta, T *c, int ldc) {
  return ffi::Error::InvalidArgument("Unsupported type for syrk");
}

#define SYRK_SPECIALIZATION(T, cublas_func)                                    \
  template <>                                                                  \
  ffi::Error Syrk<T>(cublasHandle_t handle, cublasFillMode_t uplo,             \
                     cublasOperation_t trans, int n, int k, const T *alpha,    \
                     const T *a, int lda, const T *beta, T *c, int ldc) {      \
    cublasStatus_t status =                                                    \
        cublas_func(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);   \
    return CublasStatusToError(status, #cublas_func);                          \
  }

SYRK_SPECIALIZATION(float, cublasSsyrk)
SYRK_SPECIALIZATION(double, cublasDsyrk)
SYRK_SPECIALIZATION(cuComplex, cublasCsyrk)
SYRK_SPECIALIZATION(cuDoubleComplex, cublasZsyrk)

#undef SYRK_SPECIALIZATION

} // namespace blas

// Symmetric rank-k update: syrk

template <typename T>
ffi::Error SyrkImpl(CUstream stream, bool transpose, bool uplo_,
                    ffi::AnyBuffer a, const T *alpha, const T *beta,
                    ffi::Result<ffi::AnyBuffer> c_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  auto size = transpose ? cols : rows;
  FFI_RETURN_IF_ERROR(
      CheckShape(c_out->dimensions(), {batch, size, size}, "c_out", "syrk"));

  FFI_ASSIGN_OR_RETURN(auto n,
                       MaybeCastNoOverflow<int>(transpose ? cols : rows));
  FFI_ASSIGN_OR_RETURN(auto k,
                       MaybeCastNoOverflow<int>(transpose ? rows : cols));
  // We flip uplo here because C is passed in row-major format.
  // Row-major C is equivalent to C^T in column-major, and since C is
  // symmetric, this means we need to swap upper/lower triangular.
  cublasFillMode_t uplo =
      uplo_ ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  // We intentionally flip transpose here, this allows us to pass in
  // the data as a row-major format without paying the cost of
  // layout transformation to a col-major (which cuBLAS uses)
  cublasOperation_t trans = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;

  const T *a_data = static_cast<const T *>(a.untyped_data());
  T *c_out_data = static_cast<T *>(c_out->untyped_data());

  FFI_ASSIGN_OR_RETURN(auto handle, BlasHandlePool::Borrow(stream));
  // lda is the leading dimension of a, ldc is the leading dimension of c
  // For column-major (which cuBLAS uses), lda = number of rows of a, ldc = n
  int lda = trans == CUBLAS_OP_N ? n : k;
  int ldc = n;
  for (int i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR(blas::Syrk<T>(handle.get(), uplo, trans, n, k, alpha,
                                      a_data, lda, beta, c_out_data, ldc));
    a_data += k * n;
    c_out_data += n * n;
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error SyrkImpl(CUstream stream, bool transpose, bool uplo_,
                    ffi::AnyBuffer a, ffi::AnyBuffer c_in, const T *alpha,
                    const T *beta, ffi::Result<ffi::AnyBuffer> c_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  auto size = transpose ? cols : rows;
  FFI_RETURN_IF_ERROR(
      CheckShape(c_in.dimensions(), {batch, size, size}, "c_in", "syrk"));

  T *c_data = static_cast<T *>(c_in.untyped_data());
  T *c_out_data = static_cast<T *>(c_out->untyped_data());

  if (c_data != c_out_data) {
    cudaError_t err = cudaMemcpyAsync(c_out_data, c_data, c_in.size_bytes(),
                                      cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      return ffi::Error::InvalidArgument(absl::StrFormat(
          "cudaMemcpyAsync failed: %s", cudaGetErrorString(err)));
    }
  }
  return SyrkImpl<T>(stream, transpose, uplo_, a, alpha, beta, c_out);
}

template <typename T>
ffi::Error SyrkImpl(CUstream stream, bool transpose, bool uplo,
                    bool use_alpha_attribute, double alpha_real,
                    double alpha_imag, bool use_beta_attribute,
                    double beta_real, double beta_imag, ffi::AnyBuffer a,
                    ffi::AnyBuffer c_in, ffi::AnyBuffer alpha_,
                    ffi::AnyBuffer beta_, ffi::Result<ffi::AnyBuffer> c_out) {
  T host_alpha, host_beta;
  FFI_RETURN_IF_ERROR(GetHostScalar<T>(stream, use_alpha_attribute, alpha_real,
                                       alpha_imag, alpha_, &host_alpha));
  FFI_RETURN_IF_ERROR(GetHostScalar<T>(stream, use_beta_attribute, beta_real,
                                       beta_imag, beta_, &host_beta));
  return SyrkImpl<T>(stream, transpose, uplo, a, c_in, &host_alpha, &host_beta,
                     c_out);
}

template <typename T>
ffi::Error SyrkImpl(CUstream stream, bool transpose, bool uplo,
                    bool use_alpha_attribute, double alpha_real,
                    double alpha_imag, ffi::AnyBuffer a, ffi::AnyBuffer alpha_,
                    ffi::Result<ffi::AnyBuffer> c_out) {
  T host_alpha, host_beta;
  FFI_RETURN_IF_ERROR(GetHostScalar<T>(stream, use_alpha_attribute, alpha_real,
                                       alpha_imag, alpha_, &host_alpha));
  FFI_RETURN_IF_ERROR(GetHostScalar<T>(0.0, 0.0, &host_beta));
  return SyrkImpl<T>(stream, transpose, uplo, a, &host_alpha, &host_beta,
                     c_out);
}

ffi::Error SyrkDispatch(CUstream stream, bool transpose, bool uplo,
                        bool use_alpha_attribute, double alpha_real,
                        double alpha_imag, bool use_beta_attribute,
                        double beta_real, double beta_imag, ffi::AnyBuffer a,
                        ffi::AnyBuffer c_in, ffi::AnyBuffer alpha_,
                        ffi::AnyBuffer beta_,
                        ffi::Result<ffi::AnyBuffer> c_out) {
  auto dataType = c_in.element_type();
  SOLVER_BLAS_DISPATCH_IMPL(SyrkImpl, stream, transpose, uplo,
                            use_alpha_attribute, alpha_real, alpha_imag,
                            use_beta_attribute, beta_real, beta_imag, a, c_in,
                            alpha_, beta_, c_out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in syrk", absl::FormatStreamed(dataType)));
}

ffi::Error SyrkNoCDispatch(CUstream stream, bool transpose, bool uplo,
                           bool use_alpha_attribute, double alpha_real,
                           double alpha_imag, ffi::AnyBuffer a,
                           ffi::AnyBuffer alpha_,
                           ffi::Result<ffi::AnyBuffer> c_out) {
  auto dataType = a.element_type();
  SOLVER_BLAS_DISPATCH_IMPL(SyrkImpl, stream, transpose, uplo,
                            use_alpha_attribute, alpha_real, alpha_imag, a,
                            alpha_, c_out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in syrk", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER(
    SyrkFfi, SyrkDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<CUstream>>()
        .Attr<bool>("transpose")           // transpose
        .Attr<bool>("uplo")                // uplo
        .Attr<bool>("use_alpha_attribute") // use_alpha_attribute
        .Attr<double>("alpha_real")        // alpha_real
        .Attr<double>("alpha_imag")        // alpha_imag
        .Attr<bool>("use_beta_attribute")  // use_beta_attribute
        .Attr<double>("beta_real")         // beta_real
        .Attr<double>("beta_imag")         // beta_imag
        .Arg<ffi::AnyBuffer>()             // a
        .Arg<ffi::AnyBuffer>()             // c_in
        .Arg<ffi::AnyBuffer>()             // alpha
        .Arg<ffi::AnyBuffer>()             // beta
        .Ret<ffi::AnyBuffer>()             // c_out
);

XLA_FFI_DEFINE_HANDLER(
    SyrkNoCFfi, SyrkNoCDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<CUstream>>()
        .Attr<bool>("transpose")           // transpose
        .Attr<bool>("uplo")                // uplo
        .Attr<bool>("use_alpha_attribute") // use_alpha_attribute
        .Attr<double>("alpha_real")        // alpha_real
        .Attr<double>("alpha_imag")        // alpha_imag
        .Arg<ffi::AnyBuffer>()             // a
        .Arg<ffi::AnyBuffer>()             // alpha
        .Ret<ffi::AnyBuffer>()             // c_out
);

void registerReactantXLACUDAFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_cublas_syrk_ffi",
                           "CUDA", SyrkFfi);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                           "reactant_cublas_syrk_no_c_ffi", "CUDA", SyrkNoCFfi);
}

#undef SOLVER_BLAS_DISPATCH_IMPL

#else

void registerReactantXLACUDAFFI() {}

#endif

} // namespace cuda
} // namespace reactant

REACTANT_ABI void registerReactantXLAFFI() {
  reactant::cuda::registerReactantXLACUDAFFI();
  return;
}
