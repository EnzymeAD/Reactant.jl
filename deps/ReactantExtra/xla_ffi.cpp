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

// ============================================================================
// CSR sparse matrix products (spmv / spmm) via cuSPARSE / hipSPARSE.
//
// The Enzyme-JAX pass `lower-sparse-csr` emits stablehlo.custom_calls with
// api_version = 4 (TYPED_FFI) targeting either
//   - "reactant_csr_matmul":     out = alpha * A * dense, with operands
//     (rowptr, colind, nzval, dense), or
//   - "reactant_csr_matmul_acc": out = alpha * A * dense + beta * acc, with
//     operands (rowptr, colind, nzval, dense, acc) and acc aliased to out.
// Column-major layouts are pinned; the result is dense. The backend_config
// dict carries i64 attributes "m", "n", "transpose" (must be 0 for now) and
// "index_base" (0 or 1; sparse_tensor-dialect buffers are 0-based), plus f64
// "alpha" (and "beta" for the accumulating variant).
// ============================================================================

#if defined(REACTANT_CUDA)
#include <cuda_runtime_api.h>
#include <cusparse.h>

#define REACTANT_CUSPARSE_RET(expr)                                            \
  do {                                                                         \
    cusparseStatus_t status__ = (expr);                                        \
    if (status__ != CUSPARSE_STATUS_SUCCESS) {                                 \
      return ffi::Error(ffi::ErrorCode::kInternal,                             \
                        absl::StrFormat("reactant_csr_matmul: %s failed: %s",  \
                                        #expr,                                 \
                                        cusparseGetErrorString(status__)));    \
    }                                                                          \
  } while (0)

// Computes out = alpha_v * A * dense (+ beta_v * *acc when acc != nullptr;
// the accumulated-into operand is aliased to out by the lowering, so it is
// copied into out first if XLA did not reuse the buffer).
static ffi::Error csrMatmulCudaImpl(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, ffi::AnyBuffer rowptr,
    ffi::AnyBuffer colind, ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
    ffi::AnyBuffer *acc, ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
    int64_t transpose, int64_t index_base, double alpha_v, double beta_v) {
  if (transpose != 0) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "reactant_csr_matmul: transposed products are not supported");
  }
  if (colind.element_type() != rowptr.element_type()) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "reactant_csr_matmul: rowptr and colind dtypes must match");
  }

  cusparseIndexType_t index_type;
  switch (rowptr.element_type()) {
  case ffi::DataType::S32:
    index_type = CUSPARSE_INDEX_32I;
    break;
  case ffi::DataType::S64:
    index_type = CUSPARSE_INDEX_64I;
    break;
  default:
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "reactant_csr_matmul: index buffers must be i32 or i64");
  }

  const float alpha_f = static_cast<float>(alpha_v),
              beta_f = static_cast<float>(beta_v);
  const double alpha_d = alpha_v, beta_d = beta_v;
  cudaDataType value_type;
  size_t value_bytes;
  const void *alpha, *beta;
  switch (nzval.element_type()) {
  case ffi::DataType::F32:
    value_type = CUDA_R_32F;
    value_bytes = sizeof(float);
    alpha = &alpha_f;
    beta = &beta_f;
    break;
  case ffi::DataType::F64:
    value_type = CUDA_R_64F;
    value_bytes = sizeof(double);
    alpha = &alpha_d;
    beta = &beta_d;
    break;
  default:
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "reactant_csr_matmul: only f32 and f64 values are supported");
  }
  if (dense.element_type() != nzval.element_type() ||
      out->element_type() != nzval.element_type() ||
      (acc != nullptr && acc->element_type() != nzval.element_type())) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "reactant_csr_matmul: value, operand and result dtypes must match");
  }
  if (acc != nullptr && acc->untyped_data() != out->untyped_data()) {
    cudaError_t copy_status =
        cudaMemcpyAsync(out->untyped_data(), acc->untyped_data(),
                        static_cast<size_t>(out->element_count()) * value_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    if (copy_status != cudaSuccess) {
      return ffi::Error(
          ffi::ErrorCode::kInternal,
          absl::StrFormat(
              "reactant_csr_matmul: copying the accumulated-into operand "
              "failed: %s",
              cudaGetErrorString(copy_status)));
    }
  }

  static thread_local cusparseHandle_t handle = nullptr;
  if (handle == nullptr) {
    REACTANT_CUSPARSE_RET(cusparseCreate(&handle));
  }
  REACTANT_CUSPARSE_RET(cusparseSetStream(handle, stream));

  int64_t nnz = colind.element_count();
  cusparseSpMatDescr_t mat_a;
  REACTANT_CUSPARSE_RET(cusparseCreateCsr(
      &mat_a, m, n, nnz, rowptr.untyped_data(), colind.untyped_data(),
      nzval.untyped_data(), index_type, index_type,
      index_base == 1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
      value_type));

  auto with_workspace = [&](size_t buffer_size, auto &&compute) -> ffi::Error {
    void *workspace = nullptr;
    if (buffer_size > 0) {
      auto maybe_workspace = scratch.Allocate(buffer_size);
      if (!maybe_workspace.has_value()) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "reactant_csr_matmul: failed to allocate workspace");
      }
      workspace = *maybe_workspace;
    }
    return compute(workspace);
  };

  ffi::Error err = ffi::Error::Success();
  int64_t rank = dense.dimensions().size();
  if (rank == 1) {
    cusparseDnVecDescr_t vec_x, vec_y;
    REACTANT_CUSPARSE_RET(
        cusparseCreateDnVec(&vec_x, n, dense.untyped_data(), value_type));
    REACTANT_CUSPARSE_RET(
        cusparseCreateDnVec(&vec_y, m, out->untyped_data(), value_type));
    err = [&]() -> ffi::Error {
      size_t buffer_size = 0;
      REACTANT_CUSPARSE_RET(cusparseSpMV_bufferSize(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, vec_x, beta,
          vec_y, value_type, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
      return with_workspace(buffer_size, [&](void *workspace) -> ffi::Error {
        REACTANT_CUSPARSE_RET(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, vec_x, beta,
            vec_y, value_type, CUSPARSE_SPMV_ALG_DEFAULT, workspace));
        return ffi::Error::Success();
      });
    }();
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
  } else if (rank == 2) {
    // Layouts are pinned column-major by the Julia rewrite.
    int64_t k = dense.dimensions()[0];
    int64_t c = dense.dimensions()[1];
    cusparseDnMatDescr_t mat_b, mat_c;
    REACTANT_CUSPARSE_RET(cusparseCreateDnMat(&mat_b, k, c, /*ld=*/k,
                                              dense.untyped_data(), value_type,
                                              CUSPARSE_ORDER_COL));
    REACTANT_CUSPARSE_RET(cusparseCreateDnMat(&mat_c, m, c, /*ld=*/m,
                                              out->untyped_data(), value_type,
                                              CUSPARSE_ORDER_COL));
    err = [&]() -> ffi::Error {
      size_t buffer_size = 0;
      REACTANT_CUSPARSE_RET(cusparseSpMM_bufferSize(
          handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, mat_b, beta, mat_c,
          value_type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
      return with_workspace(buffer_size, [&](void *workspace) -> ffi::Error {
        REACTANT_CUSPARSE_RET(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, mat_b, beta, mat_c,
            value_type, CUSPARSE_SPMM_ALG_DEFAULT, workspace));
        return ffi::Error::Success();
      });
    }();
    cusparseDestroyDnMat(mat_b);
    cusparseDestroyDnMat(mat_c);
  } else {
    err =
        ffi::Error(ffi::ErrorCode::kInvalidArgument,
                   "reactant_csr_matmul: dense operand must have rank 1 or 2");
  }
  cusparseDestroySpMat(mat_a);
  return err;
}

static ffi::Error csrMatmulCuda(cudaStream_t stream,
                                ffi::ScratchAllocator scratch,
                                ffi::AnyBuffer rowptr, ffi::AnyBuffer colind,
                                ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
                                ffi::Result<ffi::AnyBuffer> out, int64_t m,
                                int64_t n, int64_t transpose,
                                int64_t index_base, double alpha) {
  return csrMatmulCudaImpl(stream, scratch, rowptr, colind, nzval, dense,
                           /*acc=*/nullptr, out, m, n, transpose, index_base,
                           alpha, /*beta_v=*/0.0);
}

static ffi::Error csrMatmulAccCuda(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer rowptr,
    ffi::AnyBuffer colind, ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
    ffi::AnyBuffer acc, ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
    int64_t transpose, int64_t index_base, double alpha, double beta) {
  return csrMatmulCudaImpl(stream, scratch, rowptr, colind, nzval, dense, &acc,
                           out, m, n, transpose, index_base, alpha, beta);
}

XLA_FFI_DEFINE_HANDLER(csrMatmulHandlerCUDA, csrMatmulCuda,
                       xla::ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::AnyBuffer>() // rowptr
                           .Arg<ffi::AnyBuffer>() // colind
                           .Arg<ffi::AnyBuffer>() // nzval
                           .Arg<ffi::AnyBuffer>() // dense
                           .Ret<ffi::AnyBuffer>() // out
                           .Attr<int64_t>("m")
                           .Attr<int64_t>("n")
                           .Attr<int64_t>("transpose")
                           .Attr<int64_t>("index_base")
                           .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER(csrMatmulAccHandlerCUDA, csrMatmulAccCuda,
                       xla::ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::AnyBuffer>() // rowptr
                           .Arg<ffi::AnyBuffer>() // colind
                           .Arg<ffi::AnyBuffer>() // nzval
                           .Arg<ffi::AnyBuffer>() // dense
                           .Arg<ffi::AnyBuffer>() // acc (aliased to out)
                           .Ret<ffi::AnyBuffer>() // out
                           .Attr<int64_t>("m")
                           .Attr<int64_t>("n")
                           .Attr<int64_t>("transpose")
                           .Attr<int64_t>("index_base")
                           .Attr<double>("alpha")
                           .Attr<double>("beta"));
#endif // REACTANT_CUDA

#if defined(REACTANT_ROCM)
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>

#define REACTANT_HIPSPARSE_RET(expr)                                           \
  do {                                                                         \
    hipsparseStatus_t status__ = (expr);                                       \
    if (status__ != HIPSPARSE_STATUS_SUCCESS) {                                \
      return ffi::Error(                                                       \
          ffi::ErrorCode::kInternal,                                           \
          absl::StrFormat("reactant_csr_matmul: %s failed: hipSPARSE status "  \
                          "%d",                                                \
                          #expr, static_cast<int>(status__)));                 \
    }                                                                          \
  } while (0)

// Computes out = alpha_v * A * dense (+ beta_v * *acc when acc != nullptr;
// the accumulated-into operand is aliased to out by the lowering, so it is
// copied into out first if XLA did not reuse the buffer).
static ffi::Error csrMatmulRocmImpl(
    hipStream_t stream, ffi::ScratchAllocator &scratch, ffi::AnyBuffer rowptr,
    ffi::AnyBuffer colind, ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
    ffi::AnyBuffer *acc, ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
    int64_t transpose, int64_t index_base, double alpha_v, double beta_v) {
  if (transpose != 0) {
    return ffi::Error(
        ffi::ErrorCode::kUnimplemented,
        "reactant_csr_matmul: transposed products are not supported");
  }
  if (colind.element_type() != rowptr.element_type()) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "reactant_csr_matmul: rowptr and colind dtypes must match");
  }

  hipsparseIndexType_t index_type;
  switch (rowptr.element_type()) {
  case ffi::DataType::S32:
    index_type = HIPSPARSE_INDEX_32I;
    break;
  case ffi::DataType::S64:
    index_type = HIPSPARSE_INDEX_64I;
    break;
  default:
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "reactant_csr_matmul: index buffers must be i32 or i64");
  }

  const float alpha_f = static_cast<float>(alpha_v),
              beta_f = static_cast<float>(beta_v);
  const double alpha_d = alpha_v, beta_d = beta_v;
  hipDataType value_type;
  size_t value_bytes;
  const void *alpha, *beta;
  switch (nzval.element_type()) {
  case ffi::DataType::F32:
    value_type = HIP_R_32F;
    value_bytes = sizeof(float);
    alpha = &alpha_f;
    beta = &beta_f;
    break;
  case ffi::DataType::F64:
    value_type = HIP_R_64F;
    value_bytes = sizeof(double);
    alpha = &alpha_d;
    beta = &beta_d;
    break;
  default:
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "reactant_csr_matmul: only f32 and f64 values are supported");
  }
  if (dense.element_type() != nzval.element_type() ||
      out->element_type() != nzval.element_type() ||
      (acc != nullptr && acc->element_type() != nzval.element_type())) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "reactant_csr_matmul: value, operand and result dtypes must match");
  }
  if (acc != nullptr && acc->untyped_data() != out->untyped_data()) {
    hipError_t copy_status =
        hipMemcpyAsync(out->untyped_data(), acc->untyped_data(),
                       static_cast<size_t>(out->element_count()) * value_bytes,
                       hipMemcpyDeviceToDevice, stream);
    if (copy_status != hipSuccess) {
      return ffi::Error(
          ffi::ErrorCode::kInternal,
          absl::StrFormat(
              "reactant_csr_matmul: copying the accumulated-into operand "
              "failed: %s",
              hipGetErrorString(copy_status)));
    }
  }

  static thread_local hipsparseHandle_t handle = nullptr;
  if (handle == nullptr) {
    REACTANT_HIPSPARSE_RET(hipsparseCreate(&handle));
  }
  REACTANT_HIPSPARSE_RET(hipsparseSetStream(handle, stream));

  int64_t nnz = colind.element_count();
  hipsparseSpMatDescr_t mat_a;
  REACTANT_HIPSPARSE_RET(hipsparseCreateCsr(
      &mat_a, m, n, nnz, rowptr.untyped_data(), colind.untyped_data(),
      nzval.untyped_data(), index_type, index_type,
      index_base == 1 ? HIPSPARSE_INDEX_BASE_ONE : HIPSPARSE_INDEX_BASE_ZERO,
      value_type));

  auto with_workspace = [&](size_t buffer_size, auto &&compute) -> ffi::Error {
    void *workspace = nullptr;
    if (buffer_size > 0) {
      auto maybe_workspace = scratch.Allocate(buffer_size);
      if (!maybe_workspace.has_value()) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "reactant_csr_matmul: failed to allocate workspace");
      }
      workspace = *maybe_workspace;
    }
    return compute(workspace);
  };

  ffi::Error err = ffi::Error::Success();
  int64_t rank = dense.dimensions().size();
  if (rank == 1) {
    hipsparseDnVecDescr_t vec_x, vec_y;
    REACTANT_HIPSPARSE_RET(
        hipsparseCreateDnVec(&vec_x, n, dense.untyped_data(), value_type));
    REACTANT_HIPSPARSE_RET(
        hipsparseCreateDnVec(&vec_y, m, out->untyped_data(), value_type));
    err = [&]() -> ffi::Error {
      size_t buffer_size = 0;
      REACTANT_HIPSPARSE_RET(hipsparseSpMV_bufferSize(
          handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, vec_x, beta,
          vec_y, value_type, HIPSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
      return with_workspace(buffer_size, [&](void *workspace) -> ffi::Error {
        REACTANT_HIPSPARSE_RET(hipsparseSpMV(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, vec_x,
            beta, vec_y, value_type, HIPSPARSE_SPMV_ALG_DEFAULT, workspace));
        return ffi::Error::Success();
      });
    }();
    hipsparseDestroyDnVec(vec_x);
    hipsparseDestroyDnVec(vec_y);
  } else if (rank == 2) {
    // Layouts are pinned column-major by the Julia rewrite.
    int64_t k = dense.dimensions()[0];
    int64_t c = dense.dimensions()[1];
    hipsparseDnMatDescr_t mat_b, mat_c;
    REACTANT_HIPSPARSE_RET(
        hipsparseCreateDnMat(&mat_b, k, c, /*ld=*/k, dense.untyped_data(),
                             value_type, HIPSPARSE_ORDER_COL));
    REACTANT_HIPSPARSE_RET(hipsparseCreateDnMat(&mat_c, m, c, /*ld=*/m,
                                                out->untyped_data(), value_type,
                                                HIPSPARSE_ORDER_COL));
    err = [&]() -> ffi::Error {
      size_t buffer_size = 0;
      REACTANT_HIPSPARSE_RET(hipsparseSpMM_bufferSize(
          handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
          HIPSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, mat_b, beta, mat_c,
          value_type, HIPSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
      return with_workspace(buffer_size, [&](void *workspace) -> ffi::Error {
        REACTANT_HIPSPARSE_RET(hipsparseSpMM(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            HIPSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_a, mat_b, beta, mat_c,
            value_type, HIPSPARSE_SPMM_ALG_DEFAULT, workspace));
        return ffi::Error::Success();
      });
    }();
    hipsparseDestroyDnMat(mat_b);
    hipsparseDestroyDnMat(mat_c);
  } else {
    err =
        ffi::Error(ffi::ErrorCode::kInvalidArgument,
                   "reactant_csr_matmul: dense operand must have rank 1 or 2");
  }
  hipsparseDestroySpMat(mat_a);
  return err;
}

static ffi::Error csrMatmulRocm(hipStream_t stream,
                                ffi::ScratchAllocator scratch,
                                ffi::AnyBuffer rowptr, ffi::AnyBuffer colind,
                                ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
                                ffi::Result<ffi::AnyBuffer> out, int64_t m,
                                int64_t n, int64_t transpose,
                                int64_t index_base, double alpha) {
  return csrMatmulRocmImpl(stream, scratch, rowptr, colind, nzval, dense,
                           /*acc=*/nullptr, out, m, n, transpose, index_base,
                           alpha, /*beta_v=*/0.0);
}

static ffi::Error csrMatmulAccRocm(
    hipStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer rowptr,
    ffi::AnyBuffer colind, ffi::AnyBuffer nzval, ffi::AnyBuffer dense,
    ffi::AnyBuffer acc, ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
    int64_t transpose, int64_t index_base, double alpha, double beta) {
  return csrMatmulRocmImpl(stream, scratch, rowptr, colind, nzval, dense, &acc,
                           out, m, n, transpose, index_base, alpha, beta);
}

XLA_FFI_DEFINE_HANDLER(csrMatmulHandlerROCM, csrMatmulRocm,
                       xla::ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<hipStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::AnyBuffer>() // rowptr
                           .Arg<ffi::AnyBuffer>() // colind
                           .Arg<ffi::AnyBuffer>() // nzval
                           .Arg<ffi::AnyBuffer>() // dense
                           .Ret<ffi::AnyBuffer>() // out
                           .Attr<int64_t>("m")
                           .Attr<int64_t>("n")
                           .Attr<int64_t>("transpose")
                           .Attr<int64_t>("index_base")
                           .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER(csrMatmulAccHandlerROCM, csrMatmulAccRocm,
                       xla::ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<hipStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::AnyBuffer>() // rowptr
                           .Arg<ffi::AnyBuffer>() // colind
                           .Arg<ffi::AnyBuffer>() // nzval
                           .Arg<ffi::AnyBuffer>() // dense
                           .Arg<ffi::AnyBuffer>() // acc (aliased to out)
                           .Ret<ffi::AnyBuffer>() // out
                           .Attr<int64_t>("m")
                           .Attr<int64_t>("n")
                           .Attr<int64_t>("transpose")
                           .Attr<int64_t>("index_base")
                           .Attr<double>("alpha")
                           .Attr<double>("beta"));
#endif // REACTANT_ROCM

void registerReactantXLAInternalFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "Host", juliaCallbackHandlerHost);
#if defined(REACTANT_CUDA)
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_julia_callback",
                           "CUDA", juliaCallbackHandlerCUDA);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_csr_matmul",
                           "CUDA", csrMatmulHandlerCUDA);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_csr_matmul_acc",
                           "CUDA", csrMatmulAccHandlerCUDA);
#endif
#if defined(REACTANT_ROCM)
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_csr_matmul",
                           "ROCM", csrMatmulHandlerROCM);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "reactant_csr_matmul_acc",
                           "ROCM", csrMatmulAccHandlerROCM);
#endif
}

} // namespace reactant_ffi
} // namespace reactant

REACTANT_ABI void registerReactantXLAFFI() {
  reactant::reactant_ffi::registerReactantXLAInternalFFI();
}
