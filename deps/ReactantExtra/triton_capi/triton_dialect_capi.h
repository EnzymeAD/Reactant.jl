#ifndef REACTANT_EXTRA_TRITON_CAPI_TRITON_DIALECT_CAPI_H_
#define REACTANT_EXTRA_TRITON_CAPI_TRITON_DIALECT_CAPI_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, triton);

MLIR_CAPI_EXPORTED MlirTypeID mlirTritonPointerTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirTritonPointerTypeGet(MlirType pointeeType,
                                                     int addressSpace);
MLIR_CAPI_EXPORTED bool mlirTritonIsAPointer(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirTritonPointerTypeGetPointeeType(MlirType pointerType);
MLIR_CAPI_EXPORTED int mlirTritonPointerTypeGetAddressSpace(
    MlirType pointerType);

MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonInferReduceOpEncoding(MlirAttribute operandEncoding, int axis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // REACTANT_EXTRA_TRITON_CAPI_TRITON_DIALECT_CAPI_H_
