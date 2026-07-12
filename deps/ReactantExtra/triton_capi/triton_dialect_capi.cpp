#include "triton_dialect_capi.h"

#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, triton,
                                      mlir::triton::TritonDialect);

MlirType mlirTritonPointerTypeGet(MlirType pointeeType, int addressSpace) {
  return wrap(
      mlir::triton::PointerType::get(unwrap(pointeeType), addressSpace));
}

bool mlirTritonIsAPointer(MlirType type) {
  return llvm::isa<mlir::triton::PointerType>(unwrap(type));
}

MlirType mlirTritonPointerTypeGetPointeeType(MlirType pointerType) {
  return wrap(llvm::cast<mlir::triton::PointerType>(unwrap(pointerType))
                  .getPointeeType());
}

int mlirTritonPointerTypeGetAddressSpace(MlirType pointerType) {
  return llvm::cast<mlir::triton::PointerType>(unwrap(pointerType))
      .getAddressSpace();
}

MlirAttribute mlirTritonInferReduceOpEncoding(MlirAttribute operandEncoding,
                                              int axis) {
  auto opEncoding = unwrap(operandEncoding);
  mlir::Dialect& dialect = opEncoding.getDialect();
  auto inferLayoutInterface =
      llvm::dyn_cast<mlir::triton::DialectInferLayoutInterface>(&dialect);
  mlir::Attribute retEncoding;
  (void)inferLayoutInterface->inferReduceOpEncoding(opEncoding, axis,
                                                    retEncoding, std::nullopt);
  return wrap(retEncoding);
}

MlirTypeID mlirTritonPointerTypeGetTypeID(void) {
  return wrap(mlir::triton::PointerType::getTypeID());
}

}  // extern "C"
