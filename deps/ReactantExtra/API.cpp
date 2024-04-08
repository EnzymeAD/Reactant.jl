#include "mlir-c/IR.h"

extern "C" MlirContext foo() {
 return mlirContextCreate();
}