// xla_ffi.h

#ifndef REACTANTEXTRA_XLA_FFI_H
#define REACTANTEXTRA_XLA_FFI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register Reactant XLA FFI endpoints.
MLIR_CAPI_EXPORTED void registerReactantXLAFFI(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // REACTANTEXTRA_XLA_FFI_H
