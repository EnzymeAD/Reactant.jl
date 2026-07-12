#include "triton_dialect_capi.h"

#include <gtest/gtest.h>
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

TEST(TritonDialectCapiTest, DialectRegistrationAndPointerTypes) {
  MlirContext context = mlirContextCreate();
  MlirDialectHandle tritonHandle = mlirGetDialectHandle__triton__();
  mlirDialectHandleLoadDialect(tritonHandle, context);

  // Test non-pointer type verification
  MlirType f32 = mlirF32TypeGet(context);
  EXPECT_FALSE(mlirTritonIsAPointer(f32));

  // Test PointerType creation and inspection
  int addressSpace = 1;
  MlirType ptrType = mlirTritonPointerTypeGet(f32, addressSpace);
  EXPECT_FALSE(mlirTypeIsNull(ptrType));
  EXPECT_TRUE(mlirTritonIsAPointer(ptrType));

  MlirType pointeeType = mlirTritonPointerTypeGetPointeeType(ptrType);
  EXPECT_TRUE(mlirTypeEqual(pointeeType, f32));
  EXPECT_EQ(mlirTritonPointerTypeGetAddressSpace(ptrType), addressSpace);

  // Test TypeID
  MlirTypeID ptrTypeId = mlirTritonPointerTypeGetTypeID();
  EXPECT_FALSE(mlirTypeIDIsNull(ptrTypeId));

  mlirContextDestroy(context);
}
