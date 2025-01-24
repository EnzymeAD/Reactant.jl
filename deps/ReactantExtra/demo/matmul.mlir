module {
  func.func @main(%arg0: tensor<4x4xf64>, %arg1: tensor<8x4xf64>) -> tensor<8x4xf64> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<8x4xf64>, tensor<4x4xf64>) -> tensor<8x4xf64>
    return %0 : tensor<8x4xf64>
  }
}