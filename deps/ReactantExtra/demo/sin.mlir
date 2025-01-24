module {
  func.func @main(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
    %0 = stablehlo.sine %arg0 : tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }
}