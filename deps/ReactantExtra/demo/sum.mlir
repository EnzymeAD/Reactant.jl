module {
  func.func @main(%arg0: tensor<4x4xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x4xf64>, tensor<f64>) -> tensor<f64>
    return %1 : tensor<f64>
  }
}