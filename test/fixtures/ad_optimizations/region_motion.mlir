module {
  func.func private @scaled_square(%x: tensor<f64>, %c: tensor<f64>) -> tensor<f64> {
    %coefficient = stablehlo.exponential %c : tensor<f64>
    %xx = stablehlo.multiply %x, %x : tensor<f64>
    %r = stablehlo.multiply %xx, %coefficient : tensor<f64>
    return %r : tensor<f64>
  }
  func.func @main(%x: tensor<f64>, %dx1: tensor<f64>, %dx2: tensor<f64>, %c: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %d1 = enzyme.fwddiff @scaled_square(%x, %dx1, %c) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %d2 = enzyme.fwddiff @scaled_square(%x, %dx2, %c) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    return %d1, %d2 : tensor<f64>, tensor<f64>
  }
}
