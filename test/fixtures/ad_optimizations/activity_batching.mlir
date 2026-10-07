module {
  func.func private @square_and_constant(%x: tensor<f64>, %c: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %xx = stablehlo.multiply %x, %x : tensor<f64>
    return %xx, %c : tensor<f64>, tensor<f64>
  }
  func.func @main(%x: tensor<f64>, %dx1: tensor<f64>, %dx2: tensor<f64>, %c: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %dxx1, %p1, %dc1 = enzyme.fwddiff @square_and_constant(%x, %dx1, %c) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    %dxx2, %p2 = enzyme.fwddiff @square_and_constant(%x, %dx2, %c) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_const>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %dxx1, %p1, %dc1, %dxx2, %p2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}
