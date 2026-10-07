module {
  func.func private @square_and_constant(%x: tensor<f64>, %c: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %xx = stablehlo.multiply %x, %x : tensor<f64>
    return %xx, %c : tensor<f64>, tensor<f64>
  }
  func.func @main(%x: tensor<f64>, %c: tensor<f64>, %s1: tensor<f64>, %s2: tensor<f64>, %sc: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %p1, %dx1 = enzyme.autodiff @square_and_constant(%x, %c, %s1, %sc) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %p2, %dx2 = enzyme.autodiff @square_and_constant(%x, %c, %s2) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %p1, %dx1, %p2, %dx2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}
