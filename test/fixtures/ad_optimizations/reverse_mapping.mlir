module {
  func.func private @muladd(%x: tensor<f64>, %c: tensor<f64>, %y: tensor<f64>) -> tensor<f64> {
    %xy = stablehlo.multiply %x, %y : tensor<f64>
    %r = stablehlo.add %xy, %c : tensor<f64>
    return %r : tensor<f64>
  }
  func.func @main(%x: tensor<f64>, %c: tensor<f64>, %y: tensor<f64>, %s1: tensor<f64>, %s2: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %p1, %dx1, %dy1 = enzyme.autodiff @muladd(%x, %c, %y, %s1) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    %p2, %dx2, %dy2 = enzyme.autodiff @muladd(%x, %c, %y, %s2) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    return %p1, %dx1, %dy1, %p2, %dx2, %dy2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}
