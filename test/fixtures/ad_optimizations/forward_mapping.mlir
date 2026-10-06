module {
  func.func private @muladd(%x: tensor<f64>, %c: tensor<f64>, %y: tensor<f64>) -> tensor<f64> {
    %xy = stablehlo.multiply %x, %y : tensor<f64>
    %r = stablehlo.add %xy, %c : tensor<f64>
    return %r : tensor<f64>
  }
  func.func @main(%x: tensor<f64>, %c: tensor<f64>, %y: tensor<f64>, %s1: tensor<f64>, %s2: tensor<f64>, %t1: tensor<f64>, %t2: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %p1, %dx1 = enzyme.fwddiff @muladd(%x, %s1, %c, %y, %t1) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %p2, %dx2 = enzyme.fwddiff @muladd(%x, %s2, %c, %y, %t2) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %p1, %dx1, %p2, %dx2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}
