module {
  func.func @square(%x: tensor<f64>) -> tensor<f64> {
    %r = stablehlo.multiply %x, %x : tensor<f64>
    return %r : tensor<f64>
  }
  func.func @main(%x: tensor<f64>, %s1: tensor<f64>, %s2: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %p1, %d1 = enzyme.fwddiff @square(%x, %s1) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dup>]}> : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %p2, %d2 = enzyme.fwddiff @square(%x, %s2) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dup>]}> : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %q1 = enzyme.fwddiff @square(%p1, %s1) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %q2 = enzyme.fwddiff @square(%p1, %s2) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (tensor<f64>, tensor<f64>) -> tensor<f64>
    return %d1, %d2, %q1, %q2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}
