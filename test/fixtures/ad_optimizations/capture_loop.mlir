// Enzyme-JAX issue #3204. Only the Enzyme attribute syntax is updated.
module @root {
  func.func @main() -> tensor<f64> {
    %cst = stablehlo.constant dense<2.000000e-01> : tensor<f64>
    %0 = stablehlo.while(%iterArg = %cst) : tensor<f64>
    cond {
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1 = stablehlo.compare LT, %iterArg, %cst_0 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1 = enzyme.autodiff_region(%iterArg, %cst_0) {
      ^bb0(%arg0: tensor<f64>):
        %cst_1 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
        %2 = stablehlo.multiply %cst_1, %arg0 : tensor<f64>
        %3 = stablehlo.multiply %2, %iterArg : tensor<f64>
        enzyme.yield %3 : tensor<f64>
      } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>) -> tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }
    return %0 : tensor<f64>
  }
}
