module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %c_false = stablehlo.constant dense<false> : tensor<i1>
    %c_0i64 = stablehlo.constant dense<0> : tensor<i64>
    %c_1i64 = stablehlo.constant dense<1> : tensor<i64>
    %c_3i64 = stablehlo.constant dense<3> : tensor<i64>
    %c_0_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %c_0_rng = stablehlo.constant dense<0> : tensor<2xui64>

    // Outer while
    %out:5 = stablehlo.while(%depth = %c_0i64, %turning = %c_false, %diverging = %c_false, %rng = %arg0, %q = %c_0_1x2) : tensor<i64>, tensor<i1>, tensor<i1>, tensor<2xui64>, tensor<1x2xf64>
    cond {
      %lt = stablehlo.compare LT, %depth, %c_3i64, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %not_turn = stablehlo.not %turning : tensor<i1>
      %c1 = stablehlo.and %lt, %not_turn : tensor<i1>
      stablehlo.return %c1 : tensor<i1>
    } do {
      %num_leaves = stablehlo.shift_left %c_1i64, %depth : tensor<i64>

      // Middle while
      %mid:3 = stablehlo.while(%leaf_idx = %c_0i64, %mid_rng = %rng, %mid_turn = %c_false) : tensor<i64>, tensor<2xui64>, tensor<i1>
      cond {
        %lt2 = stablehlo.compare LT, %leaf_idx, %num_leaves, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %not_turn2 = stablehlo.not %mid_turn : tensor<i1>
        %c3 = stablehlo.and %lt2, %not_turn2 : tensor<i1>
        stablehlo.return %c3 : tensor<i1>
      } do {
        // bitcast_convert: tensor<2xui64> → tensor<2x2xui32>
        %decomposed = stablehlo.bitcast_convert %mid_rng : (tensor<2xui64>) -> tensor<2x2xui32>
        // bitcast_convert back: tensor<2x2xui32> → tensor<2xui64>
        %new_rng = stablehlo.bitcast_convert %decomposed : (tensor<2x2xui32>) -> tensor<2xui64>

        %new_idx = stablehlo.add %leaf_idx, %c_1i64 : tensor<i64>
        stablehlo.return %new_idx, %new_rng, %mid_turn : tensor<i64>, tensor<2xui64>, tensor<i1>
      }

      %new_depth = stablehlo.add %depth, %c_1i64 : tensor<i64>
      stablehlo.return %new_depth, %mid#2, %diverging, %mid#1, %q : tensor<i64>, tensor<i1>, tensor<i1>, tensor<2xui64>, tensor<1x2xf64>
    }
    %result = stablehlo.reshape %out#4 : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %c_0_rng : tensor<2x1xf64>, tensor<2xui64>
  }
}
