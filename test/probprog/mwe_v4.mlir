module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %c_false = stablehlo.constant dense<false> : tensor<i1>
    %c_0i64 = stablehlo.constant dense<0> : tensor<i64>
    %c_1i64 = stablehlo.constant dense<1> : tensor<i64>
    %c_3i64 = stablehlo.constant dense<3> : tensor<i64>
    %c_0_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %c_0_rng = stablehlo.constant dense<0> : tensor<2xui64>
    %c_466 = stablehlo.constant dense<466688986> : tensor<2xui32>
    %c_13u = stablehlo.constant dense<13> : tensor<2xui32>
    %c_15u = stablehlo.constant dense<15> : tensor<2xui32>
    %c_26u = stablehlo.constant dense<26> : tensor<2xui32>
    %c_6u = stablehlo.constant dense<6> : tensor<2xui32>
    %c_1u = stablehlo.constant dense<1> : tensor<2xui32>
    %c_17u = stablehlo.constant dense<17> : tensor<2xui32>
    %c_19u = stablehlo.constant dense<19> : tensor<2xui32>
    %c_idx = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %c_0f64 = stablehlo.constant dense<0.0> : tensor<f64>

    // Outer while
    %out:6 = stablehlo.while(%depth = %c_0i64, %turning = %c_false, %diverging = %c_false, %rng = %arg0, %q = %c_0_1x2, %weight = %c_0f64) : tensor<i64>, tensor<i1>, tensor<i1>, tensor<2xui64>, tensor<1x2xf64>, tensor<f64>
    cond {
      %lt = stablehlo.compare LT, %depth, %c_3i64, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %not_turn = stablehlo.not %turning : tensor<i1>
      %c1 = stablehlo.and %lt, %not_turn : tensor<i1>
      stablehlo.return %c1 : tensor<i1>
    } do {
      %num_leaves = stablehlo.shift_left %c_1i64, %depth : tensor<i64>

      // Middle while
      %mid:4 = stablehlo.while(%leaf_idx = %c_0i64, %mid_rng = %rng, %mid_turn = %c_false, %mid_weight = %c_0f64) : tensor<i64>, tensor<2xui64>, tensor<i1>, tensor<f64>
      cond {
        %lt2 = stablehlo.compare LT, %leaf_idx, %num_leaves, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %not_turn2 = stablehlo.not %mid_turn : tensor<i1>
        %c3 = stablehlo.and %lt2, %not_turn2 : tensor<i1>
        stablehlo.return %c3 : tensor<i1>
      } do {
        // Threefry decomposition
        %bc = stablehlo.bitcast_convert %mid_rng : (tensor<2xui64>) -> tensor<2x2xui32>
        %flat = stablehlo.reshape %bc : (tensor<2x2xui32>) -> tensor<4xui32>
        %s0 = stablehlo.slice %flat [0:1] : (tensor<4xui32>) -> tensor<1xui32>
        %s1 = stablehlo.slice %flat [1:2] : (tensor<4xui32>) -> tensor<1xui32>
        %s2 = stablehlo.slice %flat [2:3] : (tensor<4xui32>) -> tensor<1xui32>
        %s3 = stablehlo.slice %flat [3:4] : (tensor<4xui32>) -> tensor<1xui32>
        %k0 = stablehlo.broadcast_in_dim %s0, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %k1 = stablehlo.broadcast_in_dim %s1, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %k2 = stablehlo.broadcast_in_dim %s2, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %k3 = stablehlo.broadcast_in_dim %s3, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %xk = stablehlo.xor %k0, %k1 : tensor<2xui32>
        %ks = stablehlo.xor %xk, %c_466 : tensor<2xui32>
        // Minimal hash round
        %r1a = stablehlo.add %c_idx, %k1 : tensor<2xui32>
        %r1b = stablehlo.add %k0, %r1a : tensor<2xui32>
        %r1c = stablehlo.shift_left %r1a, %c_13u : tensor<2xui32>
        %r1d = stablehlo.shift_right_logical %r1a, %c_19u : tensor<2xui32>
        %r1e = stablehlo.or %r1c, %r1d : tensor<2xui32>
        %r1f = stablehlo.xor %r1b, %r1e : tensor<2xui32>
        // Key injection
        %ki0 = stablehlo.add %r1b, %k1 : tensor<2xui32>
        %ki1 = stablehlo.add %r1f, %ks : tensor<2xui32>
        %ki2 = stablehlo.add %ki1, %c_1u : tensor<2xui32>
        // Reassemble
        %out0a = stablehlo.slice %ki0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %out0b = stablehlo.slice %ki2 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %out1a = stablehlo.slice %ki0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %out1b = stablehlo.slice %ki2 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %cat = stablehlo.concatenate %out0a, %out0b, %out1a, %out1b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh = stablehlo.reshape %cat : (tensor<4xui32>) -> tensor<2x2xui32>
        %key1 = stablehlo.bitcast_convert %rsh : (tensor<2x2xui32>) -> tensor<2xui64>

        // rng_bit_generator consuming the key
        %rng_out, %bits = stablehlo.rng_bit_generator %key1, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)

        // stablehlo.if (base tree vs combine)
        %is_first = stablehlo.compare EQ, %leaf_idx, %c_0i64, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %if_result:2 = "stablehlo.if"(%is_first) ({
          stablehlo.return %c_0f64, %key1 : tensor<f64>, tensor<2xui64>
        }, {
          // Use rng_bit_generator in else branch (like the original)
          %rng2, %bits2 = stablehlo.rng_bit_generator %rng_out, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
          stablehlo.return %mid_weight, %rng2 : tensor<f64>, tensor<2xui64>
        }) : (tensor<i1>) -> (tensor<f64>, tensor<2xui64>)

        %new_idx = stablehlo.add %leaf_idx, %c_1i64 : tensor<i64>
        stablehlo.return %new_idx, %if_result#1, %mid_turn, %if_result#0 : tensor<i64>, tensor<2xui64>, tensor<i1>, tensor<f64>
      }

      %new_depth = stablehlo.add %depth, %c_1i64 : tensor<i64>
      stablehlo.return %new_depth, %mid#2, %diverging, %mid#1, %q, %mid#3 : tensor<i64>, tensor<i1>, tensor<i1>, tensor<2xui64>, tensor<1x2xf64>, tensor<f64>
    }
    %result = stablehlo.reshape %out#4 : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %c_0_rng : tensor<2x1xf64>, tensor<2xui64>
  }
}
