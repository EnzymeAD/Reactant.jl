module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %c_false = stablehlo.constant dense<false> : tensor<i1>
    %c_0i64 = stablehlo.constant dense<0> : tensor<i64>
    %c_1i64 = stablehlo.constant dense<1> : tensor<i64>
    %c_3i64 = stablehlo.constant dense<3> : tensor<i64>
    %c_0f64 = stablehlo.constant dense<0.0> : tensor<f64>
    %c_half = stablehlo.constant dense<0.5> : tensor<f64>
    %c_1f64 = stablehlo.constant dense<1.0> : tensor<f64>
    %c_0_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %c_0_rng = stablehlo.constant dense<0> : tensor<2xui64>
    %c_0_3x2 = stablehlo.constant dense<0.0> : tensor<3x2xf64>
    %c_466 = stablehlo.constant dense<466688986> : tensor<2xui32>
    %c_13u = stablehlo.constant dense<13> : tensor<2xui32>
    %c_19u = stablehlo.constant dense<19> : tensor<2xui32>
    %c_1u = stablehlo.constant dense<1> : tensor<2xui32>
    %c_idx = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %c_12ui64 = stablehlo.constant dense<12> : tensor<ui64>
    %c_magic = stablehlo.constant dense<4607182418800017408> : tensor<ui64>

    // Outer while (17 args)
    %out:17 = stablehlo.while(
      %iA0 = %c_0_1x2, %iA1 = %c_0_1x2, %iA2 = %c_0_1x2, %iA3 = %c_0_1x2,
      %iA4 = %c_0_1x2, %iA5 = %c_0_1x2, %iA6 = %c_0_1x2, %iA7 = %c_0_1x2,
      %iA8 = %c_0f64, %iA9 = %c_0f64, %iA10 = %c_0i64, %iA11 = %c_0f64,
      %iA12 = %c_false, %iA13 = %c_false, %iA14 = %c_0f64,
      %iA15 = %c_0_1x2, %iA16 = %arg0
    ) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>,
        tensor<i1>, tensor<i1>, tensor<f64>,
        tensor<1x2xf64>, tensor<2xui64>
    cond {
      %lt = stablehlo.compare LT, %iA10, %c_3i64, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %n1 = stablehlo.not %iA12 : tensor<i1>
      %n2 = stablehlo.not %iA13 : tensor<i1>
      %a1 = stablehlo.and %lt, %n1 : tensor<i1>
      %a2 = stablehlo.and %a1, %n2 : tensor<i1>
      stablehlo.return %a2 : tensor<i1>
    } do {
      // Direction sampling
      %rng1, %bits1 = stablehlo.rng_bit_generator %iA16, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %shifted = stablehlo.shift_right_logical %bits1, %c_12ui64 : tensor<ui64>
      %ored = stablehlo.or %shifted, %c_magic : tensor<ui64>
      %bc_f = stablehlo.bitcast_convert %ored : (tensor<ui64>) -> tensor<f64>
      %uniform = stablehlo.subtract %bc_f, %c_1f64 : tensor<f64>
      %direction = stablehlo.compare LT, %uniform, %c_half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>

      %rng2, %bits2 = stablehlo.rng_bit_generator %rng1, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)

      %num_leaves = stablehlo.shift_left %c_1i64, %iA10 : tensor<i64>

      // Middle while (21 args)
      %mid:21 = stablehlo.while(
        %mA0 = %iA0, %mA1 = %iA1, %mA2 = %iA2, %mA3 = %iA3,
        %mA4 = %iA4, %mA5 = %iA5, %mA6 = %iA6, %mA7 = %iA7,
        %mA8 = %iA8, %mA9 = %iA9, %mA10 = %iA10, %mA11 = %iA11,
        %mA12 = %iA12, %mA13 = %iA13, %mA14 = %iA14,
        %mA15 = %c_0i64, %mA16 = %iA15, %mA17 = %rng2,
        %mA18 = %c_0_3x2, %mA19 = %c_0_3x2, %mA20 = %c_0i64
      ) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>,
          tensor<i1>, tensor<i1>, tensor<f64>,
          tensor<i64>, tensor<1x2xf64>, tensor<2xui64>,
          tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      cond {
        %lt2 = stablehlo.compare LT, %mA15, %num_leaves, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %nt = stablehlo.not %mA13 : tensor<i1>
        %c3 = stablehlo.and %lt2, %nt : tensor<i1>
        stablehlo.return %c3 : tensor<i1>
      } do {
        // Threefry decomposition
        %bc = stablehlo.bitcast_convert %mA17 : (tensor<2xui64>) -> tensor<2x2xui32>
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
        %r1a = stablehlo.add %c_idx, %k1 : tensor<2xui32>
        %r1b = stablehlo.add %k0, %r1a : tensor<2xui32>
        %r1c = stablehlo.shift_left %r1a, %c_13u : tensor<2xui32>
        %r1d = stablehlo.shift_right_logical %r1a, %c_19u : tensor<2xui32>
        %r1e = stablehlo.or %r1c, %r1d : tensor<2xui32>
        %r1f = stablehlo.xor %r1b, %r1e : tensor<2xui32>
        %ki0 = stablehlo.add %r1b, %k1 : tensor<2xui32>
        %ki1 = stablehlo.add %r1f, %ks : tensor<2xui32>
        %ki2 = stablehlo.add %ki1, %c_1u : tensor<2xui32>
        // Second half
        %xk2 = stablehlo.xor %k2, %k3 : tensor<2xui32>
        %ks2 = stablehlo.xor %xk2, %c_466 : tensor<2xui32>
        %s2a = stablehlo.add %c_idx, %k3 : tensor<2xui32>
        %s2b = stablehlo.add %k2, %s2a : tensor<2xui32>
        %s2c = stablehlo.shift_left %s2a, %c_13u : tensor<2xui32>
        %s2d = stablehlo.shift_right_logical %s2a, %c_19u : tensor<2xui32>
        %s2e = stablehlo.or %s2c, %s2d : tensor<2xui32>
        %s2f = stablehlo.xor %s2b, %s2e : tensor<2xui32>
        %ski0 = stablehlo.add %s2b, %k3 : tensor<2xui32>
        %ski1 = stablehlo.add %s2f, %ks2 : tensor<2xui32>
        %ski2 = stablehlo.add %ski1, %c_1u : tensor<2xui32>
        // Reassemble key 1
        %o0a = stablehlo.slice %ki0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %o0b = stablehlo.slice %ki2 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %o1a = stablehlo.slice %ski0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %o1b = stablehlo.slice %ski2 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %cat0 = stablehlo.concatenate %o0a, %o0b, %o1a, %o1b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh0 = stablehlo.reshape %cat0 : (tensor<4xui32>) -> tensor<2x2xui32>
        %key1 = stablehlo.bitcast_convert %rsh0 : (tensor<2x2xui32>) -> tensor<2xui64>
        // Reassemble key 2
        %o2a = stablehlo.slice %ki0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %o2b = stablehlo.slice %ki2 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %o3a = stablehlo.slice %ski0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %o3b = stablehlo.slice %ski2 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %cat1 = stablehlo.concatenate %o2a, %o2b, %o3a, %o3b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh1 = stablehlo.reshape %cat1 : (tensor<4xui32>) -> tensor<2x2xui32>
        %key2 = stablehlo.bitcast_convert %rsh1 : (tensor<2x2xui32>) -> tensor<2xui64>

        // stablehlo.if with 16 return values + rng_bit_generator in else branch
        %is_first = stablehlo.compare EQ, %mA15, %c_0i64, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %if_res:16 = "stablehlo.if"(%is_first) ({
          stablehlo.return %c_0_1x2, %c_0_1x2, %c_0_1x2, %c_0_1x2, %c_0_1x2, %c_0_1x2, %c_0_1x2, %c_0_1x2, %c_0f64, %c_0f64, %c_0i64, %c_0f64, %c_false, %c_0f64, %c_1i64, %c_0_1x2 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>
        }, {
          // rng_bit_generator consuming Threefry output (key2) inside if else
          %rng_if, %bits_if = stablehlo.rng_bit_generator %key2, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
          stablehlo.return %mA0, %mA1, %mA2, %mA3, %mA4, %mA5, %mA6, %mA7, %mA8, %mA9, %mA10, %mA11, %mA12, %mA14, %mA15, %mA16 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>
        }) : (tensor<i1>) -> (tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>)

        %new_idx = stablehlo.add %mA15, %c_1i64 : tensor<i64>
        %new_cnt = stablehlo.add %mA20, %c_1i64 : tensor<i64>
        stablehlo.return %if_res#0, %if_res#1, %if_res#2, %if_res#3, %if_res#4, %if_res#5, %if_res#6, %if_res#7, %if_res#8, %if_res#9, %if_res#10, %if_res#11, %c_false, %if_res#12, %if_res#13, %new_idx, %if_res#15, %key1, %mA18, %mA19, %new_cnt : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>, tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      }

      %new_depth = stablehlo.add %iA10, %c_1i64 : tensor<i64>
      stablehlo.return %mid#0, %mid#1, %mid#2, %mid#3, %mid#4, %mid#5, %mid#6, %mid#7, %mid#8, %mid#9, %new_depth, %mid#11, %mid#12, %mid#13, %mid#14, %mid#16, %rng1 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    }
    %result = stablehlo.reshape %out#0 : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %c_0_rng : tensor<2x1xf64>, tensor<2xui64>
  }
}
