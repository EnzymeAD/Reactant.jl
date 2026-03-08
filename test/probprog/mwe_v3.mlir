module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %c_false = stablehlo.constant dense<false> : tensor<i1>
    %c_0i64 = stablehlo.constant dense<0> : tensor<i64>
    %c_1i64 = stablehlo.constant dense<1> : tensor<i64>
    %c_3i64 = stablehlo.constant dense<3> : tensor<i64>
    %c_0_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %c_0_rng = stablehlo.constant dense<0> : tensor<2xui64>
    %c_12ui64 = stablehlo.constant dense<12> : tensor<ui64>
    %c_magic = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %c_one_f64 = stablehlo.constant dense<1.0> : tensor<f64>
    %c_half = stablehlo.constant dense<0.5> : tensor<f64>
    %c_466 = stablehlo.constant dense<466688986> : tensor<2xui32>
    %c_13u = stablehlo.constant dense<13> : tensor<2xui32>
    %c_15u = stablehlo.constant dense<15> : tensor<2xui32>
    %c_26u = stablehlo.constant dense<26> : tensor<2xui32>
    %c_6u = stablehlo.constant dense<6> : tensor<2xui32>
    %c_1u = stablehlo.constant dense<1> : tensor<2xui32>
    %c_17u = stablehlo.constant dense<17> : tensor<2xui32>
    %c_29u = stablehlo.constant dense<29> : tensor<2xui32>
    %c_16u = stablehlo.constant dense<16> : tensor<2xui32>
    %c_24u = stablehlo.constant dense<24> : tensor<2xui32>
    %c_2u = stablehlo.constant dense<2> : tensor<2xui32>
    %c_3u = stablehlo.constant dense<3> : tensor<2xui32>
    %c_4u = stablehlo.constant dense<4> : tensor<2xui32>
    %c_5u = stablehlo.constant dense<5> : tensor<2xui32>
    %c_8u = stablehlo.constant dense<8> : tensor<2xui32>
    %c_19u = stablehlo.constant dense<19> : tensor<2xui32>
    %c_idx = stablehlo.constant dense<[0, 1]> : tensor<2xui32>

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
        // === Threefry 2-way key split (full 5-round hash) ===
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
        // ks = k0 ^ k1 ^ magic
        %xk = stablehlo.xor %k0, %k1 : tensor<2xui32>
        %ks = stablehlo.xor %xk, %c_466 : tensor<2xui32>
        // Round 1: x0 = idx + k1, x1 = k0 + x0
        %r1a = stablehlo.add %c_idx, %k1 : tensor<2xui32>
        %r1b = stablehlo.add %k0, %r1a : tensor<2xui32>
        %r1c = stablehlo.shift_left %r1a, %c_13u : tensor<2xui32>
        %r1d = stablehlo.shift_right_logical %r1a, %c_19u : tensor<2xui32>
        %r1e = stablehlo.or %r1c, %r1d : tensor<2xui32>
        %r1f = stablehlo.xor %r1b, %r1e : tensor<2xui32>
        // Round 2
        %r2a = stablehlo.add %r1b, %r1f : tensor<2xui32>
        %r2b = stablehlo.shift_left %r1f, %c_15u : tensor<2xui32>
        %r2c = stablehlo.shift_right_logical %r1f, %c_17u : tensor<2xui32>
        %r2d = stablehlo.or %r2b, %r2c : tensor<2xui32>
        %r2e = stablehlo.xor %r2a, %r2d : tensor<2xui32>
        // Round 3
        %r3a = stablehlo.add %r2a, %r2e : tensor<2xui32>
        %r3b = stablehlo.shift_left %r2e, %c_26u : tensor<2xui32>
        %r3c = stablehlo.shift_right_logical %r2e, %c_6u : tensor<2xui32>
        %r3d = stablehlo.or %r3b, %r3c : tensor<2xui32>
        %r3e = stablehlo.xor %r3a, %r3d : tensor<2xui32>
        // Round 4
        %r4a = stablehlo.add %r3a, %r3e : tensor<2xui32>
        %r4b = stablehlo.shift_left %r3e, %c_6u : tensor<2xui32>
        %r4c = stablehlo.shift_right_logical %r3e, %c_26u : tensor<2xui32>
        %r4d = stablehlo.or %r4b, %r4c : tensor<2xui32>
        %r4e = stablehlo.xor %r4a, %r4d : tensor<2xui32>
        // Key injection
        %ki0 = stablehlo.add %r4a, %k1 : tensor<2xui32>
        %ki1 = stablehlo.add %r4e, %ks : tensor<2xui32>
        %ki2 = stablehlo.add %ki1, %c_1u : tensor<2xui32>
        // === Second half (produces second key) ===
        %xk2 = stablehlo.xor %k2, %k3 : tensor<2xui32>
        %ks2 = stablehlo.xor %xk2, %c_466 : tensor<2xui32>
        %s2a = stablehlo.add %c_idx, %k3 : tensor<2xui32>
        %s2b = stablehlo.add %k2, %s2a : tensor<2xui32>
        %s2c = stablehlo.shift_left %s2a, %c_13u : tensor<2xui32>
        %s2d = stablehlo.shift_right_logical %s2a, %c_19u : tensor<2xui32>
        %s2e = stablehlo.or %s2c, %s2d : tensor<2xui32>
        %s2f = stablehlo.xor %s2b, %s2e : tensor<2xui32>
        %s3a = stablehlo.add %s2b, %s2f : tensor<2xui32>
        %s3b = stablehlo.shift_left %s2f, %c_15u : tensor<2xui32>
        %s3c = stablehlo.shift_right_logical %s2f, %c_17u : tensor<2xui32>
        %s3d = stablehlo.or %s3b, %s3c : tensor<2xui32>
        %s3e = stablehlo.xor %s3a, %s3d : tensor<2xui32>
        %s4a = stablehlo.add %s3a, %s3e : tensor<2xui32>
        %s4b = stablehlo.shift_left %s3e, %c_26u : tensor<2xui32>
        %s4c = stablehlo.shift_right_logical %s3e, %c_6u : tensor<2xui32>
        %s4d = stablehlo.or %s4b, %s4c : tensor<2xui32>
        %s4e = stablehlo.xor %s4a, %s4d : tensor<2xui32>
        %s5a = stablehlo.add %s4a, %s4e : tensor<2xui32>
        %s5b = stablehlo.shift_left %s4e, %c_6u : tensor<2xui32>
        %s5c = stablehlo.shift_right_logical %s4e, %c_26u : tensor<2xui32>
        %s5d = stablehlo.or %s5b, %s5c : tensor<2xui32>
        %s5e = stablehlo.xor %s5a, %s5d : tensor<2xui32>
        %ski0 = stablehlo.add %s5a, %k3 : tensor<2xui32>
        %ski1 = stablehlo.add %s5e, %k2 : tensor<2xui32>
        %ski2 = stablehlo.add %ski1, %c_1u : tensor<2xui32>
        // Reassemble first key
        %out0_a = stablehlo.slice %ki0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %out0_b = stablehlo.slice %ki2 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %out1_a = stablehlo.slice %ski0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %out1_b = stablehlo.slice %ski2 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %cat0 = stablehlo.concatenate %out0_a, %out0_b, %out1_a, %out1_b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh0 = stablehlo.reshape %cat0 : (tensor<4xui32>) -> tensor<2x2xui32>
        %key1 = stablehlo.bitcast_convert %rsh0 : (tensor<2x2xui32>) -> tensor<2xui64>
        // Reassemble second key
        %out2_a = stablehlo.slice %ki0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %out2_b = stablehlo.slice %ki2 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %out3_a = stablehlo.slice %ski0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %out3_b = stablehlo.slice %ski2 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %cat1 = stablehlo.concatenate %out2_a, %out2_b, %out3_a, %out3_b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh1 = stablehlo.reshape %cat1 : (tensor<4xui32>) -> tensor<2x2xui32>
        %key2 = stablehlo.bitcast_convert %rsh1 : (tensor<2x2xui32>) -> tensor<2xui64>

        // Use key2 to sample via rng_bit_generator
        %rng_out, %bits = stablehlo.rng_bit_generator %key2, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)

        %new_idx = stablehlo.add %leaf_idx, %c_1i64 : tensor<i64>
        stablehlo.return %new_idx, %key1, %mid_turn : tensor<i64>, tensor<2xui64>, tensor<i1>
      }

      %new_depth = stablehlo.add %depth, %c_1i64 : tensor<i64>
      stablehlo.return %new_depth, %mid#2, %diverging, %mid#1, %q : tensor<i64>, tensor<i1>, tensor<i1>, tensor<2xui64>, tensor<1x2xf64>
    }
    %result = stablehlo.reshape %out#4 : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %c_0_rng : tensor<2x1xf64>, tensor<2xui64>
  }
}
