module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %cst_f64 = stablehlo.constant dense<0.0> : tensor<f64>
    %cst_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %cst_3x2 = stablehlo.constant dense<0.0> : tensor<3x2xf64>
    %c_i64_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_i64_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_i64_3 = stablehlo.constant dense<3> : tensor<i64>
    %false = stablehlo.constant dense<false> : tensor<i1>

    // Momentum sampling
    %rng0_state, %rng0_out = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT
      : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1x2xui64>)

    // Outer while: 17 args matching NUTS tree state
    // Types: 7×tensor<1x2xf64>, 2×tensor<f64>, tensor<i64>, tensor<f64>, 2×tensor<i1>, tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    %outer:17 = stablehlo.while(
      %o0 = %cst_1x2, %o1 = %cst_1x2, %o2 = %cst_1x2,
      %o3 = %cst_1x2, %o4 = %cst_1x2, %o5 = %cst_1x2,
      %o6 = %cst_1x2,
      %o7 = %cst_1x2,
      %o8 = %cst_f64, %o9 = %cst_f64,
      %o10 = %c_i64_0,
      %o11 = %cst_f64,
      %o12 = %false, %o13 = %false,
      %o14 = %cst_f64,
      %o15 = %cst_1x2,
      %o16 = %rng0_state
    ) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<1x2xf64>,
        tensor<1x2xf64>,
        tensor<f64>, tensor<f64>,
        tensor<i64>,
        tensor<f64>,
        tensor<i1>, tensor<i1>,
        tensor<f64>,
        tensor<1x2xf64>,
        tensor<2xui64>
    cond {
      %cmp = stablehlo.compare LT, %o10, %c_i64_1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %not12 = stablehlo.not %o12 : tensor<i1>
      %not13 = stablehlo.not %o13 : tensor<i1>
      %and1 = stablehlo.and %cmp, %not12 : tensor<i1>
      %and2 = stablehlo.and %and1, %not13 : tensor<i1>
      stablehlo.return %and2 : tensor<i1>
    } do {
      // Direction sampling
      %rng1_state, %rng1_out = stablehlo.rng_bit_generator %o16, algorithm = DEFAULT
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)

      // Middle while: 21 args matching NUTS doubling state
      %mid:21 = stablehlo.while(
        %m0 = %o0, %m1 = %o1, %m2 = %o2,
        %m3 = %o3, %m4 = %o4, %m5 = %o5,
        %m6 = %o6,
        %m7 = %o7,
        %m8 = %o8, %m9 = %o9,
        %m10 = %o10,
        %m11 = %o11,
        %m12 = %o12, %m13 = %o13,
        %m14 = %o14,
        %m15 = %c_i64_0,
        %m16 = %o15,
        %m17 = %rng1_state,
        %m18 = %cst_3x2, %m19 = %cst_3x2,
        %m20 = %c_i64_0
      ) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>,
          tensor<1x2xf64>,
          tensor<f64>, tensor<f64>,
          tensor<i64>,
          tensor<f64>,
          tensor<i1>, tensor<i1>,
          tensor<f64>,
          tensor<i64>,
          tensor<1x2xf64>,
          tensor<2xui64>,
          tensor<3x2xf64>, tensor<3x2xf64>,
          tensor<i64>
      cond {
        %cmp_m = stablehlo.compare LT, %m15, %c_i64_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %not_m12 = stablehlo.not %m12 : tensor<i1>
        %not_m13 = stablehlo.not %m13 : tensor<i1>
        %and_m1 = stablehlo.and %cmp_m, %not_m12 : tensor<i1>
        %and_m2 = stablehlo.and %and_m1, %not_m13 : tensor<i1>
        stablehlo.return %and_m2 : tensor<i1>
      } do {
        // Subtree combine rng
        %rng2_state, %rng2_out = stablehlo.rng_bit_generator %m17, algorithm = DEFAULT
          : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)

        %new_depth = stablehlo.add %m20, %c_i64_1 : tensor<i64>

        stablehlo.return %m0, %m1, %m2, %m3, %m4, %m5, %m6, %m7,
          %m8, %m9, %m10, %m11, %m12, %m13, %m14, %m15, %m16,
          %rng2_state, %m18, %m19, %new_depth
          : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
            tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
            tensor<1x2xf64>, tensor<1x2xf64>,
            tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>,
            tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>,
            tensor<1x2xf64>, tensor<2xui64>,
            tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      }

      // Post-doubling rng
      %rng3_state, %rng3_out = stablehlo.rng_bit_generator %mid#17, algorithm = DEFAULT
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)

      %new_iter = stablehlo.add %o10, %c_i64_1 : tensor<i64>

      stablehlo.return %mid#0, %mid#1, %mid#2, %mid#3, %mid#4, %mid#5,
        %mid#6, %mid#7, %mid#8, %mid#9, %new_iter, %mid#11,
        %mid#12, %mid#13, %mid#14, %mid#16, %rng3_state
        : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>,
          tensor<i1>, tensor<i1>, tensor<f64>,
          tensor<1x2xf64>, tensor<2xui64>
    }

    %result = stablehlo.transpose %outer#0, dims = [1, 0] : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %outer#16 : tensor<2x1xf64>, tensor<2xui64>
  }
}
