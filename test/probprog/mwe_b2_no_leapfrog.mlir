module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %c_22 = stablehlo.constant dense<8> : tensor<2xui32>
    %c_23 = stablehlo.constant dense<19> : tensor<2xui32>
    %c_24 = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %c_25 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %c_26 = stablehlo.constant dense<12> : tensor<ui64>
    %c_46 = stablehlo.constant dense<5> : tensor<2xui32>
    %c_47 = stablehlo.constant dense<4> : tensor<2xui32>
    %c_48 = stablehlo.constant dense<3> : tensor<2xui32>
    %c_49 = stablehlo.constant dense<2> : tensor<2xui32>
    %c_50 = stablehlo.constant dense<24> : tensor<2xui32>
    %c_51 = stablehlo.constant dense<16> : tensor<2xui32>
    %c_52 = stablehlo.constant dense<29> : tensor<2xui32>
    %c_53 = stablehlo.constant dense<17> : tensor<2xui32>
    %c_54 = stablehlo.constant dense<1> : tensor<2xui32>
    %c_55 = stablehlo.constant dense<6> : tensor<2xui32>
    %c_56 = stablehlo.constant dense<26> : tensor<2xui32>
    %c_57 = stablehlo.constant dense<15> : tensor<2xui32>
    %c_58 = stablehlo.constant dense<13> : tensor<2xui32>
    %c_59 = stablehlo.constant dense<466688986> : tensor<2xui32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf64>
    %c_62 = stablehlo.constant dense<3> : tensor<i64>
    %c_63 = stablehlo.constant dense<false> : tensor<i1>
    %cst_64 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_66 = stablehlo.constant dense<1> : tensor<i64>
    %c_67 = stablehlo.constant dense<0> : tensor<i64>
    %cst_68 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %init_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %init_f64 = stablehlo.constant dense<0.0> : tensor<f64>
    %init_i1 = stablehlo.constant dense<false> : tensor<i1>
    %c_0_rng = stablehlo.constant dense<0> : tensor<2xui64>

    // Outer while (17 args)
    %out:17 = stablehlo.while(%iterArg = %init_1x2, %iterArg_72 = %init_1x2, %iterArg_73 = %init_1x2, %iterArg_74 = %init_1x2, %iterArg_75 = %init_1x2, %iterArg_76 = %init_1x2, %iterArg_77 = %init_1x2, %iterArg_78 = %init_1x2, %iterArg_79 = %init_f64, %iterArg_80 = %init_f64, %iterArg_81 = %c_67, %iterArg_82 = %init_f64, %iterArg_83 = %init_i1, %iterArg_84 = %init_i1, %iterArg_85 = %init_f64, %iterArg_86 = %init_1x2, %iterArg_87 = %arg0) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    cond {
      %lt = stablehlo.compare LT, %iterArg_81, %c_62, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %n1 = stablehlo.not %iterArg_83 : tensor<i1>
      %n2 = stablehlo.not %iterArg_84 : tensor<i1>
      %a1 = stablehlo.and %lt, %n1 : tensor<i1>
      %a2 = stablehlo.and %a1, %n2 : tensor<i1>
      stablehlo.return %a2 : tensor<i1>
    } do {
      %rng1, %tmp1 = stablehlo.rng_bit_generator %iterArg_87, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %rng2, %out89 = stablehlo.rng_bit_generator %rng1, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %s1 = stablehlo.shift_right_logical %out89, %c_26 : tensor<ui64>
      %s2 = stablehlo.or %s1, %c_25 : tensor<ui64>
      %s3 = stablehlo.bitcast_convert %s2 : (tensor<ui64>) -> tensor<f64>
      %s4 = stablehlo.subtract %s3, %cst_68 : tensor<f64>
      %direction = stablehlo.compare LT, %s4, %cst_64, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %rng3, %tmp3 = stablehlo.rng_bit_generator %rng2, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %rng4, %tmp4 = stablehlo.rng_bit_generator %rng3, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %num_leaves = stablehlo.shift_left %c_66, %iterArg_81 : tensor<i64>

      // Middle while (21 args)
      %mid:21 = stablehlo.while(%mA0 = %iterArg, %mA1 = %iterArg_72, %mA2 = %iterArg_73, %mA3 = %iterArg_74, %mA4 = %iterArg_75, %mA5 = %iterArg_76, %mA6 = %iterArg_77, %mA7 = %iterArg_78, %mA8 = %iterArg_79, %mA9 = %iterArg_80, %mA10 = %iterArg_81, %mA11 = %iterArg_82, %mA12 = %iterArg_83, %mA13 = %iterArg_84, %mA14 = %iterArg_85, %mA15 = %c_67, %mA16 = %iterArg_86, %mA17 = %rng3, %mA18 = %cst_61, %mA19 = %cst_61, %mA20 = %c_67) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>, tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      cond {
        %lt2 = stablehlo.compare LT, %mA15, %num_leaves, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %nt = stablehlo.not %mA13 : tensor<i1>
        %c3 = stablehlo.and %lt2, %nt : tensor<i1>
        stablehlo.return %c3 : tensor<i1>
      } do {
        // Direction-based leaf select
        %leaf_q = stablehlo.select %direction, %mA3, %mA0 : tensor<i1>, tensor<1x2xf64>
        %leaf_p = stablehlo.select %direction, %mA4, %mA1 : tensor<i1>, tensor<1x2xf64>
        %leaf_g = stablehlo.select %direction, %mA5, %mA2 : tensor<i1>, tensor<1x2xf64>

        // === FULL THREEFRY (same as b2 lines 92-351) ===
        %1680 = stablehlo.bitcast_convert %mA17 : (tensor<2xui64>) -> tensor<2x2xui32>
        %1681 = stablehlo.reshape %1680 : (tensor<2x2xui32>) -> tensor<4xui32>
        %1682 = stablehlo.slice %1681 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
        %1683 = stablehlo.slice %1681 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
        %1684 = stablehlo.slice %1681 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
        %1685 = stablehlo.slice %1681 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
        %k0 = stablehlo.broadcast_in_dim %1682, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %k1 = stablehlo.broadcast_in_dim %1683, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %k2 = stablehlo.broadcast_in_dim %1684, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %k3 = stablehlo.broadcast_in_dim %1685, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        // First half: ks, 5 rounds, key injection
        %xk0 = stablehlo.xor %k0, %k1 : tensor<2xui32>
        %ks0 = stablehlo.xor %xk0, %c_59 : tensor<2xui32>
        %r1a = stablehlo.add %c_24, %k1 : tensor<2xui32>
        %r1b = stablehlo.add %k0, %r1a : tensor<2xui32>
        %r1c = stablehlo.shift_left %r1a, %c_58 : tensor<2xui32>
        %r1d = stablehlo.shift_right_logical %r1a, %c_23 : tensor<2xui32>
        %r1e = stablehlo.or %r1c, %r1d : tensor<2xui32>
        %r1f = stablehlo.xor %r1b, %r1e : tensor<2xui32>
        %r2a = stablehlo.add %r1b, %r1f : tensor<2xui32>
        %r2b = stablehlo.shift_left %r1f, %c_57 : tensor<2xui32>
        %r2c = stablehlo.shift_right_logical %r1f, %c_53 : tensor<2xui32>
        %r2d = stablehlo.or %r2b, %r2c : tensor<2xui32>
        %r2e = stablehlo.xor %r2a, %r2d : tensor<2xui32>
        %r3a = stablehlo.add %r2a, %r2e : tensor<2xui32>
        %r3b = stablehlo.shift_left %r2e, %c_56 : tensor<2xui32>
        %r3c = stablehlo.shift_right_logical %r2e, %c_55 : tensor<2xui32>
        %r3d = stablehlo.or %r3b, %r3c : tensor<2xui32>
        %r3e = stablehlo.xor %r3a, %r3d : tensor<2xui32>
        %r4a = stablehlo.add %r3a, %r3e : tensor<2xui32>
        %r4b = stablehlo.shift_left %r3e, %c_55 : tensor<2xui32>
        %r4c = stablehlo.shift_right_logical %r3e, %c_56 : tensor<2xui32>
        %r4d = stablehlo.or %r4b, %r4c : tensor<2xui32>
        %r4e = stablehlo.xor %r4a, %r4d : tensor<2xui32>
        %ki1_0 = stablehlo.add %r4a, %k1 : tensor<2xui32>
        %ki1_1t = stablehlo.add %r4e, %ks0 : tensor<2xui32>
        %ki1_1 = stablehlo.add %ki1_1t, %c_54 : tensor<2xui32>
        // Rounds 6-10 (second key injection block)
        %r5a0 = stablehlo.add %ki1_0, %ki1_1 : tensor<2xui32>
        %r5b = stablehlo.shift_left %ki1_1, %c_53 : tensor<2xui32>
        %r5c = stablehlo.shift_right_logical %ki1_1, %c_57 : tensor<2xui32>
        %r5d = stablehlo.or %r5b, %r5c : tensor<2xui32>
        %r5e = stablehlo.xor %r5a0, %r5d : tensor<2xui32>
        %r6a = stablehlo.add %r5a0, %r5e : tensor<2xui32>
        %r6b = stablehlo.shift_left %r5e, %c_52 : tensor<2xui32>
        %r6c = stablehlo.shift_right_logical %r5e, %c_48 : tensor<2xui32>
        %r6d = stablehlo.or %r6b, %r6c : tensor<2xui32>
        %r6e = stablehlo.xor %r6a, %r6d : tensor<2xui32>
        %r7a = stablehlo.add %r6a, %r6e : tensor<2xui32>
        %r7b = stablehlo.shift_left %r6e, %c_51 : tensor<2xui32>
        %r7c = stablehlo.shift_right_logical %r6e, %c_51 : tensor<2xui32>
        %r7d = stablehlo.or %r7b, %r7c : tensor<2xui32>
        %r7e = stablehlo.xor %r7a, %r7d : tensor<2xui32>
        %r8a = stablehlo.add %r7a, %r7e : tensor<2xui32>
        %r8b = stablehlo.shift_left %r7e, %c_50 : tensor<2xui32>
        %r8c = stablehlo.shift_right_logical %r7e, %c_22 : tensor<2xui32>
        %r8d = stablehlo.or %r8b, %r8c : tensor<2xui32>
        %r8e = stablehlo.xor %r8a, %r8d : tensor<2xui32>
        %ki2_0 = stablehlo.add %r8a, %ks0 : tensor<2xui32>
        %ki2_1t = stablehlo.add %r8e, %k0 : tensor<2xui32>
        %ki2_1 = stablehlo.add %ki2_1t, %c_49 : tensor<2xui32>
        // Rounds 11-15
        %ra0 = stablehlo.add %ki2_0, %ki2_1 : tensor<2xui32>
        %rab = stablehlo.shift_left %ki2_1, %c_58 : tensor<2xui32>
        %rac = stablehlo.shift_right_logical %ki2_1, %c_23 : tensor<2xui32>
        %rad = stablehlo.or %rab, %rac : tensor<2xui32>
        %rae = stablehlo.xor %ra0, %rad : tensor<2xui32>
        %rb0 = stablehlo.add %ra0, %rae : tensor<2xui32>
        %rbb = stablehlo.shift_left %rae, %c_57 : tensor<2xui32>
        %rbc = stablehlo.shift_right_logical %rae, %c_53 : tensor<2xui32>
        %rbd = stablehlo.or %rbb, %rbc : tensor<2xui32>
        %rbe = stablehlo.xor %rb0, %rbd : tensor<2xui32>
        %rc0 = stablehlo.add %rb0, %rbe : tensor<2xui32>
        %rcb = stablehlo.shift_left %rbe, %c_56 : tensor<2xui32>
        %rcc = stablehlo.shift_right_logical %rbe, %c_55 : tensor<2xui32>
        %rcd = stablehlo.or %rcb, %rcc : tensor<2xui32>
        %rce = stablehlo.xor %rc0, %rcd : tensor<2xui32>
        %rd0 = stablehlo.add %rc0, %rce : tensor<2xui32>
        %rdb = stablehlo.shift_left %rce, %c_55 : tensor<2xui32>
        %rdc = stablehlo.shift_right_logical %rce, %c_56 : tensor<2xui32>
        %rdd = stablehlo.or %rdb, %rdc : tensor<2xui32>
        %rde = stablehlo.xor %rd0, %rdd : tensor<2xui32>
        %ki3_0 = stablehlo.add %rd0, %k0 : tensor<2xui32>
        %ki3_1t = stablehlo.add %rde, %k1 : tensor<2xui32>
        %ki3_1 = stablehlo.add %ki3_1t, %c_48 : tensor<2xui32>
        // Rounds 16-20
        %sa0 = stablehlo.add %ki3_0, %ki3_1 : tensor<2xui32>
        %sab = stablehlo.shift_left %ki3_1, %c_53 : tensor<2xui32>
        %sac = stablehlo.shift_right_logical %ki3_1, %c_57 : tensor<2xui32>
        %sad = stablehlo.or %sab, %sac : tensor<2xui32>
        %sae = stablehlo.xor %sa0, %sad : tensor<2xui32>
        %sb0 = stablehlo.add %sa0, %sae : tensor<2xui32>
        %sbb = stablehlo.shift_left %sae, %c_52 : tensor<2xui32>
        %sbc = stablehlo.shift_right_logical %sae, %c_48 : tensor<2xui32>
        %sbd = stablehlo.or %sbb, %sbc : tensor<2xui32>
        %sbe = stablehlo.xor %sb0, %sbd : tensor<2xui32>
        %sc0 = stablehlo.add %sb0, %sbe : tensor<2xui32>
        %scb = stablehlo.shift_left %sbe, %c_51 : tensor<2xui32>
        %scc = stablehlo.shift_right_logical %sbe, %c_51 : tensor<2xui32>
        %scd = stablehlo.or %scb, %scc : tensor<2xui32>
        %sce = stablehlo.xor %sc0, %scd : tensor<2xui32>
        %sd0 = stablehlo.add %sc0, %sce : tensor<2xui32>
        %sdb = stablehlo.shift_left %sce, %c_50 : tensor<2xui32>
        %sdc = stablehlo.shift_right_logical %sce, %c_22 : tensor<2xui32>
        %sdd = stablehlo.or %sdb, %sdc : tensor<2xui32>
        %sde = stablehlo.xor %sd0, %sdd : tensor<2xui32>
        %ki4_0 = stablehlo.add %sd0, %k1 : tensor<2xui32>
        %ki4_1t = stablehlo.add %sde, %ks0 : tensor<2xui32>
        %ki4_1 = stablehlo.add %ki4_1t, %c_47 : tensor<2xui32>
        // Second half (for key2) - same structure with k2/k3
        %xk1 = stablehlo.xor %k2, %k3 : tensor<2xui32>
        %ks1 = stablehlo.xor %xk1, %c_59 : tensor<2xui32>
        %h1a = stablehlo.add %c_24, %k3 : tensor<2xui32>
        %h1b = stablehlo.add %k2, %h1a : tensor<2xui32>
        %h1c = stablehlo.shift_left %h1a, %c_58 : tensor<2xui32>
        %h1d = stablehlo.shift_right_logical %h1a, %c_23 : tensor<2xui32>
        %h1e = stablehlo.or %h1c, %h1d : tensor<2xui32>
        %h1f = stablehlo.xor %h1b, %h1e : tensor<2xui32>
        %h2a = stablehlo.add %h1b, %h1f : tensor<2xui32>
        %h2b = stablehlo.shift_left %h1f, %c_57 : tensor<2xui32>
        %h2c = stablehlo.shift_right_logical %h1f, %c_53 : tensor<2xui32>
        %h2d = stablehlo.or %h2b, %h2c : tensor<2xui32>
        %h2e = stablehlo.xor %h2a, %h2d : tensor<2xui32>
        %h3a = stablehlo.add %h2a, %h2e : tensor<2xui32>
        %h3b = stablehlo.shift_left %h2e, %c_56 : tensor<2xui32>
        %h3c = stablehlo.shift_right_logical %h2e, %c_55 : tensor<2xui32>
        %h3d = stablehlo.or %h3b, %h3c : tensor<2xui32>
        %h3e = stablehlo.xor %h3a, %h3d : tensor<2xui32>
        %h4a = stablehlo.add %h3a, %h3e : tensor<2xui32>
        %h4b = stablehlo.shift_left %h3e, %c_55 : tensor<2xui32>
        %h4c = stablehlo.shift_right_logical %h3e, %c_56 : tensor<2xui32>
        %h4d = stablehlo.or %h4b, %h4c : tensor<2xui32>
        %h4e = stablehlo.xor %h4a, %h4d : tensor<2xui32>
        %hki1_0 = stablehlo.add %h4a, %k3 : tensor<2xui32>
        %hki1_1t = stablehlo.add %h4e, %ks1 : tensor<2xui32>
        %hki1_1 = stablehlo.add %hki1_1t, %c_54 : tensor<2xui32>
        %h5a = stablehlo.add %hki1_0, %hki1_1 : tensor<2xui32>
        %h5b = stablehlo.shift_left %hki1_1, %c_53 : tensor<2xui32>
        %h5c = stablehlo.shift_right_logical %hki1_1, %c_57 : tensor<2xui32>
        %h5d = stablehlo.or %h5b, %h5c : tensor<2xui32>
        %h5e = stablehlo.xor %h5a, %h5d : tensor<2xui32>
        %h6a = stablehlo.add %h5a, %h5e : tensor<2xui32>
        %h6b = stablehlo.shift_left %h5e, %c_52 : tensor<2xui32>
        %h6c = stablehlo.shift_right_logical %h5e, %c_48 : tensor<2xui32>
        %h6d = stablehlo.or %h6b, %h6c : tensor<2xui32>
        %h6e = stablehlo.xor %h6a, %h6d : tensor<2xui32>
        %h7a = stablehlo.add %h6a, %h6e : tensor<2xui32>
        %h7b = stablehlo.shift_left %h6e, %c_51 : tensor<2xui32>
        %h7c = stablehlo.shift_right_logical %h6e, %c_51 : tensor<2xui32>
        %h7d = stablehlo.or %h7b, %h7c : tensor<2xui32>
        %h7e = stablehlo.xor %h7a, %h7d : tensor<2xui32>
        %h8a = stablehlo.add %h7a, %h7e : tensor<2xui32>
        %h8b = stablehlo.shift_left %h7e, %c_50 : tensor<2xui32>
        %h8c = stablehlo.shift_right_logical %h7e, %c_22 : tensor<2xui32>
        %h8d = stablehlo.or %h8b, %h8c : tensor<2xui32>
        %h8e = stablehlo.xor %h8a, %h8d : tensor<2xui32>
        %hki2_0 = stablehlo.add %h8a, %ks1 : tensor<2xui32>
        %hki2_1t = stablehlo.add %h8e, %k2 : tensor<2xui32>
        %hki2_1 = stablehlo.add %hki2_1t, %c_49 : tensor<2xui32>
        %ha0 = stablehlo.add %hki2_0, %hki2_1 : tensor<2xui32>
        %hab = stablehlo.shift_left %hki2_1, %c_58 : tensor<2xui32>
        %hac = stablehlo.shift_right_logical %hki2_1, %c_23 : tensor<2xui32>
        %had = stablehlo.or %hab, %hac : tensor<2xui32>
        %hae = stablehlo.xor %ha0, %had : tensor<2xui32>
        %hb0 = stablehlo.add %ha0, %hae : tensor<2xui32>
        %hbb = stablehlo.shift_left %hae, %c_57 : tensor<2xui32>
        %hbc = stablehlo.shift_right_logical %hae, %c_53 : tensor<2xui32>
        %hbd = stablehlo.or %hbb, %hbc : tensor<2xui32>
        %hbe = stablehlo.xor %hb0, %hbd : tensor<2xui32>
        %hc0 = stablehlo.add %hb0, %hbe : tensor<2xui32>
        %hcb = stablehlo.shift_left %hbe, %c_56 : tensor<2xui32>
        %hcc = stablehlo.shift_right_logical %hbe, %c_55 : tensor<2xui32>
        %hcd = stablehlo.or %hcb, %hcc : tensor<2xui32>
        %hce = stablehlo.xor %hc0, %hcd : tensor<2xui32>
        %hd0 = stablehlo.add %hc0, %hce : tensor<2xui32>
        %hdb = stablehlo.shift_left %hce, %c_55 : tensor<2xui32>
        %hdc = stablehlo.shift_right_logical %hce, %c_56 : tensor<2xui32>
        %hdd = stablehlo.or %hdb, %hdc : tensor<2xui32>
        %hde = stablehlo.xor %hd0, %hdd : tensor<2xui32>
        %hki3_0 = stablehlo.add %hd0, %k2 : tensor<2xui32>
        %hki3_1t = stablehlo.add %hde, %k3 : tensor<2xui32>
        %hki3_1 = stablehlo.add %hki3_1t, %c_48 : tensor<2xui32>
        %hsa0 = stablehlo.add %hki3_0, %hki3_1 : tensor<2xui32>
        %hsab = stablehlo.shift_left %hki3_1, %c_53 : tensor<2xui32>
        %hsac = stablehlo.shift_right_logical %hki3_1, %c_57 : tensor<2xui32>
        %hsad = stablehlo.or %hsab, %hsac : tensor<2xui32>
        %hsae = stablehlo.xor %hsa0, %hsad : tensor<2xui32>
        %hsb0 = stablehlo.add %hsa0, %hsae : tensor<2xui32>
        %hsbb = stablehlo.shift_left %hsae, %c_52 : tensor<2xui32>
        %hsbc = stablehlo.shift_right_logical %hsae, %c_48 : tensor<2xui32>
        %hsbd = stablehlo.or %hsbb, %hsbc : tensor<2xui32>
        %hsbe = stablehlo.xor %hsb0, %hsbd : tensor<2xui32>
        %hsc0 = stablehlo.add %hsb0, %hsbe : tensor<2xui32>
        %hscb = stablehlo.shift_left %hsbe, %c_51 : tensor<2xui32>
        %hscc = stablehlo.shift_right_logical %hsbe, %c_51 : tensor<2xui32>
        %hscd = stablehlo.or %hscb, %hscc : tensor<2xui32>
        %hsce = stablehlo.xor %hsc0, %hscd : tensor<2xui32>
        %hsd0 = stablehlo.add %hsc0, %hsce : tensor<2xui32>
        %hsdb = stablehlo.shift_left %hsce, %c_50 : tensor<2xui32>
        %hsdc = stablehlo.shift_right_logical %hsce, %c_22 : tensor<2xui32>
        %hsdd = stablehlo.or %hsdb, %hsdc : tensor<2xui32>
        %hsde = stablehlo.xor %hsd0, %hsdd : tensor<2xui32>
        %hki4_0 = stablehlo.add %hsd0, %k3 : tensor<2xui32>
        %hki4_1t = stablehlo.add %hsde, %ks1 : tensor<2xui32>
        %hki4_1 = stablehlo.add %hki4_1t, %c_47 : tensor<2xui32>
        // Reassemble key1
        %o0a = stablehlo.slice %ki4_0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %o0b = stablehlo.slice %ki4_1 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %o1a = stablehlo.slice %hki4_0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %o1b = stablehlo.slice %hki4_1 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %cat0 = stablehlo.concatenate %o0a, %o0b, %o1a, %o1b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh0 = stablehlo.reshape %cat0 : (tensor<4xui32>) -> tensor<2x2xui32>
        %key1 = stablehlo.bitcast_convert %rsh0 : (tensor<2x2xui32>) -> tensor<2xui64>
        // Reassemble key2
        %o2a = stablehlo.slice %ki4_0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %o2b = stablehlo.slice %ki4_1 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %o3a = stablehlo.slice %hki4_0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %o3b = stablehlo.slice %hki4_1 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %cat1 = stablehlo.concatenate %o2a, %o2b, %o3a, %o3b, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %rsh1 = stablehlo.reshape %cat1 : (tensor<4xui32>) -> tensor<2x2xui32>
        %key2 = stablehlo.bitcast_convert %rsh1 : (tensor<2x2xui32>) -> tensor<2xui64>

        // NO leapfrog — stablehlo.if uses passthrough values
        %is_first = stablehlo.compare EQ, %mA15, %c_67, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %if_res:16 = "stablehlo.if"(%is_first) ({
          stablehlo.return %leaf_q, %leaf_p, %leaf_g, %leaf_q, %leaf_p, %leaf_g, %leaf_q, %leaf_g, %mA8, %mA9, %c_67, %mA11, %c_63, %mA14, %c_66, %leaf_p : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>
        }, {
          %rng_if, %bits_if = stablehlo.rng_bit_generator %key2, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
          stablehlo.return %mA0, %mA1, %mA2, %mA3, %mA4, %mA5, %mA6, %mA7, %mA8, %mA9, %mA10, %mA11, %mA12, %mA14, %mA15, %mA16 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>
        }) : (tensor<i1>) -> (tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>)

        %new_idx = stablehlo.add %mA15, %c_66 : tensor<i64>
        %new_cnt = stablehlo.add %mA20, %c_66 : tensor<i64>
        stablehlo.return %if_res#0, %if_res#1, %if_res#2, %if_res#3, %if_res#4, %if_res#5, %if_res#6, %if_res#7, %if_res#8, %if_res#9, %if_res#10, %if_res#11, %c_63, %if_res#12, %if_res#13, %new_idx, %if_res#15, %key1, %mA18, %mA19, %new_cnt : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>, tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      }

      // Simplified outer while body (no turning check with dot_general)
      %new_depth = stablehlo.add %iterArg_81, %c_66 : tensor<i64>
      stablehlo.return %mid#0, %mid#1, %mid#2, %mid#3, %mid#4, %mid#5, %mid#6, %mid#7, %mid#8, %mid#9, %new_depth, %mid#11, %mid#12, %mid#13, %mid#14, %mid#16, %rng1 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    }
    %result = stablehlo.reshape %out#0 : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %c_0_rng : tensor<2x1xf64>, tensor<2xui64>
  }
}
