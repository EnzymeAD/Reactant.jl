module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %cst_neg_half = stablehlo.constant dense<-5.000000e-01> : tensor<1x2xf64>
    %cst_half_1x2 = stablehlo.constant dense<5.000000e-01> : tensor<1x2xf64>
    %c_magic = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %c_shift12 = stablehlo.constant dense<12> : tensor<ui64>
    %cst_inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %cst_3x2 = stablehlo.constant dense<0.0> : tensor<3x2xf64>
    %c_3 = stablehlo.constant dense<3> : tensor<i64>
    %c_false = stablehlo.constant dense<false> : tensor<i1>
    %cst_half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %cst_zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %cst_one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_1000 = stablehlo.constant dense<1.000000e+03> : tensor<f64>
    %cst_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
    %cst_f64 = stablehlo.constant dense<0.0> : tensor<f64>
    // Initial RNG call
    %rng0, %rng0_out = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT
      : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
    // Outer while: 17 args matching NUTS tree state
    %outer:17 = stablehlo.while(
      %o0 = %cst_1x2, %o1 = %cst_1x2, %o2 = %cst_1x2,
      %o3 = %cst_1x2, %o4 = %cst_1x2, %o5 = %cst_1x2,
      %o6 = %cst_1x2, %o7 = %cst_1x2,
      %o8 = %cst_f64, %o9 = %cst_f64,
      %o10 = %c_0, %o11 = %cst_f64,
      %o12 = %c_false, %o13 = %c_false,
      %o14 = %cst_f64, %o15 = %cst_1x2,
      %o16 = %rng0
    ) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<1x2xf64>, tensor<1x2xf64>,
        tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>,
        tensor<i1>, tensor<i1>, tensor<f64>,
        tensor<1x2xf64>, tensor<2xui64>
    cond {
      %cmp = stablehlo.compare LT, %o10, %c_3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %n12 = stablehlo.not %o12 : tensor<i1>
      %n13 = stablehlo.not %o13 : tensor<i1>
      %a1 = stablehlo.and %cmp, %n12 : tensor<i1>
      %a2 = stablehlo.and %a1, %n13 : tensor<i1>
      stablehlo.return %a2 : tensor<i1>
    } do {
      // Direction sampling via rng_bit_generator (replaces ~270 lines of Threefry)
      %rng1, %rng1_out = stablehlo.rng_bit_generator %o16, algorithm = DEFAULT
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %ds = stablehlo.shift_right_logical %rng1_out, %c_shift12 : tensor<ui64>
      %do2 = stablehlo.or %ds, %c_magic : tensor<ui64>
      %df = stablehlo.bitcast_convert %do2 : (tensor<ui64>) -> tensor<f64>
      %dv = stablehlo.subtract %df, %cst_one : tensor<f64>
      %dir = stablehlo.compare LT, %dv, %cst_half, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      // Additional RNG keys (replaces ~270 lines of Threefry)
      %rng2, %rng2_out = stablehlo.rng_bit_generator %rng1, algorithm = DEFAULT
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %rng3, %rng3_out = stablehlo.rng_bit_generator %rng2, algorithm = DEFAULT
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %depth_lim = stablehlo.shift_left %c_1, %o10 : tensor<i64>
      // Middle while: 21 args matching NUTS doubling state
      %mid:21 = stablehlo.while(
        %m0 = %o0, %m1 = %o1, %m2 = %o2,
        %m3 = %o3, %m4 = %o4, %m5 = %o5,
        %m6 = %o6, %m7 = %o7,
        %m8 = %o8, %m9 = %o9, %m10 = %o10, %m11 = %o11,
        %m12 = %o12, %m13 = %o13, %m14 = %o14,
        %m15 = %c_0, %m16 = %o15,
        %m17 = %rng2, %m18 = %cst_3x2, %m19 = %cst_3x2, %m20 = %c_0
      ) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>,
          tensor<i1>, tensor<i1>, tensor<f64>,
          tensor<i64>, tensor<1x2xf64>,
          tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      cond {
        %mc = stablehlo.compare LT, %m15, %depth_lim, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %mn1 = stablehlo.not %m12 : tensor<i1>
        %mn2 = stablehlo.not %m13 : tensor<i1>
        %ma1 = stablehlo.and %mc, %mn1 : tensor<i1>
        %ma2 = stablehlo.and %ma1, %mn2 : tensor<i1>
        stablehlo.return %ma2 : tensor<i1>
      } do {
        // Select leaf based on direction
        %lq = stablehlo.select %dir, %m3, %m0 : tensor<i1>, tensor<1x2xf64>
        %lp = stablehlo.select %dir, %m4, %m1 : tensor<i1>, tensor<1x2xf64>
        %lg = stablehlo.select %dir, %m5, %m2 : tensor<i1>, tensor<1x2xf64>
        // RNG in middle while (replaces ~260 lines of Threefry)
        %mr1, %mr1o = stablehlo.rng_bit_generator %m17, algorithm = DEFAULT
          : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
        %mr2, %mr2o = stablehlo.rng_bit_generator %mr1, algorithm = DEFAULT
          : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
        // Leapfrog integration step
        %ss = stablehlo.select %dir, %arg2, %cst_f64 : tensor<i1>, tensor<f64>
        %hs = stablehlo.multiply %ss, %cst_half : tensor<f64>
        %hsb = stablehlo.broadcast_in_dim %hs, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
        %ssb = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
        %gs = stablehlo.multiply %hsb, %lg : tensor<1x2xf64>
        %ph = stablehlo.subtract %lp, %gs : tensor<1x2xf64>
        %vel = stablehlo.dot_general %ph, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
        %qs = stablehlo.multiply %ssb, %vel : tensor<1x2xf64>
        %qn = stablehlo.add %lq, %qs : tensor<1x2xf64>
        %ke = stablehlo.dot_general %qn, %qn, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
        %kh = stablehlo.multiply %cst_half, %ke : tensor<f64>
        %qs2 = stablehlo.add %qn, %qn : tensor<1x2xf64>
        %ng = stablehlo.multiply %cst_half_1x2, %qs2 : tensor<1x2xf64>
        %ngs = stablehlo.multiply %hsb, %ng : tensor<1x2xf64>
        %pn = stablehlo.subtract %ph, %ngs : tensor<1x2xf64>
        %pv = stablehlo.dot_general %pn, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
        %pk = stablehlo.dot_general %pn, %pv, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
        %te = stablehlo.add %ke, %pk : tensor<f64>
        %H2 = stablehlo.multiply %cst_half, %te : tensor<f64>
        %de = stablehlo.subtract %te, %cst_f64 : tensor<f64>
        %hd = stablehlo.multiply %cst_half, %de : tensor<f64>
        %isn = stablehlo.compare NE, %hd, %hd, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %saf = stablehlo.select %isn, %cst_inf, %hd : tensor<i1>, tensor<f64>
        %ns = stablehlo.negate %saf : tensor<f64>
        %idv = stablehlo.compare GT, %saf, %cst_1000, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %en = stablehlo.exponential %ns : tensor<f64>
        %ap = stablehlo.minimum %en, %cst_one : tensor<f64>
        // Base tree vs combine trees
        %is1 = stablehlo.compare EQ, %m15, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %ifr:16 = "stablehlo.if"(%is1) ({
          stablehlo.return %qn, %pn, %ng, %qn, %pn, %ng,
            %qn, %ng, %kh, %H2, %c_0, %ns, %idv, %ap, %c_1, %pn
            : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
              tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
              tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>,
              tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>,
              tensor<i64>, tensor<1x2xf64>
        }, {
          %xl = stablehlo.select %dir, %m0, %qn : tensor<i1>, tensor<1x2xf64>
          %xpl = stablehlo.select %dir, %m1, %pn : tensor<i1>, tensor<1x2xf64>
          %xgl = stablehlo.select %dir, %m2, %ng : tensor<i1>, tensor<1x2xf64>
          %xr = stablehlo.select %dir, %qn, %m3 : tensor<i1>, tensor<1x2xf64>
          %xpr = stablehlo.select %dir, %pn, %m4 : tensor<i1>, tensor<1x2xf64>
          %xgr = stablehlo.select %dir, %ng, %m5 : tensor<i1>, tensor<1x2xf64>
          // Log-sum-exp weight combination
          %mw = stablehlo.maximum %m11, %ns : tensor<f64>
          %dw = stablehlo.subtract %m11, %ns : tensor<f64>
          %nw = stablehlo.compare NE, %dw, %dw : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %sw = stablehlo.add %m11, %ns : tensor<f64>
          %aw = stablehlo.abs %dw : tensor<f64>
          %naw = stablehlo.negate %aw : tensor<f64>
          %ew = stablehlo.exponential %naw : tensor<f64>
          %lw = stablehlo.log_plus_one %ew : tensor<f64>
          %lsw = stablehlo.add %mw, %lw : tensor<f64>
          %wt = stablehlo.select %nw, %sw, %lsw : tensor<i1>, tensor<f64>
          // Uniform transition probability
          %td = stablehlo.subtract %ns, %m11 : tensor<f64>
          %tp = stablehlo.logistic %td : tensor<f64>
          // Accept/reject RNG inside stablehlo.if
          %ar, %aro = stablehlo.rng_bit_generator %mr2, algorithm = DEFAULT
            : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
          %ars = stablehlo.shift_right_logical %aro, %c_shift12 : tensor<ui64>
          %aro2 = stablehlo.or %ars, %c_magic : tensor<ui64>
          %arf = stablehlo.bitcast_convert %aro2 : (tensor<ui64>) -> tensor<f64>
          %arv = stablehlo.subtract %arf, %cst_one : tensor<f64>
          %da = stablehlo.compare LT, %arv, %tp, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %xq = stablehlo.select %da, %qn, %m6 : tensor<i1>, tensor<1x2xf64>
          %xg2 = stablehlo.select %da, %ng, %m7 : tensor<i1>, tensor<1x2xf64>
          %xu = stablehlo.select %da, %kh, %m8 : tensor<i1>, tensor<f64>
          %xh = stablehlo.select %da, %H2, %m9 : tensor<i1>, tensor<f64>
          %xd = stablehlo.add %m10, %c_1 : tensor<i64>
          %xdv = stablehlo.or %m13, %idv : tensor<i1>
          %xac = stablehlo.add %m14, %ap : tensor<f64>
          %xid = stablehlo.add %m15, %c_1 : tensor<i64>
          %xps = stablehlo.add %m16, %pn : tensor<1x2xf64>
          stablehlo.return %xl, %xpl, %xgl, %xr, %xpr, %xgr,
            %xq, %xg2, %xu, %xh, %xd, %wt, %xdv, %xac, %xid, %xps
            : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
              tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
              tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>,
              tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>,
              tensor<i64>, tensor<1x2xf64>
        }) : (tensor<i1>) -> (tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
              tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
              tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>,
              tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>,
              tensor<i64>, tensor<1x2xf64>)
        // Checkpoint index computation
        %cshr = stablehlo.shift_right_logical %m20, %c_1 : tensor<i64>
        %cpop = stablehlo.popcnt %cshr : tensor<i64>
        %lp1 = stablehlo.add %m20, %c_1 : tensor<i64>
        %lnot = stablehlo.not %m20 : tensor<i64>
        %lnd = stablehlo.and %lnot, %lp1 : tensor<i64>
        %lsub = stablehlo.subtract %lnd, %c_1 : tensor<i64>
        %lpop = stablehlo.popcnt %lsub : tensor<i64>
        %idiff = stablehlo.subtract %cpop, %lpop : tensor<i64>
        %imax = stablehlo.add %idiff, %c_1 : tensor<i64>
        %iev = stablehlo.and %m20, %c_1 : tensor<i64>
        %ieve = stablehlo.compare EQ, %iev, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        // Update checkpoints at even leaf indices
        %ck:2 = "stablehlo.if"(%ieve) ({
          %up = stablehlo.dynamic_update_slice %m18, %pn, %cpop, %c_0
            : (tensor<3x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<3x2xf64>
          %ups = stablehlo.dynamic_update_slice %m19, %ifr#15, %cpop, %c_0
            : (tensor<3x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<3x2xf64>
          stablehlo.return %up, %ups : tensor<3x2xf64>, tensor<3x2xf64>
        }, {
          stablehlo.return %m18, %m19 : tensor<3x2xf64>, tensor<3x2xf64>
        }) : (tensor<i1>) -> (tensor<3x2xf64>, tensor<3x2xf64>)
        // Inner turning-check while (3rd level of nesting)
        %tw:2 = stablehlo.while(%ti = %cpop, %tt = %c_false) : tensor<i64>, tensor<i1>
        cond {
          %tge = stablehlo.compare GE, %ti, %imax, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
          %tnt = stablehlo.not %tt : tensor<i1>
          %ta = stablehlo.and %tge, %tnt : tensor<i1>
          stablehlo.return %ta : tensor<i1>
        } do {
          %tcp = stablehlo.dynamic_slice %ck#0, %ti, %c_0, sizes = [1, 2]
            : (tensor<3x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
          %tcs = stablehlo.dynamic_slice %ck#1, %ti, %c_0, sizes = [1, 2]
            : (tensor<3x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
          %td1 = stablehlo.subtract %ifr#15, %tcs : tensor<1x2xf64>
          %ts1 = stablehlo.add %td1, %tcp : tensor<1x2xf64>
          %tv1 = stablehlo.dot_general %tcp, %arg3, contracting_dims = [1] x [1]
            : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
          %tps = stablehlo.add %tcp, %pn : tensor<1x2xf64>
          %tc = stablehlo.multiply %cst_neg_half, %tps : tensor<1x2xf64>
          %targ = stablehlo.add %ts1, %tc : tensor<1x2xf64>
          %td2 = stablehlo.dot_general %tv1, %targ, contracting_dims = [0, 1] x [0, 1]
            : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
          %td3 = stablehlo.dot_general %pv, %targ, contracting_dims = [0, 1] x [0, 1]
            : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
          %tl1 = stablehlo.compare LE, %td2, %cst_zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %tl2 = stablehlo.compare LE, %td3, %cst_zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %ttr = stablehlo.or %tl1, %tl2 : tensor<i1>
          %tni = stablehlo.subtract %ti, %c_1 : tensor<i64>
          stablehlo.return %tni, %ttr : tensor<i64>, tensor<i1>
        }
        %tfn = stablehlo.select %is1, %c_false, %tw#1 : tensor<i1>, tensor<i1>
        // Middle while return: 21 values
        stablehlo.return %ifr#0, %ifr#1, %ifr#2, %ifr#3, %ifr#4, %ifr#5,
          %ifr#6, %ifr#7, %ifr#8, %ifr#9, %ifr#10, %ifr#11,
          %tfn, %ifr#12, %ifr#13, %ifr#14, %ifr#15,
          %mr1, %ck#0, %ck#1, %lp1
          : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
            tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
            tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>,
            tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>,
            tensor<f64>, tensor<i64>, tensor<1x2xf64>,
            tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      }
      // Post-doubling: biased transition for outer tree combination
      %pl = stablehlo.select %dir, %o0, %mid#0 : tensor<i1>, tensor<1x2xf64>
      %ppl = stablehlo.select %dir, %o1, %mid#1 : tensor<i1>, tensor<1x2xf64>
      %pgl = stablehlo.select %dir, %o2, %mid#2 : tensor<i1>, tensor<1x2xf64>
      %pr = stablehlo.select %dir, %mid#3, %o3 : tensor<i1>, tensor<1x2xf64>
      %ppr = stablehlo.select %dir, %mid#4, %o4 : tensor<i1>, tensor<1x2xf64>
      %pgr = stablehlo.select %dir, %mid#5, %o5 : tensor<i1>, tensor<1x2xf64>
      // Log-sum-exp weight combination
      %pmw = stablehlo.maximum %o11, %mid#11 : tensor<f64>
      %pdw = stablehlo.subtract %o11, %mid#11 : tensor<f64>
      %pnw = stablehlo.compare NE, %pdw, %pdw : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %psw = stablehlo.add %o11, %mid#11 : tensor<f64>
      %paw = stablehlo.abs %pdw : tensor<f64>
      %pnaw = stablehlo.negate %paw : tensor<f64>
      %pew = stablehlo.exponential %pnaw : tensor<f64>
      %plw = stablehlo.log_plus_one %pew : tensor<f64>
      %plsw = stablehlo.add %pmw, %plw : tensor<f64>
      %pwt = stablehlo.select %pnw, %psw, %plsw : tensor<i1>, tensor<f64>
      // Biased transition probability
      %ptd = stablehlo.subtract %mid#11, %o11 : tensor<f64>
      %pte = stablehlo.exponential %ptd : tensor<f64>
      %ptp = stablehlo.minimum %pte, %cst_one : tensor<f64>
      %pfl = stablehlo.or %mid#12, %mid#13 : tensor<i1>
      %ptp2 = stablehlo.select %pfl, %cst_zero, %ptp : tensor<i1>, tensor<f64>
      // Post-doubling accept/reject RNG
      %prng, %prng_out = stablehlo.rng_bit_generator %rng3, algorithm = DEFAULT
        : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %prs = stablehlo.shift_right_logical %prng_out, %c_shift12 : tensor<ui64>
      %pro = stablehlo.or %prs, %c_magic : tensor<ui64>
      %prf = stablehlo.bitcast_convert %pro : (tensor<ui64>) -> tensor<f64>
      %prv = stablehlo.subtract %prf, %cst_one : tensor<f64>
      %pra = stablehlo.compare LT, %prv, %ptp2, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      // Select proposal
      %pq = stablehlo.select %pra, %mid#6, %o6 : tensor<i1>, tensor<1x2xf64>
      %pg = stablehlo.select %pra, %mid#7, %o7 : tensor<i1>, tensor<1x2xf64>
      %pu = stablehlo.select %pra, %mid#8, %o8 : tensor<i1>, tensor<f64>
      %ph2 = stablehlo.select %pra, %mid#9, %o9 : tensor<i1>, tensor<f64>
      // Update outer state
      %pnd = stablehlo.add %o10, %c_1 : tensor<i64>
      %pps = stablehlo.add %o15, %mid#16 : tensor<1x2xf64>
      // Outer turning check
      %ov1 = stablehlo.dot_general %ppl, %arg3, contracting_dims = [1] x [1]
        : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
      %ov2 = stablehlo.dot_general %ppr, %arg3, contracting_dims = [1] x [1]
        : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
      %osum = stablehlo.add %ppl, %ppr : tensor<1x2xf64>
      %octr = stablehlo.multiply %cst_neg_half, %osum : tensor<1x2xf64>
      %oarg = stablehlo.add %pps, %octr : tensor<1x2xf64>
      %od1 = stablehlo.dot_general %ov1, %oarg, contracting_dims = [0, 1] x [0, 1]
        : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
      %od2 = stablehlo.dot_general %ov2, %oarg, contracting_dims = [0, 1] x [0, 1]
        : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
      %ol1 = stablehlo.compare LE, %od1, %cst_zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %ol2 = stablehlo.compare LE, %od2, %cst_zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %otr = stablehlo.or %ol1, %ol2 : tensor<i1>
      %otrn = stablehlo.or %mid#12, %otr : tensor<i1>
      %odiv = stablehlo.or %o13, %mid#13 : tensor<i1>
      %oacc = stablehlo.add %o14, %mid#14 : tensor<f64>
      // Outer while return: 17 values
      stablehlo.return %pl, %ppl, %pgl, %pr, %ppr, %pgr,
        %pq, %pg, %pu, %ph2, %pnd, %pwt,
        %otrn, %odiv, %oacc, %pps, %rng1
        : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>,
          tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>,
          tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>,
          tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    }
    %result = stablehlo.transpose %outer#6, dims = [1, 0] : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %result, %outer#16 : tensor<2x1xf64>, tensor<2xui64>
  }
}
