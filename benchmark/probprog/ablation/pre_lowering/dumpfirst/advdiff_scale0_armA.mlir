module @reactant_run_mcmc attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.func @enzymexla_lapack_dpotrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @enzymexla_blas_dtrsm_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64)
  func.func @main(%arg0: tensor<2xui64> {enzymexla.memory_effects = [], tf.aliasing_output = 10 : i32}, %arg1: tensor<3x1xf64> {enzymexla.memory_effects = []}, %arg2: tensor<f64> {enzymexla.memory_effects = []}, %arg3: tensor<3x3xf64> {enzymexla.memory_effects = []}, %arg4: tensor<20x20xf64> {enzymexla.memory_effects = []}, %arg5: tensor<20x20xf64> {enzymexla.memory_effects = []}, %arg6: tensor<1x20xf64> {enzymexla.memory_effects = []}, %arg7: tensor<8x5xf64> {enzymexla.memory_effects = []}) -> (tensor<3x20xf64>, tensor<2x20xi1>, tensor<20xf64>, tensor<2xui64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<2xui64>, tensor<2xui64>) attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %c_0 = stablehlo.constant dense<67> : tensor<i8>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %c_2 = stablehlo.constant dense<78> : tensor<i8>
    %c_3 = stablehlo.constant dense<84> : tensor<i8>
    %c_4 = stablehlo.constant dense<85> : tensor<i8>
    %c_5 = stablehlo.constant dense<76> : tensor<i8>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf64>
    %c_6 = stablehlo.constant dense<466688986> : tensor<1xui32>
    %c_7 = stablehlo.constant dense<1> : tensor<1xui32>
    %c_8 = stablehlo.constant dense<2> : tensor<1xui32>
    %c_9 = stablehlo.constant dense<0> : tensor<ui32>
    %c_10 = stablehlo.constant dense<3> : tensor<1xui32>
    %c_11 = stablehlo.constant dense<29> : tensor<1xui32>
    %c_12 = stablehlo.constant dense<16> : tensor<1xui32>
    %c_13 = stablehlo.constant dense<8> : tensor<1xui32>
    %c_14 = stablehlo.constant dense<24> : tensor<1xui32>
    %c_15 = stablehlo.constant dense<4> : tensor<1xui32>
    %c_16 = stablehlo.constant dense<19> : tensor<1xui32>
    %c_17 = stablehlo.constant dense<13> : tensor<1xui32>
    %c_18 = stablehlo.constant dense<17> : tensor<1xui32>
    %c_19 = stablehlo.constant dense<15> : tensor<1xui32>
    %c_20 = stablehlo.constant dense<26> : tensor<1xui32>
    %c_21 = stablehlo.constant dense<6> : tensor<1xui32>
    %cst_22 = stablehlo.constant dense<-5.000000e-01> : tensor<1x3xf64>
    %c_23 = stablehlo.constant dense<true> : tensor<i1>
    %cst_24 = stablehlo.constant dense<1.4142135623730951> : tensor<1x3xf64>
    %c_25 = stablehlo.constant dense<true> : tensor<2x20xi1>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<3x20xf64>
    %cst_27 = stablehlo.constant dense<4.000000e+01> : tensor<1x1xf64>
    %c_28 = stablehlo.constant dense<"0x01000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001000000000000000000000000000000000000000001"> : tensor<20x20xi1>
    %c_29 = stablehlo.constant dense<"0x01010101010101010101010101010101010101010001010101010101010101010101010101010101000001010101010101010101010101010101010100000001010101010101010101010101010101010000000001010101010101010101010101010101000000000001010101010101010101010101010100000000000001010101010101010101010101010000000000000001010101010101010101010101000000000000000001010101010101010101010100000000000000000001010101010101010101010000000000000000000001010101010101010101000000000000000000000001010101010101010100000000000000000000000001010101010101010000000000000000000000000001010101010101000000000000000000000000000001010101010100000000000000000000000000000001010101010000000000000000000000000000000001010101000000000000000000000000000000000001010100000000000000000000000000000000000001010000000000000000000000000000000000000001"> : tensor<20x20xi1>
    %c_30 = stablehlo.constant dense<8> : tensor<3xui32>
    %c_31 = stablehlo.constant dense<19> : tensor<3xui32>
    %c_32 = stablehlo.constant dense<[0, 1, 2]> : tensor<3xui32>
    %c_33 = stablehlo.constant dense<5> : tensor<ui32>
    %c_34 = stablehlo.constant dense<5> : tensor<1xui32>
    %c_35 = stablehlo.constant dense<8> : tensor<2xui32>
    %c_36 = stablehlo.constant dense<19> : tensor<2xui32>
    %c_37 = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %cst_38 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf64>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<20x1xf64>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf64>
    %cst_41 = stablehlo.constant dense<0.91893853320467266> : tensor<f64>
    %cst_42 = stablehlo.constant dense<-4.000000e+01> : tensor<f64>
    %cst_43 = stablehlo.constant dense<2.500000e-02> : tensor<f64>
    %c_44 = stablehlo.constant dense<[[2], [5], [7], [9], [12], [14], [16], [19]]> : tensor<8x1xi64>
    %cst_45 = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %cst_46 = stablehlo.constant dense<2.756815599614018> : tensor<f64>
    %cst_47 = stablehlo.constant dense<"0x000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000F03F"> : tensor<20x20xf64>
    %c_48 = stablehlo.constant dense<1> : tensor<8x1xi64>
    %c_49 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %c_50 = stablehlo.constant dense<12> : tensor<ui64>
    %cst_51 = stablehlo.constant dense<-0.99999999999999988> : tensor<1x3xf64>
    %cst_52 = stablehlo.constant dense<2.000000e+00> : tensor<1x3xf64>
    %cst_53 = stablehlo.constant dense<1.000000e+00> : tensor<1x3xf64>
    %c_54 = stablehlo.constant dense<4607182418800017408> : tensor<1x3xui64>
    %c_55 = stablehlo.constant dense<12> : tensor<1x3xui64>
    %c_56 = stablehlo.constant dense<5> : tensor<3xui32>
    %c_57 = stablehlo.constant dense<4> : tensor<3xui32>
    %c_58 = stablehlo.constant dense<3> : tensor<3xui32>
    %c_59 = stablehlo.constant dense<2> : tensor<3xui32>
    %c_60 = stablehlo.constant dense<24> : tensor<3xui32>
    %c_61 = stablehlo.constant dense<16> : tensor<3xui32>
    %c_62 = stablehlo.constant dense<29> : tensor<3xui32>
    %c_63 = stablehlo.constant dense<17> : tensor<3xui32>
    %c_64 = stablehlo.constant dense<1> : tensor<3xui32>
    %c_65 = stablehlo.constant dense<6> : tensor<3xui32>
    %c_66 = stablehlo.constant dense<26> : tensor<3xui32>
    %c_67 = stablehlo.constant dense<15> : tensor<3xui32>
    %c_68 = stablehlo.constant dense<13> : tensor<3xui32>
    %c_69 = stablehlo.constant dense<466688986> : tensor<3xui32>
    %c_70 = stablehlo.constant dense<5> : tensor<2xui32>
    %c_71 = stablehlo.constant dense<4> : tensor<2xui32>
    %c_72 = stablehlo.constant dense<3> : tensor<2xui32>
    %c_73 = stablehlo.constant dense<2> : tensor<2xui32>
    %c_74 = stablehlo.constant dense<24> : tensor<2xui32>
    %c_75 = stablehlo.constant dense<16> : tensor<2xui32>
    %c_76 = stablehlo.constant dense<29> : tensor<2xui32>
    %c_77 = stablehlo.constant dense<17> : tensor<2xui32>
    %c_78 = stablehlo.constant dense<1> : tensor<2xui32>
    %c_79 = stablehlo.constant dense<6> : tensor<2xui32>
    %c_80 = stablehlo.constant dense<26> : tensor<2xui32>
    %c_81 = stablehlo.constant dense<15> : tensor<2xui32>
    %c_82 = stablehlo.constant dense<13> : tensor<2xui32>
    %c_83 = stablehlo.constant dense<466688986> : tensor<2xui32>
    %cst_84 = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<0x7FF0000000000000> : tensor<f64>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<5x3xf64>
    %c_86 = stablehlo.constant dense<5> : tensor<i64>
    %c_87 = stablehlo.constant dense<false> : tensor<i1>
    %cst_88 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %cst_89 = stablehlo.constant dense<5.000000e-01> : tensor<1x1xf64>
    %cst_90 = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<f64>
    %c_91 = stablehlo.constant dense<20> : tensor<i64>
    %c_92 = stablehlo.constant dense<1> : tensor<i64>
    %c_93 = stablehlo.constant dense<0> : tensor<i64>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<20xf64>
    %cst_95 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_96 = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    %cst_97 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<3x3xf64>
    %cst_98 = stablehlo.constant dense<1.000000e+03> : tensor<f64>
    %0 = stablehlo.reshape %arg1 : (tensor<3x1xf64>) -> tensor<1x3xf64>
    %1 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %2 = stablehlo.reshape %arg6 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %3 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<8x5xf64>) -> tensor<5x8xf64>
    %4 = stablehlo.dot_general %cst_97, %arg3, contracting_dims = [1] x [1] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %5 = stablehlo.dot_general %4, %cst_97, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %6:2 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_1, %5, %c_1, %c) {api_version = 3 : i32, backend_config = "\00p~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i8>, tensor<i64>, tensor<3x3xf64>, tensor<i64>, tensor<i64>) -> (tensor<3x3xf64>, tensor<i64>)
    %7 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %9 = stablehlo.compare GE, %7, %8 : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %10 = stablehlo.select %9, %6#0, %cst : tensor<3x3xi1>, tensor<3x3xf64>
    %11 = stablehlo.dot_general %cst_97, %10, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %12 = stablehlo.dot_general %11, %cst_97, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %13 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_1, %c_1, %cst_95, %12, %c_1, %cst_96, %c_1) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<3x3xf64>, tensor<i64>, tensor<3x3xf64>, tensor<i64>) -> tensor<3x3xf64>
    %14 = stablehlo.bitcast_convert %arg0 : (tensor<2xui64>) -> tensor<2x2xui32>
    %15 = stablehlo.reshape %14 : (tensor<2x2xui32>) -> tensor<4xui32>
    %16 = stablehlo.slice %15 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %17 = stablehlo.slice %15 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %18 = stablehlo.slice %15 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %19 = stablehlo.slice %15 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %20 = stablehlo.slice %15 [0:2] : (tensor<4xui32>) -> tensor<2xui32>
    %21 = stablehlo.reduce(%20 init: %c_9) applies stablehlo.add across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %22 = stablehlo.reshape %21 : (tensor<ui32>) -> tensor<1xui32>
    %23 = stablehlo.shift_left %17, %c_17 : tensor<1xui32>
    %24 = stablehlo.shift_right_logical %17, %c_16 : tensor<1xui32>
    %25 = stablehlo.or %23, %24 : tensor<1xui32>
    %26 = stablehlo.xor %22, %25 : tensor<1xui32>
    %27 = stablehlo.add %22, %26 : tensor<1xui32>
    %28 = stablehlo.shift_left %26, %c_19 : tensor<1xui32>
    %29 = stablehlo.shift_right_logical %26, %c_18 : tensor<1xui32>
    %30 = stablehlo.or %28, %29 : tensor<1xui32>
    %31 = stablehlo.xor %27, %30 : tensor<1xui32>
    %32 = stablehlo.add %27, %31 : tensor<1xui32>
    %33 = stablehlo.shift_left %31, %c_20 : tensor<1xui32>
    %34 = stablehlo.shift_right_logical %31, %c_21 : tensor<1xui32>
    %35 = stablehlo.or %33, %34 : tensor<1xui32>
    %36 = stablehlo.xor %32, %35 : tensor<1xui32>
    %37 = stablehlo.add %32, %36 : tensor<1xui32>
    %38 = stablehlo.shift_left %36, %c_21 : tensor<1xui32>
    %39 = stablehlo.shift_right_logical %36, %c_20 : tensor<1xui32>
    %40 = stablehlo.or %38, %39 : tensor<1xui32>
    %41 = stablehlo.xor %37, %40 : tensor<1xui32>
    %42 = stablehlo.reduce(%20 init: %c_9) applies stablehlo.xor across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %43 = stablehlo.reshape %42 : (tensor<ui32>) -> tensor<1xui32>
    %44 = stablehlo.xor %43, %c_6 : tensor<1xui32>
    %45 = stablehlo.add %41, %44 : tensor<1xui32>
    %46 = stablehlo.add %45, %c_7 : tensor<1xui32>
    %47 = stablehlo.add %22, %c_7 : tensor<1xui32>
    %48 = stablehlo.add %47, %44 : tensor<1xui32>
    %49 = stablehlo.add %48, %41 : tensor<1xui32>
    %50 = stablehlo.add %49, %36 : tensor<1xui32>
    %51 = stablehlo.add %50, %31 : tensor<1xui32>
    %52 = stablehlo.add %51, %26 : tensor<1xui32>
    %53 = stablehlo.add %52, %17 : tensor<1xui32>
    %54 = stablehlo.shift_left %46, %c_18 : tensor<1xui32>
    %55 = stablehlo.shift_right_logical %46, %c_19 : tensor<1xui32>
    %56 = stablehlo.or %54, %55 : tensor<1xui32>
    %57 = stablehlo.xor %53, %56 : tensor<1xui32>
    %58 = stablehlo.add %22, %57 : tensor<1xui32>
    %59 = stablehlo.add %58, %c_7 : tensor<1xui32>
    %60 = stablehlo.add %59, %44 : tensor<1xui32>
    %61 = stablehlo.add %60, %41 : tensor<1xui32>
    %62 = stablehlo.add %61, %36 : tensor<1xui32>
    %63 = stablehlo.add %62, %31 : tensor<1xui32>
    %64 = stablehlo.add %63, %26 : tensor<1xui32>
    %65 = stablehlo.add %64, %17 : tensor<1xui32>
    %66 = stablehlo.shift_left %57, %c_11 : tensor<1xui32>
    %67 = stablehlo.shift_right_logical %57, %c_10 : tensor<1xui32>
    %68 = stablehlo.or %66, %67 : tensor<1xui32>
    %69 = stablehlo.xor %65, %68 : tensor<1xui32>
    %70 = stablehlo.add %22, %69 : tensor<1xui32>
    %71 = stablehlo.add %70, %57 : tensor<1xui32>
    %72 = stablehlo.add %71, %c_7 : tensor<1xui32>
    %73 = stablehlo.add %72, %44 : tensor<1xui32>
    %74 = stablehlo.add %73, %41 : tensor<1xui32>
    %75 = stablehlo.add %74, %36 : tensor<1xui32>
    %76 = stablehlo.add %75, %31 : tensor<1xui32>
    %77 = stablehlo.add %76, %26 : tensor<1xui32>
    %78 = stablehlo.add %77, %17 : tensor<1xui32>
    %79 = stablehlo.shift_left %69, %c_12 : tensor<1xui32>
    %80 = stablehlo.shift_right_logical %69, %c_12 : tensor<1xui32>
    %81 = stablehlo.or %79, %80 : tensor<1xui32>
    %82 = stablehlo.xor %78, %81 : tensor<1xui32>
    %83 = stablehlo.add %22, %82 : tensor<1xui32>
    %84 = stablehlo.add %83, %69 : tensor<1xui32>
    %85 = stablehlo.add %84, %57 : tensor<1xui32>
    %86 = stablehlo.add %85, %c_7 : tensor<1xui32>
    %87 = stablehlo.add %86, %44 : tensor<1xui32>
    %88 = stablehlo.add %87, %41 : tensor<1xui32>
    %89 = stablehlo.add %88, %36 : tensor<1xui32>
    %90 = stablehlo.add %89, %31 : tensor<1xui32>
    %91 = stablehlo.add %90, %26 : tensor<1xui32>
    %92 = stablehlo.add %91, %17 : tensor<1xui32>
    %93 = stablehlo.shift_left %82, %c_14 : tensor<1xui32>
    %94 = stablehlo.shift_right_logical %82, %c_13 : tensor<1xui32>
    %95 = stablehlo.or %93, %94 : tensor<1xui32>
    %96 = stablehlo.xor %92, %95 : tensor<1xui32>
    %97 = stablehlo.add %96, %16 : tensor<1xui32>
    %98 = stablehlo.add %97, %c_8 : tensor<1xui32>
    %99 = stablehlo.add %22, %c_8 : tensor<1xui32>
    %100 = stablehlo.add %99, %96 : tensor<1xui32>
    %101 = stablehlo.add %100, %44 : tensor<1xui32>
    %102 = stablehlo.add %101, %82 : tensor<1xui32>
    %103 = stablehlo.add %102, %69 : tensor<1xui32>
    %104 = stablehlo.add %103, %57 : tensor<1xui32>
    %105 = stablehlo.add %104, %46 : tensor<1xui32>
    %106 = stablehlo.add %105, %37 : tensor<1xui32>
    %107 = stablehlo.shift_left %98, %c_17 : tensor<1xui32>
    %108 = stablehlo.shift_right_logical %98, %c_16 : tensor<1xui32>
    %109 = stablehlo.or %107, %108 : tensor<1xui32>
    %110 = stablehlo.xor %106, %109 : tensor<1xui32>
    %111 = stablehlo.add %22, %110 : tensor<1xui32>
    %112 = stablehlo.add %111, %c_8 : tensor<1xui32>
    %113 = stablehlo.add %112, %96 : tensor<1xui32>
    %114 = stablehlo.add %113, %44 : tensor<1xui32>
    %115 = stablehlo.add %114, %82 : tensor<1xui32>
    %116 = stablehlo.add %115, %69 : tensor<1xui32>
    %117 = stablehlo.add %116, %57 : tensor<1xui32>
    %118 = stablehlo.add %117, %46 : tensor<1xui32>
    %119 = stablehlo.add %118, %37 : tensor<1xui32>
    %120 = stablehlo.shift_left %110, %c_19 : tensor<1xui32>
    %121 = stablehlo.shift_right_logical %110, %c_18 : tensor<1xui32>
    %122 = stablehlo.or %120, %121 : tensor<1xui32>
    %123 = stablehlo.xor %119, %122 : tensor<1xui32>
    %124 = stablehlo.add %22, %123 : tensor<1xui32>
    %125 = stablehlo.add %124, %110 : tensor<1xui32>
    %126 = stablehlo.add %125, %c_8 : tensor<1xui32>
    %127 = stablehlo.add %126, %96 : tensor<1xui32>
    %128 = stablehlo.add %127, %44 : tensor<1xui32>
    %129 = stablehlo.add %128, %82 : tensor<1xui32>
    %130 = stablehlo.add %129, %69 : tensor<1xui32>
    %131 = stablehlo.add %130, %57 : tensor<1xui32>
    %132 = stablehlo.add %131, %46 : tensor<1xui32>
    %133 = stablehlo.add %132, %37 : tensor<1xui32>
    %134 = stablehlo.shift_left %123, %c_20 : tensor<1xui32>
    %135 = stablehlo.shift_right_logical %123, %c_21 : tensor<1xui32>
    %136 = stablehlo.or %134, %135 : tensor<1xui32>
    %137 = stablehlo.xor %133, %136 : tensor<1xui32>
    %138 = stablehlo.add %22, %137 : tensor<1xui32>
    %139 = stablehlo.add %138, %123 : tensor<1xui32>
    %140 = stablehlo.add %139, %110 : tensor<1xui32>
    %141 = stablehlo.add %140, %c_8 : tensor<1xui32>
    %142 = stablehlo.add %141, %96 : tensor<1xui32>
    %143 = stablehlo.add %142, %44 : tensor<1xui32>
    %144 = stablehlo.add %143, %82 : tensor<1xui32>
    %145 = stablehlo.add %144, %69 : tensor<1xui32>
    %146 = stablehlo.add %145, %57 : tensor<1xui32>
    %147 = stablehlo.add %146, %46 : tensor<1xui32>
    %148 = stablehlo.add %147, %37 : tensor<1xui32>
    %149 = stablehlo.shift_left %137, %c_21 : tensor<1xui32>
    %150 = stablehlo.shift_right_logical %137, %c_20 : tensor<1xui32>
    %151 = stablehlo.or %149, %150 : tensor<1xui32>
    %152 = stablehlo.xor %148, %151 : tensor<1xui32>
    %153 = stablehlo.add %152, %17 : tensor<1xui32>
    %154 = stablehlo.add %153, %c_10 : tensor<1xui32>
    %155 = stablehlo.add %22, %c_10 : tensor<1xui32>
    %156 = stablehlo.add %155, %152 : tensor<1xui32>
    %157 = stablehlo.add %156, %148 : tensor<1xui32>
    %158 = stablehlo.shift_left %154, %c_18 : tensor<1xui32>
    %159 = stablehlo.shift_right_logical %154, %c_19 : tensor<1xui32>
    %160 = stablehlo.or %158, %159 : tensor<1xui32>
    %161 = stablehlo.xor %157, %160 : tensor<1xui32>
    %162 = stablehlo.add %22, %161 : tensor<1xui32>
    %163 = stablehlo.add %162, %c_10 : tensor<1xui32>
    %164 = stablehlo.add %163, %152 : tensor<1xui32>
    %165 = stablehlo.add %164, %148 : tensor<1xui32>
    %166 = stablehlo.shift_left %161, %c_11 : tensor<1xui32>
    %167 = stablehlo.shift_right_logical %161, %c_10 : tensor<1xui32>
    %168 = stablehlo.or %166, %167 : tensor<1xui32>
    %169 = stablehlo.xor %165, %168 : tensor<1xui32>
    %170 = stablehlo.add %22, %169 : tensor<1xui32>
    %171 = stablehlo.add %170, %161 : tensor<1xui32>
    %172 = stablehlo.add %171, %c_10 : tensor<1xui32>
    %173 = stablehlo.add %172, %152 : tensor<1xui32>
    %174 = stablehlo.add %173, %148 : tensor<1xui32>
    %175 = stablehlo.shift_left %169, %c_12 : tensor<1xui32>
    %176 = stablehlo.shift_right_logical %169, %c_12 : tensor<1xui32>
    %177 = stablehlo.or %175, %176 : tensor<1xui32>
    %178 = stablehlo.xor %174, %177 : tensor<1xui32>
    %179 = stablehlo.add %22, %178 : tensor<1xui32>
    %180 = stablehlo.add %179, %169 : tensor<1xui32>
    %181 = stablehlo.add %180, %161 : tensor<1xui32>
    %182 = stablehlo.add %181, %c_10 : tensor<1xui32>
    %183 = stablehlo.add %182, %152 : tensor<1xui32>
    %184 = stablehlo.add %183, %148 : tensor<1xui32>
    %185 = stablehlo.add %181, %154 : tensor<1xui32>
    %186 = stablehlo.add %185, %148 : tensor<1xui32>
    %187 = stablehlo.shift_left %178, %c_14 : tensor<1xui32>
    %188 = stablehlo.shift_right_logical %178, %c_13 : tensor<1xui32>
    %189 = stablehlo.or %187, %188 : tensor<1xui32>
    %190 = stablehlo.xor %184, %189 : tensor<1xui32>
    %191 = stablehlo.add %190, %44 : tensor<1xui32>
    %192 = stablehlo.add %191, %c_15 : tensor<1xui32>
    %193 = stablehlo.add %186, %192 : tensor<1xui32>
    %194 = stablehlo.shift_left %192, %c_17 : tensor<1xui32>
    %195 = stablehlo.shift_right_logical %192, %c_16 : tensor<1xui32>
    %196 = stablehlo.or %194, %195 : tensor<1xui32>
    %197 = stablehlo.xor %193, %196 : tensor<1xui32>
    %198 = stablehlo.add %193, %197 : tensor<1xui32>
    %199 = stablehlo.shift_left %197, %c_19 : tensor<1xui32>
    %200 = stablehlo.shift_right_logical %197, %c_18 : tensor<1xui32>
    %201 = stablehlo.or %199, %200 : tensor<1xui32>
    %202 = stablehlo.xor %198, %201 : tensor<1xui32>
    %203 = stablehlo.add %198, %202 : tensor<1xui32>
    %204 = stablehlo.shift_left %202, %c_20 : tensor<1xui32>
    %205 = stablehlo.shift_right_logical %202, %c_21 : tensor<1xui32>
    %206 = stablehlo.or %204, %205 : tensor<1xui32>
    %207 = stablehlo.xor %203, %206 : tensor<1xui32>
    %208 = stablehlo.add %203, %207 : tensor<1xui32>
    %209 = stablehlo.shift_left %207, %c_21 : tensor<1xui32>
    %210 = stablehlo.shift_right_logical %207, %c_20 : tensor<1xui32>
    %211 = stablehlo.or %209, %210 : tensor<1xui32>
    %212 = stablehlo.xor %208, %211 : tensor<1xui32>
    %213 = stablehlo.add %212, %16 : tensor<1xui32>
    %214 = stablehlo.slice %15 [2:4] : (tensor<4xui32>) -> tensor<2xui32>
    %215 = stablehlo.reduce(%214 init: %c_9) applies stablehlo.add across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %216 = stablehlo.reshape %215 : (tensor<ui32>) -> tensor<1xui32>
    %217 = stablehlo.shift_left %19, %c_17 : tensor<1xui32>
    %218 = stablehlo.shift_right_logical %19, %c_16 : tensor<1xui32>
    %219 = stablehlo.or %217, %218 : tensor<1xui32>
    %220 = stablehlo.xor %216, %219 : tensor<1xui32>
    %221 = stablehlo.add %216, %220 : tensor<1xui32>
    %222 = stablehlo.shift_left %220, %c_19 : tensor<1xui32>
    %223 = stablehlo.shift_right_logical %220, %c_18 : tensor<1xui32>
    %224 = stablehlo.or %222, %223 : tensor<1xui32>
    %225 = stablehlo.xor %221, %224 : tensor<1xui32>
    %226 = stablehlo.add %221, %225 : tensor<1xui32>
    %227 = stablehlo.shift_left %225, %c_20 : tensor<1xui32>
    %228 = stablehlo.shift_right_logical %225, %c_21 : tensor<1xui32>
    %229 = stablehlo.or %227, %228 : tensor<1xui32>
    %230 = stablehlo.xor %226, %229 : tensor<1xui32>
    %231 = stablehlo.add %226, %230 : tensor<1xui32>
    %232 = stablehlo.shift_left %230, %c_21 : tensor<1xui32>
    %233 = stablehlo.shift_right_logical %230, %c_20 : tensor<1xui32>
    %234 = stablehlo.or %232, %233 : tensor<1xui32>
    %235 = stablehlo.xor %231, %234 : tensor<1xui32>
    %236 = stablehlo.reduce(%214 init: %c_9) applies stablehlo.xor across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %237 = stablehlo.reshape %236 : (tensor<ui32>) -> tensor<1xui32>
    %238 = stablehlo.xor %237, %c_6 : tensor<1xui32>
    %239 = stablehlo.add %235, %238 : tensor<1xui32>
    %240 = stablehlo.add %239, %c_7 : tensor<1xui32>
    %241 = stablehlo.add %216, %c_7 : tensor<1xui32>
    %242 = stablehlo.add %241, %238 : tensor<1xui32>
    %243 = stablehlo.add %242, %235 : tensor<1xui32>
    %244 = stablehlo.add %243, %230 : tensor<1xui32>
    %245 = stablehlo.add %244, %225 : tensor<1xui32>
    %246 = stablehlo.add %245, %220 : tensor<1xui32>
    %247 = stablehlo.add %246, %19 : tensor<1xui32>
    %248 = stablehlo.shift_left %240, %c_18 : tensor<1xui32>
    %249 = stablehlo.shift_right_logical %240, %c_19 : tensor<1xui32>
    %250 = stablehlo.or %248, %249 : tensor<1xui32>
    %251 = stablehlo.xor %247, %250 : tensor<1xui32>
    %252 = stablehlo.add %216, %251 : tensor<1xui32>
    %253 = stablehlo.add %252, %c_7 : tensor<1xui32>
    %254 = stablehlo.add %253, %238 : tensor<1xui32>
    %255 = stablehlo.add %254, %235 : tensor<1xui32>
    %256 = stablehlo.add %255, %230 : tensor<1xui32>
    %257 = stablehlo.add %256, %225 : tensor<1xui32>
    %258 = stablehlo.add %257, %220 : tensor<1xui32>
    %259 = stablehlo.add %258, %19 : tensor<1xui32>
    %260 = stablehlo.shift_left %251, %c_11 : tensor<1xui32>
    %261 = stablehlo.shift_right_logical %251, %c_10 : tensor<1xui32>
    %262 = stablehlo.or %260, %261 : tensor<1xui32>
    %263 = stablehlo.xor %259, %262 : tensor<1xui32>
    %264 = stablehlo.add %216, %263 : tensor<1xui32>
    %265 = stablehlo.add %264, %251 : tensor<1xui32>
    %266 = stablehlo.add %265, %c_7 : tensor<1xui32>
    %267 = stablehlo.add %266, %238 : tensor<1xui32>
    %268 = stablehlo.add %267, %235 : tensor<1xui32>
    %269 = stablehlo.add %268, %230 : tensor<1xui32>
    %270 = stablehlo.add %269, %225 : tensor<1xui32>
    %271 = stablehlo.add %270, %220 : tensor<1xui32>
    %272 = stablehlo.add %271, %19 : tensor<1xui32>
    %273 = stablehlo.shift_left %263, %c_12 : tensor<1xui32>
    %274 = stablehlo.shift_right_logical %263, %c_12 : tensor<1xui32>
    %275 = stablehlo.or %273, %274 : tensor<1xui32>
    %276 = stablehlo.xor %272, %275 : tensor<1xui32>
    %277 = stablehlo.add %216, %276 : tensor<1xui32>
    %278 = stablehlo.add %277, %263 : tensor<1xui32>
    %279 = stablehlo.add %278, %251 : tensor<1xui32>
    %280 = stablehlo.add %279, %c_7 : tensor<1xui32>
    %281 = stablehlo.add %280, %238 : tensor<1xui32>
    %282 = stablehlo.add %281, %235 : tensor<1xui32>
    %283 = stablehlo.add %282, %230 : tensor<1xui32>
    %284 = stablehlo.add %283, %225 : tensor<1xui32>
    %285 = stablehlo.add %284, %220 : tensor<1xui32>
    %286 = stablehlo.add %285, %19 : tensor<1xui32>
    %287 = stablehlo.shift_left %276, %c_14 : tensor<1xui32>
    %288 = stablehlo.shift_right_logical %276, %c_13 : tensor<1xui32>
    %289 = stablehlo.or %287, %288 : tensor<1xui32>
    %290 = stablehlo.xor %286, %289 : tensor<1xui32>
    %291 = stablehlo.add %290, %18 : tensor<1xui32>
    %292 = stablehlo.add %291, %c_8 : tensor<1xui32>
    %293 = stablehlo.add %216, %c_8 : tensor<1xui32>
    %294 = stablehlo.add %293, %290 : tensor<1xui32>
    %295 = stablehlo.add %294, %238 : tensor<1xui32>
    %296 = stablehlo.add %295, %276 : tensor<1xui32>
    %297 = stablehlo.add %296, %263 : tensor<1xui32>
    %298 = stablehlo.add %297, %251 : tensor<1xui32>
    %299 = stablehlo.add %298, %240 : tensor<1xui32>
    %300 = stablehlo.add %299, %231 : tensor<1xui32>
    %301 = stablehlo.shift_left %292, %c_17 : tensor<1xui32>
    %302 = stablehlo.shift_right_logical %292, %c_16 : tensor<1xui32>
    %303 = stablehlo.or %301, %302 : tensor<1xui32>
    %304 = stablehlo.xor %300, %303 : tensor<1xui32>
    %305 = stablehlo.add %216, %304 : tensor<1xui32>
    %306 = stablehlo.add %305, %c_8 : tensor<1xui32>
    %307 = stablehlo.add %306, %290 : tensor<1xui32>
    %308 = stablehlo.add %307, %238 : tensor<1xui32>
    %309 = stablehlo.add %308, %276 : tensor<1xui32>
    %310 = stablehlo.add %309, %263 : tensor<1xui32>
    %311 = stablehlo.add %310, %251 : tensor<1xui32>
    %312 = stablehlo.add %311, %240 : tensor<1xui32>
    %313 = stablehlo.add %312, %231 : tensor<1xui32>
    %314 = stablehlo.shift_left %304, %c_19 : tensor<1xui32>
    %315 = stablehlo.shift_right_logical %304, %c_18 : tensor<1xui32>
    %316 = stablehlo.or %314, %315 : tensor<1xui32>
    %317 = stablehlo.xor %313, %316 : tensor<1xui32>
    %318 = stablehlo.add %216, %317 : tensor<1xui32>
    %319 = stablehlo.add %318, %304 : tensor<1xui32>
    %320 = stablehlo.add %319, %c_8 : tensor<1xui32>
    %321 = stablehlo.add %320, %290 : tensor<1xui32>
    %322 = stablehlo.add %321, %238 : tensor<1xui32>
    %323 = stablehlo.add %322, %276 : tensor<1xui32>
    %324 = stablehlo.add %323, %263 : tensor<1xui32>
    %325 = stablehlo.add %324, %251 : tensor<1xui32>
    %326 = stablehlo.add %325, %240 : tensor<1xui32>
    %327 = stablehlo.add %326, %231 : tensor<1xui32>
    %328 = stablehlo.shift_left %317, %c_20 : tensor<1xui32>
    %329 = stablehlo.shift_right_logical %317, %c_21 : tensor<1xui32>
    %330 = stablehlo.or %328, %329 : tensor<1xui32>
    %331 = stablehlo.xor %327, %330 : tensor<1xui32>
    %332 = stablehlo.add %216, %331 : tensor<1xui32>
    %333 = stablehlo.add %332, %317 : tensor<1xui32>
    %334 = stablehlo.add %333, %304 : tensor<1xui32>
    %335 = stablehlo.add %334, %c_8 : tensor<1xui32>
    %336 = stablehlo.add %335, %290 : tensor<1xui32>
    %337 = stablehlo.add %336, %238 : tensor<1xui32>
    %338 = stablehlo.add %337, %276 : tensor<1xui32>
    %339 = stablehlo.add %338, %263 : tensor<1xui32>
    %340 = stablehlo.add %339, %251 : tensor<1xui32>
    %341 = stablehlo.add %340, %240 : tensor<1xui32>
    %342 = stablehlo.add %341, %231 : tensor<1xui32>
    %343 = stablehlo.shift_left %331, %c_21 : tensor<1xui32>
    %344 = stablehlo.shift_right_logical %331, %c_20 : tensor<1xui32>
    %345 = stablehlo.or %343, %344 : tensor<1xui32>
    %346 = stablehlo.xor %342, %345 : tensor<1xui32>
    %347 = stablehlo.add %346, %19 : tensor<1xui32>
    %348 = stablehlo.add %347, %c_10 : tensor<1xui32>
    %349 = stablehlo.add %216, %c_10 : tensor<1xui32>
    %350 = stablehlo.add %349, %346 : tensor<1xui32>
    %351 = stablehlo.add %350, %342 : tensor<1xui32>
    %352 = stablehlo.shift_left %348, %c_18 : tensor<1xui32>
    %353 = stablehlo.shift_right_logical %348, %c_19 : tensor<1xui32>
    %354 = stablehlo.or %352, %353 : tensor<1xui32>
    %355 = stablehlo.xor %351, %354 : tensor<1xui32>
    %356 = stablehlo.add %216, %355 : tensor<1xui32>
    %357 = stablehlo.add %356, %c_10 : tensor<1xui32>
    %358 = stablehlo.add %357, %346 : tensor<1xui32>
    %359 = stablehlo.add %358, %342 : tensor<1xui32>
    %360 = stablehlo.shift_left %355, %c_11 : tensor<1xui32>
    %361 = stablehlo.shift_right_logical %355, %c_10 : tensor<1xui32>
    %362 = stablehlo.or %360, %361 : tensor<1xui32>
    %363 = stablehlo.xor %359, %362 : tensor<1xui32>
    %364 = stablehlo.add %216, %363 : tensor<1xui32>
    %365 = stablehlo.add %364, %355 : tensor<1xui32>
    %366 = stablehlo.add %365, %c_10 : tensor<1xui32>
    %367 = stablehlo.add %366, %346 : tensor<1xui32>
    %368 = stablehlo.add %367, %342 : tensor<1xui32>
    %369 = stablehlo.shift_left %363, %c_12 : tensor<1xui32>
    %370 = stablehlo.shift_right_logical %363, %c_12 : tensor<1xui32>
    %371 = stablehlo.or %369, %370 : tensor<1xui32>
    %372 = stablehlo.xor %368, %371 : tensor<1xui32>
    %373 = stablehlo.add %216, %372 : tensor<1xui32>
    %374 = stablehlo.add %373, %363 : tensor<1xui32>
    %375 = stablehlo.add %374, %355 : tensor<1xui32>
    %376 = stablehlo.add %375, %c_10 : tensor<1xui32>
    %377 = stablehlo.add %376, %346 : tensor<1xui32>
    %378 = stablehlo.add %377, %342 : tensor<1xui32>
    %379 = stablehlo.add %375, %348 : tensor<1xui32>
    %380 = stablehlo.add %379, %342 : tensor<1xui32>
    %381 = stablehlo.shift_left %372, %c_14 : tensor<1xui32>
    %382 = stablehlo.shift_right_logical %372, %c_13 : tensor<1xui32>
    %383 = stablehlo.or %381, %382 : tensor<1xui32>
    %384 = stablehlo.xor %378, %383 : tensor<1xui32>
    %385 = stablehlo.add %384, %238 : tensor<1xui32>
    %386 = stablehlo.add %385, %c_15 : tensor<1xui32>
    %387 = stablehlo.add %380, %386 : tensor<1xui32>
    %388 = stablehlo.shift_left %386, %c_17 : tensor<1xui32>
    %389 = stablehlo.shift_right_logical %386, %c_16 : tensor<1xui32>
    %390 = stablehlo.or %388, %389 : tensor<1xui32>
    %391 = stablehlo.xor %387, %390 : tensor<1xui32>
    %392 = stablehlo.add %387, %391 : tensor<1xui32>
    %393 = stablehlo.shift_left %391, %c_19 : tensor<1xui32>
    %394 = stablehlo.shift_right_logical %391, %c_18 : tensor<1xui32>
    %395 = stablehlo.or %393, %394 : tensor<1xui32>
    %396 = stablehlo.xor %392, %395 : tensor<1xui32>
    %397 = stablehlo.add %392, %396 : tensor<1xui32>
    %398 = stablehlo.shift_left %396, %c_20 : tensor<1xui32>
    %399 = stablehlo.shift_right_logical %396, %c_21 : tensor<1xui32>
    %400 = stablehlo.or %398, %399 : tensor<1xui32>
    %401 = stablehlo.xor %397, %400 : tensor<1xui32>
    %402 = stablehlo.add %397, %401 : tensor<1xui32>
    %403 = stablehlo.shift_left %401, %c_21 : tensor<1xui32>
    %404 = stablehlo.shift_right_logical %401, %c_20 : tensor<1xui32>
    %405 = stablehlo.or %403, %404 : tensor<1xui32>
    %406 = stablehlo.xor %402, %405 : tensor<1xui32>
    %407 = stablehlo.add %406, %18 : tensor<1xui32>
    %408 = stablehlo.concatenate %208, %213, %402, %407, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
    %409 = stablehlo.concatenate %44, %c_34, %238, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<3xui32>
    %410 = stablehlo.pad %409, %c_33, low = [0], high = [1], interior = [0] : (tensor<3xui32>, tensor<ui32>) -> tensor<4xui32>
    %411 = stablehlo.add %408, %410 : tensor<4xui32>
    %412 = stablehlo.slice %411 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %413 = stablehlo.slice %411 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %414 = stablehlo.slice %411 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %415 = stablehlo.slice %411 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %416 = stablehlo.slice %411 [0:2] : (tensor<4xui32>) -> tensor<2xui32>
    %417 = stablehlo.reduce(%416 init: %c_9) applies stablehlo.add across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %418 = stablehlo.reshape %417 : (tensor<ui32>) -> tensor<1xui32>
    %419 = stablehlo.shift_left %413, %c_17 : tensor<1xui32>
    %420 = stablehlo.shift_right_logical %413, %c_16 : tensor<1xui32>
    %421 = stablehlo.or %419, %420 : tensor<1xui32>
    %422 = stablehlo.xor %418, %421 : tensor<1xui32>
    %423 = stablehlo.add %418, %422 : tensor<1xui32>
    %424 = stablehlo.shift_left %422, %c_19 : tensor<1xui32>
    %425 = stablehlo.shift_right_logical %422, %c_18 : tensor<1xui32>
    %426 = stablehlo.or %424, %425 : tensor<1xui32>
    %427 = stablehlo.xor %423, %426 : tensor<1xui32>
    %428 = stablehlo.add %423, %427 : tensor<1xui32>
    %429 = stablehlo.shift_left %427, %c_20 : tensor<1xui32>
    %430 = stablehlo.shift_right_logical %427, %c_21 : tensor<1xui32>
    %431 = stablehlo.or %429, %430 : tensor<1xui32>
    %432 = stablehlo.xor %428, %431 : tensor<1xui32>
    %433 = stablehlo.add %428, %432 : tensor<1xui32>
    %434 = stablehlo.shift_left %432, %c_21 : tensor<1xui32>
    %435 = stablehlo.shift_right_logical %432, %c_20 : tensor<1xui32>
    %436 = stablehlo.or %434, %435 : tensor<1xui32>
    %437 = stablehlo.xor %433, %436 : tensor<1xui32>
    %438 = stablehlo.reduce(%416 init: %c_9) applies stablehlo.xor across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %439 = stablehlo.reshape %438 : (tensor<ui32>) -> tensor<1xui32>
    %440 = stablehlo.xor %439, %c_6 : tensor<1xui32>
    %441 = stablehlo.add %437, %440 : tensor<1xui32>
    %442 = stablehlo.add %441, %c_7 : tensor<1xui32>
    %443 = stablehlo.add %418, %c_7 : tensor<1xui32>
    %444 = stablehlo.add %443, %440 : tensor<1xui32>
    %445 = stablehlo.add %444, %437 : tensor<1xui32>
    %446 = stablehlo.add %445, %432 : tensor<1xui32>
    %447 = stablehlo.add %446, %427 : tensor<1xui32>
    %448 = stablehlo.add %447, %422 : tensor<1xui32>
    %449 = stablehlo.add %448, %413 : tensor<1xui32>
    %450 = stablehlo.shift_left %442, %c_18 : tensor<1xui32>
    %451 = stablehlo.shift_right_logical %442, %c_19 : tensor<1xui32>
    %452 = stablehlo.or %450, %451 : tensor<1xui32>
    %453 = stablehlo.xor %449, %452 : tensor<1xui32>
    %454 = stablehlo.add %418, %453 : tensor<1xui32>
    %455 = stablehlo.add %454, %c_7 : tensor<1xui32>
    %456 = stablehlo.add %455, %440 : tensor<1xui32>
    %457 = stablehlo.add %456, %437 : tensor<1xui32>
    %458 = stablehlo.add %457, %432 : tensor<1xui32>
    %459 = stablehlo.add %458, %427 : tensor<1xui32>
    %460 = stablehlo.add %459, %422 : tensor<1xui32>
    %461 = stablehlo.add %460, %413 : tensor<1xui32>
    %462 = stablehlo.shift_left %453, %c_11 : tensor<1xui32>
    %463 = stablehlo.shift_right_logical %453, %c_10 : tensor<1xui32>
    %464 = stablehlo.or %462, %463 : tensor<1xui32>
    %465 = stablehlo.xor %461, %464 : tensor<1xui32>
    %466 = stablehlo.add %418, %465 : tensor<1xui32>
    %467 = stablehlo.add %466, %453 : tensor<1xui32>
    %468 = stablehlo.add %467, %c_7 : tensor<1xui32>
    %469 = stablehlo.add %468, %440 : tensor<1xui32>
    %470 = stablehlo.add %469, %437 : tensor<1xui32>
    %471 = stablehlo.add %470, %432 : tensor<1xui32>
    %472 = stablehlo.add %471, %427 : tensor<1xui32>
    %473 = stablehlo.add %472, %422 : tensor<1xui32>
    %474 = stablehlo.add %473, %413 : tensor<1xui32>
    %475 = stablehlo.shift_left %465, %c_12 : tensor<1xui32>
    %476 = stablehlo.shift_right_logical %465, %c_12 : tensor<1xui32>
    %477 = stablehlo.or %475, %476 : tensor<1xui32>
    %478 = stablehlo.xor %474, %477 : tensor<1xui32>
    %479 = stablehlo.add %418, %478 : tensor<1xui32>
    %480 = stablehlo.add %479, %465 : tensor<1xui32>
    %481 = stablehlo.add %480, %453 : tensor<1xui32>
    %482 = stablehlo.add %481, %c_7 : tensor<1xui32>
    %483 = stablehlo.add %482, %440 : tensor<1xui32>
    %484 = stablehlo.add %483, %437 : tensor<1xui32>
    %485 = stablehlo.add %484, %432 : tensor<1xui32>
    %486 = stablehlo.add %485, %427 : tensor<1xui32>
    %487 = stablehlo.add %486, %422 : tensor<1xui32>
    %488 = stablehlo.add %487, %413 : tensor<1xui32>
    %489 = stablehlo.shift_left %478, %c_14 : tensor<1xui32>
    %490 = stablehlo.shift_right_logical %478, %c_13 : tensor<1xui32>
    %491 = stablehlo.or %489, %490 : tensor<1xui32>
    %492 = stablehlo.xor %488, %491 : tensor<1xui32>
    %493 = stablehlo.add %492, %412 : tensor<1xui32>
    %494 = stablehlo.add %493, %c_8 : tensor<1xui32>
    %495 = stablehlo.add %418, %c_8 : tensor<1xui32>
    %496 = stablehlo.add %495, %492 : tensor<1xui32>
    %497 = stablehlo.add %496, %440 : tensor<1xui32>
    %498 = stablehlo.add %497, %478 : tensor<1xui32>
    %499 = stablehlo.add %498, %465 : tensor<1xui32>
    %500 = stablehlo.add %499, %453 : tensor<1xui32>
    %501 = stablehlo.add %500, %442 : tensor<1xui32>
    %502 = stablehlo.add %501, %433 : tensor<1xui32>
    %503 = stablehlo.shift_left %494, %c_17 : tensor<1xui32>
    %504 = stablehlo.shift_right_logical %494, %c_16 : tensor<1xui32>
    %505 = stablehlo.or %503, %504 : tensor<1xui32>
    %506 = stablehlo.xor %502, %505 : tensor<1xui32>
    %507 = stablehlo.add %418, %506 : tensor<1xui32>
    %508 = stablehlo.add %507, %c_8 : tensor<1xui32>
    %509 = stablehlo.add %508, %492 : tensor<1xui32>
    %510 = stablehlo.add %509, %440 : tensor<1xui32>
    %511 = stablehlo.add %510, %478 : tensor<1xui32>
    %512 = stablehlo.add %511, %465 : tensor<1xui32>
    %513 = stablehlo.add %512, %453 : tensor<1xui32>
    %514 = stablehlo.add %513, %442 : tensor<1xui32>
    %515 = stablehlo.add %514, %433 : tensor<1xui32>
    %516 = stablehlo.shift_left %506, %c_19 : tensor<1xui32>
    %517 = stablehlo.shift_right_logical %506, %c_18 : tensor<1xui32>
    %518 = stablehlo.or %516, %517 : tensor<1xui32>
    %519 = stablehlo.xor %515, %518 : tensor<1xui32>
    %520 = stablehlo.add %418, %519 : tensor<1xui32>
    %521 = stablehlo.add %520, %506 : tensor<1xui32>
    %522 = stablehlo.add %521, %c_8 : tensor<1xui32>
    %523 = stablehlo.add %522, %492 : tensor<1xui32>
    %524 = stablehlo.add %523, %440 : tensor<1xui32>
    %525 = stablehlo.add %524, %478 : tensor<1xui32>
    %526 = stablehlo.add %525, %465 : tensor<1xui32>
    %527 = stablehlo.add %526, %453 : tensor<1xui32>
    %528 = stablehlo.add %527, %442 : tensor<1xui32>
    %529 = stablehlo.add %528, %433 : tensor<1xui32>
    %530 = stablehlo.shift_left %519, %c_20 : tensor<1xui32>
    %531 = stablehlo.shift_right_logical %519, %c_21 : tensor<1xui32>
    %532 = stablehlo.or %530, %531 : tensor<1xui32>
    %533 = stablehlo.xor %529, %532 : tensor<1xui32>
    %534 = stablehlo.add %418, %533 : tensor<1xui32>
    %535 = stablehlo.add %534, %519 : tensor<1xui32>
    %536 = stablehlo.add %535, %506 : tensor<1xui32>
    %537 = stablehlo.add %536, %c_8 : tensor<1xui32>
    %538 = stablehlo.add %537, %492 : tensor<1xui32>
    %539 = stablehlo.add %538, %440 : tensor<1xui32>
    %540 = stablehlo.add %539, %478 : tensor<1xui32>
    %541 = stablehlo.add %540, %465 : tensor<1xui32>
    %542 = stablehlo.add %541, %453 : tensor<1xui32>
    %543 = stablehlo.add %542, %442 : tensor<1xui32>
    %544 = stablehlo.add %543, %433 : tensor<1xui32>
    %545 = stablehlo.shift_left %533, %c_21 : tensor<1xui32>
    %546 = stablehlo.shift_right_logical %533, %c_20 : tensor<1xui32>
    %547 = stablehlo.or %545, %546 : tensor<1xui32>
    %548 = stablehlo.xor %544, %547 : tensor<1xui32>
    %549 = stablehlo.add %548, %413 : tensor<1xui32>
    %550 = stablehlo.add %549, %c_10 : tensor<1xui32>
    %551 = stablehlo.add %418, %c_10 : tensor<1xui32>
    %552 = stablehlo.add %551, %548 : tensor<1xui32>
    %553 = stablehlo.add %552, %544 : tensor<1xui32>
    %554 = stablehlo.shift_left %550, %c_18 : tensor<1xui32>
    %555 = stablehlo.shift_right_logical %550, %c_19 : tensor<1xui32>
    %556 = stablehlo.or %554, %555 : tensor<1xui32>
    %557 = stablehlo.xor %553, %556 : tensor<1xui32>
    %558 = stablehlo.add %418, %557 : tensor<1xui32>
    %559 = stablehlo.add %558, %c_10 : tensor<1xui32>
    %560 = stablehlo.add %559, %548 : tensor<1xui32>
    %561 = stablehlo.add %560, %544 : tensor<1xui32>
    %562 = stablehlo.shift_left %557, %c_11 : tensor<1xui32>
    %563 = stablehlo.shift_right_logical %557, %c_10 : tensor<1xui32>
    %564 = stablehlo.or %562, %563 : tensor<1xui32>
    %565 = stablehlo.xor %561, %564 : tensor<1xui32>
    %566 = stablehlo.add %418, %565 : tensor<1xui32>
    %567 = stablehlo.add %566, %557 : tensor<1xui32>
    %568 = stablehlo.add %567, %c_10 : tensor<1xui32>
    %569 = stablehlo.add %568, %548 : tensor<1xui32>
    %570 = stablehlo.add %569, %544 : tensor<1xui32>
    %571 = stablehlo.shift_left %565, %c_12 : tensor<1xui32>
    %572 = stablehlo.shift_right_logical %565, %c_12 : tensor<1xui32>
    %573 = stablehlo.or %571, %572 : tensor<1xui32>
    %574 = stablehlo.xor %570, %573 : tensor<1xui32>
    %575 = stablehlo.add %418, %574 : tensor<1xui32>
    %576 = stablehlo.add %575, %565 : tensor<1xui32>
    %577 = stablehlo.add %576, %557 : tensor<1xui32>
    %578 = stablehlo.add %577, %c_10 : tensor<1xui32>
    %579 = stablehlo.add %578, %548 : tensor<1xui32>
    %580 = stablehlo.add %579, %544 : tensor<1xui32>
    %581 = stablehlo.add %577, %550 : tensor<1xui32>
    %582 = stablehlo.add %581, %544 : tensor<1xui32>
    %583 = stablehlo.shift_left %574, %c_14 : tensor<1xui32>
    %584 = stablehlo.shift_right_logical %574, %c_13 : tensor<1xui32>
    %585 = stablehlo.or %583, %584 : tensor<1xui32>
    %586 = stablehlo.xor %580, %585 : tensor<1xui32>
    %587 = stablehlo.add %586, %440 : tensor<1xui32>
    %588 = stablehlo.add %587, %c_15 : tensor<1xui32>
    %589 = stablehlo.add %582, %588 : tensor<1xui32>
    %590 = stablehlo.shift_left %588, %c_17 : tensor<1xui32>
    %591 = stablehlo.shift_right_logical %588, %c_16 : tensor<1xui32>
    %592 = stablehlo.or %590, %591 : tensor<1xui32>
    %593 = stablehlo.xor %589, %592 : tensor<1xui32>
    %594 = stablehlo.add %589, %593 : tensor<1xui32>
    %595 = stablehlo.shift_left %593, %c_19 : tensor<1xui32>
    %596 = stablehlo.shift_right_logical %593, %c_18 : tensor<1xui32>
    %597 = stablehlo.or %595, %596 : tensor<1xui32>
    %598 = stablehlo.xor %594, %597 : tensor<1xui32>
    %599 = stablehlo.add %594, %598 : tensor<1xui32>
    %600 = stablehlo.shift_left %598, %c_20 : tensor<1xui32>
    %601 = stablehlo.shift_right_logical %598, %c_21 : tensor<1xui32>
    %602 = stablehlo.or %600, %601 : tensor<1xui32>
    %603 = stablehlo.xor %599, %602 : tensor<1xui32>
    %604 = stablehlo.add %599, %603 : tensor<1xui32>
    %605 = stablehlo.shift_left %603, %c_21 : tensor<1xui32>
    %606 = stablehlo.shift_right_logical %603, %c_20 : tensor<1xui32>
    %607 = stablehlo.or %605, %606 : tensor<1xui32>
    %608 = stablehlo.xor %604, %607 : tensor<1xui32>
    %609 = stablehlo.add %608, %412 : tensor<1xui32>
    %610 = stablehlo.slice %411 [2:4] : (tensor<4xui32>) -> tensor<2xui32>
    %611 = stablehlo.reduce(%610 init: %c_9) applies stablehlo.add across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %612 = stablehlo.reshape %611 : (tensor<ui32>) -> tensor<1xui32>
    %613 = stablehlo.shift_left %415, %c_17 : tensor<1xui32>
    %614 = stablehlo.shift_right_logical %415, %c_16 : tensor<1xui32>
    %615 = stablehlo.or %613, %614 : tensor<1xui32>
    %616 = stablehlo.xor %612, %615 : tensor<1xui32>
    %617 = stablehlo.add %612, %616 : tensor<1xui32>
    %618 = stablehlo.shift_left %616, %c_19 : tensor<1xui32>
    %619 = stablehlo.shift_right_logical %616, %c_18 : tensor<1xui32>
    %620 = stablehlo.or %618, %619 : tensor<1xui32>
    %621 = stablehlo.xor %617, %620 : tensor<1xui32>
    %622 = stablehlo.add %617, %621 : tensor<1xui32>
    %623 = stablehlo.shift_left %621, %c_20 : tensor<1xui32>
    %624 = stablehlo.shift_right_logical %621, %c_21 : tensor<1xui32>
    %625 = stablehlo.or %623, %624 : tensor<1xui32>
    %626 = stablehlo.xor %622, %625 : tensor<1xui32>
    %627 = stablehlo.add %622, %626 : tensor<1xui32>
    %628 = stablehlo.shift_left %626, %c_21 : tensor<1xui32>
    %629 = stablehlo.shift_right_logical %626, %c_20 : tensor<1xui32>
    %630 = stablehlo.or %628, %629 : tensor<1xui32>
    %631 = stablehlo.xor %627, %630 : tensor<1xui32>
    %632 = stablehlo.reduce(%610 init: %c_9) applies stablehlo.xor across dimensions = [0] : (tensor<2xui32>, tensor<ui32>) -> tensor<ui32>
    %633 = stablehlo.reshape %632 : (tensor<ui32>) -> tensor<1xui32>
    %634 = stablehlo.xor %633, %c_6 : tensor<1xui32>
    %635 = stablehlo.add %631, %634 : tensor<1xui32>
    %636 = stablehlo.add %635, %c_7 : tensor<1xui32>
    %637 = stablehlo.add %612, %c_7 : tensor<1xui32>
    %638 = stablehlo.add %637, %634 : tensor<1xui32>
    %639 = stablehlo.add %638, %631 : tensor<1xui32>
    %640 = stablehlo.add %639, %626 : tensor<1xui32>
    %641 = stablehlo.add %640, %621 : tensor<1xui32>
    %642 = stablehlo.add %641, %616 : tensor<1xui32>
    %643 = stablehlo.add %642, %415 : tensor<1xui32>
    %644 = stablehlo.shift_left %636, %c_18 : tensor<1xui32>
    %645 = stablehlo.shift_right_logical %636, %c_19 : tensor<1xui32>
    %646 = stablehlo.or %644, %645 : tensor<1xui32>
    %647 = stablehlo.xor %643, %646 : tensor<1xui32>
    %648 = stablehlo.add %612, %647 : tensor<1xui32>
    %649 = stablehlo.add %648, %c_7 : tensor<1xui32>
    %650 = stablehlo.add %649, %634 : tensor<1xui32>
    %651 = stablehlo.add %650, %631 : tensor<1xui32>
    %652 = stablehlo.add %651, %626 : tensor<1xui32>
    %653 = stablehlo.add %652, %621 : tensor<1xui32>
    %654 = stablehlo.add %653, %616 : tensor<1xui32>
    %655 = stablehlo.add %654, %415 : tensor<1xui32>
    %656 = stablehlo.shift_left %647, %c_11 : tensor<1xui32>
    %657 = stablehlo.shift_right_logical %647, %c_10 : tensor<1xui32>
    %658 = stablehlo.or %656, %657 : tensor<1xui32>
    %659 = stablehlo.xor %655, %658 : tensor<1xui32>
    %660 = stablehlo.add %612, %659 : tensor<1xui32>
    %661 = stablehlo.add %660, %647 : tensor<1xui32>
    %662 = stablehlo.add %661, %c_7 : tensor<1xui32>
    %663 = stablehlo.add %662, %634 : tensor<1xui32>
    %664 = stablehlo.add %663, %631 : tensor<1xui32>
    %665 = stablehlo.add %664, %626 : tensor<1xui32>
    %666 = stablehlo.add %665, %621 : tensor<1xui32>
    %667 = stablehlo.add %666, %616 : tensor<1xui32>
    %668 = stablehlo.add %667, %415 : tensor<1xui32>
    %669 = stablehlo.shift_left %659, %c_12 : tensor<1xui32>
    %670 = stablehlo.shift_right_logical %659, %c_12 : tensor<1xui32>
    %671 = stablehlo.or %669, %670 : tensor<1xui32>
    %672 = stablehlo.xor %668, %671 : tensor<1xui32>
    %673 = stablehlo.add %612, %672 : tensor<1xui32>
    %674 = stablehlo.add %673, %659 : tensor<1xui32>
    %675 = stablehlo.add %674, %647 : tensor<1xui32>
    %676 = stablehlo.add %675, %c_7 : tensor<1xui32>
    %677 = stablehlo.add %676, %634 : tensor<1xui32>
    %678 = stablehlo.add %677, %631 : tensor<1xui32>
    %679 = stablehlo.add %678, %626 : tensor<1xui32>
    %680 = stablehlo.add %679, %621 : tensor<1xui32>
    %681 = stablehlo.add %680, %616 : tensor<1xui32>
    %682 = stablehlo.add %681, %415 : tensor<1xui32>
    %683 = stablehlo.shift_left %672, %c_14 : tensor<1xui32>
    %684 = stablehlo.shift_right_logical %672, %c_13 : tensor<1xui32>
    %685 = stablehlo.or %683, %684 : tensor<1xui32>
    %686 = stablehlo.xor %682, %685 : tensor<1xui32>
    %687 = stablehlo.add %686, %414 : tensor<1xui32>
    %688 = stablehlo.add %687, %c_8 : tensor<1xui32>
    %689 = stablehlo.add %612, %c_8 : tensor<1xui32>
    %690 = stablehlo.add %689, %686 : tensor<1xui32>
    %691 = stablehlo.add %690, %634 : tensor<1xui32>
    %692 = stablehlo.add %691, %672 : tensor<1xui32>
    %693 = stablehlo.add %692, %659 : tensor<1xui32>
    %694 = stablehlo.add %693, %647 : tensor<1xui32>
    %695 = stablehlo.add %694, %636 : tensor<1xui32>
    %696 = stablehlo.add %695, %627 : tensor<1xui32>
    %697 = stablehlo.shift_left %688, %c_17 : tensor<1xui32>
    %698 = stablehlo.shift_right_logical %688, %c_16 : tensor<1xui32>
    %699 = stablehlo.or %697, %698 : tensor<1xui32>
    %700 = stablehlo.xor %696, %699 : tensor<1xui32>
    %701 = stablehlo.add %612, %700 : tensor<1xui32>
    %702 = stablehlo.add %701, %c_8 : tensor<1xui32>
    %703 = stablehlo.add %702, %686 : tensor<1xui32>
    %704 = stablehlo.add %703, %634 : tensor<1xui32>
    %705 = stablehlo.add %704, %672 : tensor<1xui32>
    %706 = stablehlo.add %705, %659 : tensor<1xui32>
    %707 = stablehlo.add %706, %647 : tensor<1xui32>
    %708 = stablehlo.add %707, %636 : tensor<1xui32>
    %709 = stablehlo.add %708, %627 : tensor<1xui32>
    %710 = stablehlo.shift_left %700, %c_19 : tensor<1xui32>
    %711 = stablehlo.shift_right_logical %700, %c_18 : tensor<1xui32>
    %712 = stablehlo.or %710, %711 : tensor<1xui32>
    %713 = stablehlo.xor %709, %712 : tensor<1xui32>
    %714 = stablehlo.add %612, %713 : tensor<1xui32>
    %715 = stablehlo.add %714, %700 : tensor<1xui32>
    %716 = stablehlo.add %715, %c_8 : tensor<1xui32>
    %717 = stablehlo.add %716, %686 : tensor<1xui32>
    %718 = stablehlo.add %717, %634 : tensor<1xui32>
    %719 = stablehlo.add %718, %672 : tensor<1xui32>
    %720 = stablehlo.add %719, %659 : tensor<1xui32>
    %721 = stablehlo.add %720, %647 : tensor<1xui32>
    %722 = stablehlo.add %721, %636 : tensor<1xui32>
    %723 = stablehlo.add %722, %627 : tensor<1xui32>
    %724 = stablehlo.shift_left %713, %c_20 : tensor<1xui32>
    %725 = stablehlo.shift_right_logical %713, %c_21 : tensor<1xui32>
    %726 = stablehlo.or %724, %725 : tensor<1xui32>
    %727 = stablehlo.xor %723, %726 : tensor<1xui32>
    %728 = stablehlo.add %612, %727 : tensor<1xui32>
    %729 = stablehlo.add %728, %713 : tensor<1xui32>
    %730 = stablehlo.add %729, %700 : tensor<1xui32>
    %731 = stablehlo.add %730, %c_8 : tensor<1xui32>
    %732 = stablehlo.add %731, %686 : tensor<1xui32>
    %733 = stablehlo.add %732, %634 : tensor<1xui32>
    %734 = stablehlo.add %733, %672 : tensor<1xui32>
    %735 = stablehlo.add %734, %659 : tensor<1xui32>
    %736 = stablehlo.add %735, %647 : tensor<1xui32>
    %737 = stablehlo.add %736, %636 : tensor<1xui32>
    %738 = stablehlo.add %737, %627 : tensor<1xui32>
    %739 = stablehlo.shift_left %727, %c_21 : tensor<1xui32>
    %740 = stablehlo.shift_right_logical %727, %c_20 : tensor<1xui32>
    %741 = stablehlo.or %739, %740 : tensor<1xui32>
    %742 = stablehlo.xor %738, %741 : tensor<1xui32>
    %743 = stablehlo.add %742, %415 : tensor<1xui32>
    %744 = stablehlo.add %743, %c_10 : tensor<1xui32>
    %745 = stablehlo.add %612, %c_10 : tensor<1xui32>
    %746 = stablehlo.add %745, %742 : tensor<1xui32>
    %747 = stablehlo.add %746, %738 : tensor<1xui32>
    %748 = stablehlo.shift_left %744, %c_18 : tensor<1xui32>
    %749 = stablehlo.shift_right_logical %744, %c_19 : tensor<1xui32>
    %750 = stablehlo.or %748, %749 : tensor<1xui32>
    %751 = stablehlo.xor %747, %750 : tensor<1xui32>
    %752 = stablehlo.add %612, %751 : tensor<1xui32>
    %753 = stablehlo.add %752, %c_10 : tensor<1xui32>
    %754 = stablehlo.add %753, %742 : tensor<1xui32>
    %755 = stablehlo.add %754, %738 : tensor<1xui32>
    %756 = stablehlo.shift_left %751, %c_11 : tensor<1xui32>
    %757 = stablehlo.shift_right_logical %751, %c_10 : tensor<1xui32>
    %758 = stablehlo.or %756, %757 : tensor<1xui32>
    %759 = stablehlo.xor %755, %758 : tensor<1xui32>
    %760 = stablehlo.add %612, %759 : tensor<1xui32>
    %761 = stablehlo.add %760, %751 : tensor<1xui32>
    %762 = stablehlo.add %761, %c_10 : tensor<1xui32>
    %763 = stablehlo.add %762, %742 : tensor<1xui32>
    %764 = stablehlo.add %763, %738 : tensor<1xui32>
    %765 = stablehlo.shift_left %759, %c_12 : tensor<1xui32>
    %766 = stablehlo.shift_right_logical %759, %c_12 : tensor<1xui32>
    %767 = stablehlo.or %765, %766 : tensor<1xui32>
    %768 = stablehlo.xor %764, %767 : tensor<1xui32>
    %769 = stablehlo.add %612, %768 : tensor<1xui32>
    %770 = stablehlo.add %769, %759 : tensor<1xui32>
    %771 = stablehlo.add %770, %751 : tensor<1xui32>
    %772 = stablehlo.add %771, %c_10 : tensor<1xui32>
    %773 = stablehlo.add %772, %742 : tensor<1xui32>
    %774 = stablehlo.add %773, %738 : tensor<1xui32>
    %775 = stablehlo.add %771, %744 : tensor<1xui32>
    %776 = stablehlo.add %775, %738 : tensor<1xui32>
    %777 = stablehlo.shift_left %768, %c_14 : tensor<1xui32>
    %778 = stablehlo.shift_right_logical %768, %c_13 : tensor<1xui32>
    %779 = stablehlo.or %777, %778 : tensor<1xui32>
    %780 = stablehlo.xor %774, %779 : tensor<1xui32>
    %781 = stablehlo.add %780, %634 : tensor<1xui32>
    %782 = stablehlo.add %781, %c_15 : tensor<1xui32>
    %783 = stablehlo.add %776, %782 : tensor<1xui32>
    %784 = stablehlo.shift_left %782, %c_17 : tensor<1xui32>
    %785 = stablehlo.shift_right_logical %782, %c_16 : tensor<1xui32>
    %786 = stablehlo.or %784, %785 : tensor<1xui32>
    %787 = stablehlo.xor %783, %786 : tensor<1xui32>
    %788 = stablehlo.add %783, %787 : tensor<1xui32>
    %789 = stablehlo.shift_left %787, %c_19 : tensor<1xui32>
    %790 = stablehlo.shift_right_logical %787, %c_18 : tensor<1xui32>
    %791 = stablehlo.or %789, %790 : tensor<1xui32>
    %792 = stablehlo.xor %788, %791 : tensor<1xui32>
    %793 = stablehlo.add %788, %792 : tensor<1xui32>
    %794 = stablehlo.shift_left %792, %c_20 : tensor<1xui32>
    %795 = stablehlo.shift_right_logical %792, %c_21 : tensor<1xui32>
    %796 = stablehlo.or %794, %795 : tensor<1xui32>
    %797 = stablehlo.xor %793, %796 : tensor<1xui32>
    %798 = stablehlo.add %793, %797 : tensor<1xui32>
    %799 = stablehlo.shift_left %797, %c_21 : tensor<1xui32>
    %800 = stablehlo.shift_right_logical %797, %c_20 : tensor<1xui32>
    %801 = stablehlo.or %799, %800 : tensor<1xui32>
    %802 = stablehlo.xor %798, %801 : tensor<1xui32>
    %803 = stablehlo.add %802, %414 : tensor<1xui32>
    %804 = stablehlo.concatenate %604, %609, %798, %803, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
    %805 = stablehlo.reshape %804 : (tensor<4xui32>) -> tensor<2x2xui32>
    %806 = stablehlo.concatenate %440, %c_34, %634, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<3xui32>
    %807 = stablehlo.pad %806, %c_33, low = [0], high = [1], interior = [0] : (tensor<3xui32>, tensor<ui32>) -> tensor<4xui32>
    %808 = stablehlo.reshape %807 : (tensor<4xui32>) -> tensor<2x2xui32>
    %809 = stablehlo.add %805, %808 : tensor<2x2xui32>
    %810 = stablehlo.bitcast_convert %809 : (tensor<2x2xui32>) -> tensor<2xui64>
    %811 = stablehlo.reshape %arg1 : (tensor<3x1xf64>) -> tensor<3xf64>
    %812 = stablehlo.slice %811 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %813 = stablehlo.reshape %812 : (tensor<1xf64>) -> tensor<f64>
    %814 = stablehlo.reshape %812 : (tensor<1xf64>) -> tensor<1x1xf64>
    %815 = stablehlo.slice %811 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %816 = stablehlo.reshape %815 : (tensor<1xf64>) -> tensor<f64>
    %817 = stablehlo.reshape %815 : (tensor<1xf64>) -> tensor<1x1xf64>
    %818 = stablehlo.slice %811 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %819 = stablehlo.reshape %818 : (tensor<1xf64>) -> tensor<1x1xf64>
    %820 = stablehlo.reshape %818 : (tensor<1xf64>) -> tensor<f64>
    %821 = stablehlo.exponential %813 : tensor<f64>
    %822 = stablehlo.reshape %821 : (tensor<f64>) -> tensor<1x1xf64>
    %823 = stablehlo.exponential %816 : tensor<f64>
    %824 = stablehlo.reshape %823 : (tensor<f64>) -> tensor<1x1xf64>
    %825 = stablehlo.exponential %820 : tensor<f64>
    %826 = stablehlo.reshape %825 : (tensor<f64>) -> tensor<1x1xf64>
    %827 = stablehlo.add %cst_41, %820 : tensor<f64>
    %828 = stablehlo.multiply %cst_42, %827 : tensor<f64>
    %829 = stablehlo.multiply %cst_43, %821 : tensor<f64>
    %830 = stablehlo.broadcast_in_dim %829, dims = [] : (tensor<f64>) -> tensor<20x20xf64>
    %831 = stablehlo.multiply %830, %1 : tensor<20x20xf64>
    %832 = stablehlo.add %cst_47, %831 : tensor<20x20xf64>
    %833:2 = stablehlo.custom_call @enzymexla_compile_cpu(%c_4, %c_91, %832, %c_91, %c) {api_version = 3 : i32, backend_config = "\00p~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i8>, tensor<i64>, tensor<20x20xf64>, tensor<i64>, tensor<i64>) -> (tensor<20x20xf64>, tensor<i64>)
    %834 = stablehlo.iota dim = 0 : tensor<20x20xi64>
    %835 = stablehlo.iota dim = 1 : tensor<20x20xi64>
    %836 = stablehlo.compare LE, %834, %835 : (tensor<20x20xi64>, tensor<20x20xi64>) -> tensor<20x20xi1>
    %837 = stablehlo.select %836, %833#0, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %838 = stablehlo.multiply %cst_43, %823 : tensor<f64>
    %839 = stablehlo.dot_general %arg5, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %840 = stablehlo.broadcast_in_dim %838, dims = [] : (tensor<f64>) -> tensor<20x1xf64>
    %841 = stablehlo.multiply %840, %839 : tensor<20x1xf64>
    %842 = stablehlo.subtract %2, %841 : tensor<20x1xf64>
    %843 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %842, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %844 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %843, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %845 = stablehlo.dot_general %arg5, %844, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %846 = stablehlo.multiply %840, %845 : tensor<20x1xf64>
    %847 = stablehlo.subtract %844, %846 : tensor<20x1xf64>
    %848 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %847, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %849 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %848, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %850 = stablehlo.subtract %c_44, %c_48 : tensor<8x1xi64>
    %851 = "stablehlo.gather"(%849, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
    %852 = stablehlo.slice %3 [0:1, 0:8] : (tensor<5x8xf64>) -> tensor<1x8xf64>
    %853 = stablehlo.reshape %852 : (tensor<1x8xf64>) -> tensor<8xf64>
    %854 = stablehlo.subtract %853, %851 : tensor<8xf64>
    %855 = stablehlo.dot_general %854, %854, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %856 = stablehlo.multiply %cst_88, %855 : tensor<f64>
    %857 = stablehlo.multiply %825, %825 : tensor<f64>
    %858 = stablehlo.divide %856, %857 : tensor<f64>
    %859 = stablehlo.reshape %858 : (tensor<f64>) -> tensor<1x1xf64>
    %860 = stablehlo.subtract %828, %858 : tensor<f64>
    %861 = stablehlo.dot_general %arg5, %849, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %862 = stablehlo.multiply %840, %861 : tensor<20x1xf64>
    %863 = stablehlo.subtract %849, %862 : tensor<20x1xf64>
    %864 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %863, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %865 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %864, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %866 = stablehlo.dot_general %arg5, %865, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %867 = stablehlo.multiply %840, %866 : tensor<20x1xf64>
    %868 = stablehlo.subtract %865, %867 : tensor<20x1xf64>
    %869 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %868, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %870 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %869, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %871 = "stablehlo.gather"(%870, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
    %872 = stablehlo.slice %3 [1:2, 0:8] : (tensor<5x8xf64>) -> tensor<1x8xf64>
    %873 = stablehlo.reshape %872 : (tensor<1x8xf64>) -> tensor<8xf64>
    %874 = stablehlo.subtract %873, %871 : tensor<8xf64>
    %875 = stablehlo.dot_general %874, %874, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %876 = stablehlo.multiply %cst_88, %875 : tensor<f64>
    %877 = stablehlo.divide %876, %857 : tensor<f64>
    %878 = stablehlo.reshape %877 : (tensor<f64>) -> tensor<1x1xf64>
    %879 = stablehlo.subtract %860, %877 : tensor<f64>
    %880 = stablehlo.dot_general %arg5, %870, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %881 = stablehlo.multiply %840, %880 : tensor<20x1xf64>
    %882 = stablehlo.subtract %870, %881 : tensor<20x1xf64>
    %883 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %882, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %884 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %883, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %885 = stablehlo.dot_general %arg5, %884, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %886 = stablehlo.multiply %840, %885 : tensor<20x1xf64>
    %887 = stablehlo.subtract %884, %886 : tensor<20x1xf64>
    %888 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %887, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %889 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %888, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %890 = "stablehlo.gather"(%889, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
    %891 = stablehlo.slice %3 [2:3, 0:8] : (tensor<5x8xf64>) -> tensor<1x8xf64>
    %892 = stablehlo.reshape %891 : (tensor<1x8xf64>) -> tensor<8xf64>
    %893 = stablehlo.subtract %892, %890 : tensor<8xf64>
    %894 = stablehlo.dot_general %893, %893, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %895 = stablehlo.multiply %cst_88, %894 : tensor<f64>
    %896 = stablehlo.divide %895, %857 : tensor<f64>
    %897 = stablehlo.reshape %896 : (tensor<f64>) -> tensor<1x1xf64>
    %898 = stablehlo.subtract %879, %896 : tensor<f64>
    %899 = stablehlo.dot_general %arg5, %889, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %900 = stablehlo.multiply %840, %899 : tensor<20x1xf64>
    %901 = stablehlo.subtract %889, %900 : tensor<20x1xf64>
    %902 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %901, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %903 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %902, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %904 = stablehlo.dot_general %arg5, %903, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %905 = stablehlo.multiply %840, %904 : tensor<20x1xf64>
    %906 = stablehlo.subtract %903, %905 : tensor<20x1xf64>
    %907 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %906, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %908 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %907, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %909 = "stablehlo.gather"(%908, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
    %910 = stablehlo.slice %3 [3:4, 0:8] : (tensor<5x8xf64>) -> tensor<1x8xf64>
    %911 = stablehlo.reshape %910 : (tensor<1x8xf64>) -> tensor<8xf64>
    %912 = stablehlo.subtract %911, %909 : tensor<8xf64>
    %913 = stablehlo.dot_general %912, %912, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %914 = stablehlo.multiply %cst_88, %913 : tensor<f64>
    %915 = stablehlo.divide %914, %857 : tensor<f64>
    %916 = stablehlo.reshape %915 : (tensor<f64>) -> tensor<1x1xf64>
    %917 = stablehlo.subtract %898, %915 : tensor<f64>
    %918 = stablehlo.dot_general %arg5, %908, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %919 = stablehlo.multiply %840, %918 : tensor<20x1xf64>
    %920 = stablehlo.subtract %908, %919 : tensor<20x1xf64>
    %921 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %920, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %922 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %921, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %923 = stablehlo.dot_general %arg5, %922, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
    %924 = stablehlo.multiply %840, %923 : tensor<20x1xf64>
    %925 = stablehlo.subtract %922, %924 : tensor<20x1xf64>
    %926 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %925, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %927 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %926, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %928 = "stablehlo.gather"(%927, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
    %929 = stablehlo.slice %3 [4:5, 0:8] : (tensor<5x8xf64>) -> tensor<1x8xf64>
    %930 = stablehlo.reshape %929 : (tensor<1x8xf64>) -> tensor<8xf64>
    %931 = stablehlo.subtract %930, %928 : tensor<8xf64>
    %932 = stablehlo.dot_general %931, %931, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %933 = stablehlo.multiply %cst_88, %932 : tensor<f64>
    %934 = stablehlo.divide %933, %857 : tensor<f64>
    %935 = stablehlo.reshape %934 : (tensor<f64>) -> tensor<1x1xf64>
    %936 = stablehlo.subtract %917, %934 : tensor<f64>
    %937 = stablehlo.multiply %813, %813 : tensor<f64>
    %938 = stablehlo.multiply %816, %816 : tensor<f64>
    %939 = stablehlo.multiply %820, %820 : tensor<f64>
    %940 = stablehlo.add %937, %938 : tensor<f64>
    %941 = stablehlo.add %940, %939 : tensor<f64>
    %942 = stablehlo.multiply %cst_45, %941 : tensor<f64>
    %943 = stablehlo.subtract %942, %cst_46 : tensor<f64>
    %944 = stablehlo.add %943, %936 : tensor<f64>
    %945 = stablehlo.negate %944 : tensor<f64>
    %946 = stablehlo.divide %cst_95, %857 : tensor<f64>
    %947 = stablehlo.reshape %946 : (tensor<f64>) -> tensor<1x1xf64>
    %948 = stablehlo.multiply %947, %935 : tensor<1x1xf64>
    %949 = stablehlo.negate %948 : tensor<1x1xf64>
    %950 = stablehlo.multiply %946, %cst_88 : tensor<f64>
    %951 = stablehlo.dot_general %950, %931, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
    %952 = stablehlo.add %951, %951 : tensor<8xf64>
    %953 = stablehlo.negate %952 : tensor<8xf64>
    %954 = "stablehlo.scatter"(%cst_39, %850, %953) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
      %1174 = stablehlo.add %arg8, %arg9 : tensor<f64>
      stablehlo.return %1174 : tensor<f64>
    }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
    %955 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %954, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %956 = stablehlo.negate %955 : tensor<20x1xf64>
    %957 = stablehlo.dot_general %956, %927, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %958 = stablehlo.select %c_29, %957, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %959 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %955, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %960 = stablehlo.negate %926 : tensor<20x1xf64>
    %961 = stablehlo.dot_general %960, %959, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %962 = stablehlo.select %c_29, %961, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %963 = stablehlo.add %958, %962 : tensor<20x20xf64>
    %964 = stablehlo.negate %959 : tensor<20x1xf64>
    %965 = stablehlo.multiply %964, %923 : tensor<20x1xf64>
    %966 = stablehlo.dot_general %964, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %967 = stablehlo.reshape %966 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %968 = stablehlo.multiply %840, %967 : tensor<20x1xf64>
    %969 = stablehlo.add %959, %968 : tensor<20x1xf64>
    %970 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %969, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %971 = stablehlo.negate %970 : tensor<20x1xf64>
    %972 = stablehlo.dot_general %971, %922, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %973 = stablehlo.select %c_29, %972, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %974 = stablehlo.add %963, %973 : tensor<20x20xf64>
    %975 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %970, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %976 = stablehlo.negate %921 : tensor<20x1xf64>
    %977 = stablehlo.dot_general %976, %975, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %978 = stablehlo.select %c_29, %977, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %979 = stablehlo.add %974, %978 : tensor<20x20xf64>
    %980 = stablehlo.negate %975 : tensor<20x1xf64>
    %981 = stablehlo.multiply %980, %918 : tensor<20x1xf64>
    %982 = stablehlo.add %965, %981 : tensor<20x1xf64>
    %983 = stablehlo.dot_general %980, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %984 = stablehlo.reshape %983 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %985 = stablehlo.multiply %840, %984 : tensor<20x1xf64>
    %986 = stablehlo.add %975, %985 : tensor<20x1xf64>
    %987 = stablehlo.multiply %947, %916 : tensor<1x1xf64>
    %988 = stablehlo.subtract %949, %987 : tensor<1x1xf64>
    %989 = stablehlo.dot_general %950, %912, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
    %990 = stablehlo.add %989, %989 : tensor<8xf64>
    %991 = stablehlo.negate %990 : tensor<8xf64>
    %992 = "stablehlo.scatter"(%cst_39, %850, %991) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
      %1174 = stablehlo.add %arg8, %arg9 : tensor<f64>
      stablehlo.return %1174 : tensor<f64>
    }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
    %993 = stablehlo.add %986, %992 : tensor<20x1xf64>
    %994 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %993, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %995 = stablehlo.negate %994 : tensor<20x1xf64>
    %996 = stablehlo.dot_general %995, %908, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %997 = stablehlo.select %c_29, %996, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %998 = stablehlo.add %979, %997 : tensor<20x20xf64>
    %999 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %994, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1000 = stablehlo.negate %907 : tensor<20x1xf64>
    %1001 = stablehlo.dot_general %1000, %999, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1002 = stablehlo.select %c_29, %1001, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1003 = stablehlo.add %998, %1002 : tensor<20x20xf64>
    %1004 = stablehlo.negate %999 : tensor<20x1xf64>
    %1005 = stablehlo.multiply %1004, %904 : tensor<20x1xf64>
    %1006 = stablehlo.add %982, %1005 : tensor<20x1xf64>
    %1007 = stablehlo.dot_general %1004, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1008 = stablehlo.reshape %1007 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1009 = stablehlo.multiply %840, %1008 : tensor<20x1xf64>
    %1010 = stablehlo.add %999, %1009 : tensor<20x1xf64>
    %1011 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1010, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1012 = stablehlo.negate %1011 : tensor<20x1xf64>
    %1013 = stablehlo.dot_general %1012, %903, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1014 = stablehlo.select %c_29, %1013, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1015 = stablehlo.add %1003, %1014 : tensor<20x20xf64>
    %1016 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1011, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1017 = stablehlo.negate %902 : tensor<20x1xf64>
    %1018 = stablehlo.dot_general %1017, %1016, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1019 = stablehlo.select %c_29, %1018, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1020 = stablehlo.add %1015, %1019 : tensor<20x20xf64>
    %1021 = stablehlo.negate %1016 : tensor<20x1xf64>
    %1022 = stablehlo.multiply %1021, %899 : tensor<20x1xf64>
    %1023 = stablehlo.add %1006, %1022 : tensor<20x1xf64>
    %1024 = stablehlo.dot_general %1021, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1025 = stablehlo.reshape %1024 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1026 = stablehlo.multiply %840, %1025 : tensor<20x1xf64>
    %1027 = stablehlo.add %1016, %1026 : tensor<20x1xf64>
    %1028 = stablehlo.multiply %947, %897 : tensor<1x1xf64>
    %1029 = stablehlo.subtract %988, %1028 : tensor<1x1xf64>
    %1030 = stablehlo.dot_general %950, %893, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
    %1031 = stablehlo.add %1030, %1030 : tensor<8xf64>
    %1032 = stablehlo.negate %1031 : tensor<8xf64>
    %1033 = "stablehlo.scatter"(%cst_39, %850, %1032) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
      %1174 = stablehlo.add %arg8, %arg9 : tensor<f64>
      stablehlo.return %1174 : tensor<f64>
    }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
    %1034 = stablehlo.add %1027, %1033 : tensor<20x1xf64>
    %1035 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1034, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1036 = stablehlo.negate %1035 : tensor<20x1xf64>
    %1037 = stablehlo.dot_general %1036, %889, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1038 = stablehlo.select %c_29, %1037, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1039 = stablehlo.add %1020, %1038 : tensor<20x20xf64>
    %1040 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1035, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1041 = stablehlo.negate %888 : tensor<20x1xf64>
    %1042 = stablehlo.dot_general %1041, %1040, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1043 = stablehlo.select %c_29, %1042, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1044 = stablehlo.add %1039, %1043 : tensor<20x20xf64>
    %1045 = stablehlo.negate %1040 : tensor<20x1xf64>
    %1046 = stablehlo.multiply %1045, %885 : tensor<20x1xf64>
    %1047 = stablehlo.add %1023, %1046 : tensor<20x1xf64>
    %1048 = stablehlo.dot_general %1045, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1049 = stablehlo.reshape %1048 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1050 = stablehlo.multiply %840, %1049 : tensor<20x1xf64>
    %1051 = stablehlo.add %1040, %1050 : tensor<20x1xf64>
    %1052 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1051, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1053 = stablehlo.negate %1052 : tensor<20x1xf64>
    %1054 = stablehlo.dot_general %1053, %884, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1055 = stablehlo.select %c_29, %1054, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1056 = stablehlo.add %1044, %1055 : tensor<20x20xf64>
    %1057 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1052, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1058 = stablehlo.negate %883 : tensor<20x1xf64>
    %1059 = stablehlo.dot_general %1058, %1057, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1060 = stablehlo.select %c_29, %1059, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1061 = stablehlo.add %1056, %1060 : tensor<20x20xf64>
    %1062 = stablehlo.negate %1057 : tensor<20x1xf64>
    %1063 = stablehlo.multiply %1062, %880 : tensor<20x1xf64>
    %1064 = stablehlo.add %1047, %1063 : tensor<20x1xf64>
    %1065 = stablehlo.dot_general %1062, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1066 = stablehlo.reshape %1065 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1067 = stablehlo.multiply %840, %1066 : tensor<20x1xf64>
    %1068 = stablehlo.add %1057, %1067 : tensor<20x1xf64>
    %1069 = stablehlo.multiply %947, %878 : tensor<1x1xf64>
    %1070 = stablehlo.subtract %1029, %1069 : tensor<1x1xf64>
    %1071 = stablehlo.dot_general %950, %874, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
    %1072 = stablehlo.add %1071, %1071 : tensor<8xf64>
    %1073 = stablehlo.negate %1072 : tensor<8xf64>
    %1074 = "stablehlo.scatter"(%cst_39, %850, %1073) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
      %1174 = stablehlo.add %arg8, %arg9 : tensor<f64>
      stablehlo.return %1174 : tensor<f64>
    }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
    %1075 = stablehlo.add %1068, %1074 : tensor<20x1xf64>
    %1076 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1075, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1077 = stablehlo.negate %1076 : tensor<20x1xf64>
    %1078 = stablehlo.dot_general %1077, %870, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1079 = stablehlo.select %c_29, %1078, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1080 = stablehlo.add %1061, %1079 : tensor<20x20xf64>
    %1081 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1076, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1082 = stablehlo.negate %869 : tensor<20x1xf64>
    %1083 = stablehlo.dot_general %1082, %1081, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1084 = stablehlo.select %c_29, %1083, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1085 = stablehlo.add %1080, %1084 : tensor<20x20xf64>
    %1086 = stablehlo.negate %1081 : tensor<20x1xf64>
    %1087 = stablehlo.multiply %1086, %866 : tensor<20x1xf64>
    %1088 = stablehlo.add %1064, %1087 : tensor<20x1xf64>
    %1089 = stablehlo.dot_general %1086, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1090 = stablehlo.reshape %1089 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1091 = stablehlo.multiply %840, %1090 : tensor<20x1xf64>
    %1092 = stablehlo.add %1081, %1091 : tensor<20x1xf64>
    %1093 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1092, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1094 = stablehlo.negate %1093 : tensor<20x1xf64>
    %1095 = stablehlo.dot_general %1094, %865, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1096 = stablehlo.select %c_29, %1095, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1097 = stablehlo.add %1085, %1096 : tensor<20x20xf64>
    %1098 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1093, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1099 = stablehlo.negate %864 : tensor<20x1xf64>
    %1100 = stablehlo.dot_general %1099, %1098, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1101 = stablehlo.select %c_29, %1100, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1102 = stablehlo.add %1097, %1101 : tensor<20x20xf64>
    %1103 = stablehlo.negate %1098 : tensor<20x1xf64>
    %1104 = stablehlo.multiply %1103, %861 : tensor<20x1xf64>
    %1105 = stablehlo.add %1088, %1104 : tensor<20x1xf64>
    %1106 = stablehlo.dot_general %1103, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1107 = stablehlo.reshape %1106 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1108 = stablehlo.multiply %840, %1107 : tensor<20x1xf64>
    %1109 = stablehlo.add %1098, %1108 : tensor<20x1xf64>
    %1110 = stablehlo.multiply %947, %859 : tensor<1x1xf64>
    %1111 = stablehlo.subtract %1070, %1110 : tensor<1x1xf64>
    %1112 = stablehlo.add %1111, %1111 : tensor<1x1xf64>
    %1113 = stablehlo.dot_general %950, %854, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
    %1114 = stablehlo.add %1113, %1113 : tensor<8xf64>
    %1115 = stablehlo.negate %1114 : tensor<8xf64>
    %1116 = "stablehlo.scatter"(%cst_39, %850, %1115) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
      %1174 = stablehlo.add %arg8, %arg9 : tensor<f64>
      stablehlo.return %1174 : tensor<f64>
    }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
    %1117 = stablehlo.add %1109, %1116 : tensor<20x1xf64>
    %1118 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1117, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1119 = stablehlo.negate %1118 : tensor<20x1xf64>
    %1120 = stablehlo.dot_general %1119, %849, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1121 = stablehlo.select %c_29, %1120, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1122 = stablehlo.add %1102, %1121 : tensor<20x20xf64>
    %1123 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1118, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1124 = stablehlo.negate %848 : tensor<20x1xf64>
    %1125 = stablehlo.dot_general %1124, %1123, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1126 = stablehlo.select %c_29, %1125, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1127 = stablehlo.add %1122, %1126 : tensor<20x20xf64>
    %1128 = stablehlo.negate %1123 : tensor<20x1xf64>
    %1129 = stablehlo.multiply %1128, %845 : tensor<20x1xf64>
    %1130 = stablehlo.add %1105, %1129 : tensor<20x1xf64>
    %1131 = stablehlo.dot_general %1128, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<1x20xf64>) -> tensor<20x1xf64>
    %1133 = stablehlo.multiply %840, %1132 : tensor<20x1xf64>
    %1134 = stablehlo.add %1123, %1133 : tensor<20x1xf64>
    %1135 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1134, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1136 = stablehlo.negate %1135 : tensor<20x1xf64>
    %1137 = stablehlo.dot_general %1136, %844, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1138 = stablehlo.select %c_29, %1137, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1139 = stablehlo.add %1127, %1138 : tensor<20x20xf64>
    %1140 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %837, %c_91, %1135, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
    %1141 = stablehlo.negate %843 : tensor<20x1xf64>
    %1142 = stablehlo.dot_general %1141, %1140, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
    %1143 = stablehlo.select %c_29, %1142, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1144 = stablehlo.add %1139, %1143 : tensor<20x20xf64>
    %1145 = stablehlo.negate %1140 : tensor<20x1xf64>
    %1146 = stablehlo.multiply %1145, %839 : tensor<20x1xf64>
    %1147 = stablehlo.add %1130, %1146 : tensor<20x1xf64>
    %1148 = stablehlo.reduce(%1147 init: %cst_90) applies stablehlo.add across dimensions = [0, 1] : (tensor<20x1xf64>, tensor<f64>) -> tensor<f64>
    %1149 = stablehlo.reshape %1148 : (tensor<f64>) -> tensor<1x1xf64>
    %1150 = stablehlo.dot_general %837, %1144, contracting_dims = [1] x [1] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %1151 = stablehlo.select %c_29, %cst_40, %1150 : tensor<20x20xi1>, tensor<20x20xf64>
    %1152 = stablehlo.multiply %1150, %cst_38 : tensor<20x20xf64>
    %1153 = stablehlo.select %c_28, %1152, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
    %1154 = stablehlo.add %1151, %1153 : tensor<20x20xf64>
    %1155 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_91, %cst_95, %837, %c_91, %1154, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x20xf64>, tensor<i64>) -> tensor<20x20xf64>
    %1156 = stablehlo.transpose %1155, dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %1157 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_91, %cst_95, %837, %c_91, %1156, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x20xf64>, tensor<i64>) -> tensor<20x20xf64>
    %1158 = stablehlo.transpose %1157, dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %1159 = stablehlo.add %1158, %1157 : tensor<20x20xf64>
    %1160 = stablehlo.dot_general %1159, %arg4, contracting_dims = [0, 1] x [1, 0] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<f64>
    %1161 = stablehlo.reshape %1160 : (tensor<f64>) -> tensor<1x1xf64>
    %1162 = stablehlo.multiply %cst_89, %1161 : tensor<1x1xf64>
    %1163 = stablehlo.add %819, %cst_27 : tensor<1x1xf64>
    %1164 = stablehlo.concatenate %814, %817, %1163, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x3xf64>
    %1165 = stablehlo.concatenate %1162, %1149, %826, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x3xf64>
    %1166 = stablehlo.pad %1112, %cst_43, low = [0, 2], high = [0, 0], interior = [0, 0] : (tensor<1x1xf64>, tensor<f64>) -> tensor<1x3xf64>
    %1167 = stablehlo.multiply %1165, %1166 : tensor<1x3xf64>
    %1168 = stablehlo.concatenate %822, %824, %826, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x3xf64>
    %1169 = stablehlo.multiply %1167, %1168 : tensor<1x3xf64>
    %1170 = stablehlo.add %1164, %1169 : tensor<1x3xf64>
    %1171:8 = stablehlo.while(%iterArg = %c_93, %iterArg_99 = %0, %iterArg_100 = %1170, %iterArg_101 = %945, %iterArg_102 = %810, %iterArg_103 = %cst_26, %iterArg_104 = %c_25, %iterArg_105 = %cst_94) : tensor<i64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<3x20xf64>, tensor<2x20xi1>, tensor<20xf64>
    cond {
      %1174 = stablehlo.compare LT, %iterArg, %c_91 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1174 : tensor<i1>
    } do {
      %1174 = stablehlo.bitcast_convert %iterArg_102 : (tensor<2xui64>) -> tensor<2x2xui32>
      %1175 = stablehlo.reshape %1174 : (tensor<2x2xui32>) -> tensor<4xui32>
      %1176 = stablehlo.slice %1175 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
      %1177 = stablehlo.slice %1175 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
      %1178 = stablehlo.slice %1175 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
      %1179 = stablehlo.slice %1175 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
      %1180 = stablehlo.broadcast_in_dim %1176, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
      %1181 = stablehlo.broadcast_in_dim %1177, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
      %1182 = stablehlo.broadcast_in_dim %1178, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
      %1183 = stablehlo.broadcast_in_dim %1179, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
      %1184 = stablehlo.xor %1180, %1181 : tensor<3xui32>
      %1185 = stablehlo.xor %1184, %c_69 : tensor<3xui32>
      %1186 = stablehlo.add %c_32, %1181 : tensor<3xui32>
      %1187 = stablehlo.add %1180, %1186 : tensor<3xui32>
      %1188 = stablehlo.shift_left %1186, %c_68 : tensor<3xui32>
      %1189 = stablehlo.shift_right_logical %1186, %c_31 : tensor<3xui32>
      %1190 = stablehlo.or %1188, %1189 : tensor<3xui32>
      %1191 = stablehlo.xor %1187, %1190 : tensor<3xui32>
      %1192 = stablehlo.add %1187, %1191 : tensor<3xui32>
      %1193 = stablehlo.shift_left %1191, %c_67 : tensor<3xui32>
      %1194 = stablehlo.shift_right_logical %1191, %c_63 : tensor<3xui32>
      %1195 = stablehlo.or %1193, %1194 : tensor<3xui32>
      %1196 = stablehlo.xor %1192, %1195 : tensor<3xui32>
      %1197 = stablehlo.add %1192, %1196 : tensor<3xui32>
      %1198 = stablehlo.shift_left %1196, %c_66 : tensor<3xui32>
      %1199 = stablehlo.shift_right_logical %1196, %c_65 : tensor<3xui32>
      %1200 = stablehlo.or %1198, %1199 : tensor<3xui32>
      %1201 = stablehlo.xor %1197, %1200 : tensor<3xui32>
      %1202 = stablehlo.add %1197, %1201 : tensor<3xui32>
      %1203 = stablehlo.shift_left %1201, %c_65 : tensor<3xui32>
      %1204 = stablehlo.shift_right_logical %1201, %c_66 : tensor<3xui32>
      %1205 = stablehlo.or %1203, %1204 : tensor<3xui32>
      %1206 = stablehlo.xor %1202, %1205 : tensor<3xui32>
      %1207 = stablehlo.add %1202, %1181 : tensor<3xui32>
      %1208 = stablehlo.add %1206, %1185 : tensor<3xui32>
      %1209 = stablehlo.add %1208, %c_64 : tensor<3xui32>
      %1210 = stablehlo.add %1207, %1209 : tensor<3xui32>
      %1211 = stablehlo.shift_left %1209, %c_63 : tensor<3xui32>
      %1212 = stablehlo.shift_right_logical %1209, %c_67 : tensor<3xui32>
      %1213 = stablehlo.or %1211, %1212 : tensor<3xui32>
      %1214 = stablehlo.xor %1210, %1213 : tensor<3xui32>
      %1215 = stablehlo.add %1210, %1214 : tensor<3xui32>
      %1216 = stablehlo.shift_left %1214, %c_62 : tensor<3xui32>
      %1217 = stablehlo.shift_right_logical %1214, %c_58 : tensor<3xui32>
      %1218 = stablehlo.or %1216, %1217 : tensor<3xui32>
      %1219 = stablehlo.xor %1215, %1218 : tensor<3xui32>
      %1220 = stablehlo.add %1215, %1219 : tensor<3xui32>
      %1221 = stablehlo.shift_left %1219, %c_61 : tensor<3xui32>
      %1222 = stablehlo.shift_right_logical %1219, %c_61 : tensor<3xui32>
      %1223 = stablehlo.or %1221, %1222 : tensor<3xui32>
      %1224 = stablehlo.xor %1220, %1223 : tensor<3xui32>
      %1225 = stablehlo.add %1220, %1224 : tensor<3xui32>
      %1226 = stablehlo.shift_left %1224, %c_60 : tensor<3xui32>
      %1227 = stablehlo.shift_right_logical %1224, %c_30 : tensor<3xui32>
      %1228 = stablehlo.or %1226, %1227 : tensor<3xui32>
      %1229 = stablehlo.xor %1225, %1228 : tensor<3xui32>
      %1230 = stablehlo.add %1225, %1185 : tensor<3xui32>
      %1231 = stablehlo.add %1229, %1180 : tensor<3xui32>
      %1232 = stablehlo.add %1231, %c_59 : tensor<3xui32>
      %1233 = stablehlo.add %1230, %1232 : tensor<3xui32>
      %1234 = stablehlo.shift_left %1232, %c_68 : tensor<3xui32>
      %1235 = stablehlo.shift_right_logical %1232, %c_31 : tensor<3xui32>
      %1236 = stablehlo.or %1234, %1235 : tensor<3xui32>
      %1237 = stablehlo.xor %1233, %1236 : tensor<3xui32>
      %1238 = stablehlo.add %1233, %1237 : tensor<3xui32>
      %1239 = stablehlo.shift_left %1237, %c_67 : tensor<3xui32>
      %1240 = stablehlo.shift_right_logical %1237, %c_63 : tensor<3xui32>
      %1241 = stablehlo.or %1239, %1240 : tensor<3xui32>
      %1242 = stablehlo.xor %1238, %1241 : tensor<3xui32>
      %1243 = stablehlo.add %1238, %1242 : tensor<3xui32>
      %1244 = stablehlo.shift_left %1242, %c_66 : tensor<3xui32>
      %1245 = stablehlo.shift_right_logical %1242, %c_65 : tensor<3xui32>
      %1246 = stablehlo.or %1244, %1245 : tensor<3xui32>
      %1247 = stablehlo.xor %1243, %1246 : tensor<3xui32>
      %1248 = stablehlo.add %1243, %1247 : tensor<3xui32>
      %1249 = stablehlo.shift_left %1247, %c_65 : tensor<3xui32>
      %1250 = stablehlo.shift_right_logical %1247, %c_66 : tensor<3xui32>
      %1251 = stablehlo.or %1249, %1250 : tensor<3xui32>
      %1252 = stablehlo.xor %1248, %1251 : tensor<3xui32>
      %1253 = stablehlo.add %1248, %1180 : tensor<3xui32>
      %1254 = stablehlo.add %1252, %1181 : tensor<3xui32>
      %1255 = stablehlo.add %1254, %c_58 : tensor<3xui32>
      %1256 = stablehlo.add %1253, %1255 : tensor<3xui32>
      %1257 = stablehlo.shift_left %1255, %c_63 : tensor<3xui32>
      %1258 = stablehlo.shift_right_logical %1255, %c_67 : tensor<3xui32>
      %1259 = stablehlo.or %1257, %1258 : tensor<3xui32>
      %1260 = stablehlo.xor %1256, %1259 : tensor<3xui32>
      %1261 = stablehlo.add %1256, %1260 : tensor<3xui32>
      %1262 = stablehlo.shift_left %1260, %c_62 : tensor<3xui32>
      %1263 = stablehlo.shift_right_logical %1260, %c_58 : tensor<3xui32>
      %1264 = stablehlo.or %1262, %1263 : tensor<3xui32>
      %1265 = stablehlo.xor %1261, %1264 : tensor<3xui32>
      %1266 = stablehlo.add %1261, %1265 : tensor<3xui32>
      %1267 = stablehlo.shift_left %1265, %c_61 : tensor<3xui32>
      %1268 = stablehlo.shift_right_logical %1265, %c_61 : tensor<3xui32>
      %1269 = stablehlo.or %1267, %1268 : tensor<3xui32>
      %1270 = stablehlo.xor %1266, %1269 : tensor<3xui32>
      %1271 = stablehlo.add %1266, %1270 : tensor<3xui32>
      %1272 = stablehlo.shift_left %1270, %c_60 : tensor<3xui32>
      %1273 = stablehlo.shift_right_logical %1270, %c_30 : tensor<3xui32>
      %1274 = stablehlo.or %1272, %1273 : tensor<3xui32>
      %1275 = stablehlo.xor %1271, %1274 : tensor<3xui32>
      %1276 = stablehlo.add %1271, %1181 : tensor<3xui32>
      %1277 = stablehlo.add %1275, %1185 : tensor<3xui32>
      %1278 = stablehlo.add %1277, %c_57 : tensor<3xui32>
      %1279 = stablehlo.add %1276, %1278 : tensor<3xui32>
      %1280 = stablehlo.shift_left %1278, %c_68 : tensor<3xui32>
      %1281 = stablehlo.shift_right_logical %1278, %c_31 : tensor<3xui32>
      %1282 = stablehlo.or %1280, %1281 : tensor<3xui32>
      %1283 = stablehlo.xor %1279, %1282 : tensor<3xui32>
      %1284 = stablehlo.add %1279, %1283 : tensor<3xui32>
      %1285 = stablehlo.shift_left %1283, %c_67 : tensor<3xui32>
      %1286 = stablehlo.shift_right_logical %1283, %c_63 : tensor<3xui32>
      %1287 = stablehlo.or %1285, %1286 : tensor<3xui32>
      %1288 = stablehlo.xor %1284, %1287 : tensor<3xui32>
      %1289 = stablehlo.add %1284, %1288 : tensor<3xui32>
      %1290 = stablehlo.shift_left %1288, %c_66 : tensor<3xui32>
      %1291 = stablehlo.shift_right_logical %1288, %c_65 : tensor<3xui32>
      %1292 = stablehlo.or %1290, %1291 : tensor<3xui32>
      %1293 = stablehlo.xor %1289, %1292 : tensor<3xui32>
      %1294 = stablehlo.add %1289, %1293 : tensor<3xui32>
      %1295 = stablehlo.shift_left %1293, %c_65 : tensor<3xui32>
      %1296 = stablehlo.shift_right_logical %1293, %c_66 : tensor<3xui32>
      %1297 = stablehlo.or %1295, %1296 : tensor<3xui32>
      %1298 = stablehlo.xor %1294, %1297 : tensor<3xui32>
      %1299 = stablehlo.add %1294, %1185 : tensor<3xui32>
      %1300 = stablehlo.add %1298, %1180 : tensor<3xui32>
      %1301 = stablehlo.add %1300, %c_56 : tensor<3xui32>
      %1302 = stablehlo.xor %1182, %1183 : tensor<3xui32>
      %1303 = stablehlo.xor %1302, %c_69 : tensor<3xui32>
      %1304 = stablehlo.add %c_32, %1183 : tensor<3xui32>
      %1305 = stablehlo.add %1182, %1304 : tensor<3xui32>
      %1306 = stablehlo.shift_left %1304, %c_68 : tensor<3xui32>
      %1307 = stablehlo.shift_right_logical %1304, %c_31 : tensor<3xui32>
      %1308 = stablehlo.or %1306, %1307 : tensor<3xui32>
      %1309 = stablehlo.xor %1305, %1308 : tensor<3xui32>
      %1310 = stablehlo.add %1305, %1309 : tensor<3xui32>
      %1311 = stablehlo.shift_left %1309, %c_67 : tensor<3xui32>
      %1312 = stablehlo.shift_right_logical %1309, %c_63 : tensor<3xui32>
      %1313 = stablehlo.or %1311, %1312 : tensor<3xui32>
      %1314 = stablehlo.xor %1310, %1313 : tensor<3xui32>
      %1315 = stablehlo.add %1310, %1314 : tensor<3xui32>
      %1316 = stablehlo.shift_left %1314, %c_66 : tensor<3xui32>
      %1317 = stablehlo.shift_right_logical %1314, %c_65 : tensor<3xui32>
      %1318 = stablehlo.or %1316, %1317 : tensor<3xui32>
      %1319 = stablehlo.xor %1315, %1318 : tensor<3xui32>
      %1320 = stablehlo.add %1315, %1319 : tensor<3xui32>
      %1321 = stablehlo.shift_left %1319, %c_65 : tensor<3xui32>
      %1322 = stablehlo.shift_right_logical %1319, %c_66 : tensor<3xui32>
      %1323 = stablehlo.or %1321, %1322 : tensor<3xui32>
      %1324 = stablehlo.xor %1320, %1323 : tensor<3xui32>
      %1325 = stablehlo.add %1320, %1183 : tensor<3xui32>
      %1326 = stablehlo.add %1324, %1303 : tensor<3xui32>
      %1327 = stablehlo.add %1326, %c_64 : tensor<3xui32>
      %1328 = stablehlo.add %1325, %1327 : tensor<3xui32>
      %1329 = stablehlo.shift_left %1327, %c_63 : tensor<3xui32>
      %1330 = stablehlo.shift_right_logical %1327, %c_67 : tensor<3xui32>
      %1331 = stablehlo.or %1329, %1330 : tensor<3xui32>
      %1332 = stablehlo.xor %1328, %1331 : tensor<3xui32>
      %1333 = stablehlo.add %1328, %1332 : tensor<3xui32>
      %1334 = stablehlo.shift_left %1332, %c_62 : tensor<3xui32>
      %1335 = stablehlo.shift_right_logical %1332, %c_58 : tensor<3xui32>
      %1336 = stablehlo.or %1334, %1335 : tensor<3xui32>
      %1337 = stablehlo.xor %1333, %1336 : tensor<3xui32>
      %1338 = stablehlo.add %1333, %1337 : tensor<3xui32>
      %1339 = stablehlo.shift_left %1337, %c_61 : tensor<3xui32>
      %1340 = stablehlo.shift_right_logical %1337, %c_61 : tensor<3xui32>
      %1341 = stablehlo.or %1339, %1340 : tensor<3xui32>
      %1342 = stablehlo.xor %1338, %1341 : tensor<3xui32>
      %1343 = stablehlo.add %1338, %1342 : tensor<3xui32>
      %1344 = stablehlo.shift_left %1342, %c_60 : tensor<3xui32>
      %1345 = stablehlo.shift_right_logical %1342, %c_30 : tensor<3xui32>
      %1346 = stablehlo.or %1344, %1345 : tensor<3xui32>
      %1347 = stablehlo.xor %1343, %1346 : tensor<3xui32>
      %1348 = stablehlo.add %1343, %1303 : tensor<3xui32>
      %1349 = stablehlo.add %1347, %1182 : tensor<3xui32>
      %1350 = stablehlo.add %1349, %c_59 : tensor<3xui32>
      %1351 = stablehlo.add %1348, %1350 : tensor<3xui32>
      %1352 = stablehlo.shift_left %1350, %c_68 : tensor<3xui32>
      %1353 = stablehlo.shift_right_logical %1350, %c_31 : tensor<3xui32>
      %1354 = stablehlo.or %1352, %1353 : tensor<3xui32>
      %1355 = stablehlo.xor %1351, %1354 : tensor<3xui32>
      %1356 = stablehlo.add %1351, %1355 : tensor<3xui32>
      %1357 = stablehlo.shift_left %1355, %c_67 : tensor<3xui32>
      %1358 = stablehlo.shift_right_logical %1355, %c_63 : tensor<3xui32>
      %1359 = stablehlo.or %1357, %1358 : tensor<3xui32>
      %1360 = stablehlo.xor %1356, %1359 : tensor<3xui32>
      %1361 = stablehlo.add %1356, %1360 : tensor<3xui32>
      %1362 = stablehlo.shift_left %1360, %c_66 : tensor<3xui32>
      %1363 = stablehlo.shift_right_logical %1360, %c_65 : tensor<3xui32>
      %1364 = stablehlo.or %1362, %1363 : tensor<3xui32>
      %1365 = stablehlo.xor %1361, %1364 : tensor<3xui32>
      %1366 = stablehlo.add %1361, %1365 : tensor<3xui32>
      %1367 = stablehlo.shift_left %1365, %c_65 : tensor<3xui32>
      %1368 = stablehlo.shift_right_logical %1365, %c_66 : tensor<3xui32>
      %1369 = stablehlo.or %1367, %1368 : tensor<3xui32>
      %1370 = stablehlo.xor %1366, %1369 : tensor<3xui32>
      %1371 = stablehlo.add %1366, %1182 : tensor<3xui32>
      %1372 = stablehlo.add %1370, %1183 : tensor<3xui32>
      %1373 = stablehlo.add %1372, %c_58 : tensor<3xui32>
      %1374 = stablehlo.add %1371, %1373 : tensor<3xui32>
      %1375 = stablehlo.shift_left %1373, %c_63 : tensor<3xui32>
      %1376 = stablehlo.shift_right_logical %1373, %c_67 : tensor<3xui32>
      %1377 = stablehlo.or %1375, %1376 : tensor<3xui32>
      %1378 = stablehlo.xor %1374, %1377 : tensor<3xui32>
      %1379 = stablehlo.add %1374, %1378 : tensor<3xui32>
      %1380 = stablehlo.shift_left %1378, %c_62 : tensor<3xui32>
      %1381 = stablehlo.shift_right_logical %1378, %c_58 : tensor<3xui32>
      %1382 = stablehlo.or %1380, %1381 : tensor<3xui32>
      %1383 = stablehlo.xor %1379, %1382 : tensor<3xui32>
      %1384 = stablehlo.add %1379, %1383 : tensor<3xui32>
      %1385 = stablehlo.shift_left %1383, %c_61 : tensor<3xui32>
      %1386 = stablehlo.shift_right_logical %1383, %c_61 : tensor<3xui32>
      %1387 = stablehlo.or %1385, %1386 : tensor<3xui32>
      %1388 = stablehlo.xor %1384, %1387 : tensor<3xui32>
      %1389 = stablehlo.add %1384, %1388 : tensor<3xui32>
      %1390 = stablehlo.shift_left %1388, %c_60 : tensor<3xui32>
      %1391 = stablehlo.shift_right_logical %1388, %c_30 : tensor<3xui32>
      %1392 = stablehlo.or %1390, %1391 : tensor<3xui32>
      %1393 = stablehlo.xor %1389, %1392 : tensor<3xui32>
      %1394 = stablehlo.add %1389, %1183 : tensor<3xui32>
      %1395 = stablehlo.add %1393, %1303 : tensor<3xui32>
      %1396 = stablehlo.add %1395, %c_57 : tensor<3xui32>
      %1397 = stablehlo.add %1394, %1396 : tensor<3xui32>
      %1398 = stablehlo.shift_left %1396, %c_68 : tensor<3xui32>
      %1399 = stablehlo.shift_right_logical %1396, %c_31 : tensor<3xui32>
      %1400 = stablehlo.or %1398, %1399 : tensor<3xui32>
      %1401 = stablehlo.xor %1397, %1400 : tensor<3xui32>
      %1402 = stablehlo.add %1397, %1401 : tensor<3xui32>
      %1403 = stablehlo.shift_left %1401, %c_67 : tensor<3xui32>
      %1404 = stablehlo.shift_right_logical %1401, %c_63 : tensor<3xui32>
      %1405 = stablehlo.or %1403, %1404 : tensor<3xui32>
      %1406 = stablehlo.xor %1402, %1405 : tensor<3xui32>
      %1407 = stablehlo.add %1402, %1406 : tensor<3xui32>
      %1408 = stablehlo.shift_left %1406, %c_66 : tensor<3xui32>
      %1409 = stablehlo.shift_right_logical %1406, %c_65 : tensor<3xui32>
      %1410 = stablehlo.or %1408, %1409 : tensor<3xui32>
      %1411 = stablehlo.xor %1407, %1410 : tensor<3xui32>
      %1412 = stablehlo.add %1407, %1411 : tensor<3xui32>
      %1413 = stablehlo.shift_left %1411, %c_65 : tensor<3xui32>
      %1414 = stablehlo.shift_right_logical %1411, %c_66 : tensor<3xui32>
      %1415 = stablehlo.or %1413, %1414 : tensor<3xui32>
      %1416 = stablehlo.xor %1412, %1415 : tensor<3xui32>
      %1417 = stablehlo.add %1412, %1303 : tensor<3xui32>
      %1418 = stablehlo.add %1416, %1182 : tensor<3xui32>
      %1419 = stablehlo.add %1418, %c_56 : tensor<3xui32>
      %1420 = stablehlo.slice %1299 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
      %1421 = stablehlo.slice %1301 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
      %1422 = stablehlo.slice %1417 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
      %1423 = stablehlo.slice %1419 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
      %1424 = stablehlo.concatenate %1420, %1421, %1422, %1423, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
      %1425 = stablehlo.reshape %1424 : (tensor<4xui32>) -> tensor<2x2xui32>
      %1426 = stablehlo.bitcast_convert %1425 : (tensor<2x2xui32>) -> tensor<2xui64>
      %1427 = stablehlo.slice %1299 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
      %1428 = stablehlo.slice %1301 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
      %1429 = stablehlo.slice %1417 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
      %1430 = stablehlo.slice %1419 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
      %1431 = stablehlo.concatenate %1427, %1428, %1429, %1430, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
      %1432 = stablehlo.reshape %1431 : (tensor<4xui32>) -> tensor<2x2xui32>
      %1433 = stablehlo.bitcast_convert %1432 : (tensor<2x2xui32>) -> tensor<2xui64>
      %1434 = stablehlo.slice %1299 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
      %1435 = stablehlo.slice %1301 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
      %1436 = stablehlo.slice %1417 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
      %1437 = stablehlo.slice %1419 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
      %1438 = stablehlo.concatenate %1434, %1435, %1436, %1437, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
      %1439 = stablehlo.reshape %1438 : (tensor<4xui32>) -> tensor<2x2xui32>
      %1440 = stablehlo.bitcast_convert %1439 : (tensor<2x2xui32>) -> tensor<2xui64>
      %output_state, %output = stablehlo.rng_bit_generator %1433, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1x3xui64>)
      %1441 = stablehlo.shift_right_logical %output, %c_55 : tensor<1x3xui64>
      %1442 = stablehlo.or %1441, %c_54 : tensor<1x3xui64>
      %1443 = stablehlo.bitcast_convert %1442 : (tensor<1x3xui64>) -> tensor<1x3xf64>
      %1444 = stablehlo.subtract %1443, %cst_53 : tensor<1x3xf64>
      %1445 = stablehlo.multiply %1444, %cst_52 : tensor<1x3xf64>
      %1446 = stablehlo.add %1445, %cst_51 : tensor<1x3xf64>
      %1447 = stablehlo.maximum %1446, %cst_51 : tensor<1x3xf64>
      %1448 = chlo.erf_inv %1447 : tensor<1x3xf64> -> tensor<1x3xf64>
      %1449 = stablehlo.dot_general %1448, %13, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
      %1450 = stablehlo.multiply %cst_24, %1449 : tensor<1x3xf64>
      %1451 = stablehlo.dot_general %1450, %arg3, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
      %1452 = stablehlo.dot_general %1450, %1451, contracting_dims = [0, 1] x [0, 1] {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<f64>
      %1453 = stablehlo.multiply %1452, %cst_88 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
      %1454 = stablehlo.add %iterArg_101, %1453 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
      %1455:17 = stablehlo.while(%iterArg_106 = %iterArg_99, %iterArg_107 = %1450, %iterArg_108 = %iterArg_100, %iterArg_109 = %iterArg_99, %iterArg_110 = %1450, %iterArg_111 = %iterArg_100, %iterArg_112 = %iterArg_99, %iterArg_113 = %iterArg_100, %iterArg_114 = %iterArg_101, %iterArg_115 = %1454, %iterArg_116 = %c_93, %iterArg_117 = %cst_90, %iterArg_118 = %c_87, %iterArg_119 = %c_87, %iterArg_120 = %cst_90, %iterArg_121 = %1450, %iterArg_122 = %1440) : tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x3xf64>, tensor<2xui64>
      cond {
        %1465 = stablehlo.compare LT, %iterArg_116, %c_86, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %1466 = stablehlo.not %iterArg_118 : tensor<i1>
        %1467 = stablehlo.not %iterArg_119 : tensor<i1>
        %1468 = stablehlo.and %1465, %1466 : tensor<i1>
        %1469 = stablehlo.and %1468, %1467 : tensor<i1>
        stablehlo.return %1469 : tensor<i1>
      } do {
        %1465 = stablehlo.bitcast_convert %iterArg_122 : (tensor<2xui64>) -> tensor<2x2xui32>
        %1466 = stablehlo.reshape %1465 : (tensor<2x2xui32>) -> tensor<4xui32>
        %1467 = stablehlo.slice %1466 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
        %1468 = stablehlo.slice %1466 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
        %1469 = stablehlo.slice %1466 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
        %1470 = stablehlo.slice %1466 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
        %1471 = stablehlo.broadcast_in_dim %1467, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
        %1472 = stablehlo.broadcast_in_dim %1468, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
        %1473 = stablehlo.broadcast_in_dim %1469, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
        %1474 = stablehlo.broadcast_in_dim %1470, dims = [0] : (tensor<1xui32>) -> tensor<3xui32>
        %1475 = stablehlo.xor %1471, %1472 : tensor<3xui32>
        %1476 = stablehlo.xor %1475, %c_69 : tensor<3xui32>
        %1477 = stablehlo.add %c_32, %1472 : tensor<3xui32>
        %1478 = stablehlo.add %1471, %1477 : tensor<3xui32>
        %1479 = stablehlo.shift_left %1477, %c_68 : tensor<3xui32>
        %1480 = stablehlo.shift_right_logical %1477, %c_31 : tensor<3xui32>
        %1481 = stablehlo.or %1479, %1480 : tensor<3xui32>
        %1482 = stablehlo.xor %1478, %1481 : tensor<3xui32>
        %1483 = stablehlo.add %1478, %1482 : tensor<3xui32>
        %1484 = stablehlo.shift_left %1482, %c_67 : tensor<3xui32>
        %1485 = stablehlo.shift_right_logical %1482, %c_63 : tensor<3xui32>
        %1486 = stablehlo.or %1484, %1485 : tensor<3xui32>
        %1487 = stablehlo.xor %1483, %1486 : tensor<3xui32>
        %1488 = stablehlo.add %1483, %1487 : tensor<3xui32>
        %1489 = stablehlo.shift_left %1487, %c_66 : tensor<3xui32>
        %1490 = stablehlo.shift_right_logical %1487, %c_65 : tensor<3xui32>
        %1491 = stablehlo.or %1489, %1490 : tensor<3xui32>
        %1492 = stablehlo.xor %1488, %1491 : tensor<3xui32>
        %1493 = stablehlo.add %1488, %1492 : tensor<3xui32>
        %1494 = stablehlo.shift_left %1492, %c_65 : tensor<3xui32>
        %1495 = stablehlo.shift_right_logical %1492, %c_66 : tensor<3xui32>
        %1496 = stablehlo.or %1494, %1495 : tensor<3xui32>
        %1497 = stablehlo.xor %1493, %1496 : tensor<3xui32>
        %1498 = stablehlo.add %1493, %1472 : tensor<3xui32>
        %1499 = stablehlo.add %1497, %1476 : tensor<3xui32>
        %1500 = stablehlo.add %1499, %c_64 : tensor<3xui32>
        %1501 = stablehlo.add %1498, %1500 : tensor<3xui32>
        %1502 = stablehlo.shift_left %1500, %c_63 : tensor<3xui32>
        %1503 = stablehlo.shift_right_logical %1500, %c_67 : tensor<3xui32>
        %1504 = stablehlo.or %1502, %1503 : tensor<3xui32>
        %1505 = stablehlo.xor %1501, %1504 : tensor<3xui32>
        %1506 = stablehlo.add %1501, %1505 : tensor<3xui32>
        %1507 = stablehlo.shift_left %1505, %c_62 : tensor<3xui32>
        %1508 = stablehlo.shift_right_logical %1505, %c_58 : tensor<3xui32>
        %1509 = stablehlo.or %1507, %1508 : tensor<3xui32>
        %1510 = stablehlo.xor %1506, %1509 : tensor<3xui32>
        %1511 = stablehlo.add %1506, %1510 : tensor<3xui32>
        %1512 = stablehlo.shift_left %1510, %c_61 : tensor<3xui32>
        %1513 = stablehlo.shift_right_logical %1510, %c_61 : tensor<3xui32>
        %1514 = stablehlo.or %1512, %1513 : tensor<3xui32>
        %1515 = stablehlo.xor %1511, %1514 : tensor<3xui32>
        %1516 = stablehlo.add %1511, %1515 : tensor<3xui32>
        %1517 = stablehlo.shift_left %1515, %c_60 : tensor<3xui32>
        %1518 = stablehlo.shift_right_logical %1515, %c_30 : tensor<3xui32>
        %1519 = stablehlo.or %1517, %1518 : tensor<3xui32>
        %1520 = stablehlo.xor %1516, %1519 : tensor<3xui32>
        %1521 = stablehlo.add %1516, %1476 : tensor<3xui32>
        %1522 = stablehlo.add %1520, %1471 : tensor<3xui32>
        %1523 = stablehlo.add %1522, %c_59 : tensor<3xui32>
        %1524 = stablehlo.add %1521, %1523 : tensor<3xui32>
        %1525 = stablehlo.shift_left %1523, %c_68 : tensor<3xui32>
        %1526 = stablehlo.shift_right_logical %1523, %c_31 : tensor<3xui32>
        %1527 = stablehlo.or %1525, %1526 : tensor<3xui32>
        %1528 = stablehlo.xor %1524, %1527 : tensor<3xui32>
        %1529 = stablehlo.add %1524, %1528 : tensor<3xui32>
        %1530 = stablehlo.shift_left %1528, %c_67 : tensor<3xui32>
        %1531 = stablehlo.shift_right_logical %1528, %c_63 : tensor<3xui32>
        %1532 = stablehlo.or %1530, %1531 : tensor<3xui32>
        %1533 = stablehlo.xor %1529, %1532 : tensor<3xui32>
        %1534 = stablehlo.add %1529, %1533 : tensor<3xui32>
        %1535 = stablehlo.shift_left %1533, %c_66 : tensor<3xui32>
        %1536 = stablehlo.shift_right_logical %1533, %c_65 : tensor<3xui32>
        %1537 = stablehlo.or %1535, %1536 : tensor<3xui32>
        %1538 = stablehlo.xor %1534, %1537 : tensor<3xui32>
        %1539 = stablehlo.add %1534, %1538 : tensor<3xui32>
        %1540 = stablehlo.shift_left %1538, %c_65 : tensor<3xui32>
        %1541 = stablehlo.shift_right_logical %1538, %c_66 : tensor<3xui32>
        %1542 = stablehlo.or %1540, %1541 : tensor<3xui32>
        %1543 = stablehlo.xor %1539, %1542 : tensor<3xui32>
        %1544 = stablehlo.add %1539, %1471 : tensor<3xui32>
        %1545 = stablehlo.add %1543, %1472 : tensor<3xui32>
        %1546 = stablehlo.add %1545, %c_58 : tensor<3xui32>
        %1547 = stablehlo.add %1544, %1546 : tensor<3xui32>
        %1548 = stablehlo.shift_left %1546, %c_63 : tensor<3xui32>
        %1549 = stablehlo.shift_right_logical %1546, %c_67 : tensor<3xui32>
        %1550 = stablehlo.or %1548, %1549 : tensor<3xui32>
        %1551 = stablehlo.xor %1547, %1550 : tensor<3xui32>
        %1552 = stablehlo.add %1547, %1551 : tensor<3xui32>
        %1553 = stablehlo.shift_left %1551, %c_62 : tensor<3xui32>
        %1554 = stablehlo.shift_right_logical %1551, %c_58 : tensor<3xui32>
        %1555 = stablehlo.or %1553, %1554 : tensor<3xui32>
        %1556 = stablehlo.xor %1552, %1555 : tensor<3xui32>
        %1557 = stablehlo.add %1552, %1556 : tensor<3xui32>
        %1558 = stablehlo.shift_left %1556, %c_61 : tensor<3xui32>
        %1559 = stablehlo.shift_right_logical %1556, %c_61 : tensor<3xui32>
        %1560 = stablehlo.or %1558, %1559 : tensor<3xui32>
        %1561 = stablehlo.xor %1557, %1560 : tensor<3xui32>
        %1562 = stablehlo.add %1557, %1561 : tensor<3xui32>
        %1563 = stablehlo.shift_left %1561, %c_60 : tensor<3xui32>
        %1564 = stablehlo.shift_right_logical %1561, %c_30 : tensor<3xui32>
        %1565 = stablehlo.or %1563, %1564 : tensor<3xui32>
        %1566 = stablehlo.xor %1562, %1565 : tensor<3xui32>
        %1567 = stablehlo.add %1562, %1472 : tensor<3xui32>
        %1568 = stablehlo.add %1566, %1476 : tensor<3xui32>
        %1569 = stablehlo.add %1568, %c_57 : tensor<3xui32>
        %1570 = stablehlo.add %1567, %1569 : tensor<3xui32>
        %1571 = stablehlo.shift_left %1569, %c_68 : tensor<3xui32>
        %1572 = stablehlo.shift_right_logical %1569, %c_31 : tensor<3xui32>
        %1573 = stablehlo.or %1571, %1572 : tensor<3xui32>
        %1574 = stablehlo.xor %1570, %1573 : tensor<3xui32>
        %1575 = stablehlo.add %1570, %1574 : tensor<3xui32>
        %1576 = stablehlo.shift_left %1574, %c_67 : tensor<3xui32>
        %1577 = stablehlo.shift_right_logical %1574, %c_63 : tensor<3xui32>
        %1578 = stablehlo.or %1576, %1577 : tensor<3xui32>
        %1579 = stablehlo.xor %1575, %1578 : tensor<3xui32>
        %1580 = stablehlo.add %1575, %1579 : tensor<3xui32>
        %1581 = stablehlo.shift_left %1579, %c_66 : tensor<3xui32>
        %1582 = stablehlo.shift_right_logical %1579, %c_65 : tensor<3xui32>
        %1583 = stablehlo.or %1581, %1582 : tensor<3xui32>
        %1584 = stablehlo.xor %1580, %1583 : tensor<3xui32>
        %1585 = stablehlo.add %1580, %1584 : tensor<3xui32>
        %1586 = stablehlo.shift_left %1584, %c_65 : tensor<3xui32>
        %1587 = stablehlo.shift_right_logical %1584, %c_66 : tensor<3xui32>
        %1588 = stablehlo.or %1586, %1587 : tensor<3xui32>
        %1589 = stablehlo.xor %1585, %1588 : tensor<3xui32>
        %1590 = stablehlo.add %1585, %1476 : tensor<3xui32>
        %1591 = stablehlo.add %1589, %1471 : tensor<3xui32>
        %1592 = stablehlo.add %1591, %c_56 : tensor<3xui32>
        %1593 = stablehlo.xor %1473, %1474 : tensor<3xui32>
        %1594 = stablehlo.xor %1593, %c_69 : tensor<3xui32>
        %1595 = stablehlo.add %c_32, %1474 : tensor<3xui32>
        %1596 = stablehlo.add %1473, %1595 : tensor<3xui32>
        %1597 = stablehlo.shift_left %1595, %c_68 : tensor<3xui32>
        %1598 = stablehlo.shift_right_logical %1595, %c_31 : tensor<3xui32>
        %1599 = stablehlo.or %1597, %1598 : tensor<3xui32>
        %1600 = stablehlo.xor %1596, %1599 : tensor<3xui32>
        %1601 = stablehlo.add %1596, %1600 : tensor<3xui32>
        %1602 = stablehlo.shift_left %1600, %c_67 : tensor<3xui32>
        %1603 = stablehlo.shift_right_logical %1600, %c_63 : tensor<3xui32>
        %1604 = stablehlo.or %1602, %1603 : tensor<3xui32>
        %1605 = stablehlo.xor %1601, %1604 : tensor<3xui32>
        %1606 = stablehlo.add %1601, %1605 : tensor<3xui32>
        %1607 = stablehlo.shift_left %1605, %c_66 : tensor<3xui32>
        %1608 = stablehlo.shift_right_logical %1605, %c_65 : tensor<3xui32>
        %1609 = stablehlo.or %1607, %1608 : tensor<3xui32>
        %1610 = stablehlo.xor %1606, %1609 : tensor<3xui32>
        %1611 = stablehlo.add %1606, %1610 : tensor<3xui32>
        %1612 = stablehlo.shift_left %1610, %c_65 : tensor<3xui32>
        %1613 = stablehlo.shift_right_logical %1610, %c_66 : tensor<3xui32>
        %1614 = stablehlo.or %1612, %1613 : tensor<3xui32>
        %1615 = stablehlo.xor %1611, %1614 : tensor<3xui32>
        %1616 = stablehlo.add %1611, %1474 : tensor<3xui32>
        %1617 = stablehlo.add %1615, %1594 : tensor<3xui32>
        %1618 = stablehlo.add %1617, %c_64 : tensor<3xui32>
        %1619 = stablehlo.add %1616, %1618 : tensor<3xui32>
        %1620 = stablehlo.shift_left %1618, %c_63 : tensor<3xui32>
        %1621 = stablehlo.shift_right_logical %1618, %c_67 : tensor<3xui32>
        %1622 = stablehlo.or %1620, %1621 : tensor<3xui32>
        %1623 = stablehlo.xor %1619, %1622 : tensor<3xui32>
        %1624 = stablehlo.add %1619, %1623 : tensor<3xui32>
        %1625 = stablehlo.shift_left %1623, %c_62 : tensor<3xui32>
        %1626 = stablehlo.shift_right_logical %1623, %c_58 : tensor<3xui32>
        %1627 = stablehlo.or %1625, %1626 : tensor<3xui32>
        %1628 = stablehlo.xor %1624, %1627 : tensor<3xui32>
        %1629 = stablehlo.add %1624, %1628 : tensor<3xui32>
        %1630 = stablehlo.shift_left %1628, %c_61 : tensor<3xui32>
        %1631 = stablehlo.shift_right_logical %1628, %c_61 : tensor<3xui32>
        %1632 = stablehlo.or %1630, %1631 : tensor<3xui32>
        %1633 = stablehlo.xor %1629, %1632 : tensor<3xui32>
        %1634 = stablehlo.add %1629, %1633 : tensor<3xui32>
        %1635 = stablehlo.shift_left %1633, %c_60 : tensor<3xui32>
        %1636 = stablehlo.shift_right_logical %1633, %c_30 : tensor<3xui32>
        %1637 = stablehlo.or %1635, %1636 : tensor<3xui32>
        %1638 = stablehlo.xor %1634, %1637 : tensor<3xui32>
        %1639 = stablehlo.add %1634, %1594 : tensor<3xui32>
        %1640 = stablehlo.add %1638, %1473 : tensor<3xui32>
        %1641 = stablehlo.add %1640, %c_59 : tensor<3xui32>
        %1642 = stablehlo.add %1639, %1641 : tensor<3xui32>
        %1643 = stablehlo.shift_left %1641, %c_68 : tensor<3xui32>
        %1644 = stablehlo.shift_right_logical %1641, %c_31 : tensor<3xui32>
        %1645 = stablehlo.or %1643, %1644 : tensor<3xui32>
        %1646 = stablehlo.xor %1642, %1645 : tensor<3xui32>
        %1647 = stablehlo.add %1642, %1646 : tensor<3xui32>
        %1648 = stablehlo.shift_left %1646, %c_67 : tensor<3xui32>
        %1649 = stablehlo.shift_right_logical %1646, %c_63 : tensor<3xui32>
        %1650 = stablehlo.or %1648, %1649 : tensor<3xui32>
        %1651 = stablehlo.xor %1647, %1650 : tensor<3xui32>
        %1652 = stablehlo.add %1647, %1651 : tensor<3xui32>
        %1653 = stablehlo.shift_left %1651, %c_66 : tensor<3xui32>
        %1654 = stablehlo.shift_right_logical %1651, %c_65 : tensor<3xui32>
        %1655 = stablehlo.or %1653, %1654 : tensor<3xui32>
        %1656 = stablehlo.xor %1652, %1655 : tensor<3xui32>
        %1657 = stablehlo.add %1652, %1656 : tensor<3xui32>
        %1658 = stablehlo.shift_left %1656, %c_65 : tensor<3xui32>
        %1659 = stablehlo.shift_right_logical %1656, %c_66 : tensor<3xui32>
        %1660 = stablehlo.or %1658, %1659 : tensor<3xui32>
        %1661 = stablehlo.xor %1657, %1660 : tensor<3xui32>
        %1662 = stablehlo.add %1657, %1473 : tensor<3xui32>
        %1663 = stablehlo.add %1661, %1474 : tensor<3xui32>
        %1664 = stablehlo.add %1663, %c_58 : tensor<3xui32>
        %1665 = stablehlo.add %1662, %1664 : tensor<3xui32>
        %1666 = stablehlo.shift_left %1664, %c_63 : tensor<3xui32>
        %1667 = stablehlo.shift_right_logical %1664, %c_67 : tensor<3xui32>
        %1668 = stablehlo.or %1666, %1667 : tensor<3xui32>
        %1669 = stablehlo.xor %1665, %1668 : tensor<3xui32>
        %1670 = stablehlo.add %1665, %1669 : tensor<3xui32>
        %1671 = stablehlo.shift_left %1669, %c_62 : tensor<3xui32>
        %1672 = stablehlo.shift_right_logical %1669, %c_58 : tensor<3xui32>
        %1673 = stablehlo.or %1671, %1672 : tensor<3xui32>
        %1674 = stablehlo.xor %1670, %1673 : tensor<3xui32>
        %1675 = stablehlo.add %1670, %1674 : tensor<3xui32>
        %1676 = stablehlo.shift_left %1674, %c_61 : tensor<3xui32>
        %1677 = stablehlo.shift_right_logical %1674, %c_61 : tensor<3xui32>
        %1678 = stablehlo.or %1676, %1677 : tensor<3xui32>
        %1679 = stablehlo.xor %1675, %1678 : tensor<3xui32>
        %1680 = stablehlo.add %1675, %1679 : tensor<3xui32>
        %1681 = stablehlo.shift_left %1679, %c_60 : tensor<3xui32>
        %1682 = stablehlo.shift_right_logical %1679, %c_30 : tensor<3xui32>
        %1683 = stablehlo.or %1681, %1682 : tensor<3xui32>
        %1684 = stablehlo.xor %1680, %1683 : tensor<3xui32>
        %1685 = stablehlo.add %1680, %1474 : tensor<3xui32>
        %1686 = stablehlo.add %1684, %1594 : tensor<3xui32>
        %1687 = stablehlo.add %1686, %c_57 : tensor<3xui32>
        %1688 = stablehlo.add %1685, %1687 : tensor<3xui32>
        %1689 = stablehlo.shift_left %1687, %c_68 : tensor<3xui32>
        %1690 = stablehlo.shift_right_logical %1687, %c_31 : tensor<3xui32>
        %1691 = stablehlo.or %1689, %1690 : tensor<3xui32>
        %1692 = stablehlo.xor %1688, %1691 : tensor<3xui32>
        %1693 = stablehlo.add %1688, %1692 : tensor<3xui32>
        %1694 = stablehlo.shift_left %1692, %c_67 : tensor<3xui32>
        %1695 = stablehlo.shift_right_logical %1692, %c_63 : tensor<3xui32>
        %1696 = stablehlo.or %1694, %1695 : tensor<3xui32>
        %1697 = stablehlo.xor %1693, %1696 : tensor<3xui32>
        %1698 = stablehlo.add %1693, %1697 : tensor<3xui32>
        %1699 = stablehlo.shift_left %1697, %c_66 : tensor<3xui32>
        %1700 = stablehlo.shift_right_logical %1697, %c_65 : tensor<3xui32>
        %1701 = stablehlo.or %1699, %1700 : tensor<3xui32>
        %1702 = stablehlo.xor %1698, %1701 : tensor<3xui32>
        %1703 = stablehlo.add %1698, %1702 : tensor<3xui32>
        %1704 = stablehlo.shift_left %1702, %c_65 : tensor<3xui32>
        %1705 = stablehlo.shift_right_logical %1702, %c_66 : tensor<3xui32>
        %1706 = stablehlo.or %1704, %1705 : tensor<3xui32>
        %1707 = stablehlo.xor %1703, %1706 : tensor<3xui32>
        %1708 = stablehlo.add %1703, %1594 : tensor<3xui32>
        %1709 = stablehlo.add %1707, %1473 : tensor<3xui32>
        %1710 = stablehlo.add %1709, %c_56 : tensor<3xui32>
        %1711 = stablehlo.slice %1590 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
        %1712 = stablehlo.slice %1592 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
        %1713 = stablehlo.slice %1708 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
        %1714 = stablehlo.slice %1710 [0:1] : (tensor<3xui32>) -> tensor<1xui32>
        %1715 = stablehlo.concatenate %1711, %1712, %1713, %1714, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %1716 = stablehlo.reshape %1715 : (tensor<4xui32>) -> tensor<2x2xui32>
        %1717 = stablehlo.bitcast_convert %1716 : (tensor<2x2xui32>) -> tensor<2xui64>
        %1718 = stablehlo.slice %1590 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
        %1719 = stablehlo.slice %1592 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
        %1720 = stablehlo.slice %1708 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
        %1721 = stablehlo.slice %1710 [1:2] : (tensor<3xui32>) -> tensor<1xui32>
        %1722 = stablehlo.concatenate %1718, %1719, %1720, %1721, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %1723 = stablehlo.reshape %1722 : (tensor<4xui32>) -> tensor<2x2xui32>
        %1724 = stablehlo.bitcast_convert %1723 : (tensor<2x2xui32>) -> tensor<2xui64>
        %1725 = stablehlo.slice %1590 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
        %1726 = stablehlo.slice %1592 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
        %1727 = stablehlo.slice %1708 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
        %1728 = stablehlo.slice %1710 [2:3] : (tensor<3xui32>) -> tensor<1xui32>
        %output_state_123, %output_124 = stablehlo.rng_bit_generator %1724, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
        %1729 = stablehlo.shift_right_logical %output_124, %c_50 : tensor<ui64>
        %1730 = stablehlo.or %1729, %c_49 : tensor<ui64>
        %1731 = stablehlo.bitcast_convert %1730 : (tensor<ui64>) -> tensor<f64>
        %1732 = stablehlo.subtract %1731, %cst_95 {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
        %1733 = stablehlo.compare LT, %1732, %cst_88, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %1734 = stablehlo.broadcast_in_dim %1725, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1735 = stablehlo.broadcast_in_dim %1726, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1736 = stablehlo.broadcast_in_dim %1727, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1737 = stablehlo.broadcast_in_dim %1728, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1738 = stablehlo.xor %1734, %1735 : tensor<2xui32>
        %1739 = stablehlo.xor %1738, %c_83 : tensor<2xui32>
        %1740 = stablehlo.add %c_37, %1735 : tensor<2xui32>
        %1741 = stablehlo.add %1734, %1740 : tensor<2xui32>
        %1742 = stablehlo.shift_left %1740, %c_82 : tensor<2xui32>
        %1743 = stablehlo.shift_right_logical %1740, %c_36 : tensor<2xui32>
        %1744 = stablehlo.or %1742, %1743 : tensor<2xui32>
        %1745 = stablehlo.xor %1741, %1744 : tensor<2xui32>
        %1746 = stablehlo.add %1741, %1745 : tensor<2xui32>
        %1747 = stablehlo.shift_left %1745, %c_81 : tensor<2xui32>
        %1748 = stablehlo.shift_right_logical %1745, %c_77 : tensor<2xui32>
        %1749 = stablehlo.or %1747, %1748 : tensor<2xui32>
        %1750 = stablehlo.xor %1746, %1749 : tensor<2xui32>
        %1751 = stablehlo.add %1746, %1750 : tensor<2xui32>
        %1752 = stablehlo.shift_left %1750, %c_80 : tensor<2xui32>
        %1753 = stablehlo.shift_right_logical %1750, %c_79 : tensor<2xui32>
        %1754 = stablehlo.or %1752, %1753 : tensor<2xui32>
        %1755 = stablehlo.xor %1751, %1754 : tensor<2xui32>
        %1756 = stablehlo.add %1751, %1755 : tensor<2xui32>
        %1757 = stablehlo.shift_left %1755, %c_79 : tensor<2xui32>
        %1758 = stablehlo.shift_right_logical %1755, %c_80 : tensor<2xui32>
        %1759 = stablehlo.or %1757, %1758 : tensor<2xui32>
        %1760 = stablehlo.xor %1756, %1759 : tensor<2xui32>
        %1761 = stablehlo.add %1756, %1735 : tensor<2xui32>
        %1762 = stablehlo.add %1760, %1739 : tensor<2xui32>
        %1763 = stablehlo.add %1762, %c_78 : tensor<2xui32>
        %1764 = stablehlo.add %1761, %1763 : tensor<2xui32>
        %1765 = stablehlo.shift_left %1763, %c_77 : tensor<2xui32>
        %1766 = stablehlo.shift_right_logical %1763, %c_81 : tensor<2xui32>
        %1767 = stablehlo.or %1765, %1766 : tensor<2xui32>
        %1768 = stablehlo.xor %1764, %1767 : tensor<2xui32>
        %1769 = stablehlo.add %1764, %1768 : tensor<2xui32>
        %1770 = stablehlo.shift_left %1768, %c_76 : tensor<2xui32>
        %1771 = stablehlo.shift_right_logical %1768, %c_72 : tensor<2xui32>
        %1772 = stablehlo.or %1770, %1771 : tensor<2xui32>
        %1773 = stablehlo.xor %1769, %1772 : tensor<2xui32>
        %1774 = stablehlo.add %1769, %1773 : tensor<2xui32>
        %1775 = stablehlo.shift_left %1773, %c_75 : tensor<2xui32>
        %1776 = stablehlo.shift_right_logical %1773, %c_75 : tensor<2xui32>
        %1777 = stablehlo.or %1775, %1776 : tensor<2xui32>
        %1778 = stablehlo.xor %1774, %1777 : tensor<2xui32>
        %1779 = stablehlo.add %1774, %1778 : tensor<2xui32>
        %1780 = stablehlo.shift_left %1778, %c_74 : tensor<2xui32>
        %1781 = stablehlo.shift_right_logical %1778, %c_35 : tensor<2xui32>
        %1782 = stablehlo.or %1780, %1781 : tensor<2xui32>
        %1783 = stablehlo.xor %1779, %1782 : tensor<2xui32>
        %1784 = stablehlo.add %1779, %1739 : tensor<2xui32>
        %1785 = stablehlo.add %1783, %1734 : tensor<2xui32>
        %1786 = stablehlo.add %1785, %c_73 : tensor<2xui32>
        %1787 = stablehlo.add %1784, %1786 : tensor<2xui32>
        %1788 = stablehlo.shift_left %1786, %c_82 : tensor<2xui32>
        %1789 = stablehlo.shift_right_logical %1786, %c_36 : tensor<2xui32>
        %1790 = stablehlo.or %1788, %1789 : tensor<2xui32>
        %1791 = stablehlo.xor %1787, %1790 : tensor<2xui32>
        %1792 = stablehlo.add %1787, %1791 : tensor<2xui32>
        %1793 = stablehlo.shift_left %1791, %c_81 : tensor<2xui32>
        %1794 = stablehlo.shift_right_logical %1791, %c_77 : tensor<2xui32>
        %1795 = stablehlo.or %1793, %1794 : tensor<2xui32>
        %1796 = stablehlo.xor %1792, %1795 : tensor<2xui32>
        %1797 = stablehlo.add %1792, %1796 : tensor<2xui32>
        %1798 = stablehlo.shift_left %1796, %c_80 : tensor<2xui32>
        %1799 = stablehlo.shift_right_logical %1796, %c_79 : tensor<2xui32>
        %1800 = stablehlo.or %1798, %1799 : tensor<2xui32>
        %1801 = stablehlo.xor %1797, %1800 : tensor<2xui32>
        %1802 = stablehlo.add %1797, %1801 : tensor<2xui32>
        %1803 = stablehlo.shift_left %1801, %c_79 : tensor<2xui32>
        %1804 = stablehlo.shift_right_logical %1801, %c_80 : tensor<2xui32>
        %1805 = stablehlo.or %1803, %1804 : tensor<2xui32>
        %1806 = stablehlo.xor %1802, %1805 : tensor<2xui32>
        %1807 = stablehlo.add %1802, %1734 : tensor<2xui32>
        %1808 = stablehlo.add %1806, %1735 : tensor<2xui32>
        %1809 = stablehlo.add %1808, %c_72 : tensor<2xui32>
        %1810 = stablehlo.add %1807, %1809 : tensor<2xui32>
        %1811 = stablehlo.shift_left %1809, %c_77 : tensor<2xui32>
        %1812 = stablehlo.shift_right_logical %1809, %c_81 : tensor<2xui32>
        %1813 = stablehlo.or %1811, %1812 : tensor<2xui32>
        %1814 = stablehlo.xor %1810, %1813 : tensor<2xui32>
        %1815 = stablehlo.add %1810, %1814 : tensor<2xui32>
        %1816 = stablehlo.shift_left %1814, %c_76 : tensor<2xui32>
        %1817 = stablehlo.shift_right_logical %1814, %c_72 : tensor<2xui32>
        %1818 = stablehlo.or %1816, %1817 : tensor<2xui32>
        %1819 = stablehlo.xor %1815, %1818 : tensor<2xui32>
        %1820 = stablehlo.add %1815, %1819 : tensor<2xui32>
        %1821 = stablehlo.shift_left %1819, %c_75 : tensor<2xui32>
        %1822 = stablehlo.shift_right_logical %1819, %c_75 : tensor<2xui32>
        %1823 = stablehlo.or %1821, %1822 : tensor<2xui32>
        %1824 = stablehlo.xor %1820, %1823 : tensor<2xui32>
        %1825 = stablehlo.add %1820, %1824 : tensor<2xui32>
        %1826 = stablehlo.shift_left %1824, %c_74 : tensor<2xui32>
        %1827 = stablehlo.shift_right_logical %1824, %c_35 : tensor<2xui32>
        %1828 = stablehlo.or %1826, %1827 : tensor<2xui32>
        %1829 = stablehlo.xor %1825, %1828 : tensor<2xui32>
        %1830 = stablehlo.add %1825, %1735 : tensor<2xui32>
        %1831 = stablehlo.add %1829, %1739 : tensor<2xui32>
        %1832 = stablehlo.add %1831, %c_71 : tensor<2xui32>
        %1833 = stablehlo.add %1830, %1832 : tensor<2xui32>
        %1834 = stablehlo.shift_left %1832, %c_82 : tensor<2xui32>
        %1835 = stablehlo.shift_right_logical %1832, %c_36 : tensor<2xui32>
        %1836 = stablehlo.or %1834, %1835 : tensor<2xui32>
        %1837 = stablehlo.xor %1833, %1836 : tensor<2xui32>
        %1838 = stablehlo.add %1833, %1837 : tensor<2xui32>
        %1839 = stablehlo.shift_left %1837, %c_81 : tensor<2xui32>
        %1840 = stablehlo.shift_right_logical %1837, %c_77 : tensor<2xui32>
        %1841 = stablehlo.or %1839, %1840 : tensor<2xui32>
        %1842 = stablehlo.xor %1838, %1841 : tensor<2xui32>
        %1843 = stablehlo.add %1838, %1842 : tensor<2xui32>
        %1844 = stablehlo.shift_left %1842, %c_80 : tensor<2xui32>
        %1845 = stablehlo.shift_right_logical %1842, %c_79 : tensor<2xui32>
        %1846 = stablehlo.or %1844, %1845 : tensor<2xui32>
        %1847 = stablehlo.xor %1843, %1846 : tensor<2xui32>
        %1848 = stablehlo.add %1843, %1847 : tensor<2xui32>
        %1849 = stablehlo.shift_left %1847, %c_79 : tensor<2xui32>
        %1850 = stablehlo.shift_right_logical %1847, %c_80 : tensor<2xui32>
        %1851 = stablehlo.or %1849, %1850 : tensor<2xui32>
        %1852 = stablehlo.xor %1848, %1851 : tensor<2xui32>
        %1853 = stablehlo.add %1848, %1739 : tensor<2xui32>
        %1854 = stablehlo.add %1852, %1734 : tensor<2xui32>
        %1855 = stablehlo.add %1854, %c_70 : tensor<2xui32>
        %1856 = stablehlo.xor %1736, %1737 : tensor<2xui32>
        %1857 = stablehlo.xor %1856, %c_83 : tensor<2xui32>
        %1858 = stablehlo.add %c_37, %1737 : tensor<2xui32>
        %1859 = stablehlo.add %1736, %1858 : tensor<2xui32>
        %1860 = stablehlo.shift_left %1858, %c_82 : tensor<2xui32>
        %1861 = stablehlo.shift_right_logical %1858, %c_36 : tensor<2xui32>
        %1862 = stablehlo.or %1860, %1861 : tensor<2xui32>
        %1863 = stablehlo.xor %1859, %1862 : tensor<2xui32>
        %1864 = stablehlo.add %1859, %1863 : tensor<2xui32>
        %1865 = stablehlo.shift_left %1863, %c_81 : tensor<2xui32>
        %1866 = stablehlo.shift_right_logical %1863, %c_77 : tensor<2xui32>
        %1867 = stablehlo.or %1865, %1866 : tensor<2xui32>
        %1868 = stablehlo.xor %1864, %1867 : tensor<2xui32>
        %1869 = stablehlo.add %1864, %1868 : tensor<2xui32>
        %1870 = stablehlo.shift_left %1868, %c_80 : tensor<2xui32>
        %1871 = stablehlo.shift_right_logical %1868, %c_79 : tensor<2xui32>
        %1872 = stablehlo.or %1870, %1871 : tensor<2xui32>
        %1873 = stablehlo.xor %1869, %1872 : tensor<2xui32>
        %1874 = stablehlo.add %1869, %1873 : tensor<2xui32>
        %1875 = stablehlo.shift_left %1873, %c_79 : tensor<2xui32>
        %1876 = stablehlo.shift_right_logical %1873, %c_80 : tensor<2xui32>
        %1877 = stablehlo.or %1875, %1876 : tensor<2xui32>
        %1878 = stablehlo.xor %1874, %1877 : tensor<2xui32>
        %1879 = stablehlo.add %1874, %1737 : tensor<2xui32>
        %1880 = stablehlo.add %1878, %1857 : tensor<2xui32>
        %1881 = stablehlo.add %1880, %c_78 : tensor<2xui32>
        %1882 = stablehlo.add %1879, %1881 : tensor<2xui32>
        %1883 = stablehlo.shift_left %1881, %c_77 : tensor<2xui32>
        %1884 = stablehlo.shift_right_logical %1881, %c_81 : tensor<2xui32>
        %1885 = stablehlo.or %1883, %1884 : tensor<2xui32>
        %1886 = stablehlo.xor %1882, %1885 : tensor<2xui32>
        %1887 = stablehlo.add %1882, %1886 : tensor<2xui32>
        %1888 = stablehlo.shift_left %1886, %c_76 : tensor<2xui32>
        %1889 = stablehlo.shift_right_logical %1886, %c_72 : tensor<2xui32>
        %1890 = stablehlo.or %1888, %1889 : tensor<2xui32>
        %1891 = stablehlo.xor %1887, %1890 : tensor<2xui32>
        %1892 = stablehlo.add %1887, %1891 : tensor<2xui32>
        %1893 = stablehlo.shift_left %1891, %c_75 : tensor<2xui32>
        %1894 = stablehlo.shift_right_logical %1891, %c_75 : tensor<2xui32>
        %1895 = stablehlo.or %1893, %1894 : tensor<2xui32>
        %1896 = stablehlo.xor %1892, %1895 : tensor<2xui32>
        %1897 = stablehlo.add %1892, %1896 : tensor<2xui32>
        %1898 = stablehlo.shift_left %1896, %c_74 : tensor<2xui32>
        %1899 = stablehlo.shift_right_logical %1896, %c_35 : tensor<2xui32>
        %1900 = stablehlo.or %1898, %1899 : tensor<2xui32>
        %1901 = stablehlo.xor %1897, %1900 : tensor<2xui32>
        %1902 = stablehlo.add %1897, %1857 : tensor<2xui32>
        %1903 = stablehlo.add %1901, %1736 : tensor<2xui32>
        %1904 = stablehlo.add %1903, %c_73 : tensor<2xui32>
        %1905 = stablehlo.add %1902, %1904 : tensor<2xui32>
        %1906 = stablehlo.shift_left %1904, %c_82 : tensor<2xui32>
        %1907 = stablehlo.shift_right_logical %1904, %c_36 : tensor<2xui32>
        %1908 = stablehlo.or %1906, %1907 : tensor<2xui32>
        %1909 = stablehlo.xor %1905, %1908 : tensor<2xui32>
        %1910 = stablehlo.add %1905, %1909 : tensor<2xui32>
        %1911 = stablehlo.shift_left %1909, %c_81 : tensor<2xui32>
        %1912 = stablehlo.shift_right_logical %1909, %c_77 : tensor<2xui32>
        %1913 = stablehlo.or %1911, %1912 : tensor<2xui32>
        %1914 = stablehlo.xor %1910, %1913 : tensor<2xui32>
        %1915 = stablehlo.add %1910, %1914 : tensor<2xui32>
        %1916 = stablehlo.shift_left %1914, %c_80 : tensor<2xui32>
        %1917 = stablehlo.shift_right_logical %1914, %c_79 : tensor<2xui32>
        %1918 = stablehlo.or %1916, %1917 : tensor<2xui32>
        %1919 = stablehlo.xor %1915, %1918 : tensor<2xui32>
        %1920 = stablehlo.add %1915, %1919 : tensor<2xui32>
        %1921 = stablehlo.shift_left %1919, %c_79 : tensor<2xui32>
        %1922 = stablehlo.shift_right_logical %1919, %c_80 : tensor<2xui32>
        %1923 = stablehlo.or %1921, %1922 : tensor<2xui32>
        %1924 = stablehlo.xor %1920, %1923 : tensor<2xui32>
        %1925 = stablehlo.add %1920, %1736 : tensor<2xui32>
        %1926 = stablehlo.add %1924, %1737 : tensor<2xui32>
        %1927 = stablehlo.add %1926, %c_72 : tensor<2xui32>
        %1928 = stablehlo.add %1925, %1927 : tensor<2xui32>
        %1929 = stablehlo.shift_left %1927, %c_77 : tensor<2xui32>
        %1930 = stablehlo.shift_right_logical %1927, %c_81 : tensor<2xui32>
        %1931 = stablehlo.or %1929, %1930 : tensor<2xui32>
        %1932 = stablehlo.xor %1928, %1931 : tensor<2xui32>
        %1933 = stablehlo.add %1928, %1932 : tensor<2xui32>
        %1934 = stablehlo.shift_left %1932, %c_76 : tensor<2xui32>
        %1935 = stablehlo.shift_right_logical %1932, %c_72 : tensor<2xui32>
        %1936 = stablehlo.or %1934, %1935 : tensor<2xui32>
        %1937 = stablehlo.xor %1933, %1936 : tensor<2xui32>
        %1938 = stablehlo.add %1933, %1937 : tensor<2xui32>
        %1939 = stablehlo.shift_left %1937, %c_75 : tensor<2xui32>
        %1940 = stablehlo.shift_right_logical %1937, %c_75 : tensor<2xui32>
        %1941 = stablehlo.or %1939, %1940 : tensor<2xui32>
        %1942 = stablehlo.xor %1938, %1941 : tensor<2xui32>
        %1943 = stablehlo.add %1938, %1942 : tensor<2xui32>
        %1944 = stablehlo.shift_left %1942, %c_74 : tensor<2xui32>
        %1945 = stablehlo.shift_right_logical %1942, %c_35 : tensor<2xui32>
        %1946 = stablehlo.or %1944, %1945 : tensor<2xui32>
        %1947 = stablehlo.xor %1943, %1946 : tensor<2xui32>
        %1948 = stablehlo.add %1943, %1737 : tensor<2xui32>
        %1949 = stablehlo.add %1947, %1857 : tensor<2xui32>
        %1950 = stablehlo.add %1949, %c_71 : tensor<2xui32>
        %1951 = stablehlo.add %1948, %1950 : tensor<2xui32>
        %1952 = stablehlo.shift_left %1950, %c_82 : tensor<2xui32>
        %1953 = stablehlo.shift_right_logical %1950, %c_36 : tensor<2xui32>
        %1954 = stablehlo.or %1952, %1953 : tensor<2xui32>
        %1955 = stablehlo.xor %1951, %1954 : tensor<2xui32>
        %1956 = stablehlo.add %1951, %1955 : tensor<2xui32>
        %1957 = stablehlo.shift_left %1955, %c_81 : tensor<2xui32>
        %1958 = stablehlo.shift_right_logical %1955, %c_77 : tensor<2xui32>
        %1959 = stablehlo.or %1957, %1958 : tensor<2xui32>
        %1960 = stablehlo.xor %1956, %1959 : tensor<2xui32>
        %1961 = stablehlo.add %1956, %1960 : tensor<2xui32>
        %1962 = stablehlo.shift_left %1960, %c_80 : tensor<2xui32>
        %1963 = stablehlo.shift_right_logical %1960, %c_79 : tensor<2xui32>
        %1964 = stablehlo.or %1962, %1963 : tensor<2xui32>
        %1965 = stablehlo.xor %1961, %1964 : tensor<2xui32>
        %1966 = stablehlo.add %1961, %1965 : tensor<2xui32>
        %1967 = stablehlo.shift_left %1965, %c_79 : tensor<2xui32>
        %1968 = stablehlo.shift_right_logical %1965, %c_80 : tensor<2xui32>
        %1969 = stablehlo.or %1967, %1968 : tensor<2xui32>
        %1970 = stablehlo.xor %1966, %1969 : tensor<2xui32>
        %1971 = stablehlo.add %1966, %1857 : tensor<2xui32>
        %1972 = stablehlo.add %1970, %1736 : tensor<2xui32>
        %1973 = stablehlo.add %1972, %c_70 : tensor<2xui32>
        %1974 = stablehlo.slice %1853 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1975 = stablehlo.slice %1855 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1976 = stablehlo.slice %1971 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1977 = stablehlo.slice %1973 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1978 = stablehlo.concatenate %1974, %1975, %1976, %1977, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %1979 = stablehlo.reshape %1978 : (tensor<4xui32>) -> tensor<2x2xui32>
        %1980 = stablehlo.bitcast_convert %1979 : (tensor<2x2xui32>) -> tensor<2xui64>
        %1981 = stablehlo.slice %1853 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1982 = stablehlo.slice %1855 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1983 = stablehlo.slice %1971 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1984 = stablehlo.slice %1973 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1985 = stablehlo.concatenate %1981, %1982, %1983, %1984, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %1986 = stablehlo.reshape %1985 : (tensor<4xui32>) -> tensor<2x2xui32>
        %1987 = stablehlo.bitcast_convert %1986 : (tensor<2x2xui32>) -> tensor<2xui64>
        %1988 = stablehlo.shift_left %c_92, %iterArg_116 : tensor<i64>
        %1989:21 = stablehlo.while(%iterArg_127 = %iterArg_106, %iterArg_128 = %iterArg_107, %iterArg_129 = %iterArg_108, %iterArg_130 = %iterArg_109, %iterArg_131 = %iterArg_110, %iterArg_132 = %iterArg_111, %iterArg_133 = %iterArg_112, %iterArg_134 = %iterArg_113, %iterArg_135 = %iterArg_114, %iterArg_136 = %iterArg_115, %iterArg_137 = %iterArg_116, %iterArg_138 = %iterArg_117, %iterArg_139 = %iterArg_118, %iterArg_140 = %iterArg_119, %iterArg_141 = %iterArg_120, %iterArg_142 = %c_93, %iterArg_143 = %iterArg_121, %iterArg_144 = %1980, %iterArg_145 = %cst_85, %iterArg_146 = %cst_85, %iterArg_147 = %c_93) : tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x3xf64>, tensor<2xui64>, tensor<5x3xf64>, tensor<5x3xf64>, tensor<i64> attributes {enzymexla.finite = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>]}
        cond {
          %2035 = stablehlo.compare LT, %iterArg_142, %1988, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
          %2036 = stablehlo.not %iterArg_139 : tensor<i1>
          %2037 = stablehlo.not %iterArg_140 : tensor<i1>
          %2038 = stablehlo.and %2035, %2036 : tensor<i1>
          %2039 = stablehlo.and %2038, %2037 : tensor<i1>
          stablehlo.return %2039 : tensor<i1>
        } do {
          %2035 = stablehlo.select %1733, %iterArg_130, %iterArg_127 : tensor<i1>, tensor<1x3xf64>
          %2036 = stablehlo.select %1733, %iterArg_131, %iterArg_128 : tensor<i1>, tensor<1x3xf64>
          %2037 = stablehlo.select %1733, %iterArg_132, %iterArg_129 : tensor<i1>, tensor<1x3xf64>
          %2038 = stablehlo.bitcast_convert %iterArg_144 : (tensor<2xui64>) -> tensor<2x2xui32>
          %2039 = stablehlo.reshape %2038 : (tensor<2x2xui32>) -> tensor<4xui32>
          %2040 = stablehlo.slice %2039 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
          %2041 = stablehlo.slice %2039 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
          %2042 = stablehlo.slice %2039 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
          %2043 = stablehlo.slice %2039 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
          %2044 = stablehlo.broadcast_in_dim %2040, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
          %2045 = stablehlo.broadcast_in_dim %2041, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
          %2046 = stablehlo.broadcast_in_dim %2042, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
          %2047 = stablehlo.broadcast_in_dim %2043, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
          %2048 = stablehlo.xor %2044, %2045 : tensor<2xui32>
          %2049 = stablehlo.xor %2048, %c_83 : tensor<2xui32>
          %2050 = stablehlo.add %c_37, %2045 : tensor<2xui32>
          %2051 = stablehlo.add %2044, %2050 : tensor<2xui32>
          %2052 = stablehlo.shift_left %2050, %c_82 : tensor<2xui32>
          %2053 = stablehlo.shift_right_logical %2050, %c_36 : tensor<2xui32>
          %2054 = stablehlo.or %2052, %2053 : tensor<2xui32>
          %2055 = stablehlo.xor %2051, %2054 : tensor<2xui32>
          %2056 = stablehlo.add %2051, %2055 : tensor<2xui32>
          %2057 = stablehlo.shift_left %2055, %c_81 : tensor<2xui32>
          %2058 = stablehlo.shift_right_logical %2055, %c_77 : tensor<2xui32>
          %2059 = stablehlo.or %2057, %2058 : tensor<2xui32>
          %2060 = stablehlo.xor %2056, %2059 : tensor<2xui32>
          %2061 = stablehlo.add %2056, %2060 : tensor<2xui32>
          %2062 = stablehlo.shift_left %2060, %c_80 : tensor<2xui32>
          %2063 = stablehlo.shift_right_logical %2060, %c_79 : tensor<2xui32>
          %2064 = stablehlo.or %2062, %2063 : tensor<2xui32>
          %2065 = stablehlo.xor %2061, %2064 : tensor<2xui32>
          %2066 = stablehlo.add %2061, %2065 : tensor<2xui32>
          %2067 = stablehlo.shift_left %2065, %c_79 : tensor<2xui32>
          %2068 = stablehlo.shift_right_logical %2065, %c_80 : tensor<2xui32>
          %2069 = stablehlo.or %2067, %2068 : tensor<2xui32>
          %2070 = stablehlo.xor %2066, %2069 : tensor<2xui32>
          %2071 = stablehlo.add %2066, %2045 : tensor<2xui32>
          %2072 = stablehlo.add %2070, %2049 : tensor<2xui32>
          %2073 = stablehlo.add %2072, %c_78 : tensor<2xui32>
          %2074 = stablehlo.add %2071, %2073 : tensor<2xui32>
          %2075 = stablehlo.shift_left %2073, %c_77 : tensor<2xui32>
          %2076 = stablehlo.shift_right_logical %2073, %c_81 : tensor<2xui32>
          %2077 = stablehlo.or %2075, %2076 : tensor<2xui32>
          %2078 = stablehlo.xor %2074, %2077 : tensor<2xui32>
          %2079 = stablehlo.add %2074, %2078 : tensor<2xui32>
          %2080 = stablehlo.shift_left %2078, %c_76 : tensor<2xui32>
          %2081 = stablehlo.shift_right_logical %2078, %c_72 : tensor<2xui32>
          %2082 = stablehlo.or %2080, %2081 : tensor<2xui32>
          %2083 = stablehlo.xor %2079, %2082 : tensor<2xui32>
          %2084 = stablehlo.add %2079, %2083 : tensor<2xui32>
          %2085 = stablehlo.shift_left %2083, %c_75 : tensor<2xui32>
          %2086 = stablehlo.shift_right_logical %2083, %c_75 : tensor<2xui32>
          %2087 = stablehlo.or %2085, %2086 : tensor<2xui32>
          %2088 = stablehlo.xor %2084, %2087 : tensor<2xui32>
          %2089 = stablehlo.add %2084, %2088 : tensor<2xui32>
          %2090 = stablehlo.shift_left %2088, %c_74 : tensor<2xui32>
          %2091 = stablehlo.shift_right_logical %2088, %c_35 : tensor<2xui32>
          %2092 = stablehlo.or %2090, %2091 : tensor<2xui32>
          %2093 = stablehlo.xor %2089, %2092 : tensor<2xui32>
          %2094 = stablehlo.add %2089, %2049 : tensor<2xui32>
          %2095 = stablehlo.add %2093, %2044 : tensor<2xui32>
          %2096 = stablehlo.add %2095, %c_73 : tensor<2xui32>
          %2097 = stablehlo.add %2094, %2096 : tensor<2xui32>
          %2098 = stablehlo.shift_left %2096, %c_82 : tensor<2xui32>
          %2099 = stablehlo.shift_right_logical %2096, %c_36 : tensor<2xui32>
          %2100 = stablehlo.or %2098, %2099 : tensor<2xui32>
          %2101 = stablehlo.xor %2097, %2100 : tensor<2xui32>
          %2102 = stablehlo.add %2097, %2101 : tensor<2xui32>
          %2103 = stablehlo.shift_left %2101, %c_81 : tensor<2xui32>
          %2104 = stablehlo.shift_right_logical %2101, %c_77 : tensor<2xui32>
          %2105 = stablehlo.or %2103, %2104 : tensor<2xui32>
          %2106 = stablehlo.xor %2102, %2105 : tensor<2xui32>
          %2107 = stablehlo.add %2102, %2106 : tensor<2xui32>
          %2108 = stablehlo.shift_left %2106, %c_80 : tensor<2xui32>
          %2109 = stablehlo.shift_right_logical %2106, %c_79 : tensor<2xui32>
          %2110 = stablehlo.or %2108, %2109 : tensor<2xui32>
          %2111 = stablehlo.xor %2107, %2110 : tensor<2xui32>
          %2112 = stablehlo.add %2107, %2111 : tensor<2xui32>
          %2113 = stablehlo.shift_left %2111, %c_79 : tensor<2xui32>
          %2114 = stablehlo.shift_right_logical %2111, %c_80 : tensor<2xui32>
          %2115 = stablehlo.or %2113, %2114 : tensor<2xui32>
          %2116 = stablehlo.xor %2112, %2115 : tensor<2xui32>
          %2117 = stablehlo.add %2112, %2044 : tensor<2xui32>
          %2118 = stablehlo.add %2116, %2045 : tensor<2xui32>
          %2119 = stablehlo.add %2118, %c_72 : tensor<2xui32>
          %2120 = stablehlo.add %2117, %2119 : tensor<2xui32>
          %2121 = stablehlo.shift_left %2119, %c_77 : tensor<2xui32>
          %2122 = stablehlo.shift_right_logical %2119, %c_81 : tensor<2xui32>
          %2123 = stablehlo.or %2121, %2122 : tensor<2xui32>
          %2124 = stablehlo.xor %2120, %2123 : tensor<2xui32>
          %2125 = stablehlo.add %2120, %2124 : tensor<2xui32>
          %2126 = stablehlo.shift_left %2124, %c_76 : tensor<2xui32>
          %2127 = stablehlo.shift_right_logical %2124, %c_72 : tensor<2xui32>
          %2128 = stablehlo.or %2126, %2127 : tensor<2xui32>
          %2129 = stablehlo.xor %2125, %2128 : tensor<2xui32>
          %2130 = stablehlo.add %2125, %2129 : tensor<2xui32>
          %2131 = stablehlo.shift_left %2129, %c_75 : tensor<2xui32>
          %2132 = stablehlo.shift_right_logical %2129, %c_75 : tensor<2xui32>
          %2133 = stablehlo.or %2131, %2132 : tensor<2xui32>
          %2134 = stablehlo.xor %2130, %2133 : tensor<2xui32>
          %2135 = stablehlo.add %2130, %2134 : tensor<2xui32>
          %2136 = stablehlo.shift_left %2134, %c_74 : tensor<2xui32>
          %2137 = stablehlo.shift_right_logical %2134, %c_35 : tensor<2xui32>
          %2138 = stablehlo.or %2136, %2137 : tensor<2xui32>
          %2139 = stablehlo.xor %2135, %2138 : tensor<2xui32>
          %2140 = stablehlo.add %2135, %2045 : tensor<2xui32>
          %2141 = stablehlo.add %2139, %2049 : tensor<2xui32>
          %2142 = stablehlo.add %2141, %c_71 : tensor<2xui32>
          %2143 = stablehlo.add %2140, %2142 : tensor<2xui32>
          %2144 = stablehlo.shift_left %2142, %c_82 : tensor<2xui32>
          %2145 = stablehlo.shift_right_logical %2142, %c_36 : tensor<2xui32>
          %2146 = stablehlo.or %2144, %2145 : tensor<2xui32>
          %2147 = stablehlo.xor %2143, %2146 : tensor<2xui32>
          %2148 = stablehlo.add %2143, %2147 : tensor<2xui32>
          %2149 = stablehlo.shift_left %2147, %c_81 : tensor<2xui32>
          %2150 = stablehlo.shift_right_logical %2147, %c_77 : tensor<2xui32>
          %2151 = stablehlo.or %2149, %2150 : tensor<2xui32>
          %2152 = stablehlo.xor %2148, %2151 : tensor<2xui32>
          %2153 = stablehlo.add %2148, %2152 : tensor<2xui32>
          %2154 = stablehlo.shift_left %2152, %c_80 : tensor<2xui32>
          %2155 = stablehlo.shift_right_logical %2152, %c_79 : tensor<2xui32>
          %2156 = stablehlo.or %2154, %2155 : tensor<2xui32>
          %2157 = stablehlo.xor %2153, %2156 : tensor<2xui32>
          %2158 = stablehlo.add %2153, %2157 : tensor<2xui32>
          %2159 = stablehlo.shift_left %2157, %c_79 : tensor<2xui32>
          %2160 = stablehlo.shift_right_logical %2157, %c_80 : tensor<2xui32>
          %2161 = stablehlo.or %2159, %2160 : tensor<2xui32>
          %2162 = stablehlo.xor %2158, %2161 : tensor<2xui32>
          %2163 = stablehlo.add %2158, %2049 : tensor<2xui32>
          %2164 = stablehlo.add %2162, %2044 : tensor<2xui32>
          %2165 = stablehlo.add %2164, %c_70 : tensor<2xui32>
          %2166 = stablehlo.xor %2046, %2047 : tensor<2xui32>
          %2167 = stablehlo.xor %2166, %c_83 : tensor<2xui32>
          %2168 = stablehlo.add %c_37, %2047 : tensor<2xui32>
          %2169 = stablehlo.add %2046, %2168 : tensor<2xui32>
          %2170 = stablehlo.shift_left %2168, %c_82 : tensor<2xui32>
          %2171 = stablehlo.shift_right_logical %2168, %c_36 : tensor<2xui32>
          %2172 = stablehlo.or %2170, %2171 : tensor<2xui32>
          %2173 = stablehlo.xor %2169, %2172 : tensor<2xui32>
          %2174 = stablehlo.add %2169, %2173 : tensor<2xui32>
          %2175 = stablehlo.shift_left %2173, %c_81 : tensor<2xui32>
          %2176 = stablehlo.shift_right_logical %2173, %c_77 : tensor<2xui32>
          %2177 = stablehlo.or %2175, %2176 : tensor<2xui32>
          %2178 = stablehlo.xor %2174, %2177 : tensor<2xui32>
          %2179 = stablehlo.add %2174, %2178 : tensor<2xui32>
          %2180 = stablehlo.shift_left %2178, %c_80 : tensor<2xui32>
          %2181 = stablehlo.shift_right_logical %2178, %c_79 : tensor<2xui32>
          %2182 = stablehlo.or %2180, %2181 : tensor<2xui32>
          %2183 = stablehlo.xor %2179, %2182 : tensor<2xui32>
          %2184 = stablehlo.add %2179, %2183 : tensor<2xui32>
          %2185 = stablehlo.shift_left %2183, %c_79 : tensor<2xui32>
          %2186 = stablehlo.shift_right_logical %2183, %c_80 : tensor<2xui32>
          %2187 = stablehlo.or %2185, %2186 : tensor<2xui32>
          %2188 = stablehlo.xor %2184, %2187 : tensor<2xui32>
          %2189 = stablehlo.add %2184, %2047 : tensor<2xui32>
          %2190 = stablehlo.add %2188, %2167 : tensor<2xui32>
          %2191 = stablehlo.add %2190, %c_78 : tensor<2xui32>
          %2192 = stablehlo.add %2189, %2191 : tensor<2xui32>
          %2193 = stablehlo.shift_left %2191, %c_77 : tensor<2xui32>
          %2194 = stablehlo.shift_right_logical %2191, %c_81 : tensor<2xui32>
          %2195 = stablehlo.or %2193, %2194 : tensor<2xui32>
          %2196 = stablehlo.xor %2192, %2195 : tensor<2xui32>
          %2197 = stablehlo.add %2192, %2196 : tensor<2xui32>
          %2198 = stablehlo.shift_left %2196, %c_76 : tensor<2xui32>
          %2199 = stablehlo.shift_right_logical %2196, %c_72 : tensor<2xui32>
          %2200 = stablehlo.or %2198, %2199 : tensor<2xui32>
          %2201 = stablehlo.xor %2197, %2200 : tensor<2xui32>
          %2202 = stablehlo.add %2197, %2201 : tensor<2xui32>
          %2203 = stablehlo.shift_left %2201, %c_75 : tensor<2xui32>
          %2204 = stablehlo.shift_right_logical %2201, %c_75 : tensor<2xui32>
          %2205 = stablehlo.or %2203, %2204 : tensor<2xui32>
          %2206 = stablehlo.xor %2202, %2205 : tensor<2xui32>
          %2207 = stablehlo.add %2202, %2206 : tensor<2xui32>
          %2208 = stablehlo.shift_left %2206, %c_74 : tensor<2xui32>
          %2209 = stablehlo.shift_right_logical %2206, %c_35 : tensor<2xui32>
          %2210 = stablehlo.or %2208, %2209 : tensor<2xui32>
          %2211 = stablehlo.xor %2207, %2210 : tensor<2xui32>
          %2212 = stablehlo.add %2207, %2167 : tensor<2xui32>
          %2213 = stablehlo.add %2211, %2046 : tensor<2xui32>
          %2214 = stablehlo.add %2213, %c_73 : tensor<2xui32>
          %2215 = stablehlo.add %2212, %2214 : tensor<2xui32>
          %2216 = stablehlo.shift_left %2214, %c_82 : tensor<2xui32>
          %2217 = stablehlo.shift_right_logical %2214, %c_36 : tensor<2xui32>
          %2218 = stablehlo.or %2216, %2217 : tensor<2xui32>
          %2219 = stablehlo.xor %2215, %2218 : tensor<2xui32>
          %2220 = stablehlo.add %2215, %2219 : tensor<2xui32>
          %2221 = stablehlo.shift_left %2219, %c_81 : tensor<2xui32>
          %2222 = stablehlo.shift_right_logical %2219, %c_77 : tensor<2xui32>
          %2223 = stablehlo.or %2221, %2222 : tensor<2xui32>
          %2224 = stablehlo.xor %2220, %2223 : tensor<2xui32>
          %2225 = stablehlo.add %2220, %2224 : tensor<2xui32>
          %2226 = stablehlo.shift_left %2224, %c_80 : tensor<2xui32>
          %2227 = stablehlo.shift_right_logical %2224, %c_79 : tensor<2xui32>
          %2228 = stablehlo.or %2226, %2227 : tensor<2xui32>
          %2229 = stablehlo.xor %2225, %2228 : tensor<2xui32>
          %2230 = stablehlo.add %2225, %2229 : tensor<2xui32>
          %2231 = stablehlo.shift_left %2229, %c_79 : tensor<2xui32>
          %2232 = stablehlo.shift_right_logical %2229, %c_80 : tensor<2xui32>
          %2233 = stablehlo.or %2231, %2232 : tensor<2xui32>
          %2234 = stablehlo.xor %2230, %2233 : tensor<2xui32>
          %2235 = stablehlo.add %2230, %2046 : tensor<2xui32>
          %2236 = stablehlo.add %2234, %2047 : tensor<2xui32>
          %2237 = stablehlo.add %2236, %c_72 : tensor<2xui32>
          %2238 = stablehlo.add %2235, %2237 : tensor<2xui32>
          %2239 = stablehlo.shift_left %2237, %c_77 : tensor<2xui32>
          %2240 = stablehlo.shift_right_logical %2237, %c_81 : tensor<2xui32>
          %2241 = stablehlo.or %2239, %2240 : tensor<2xui32>
          %2242 = stablehlo.xor %2238, %2241 : tensor<2xui32>
          %2243 = stablehlo.add %2238, %2242 : tensor<2xui32>
          %2244 = stablehlo.shift_left %2242, %c_76 : tensor<2xui32>
          %2245 = stablehlo.shift_right_logical %2242, %c_72 : tensor<2xui32>
          %2246 = stablehlo.or %2244, %2245 : tensor<2xui32>
          %2247 = stablehlo.xor %2243, %2246 : tensor<2xui32>
          %2248 = stablehlo.add %2243, %2247 : tensor<2xui32>
          %2249 = stablehlo.shift_left %2247, %c_75 : tensor<2xui32>
          %2250 = stablehlo.shift_right_logical %2247, %c_75 : tensor<2xui32>
          %2251 = stablehlo.or %2249, %2250 : tensor<2xui32>
          %2252 = stablehlo.xor %2248, %2251 : tensor<2xui32>
          %2253 = stablehlo.add %2248, %2252 : tensor<2xui32>
          %2254 = stablehlo.shift_left %2252, %c_74 : tensor<2xui32>
          %2255 = stablehlo.shift_right_logical %2252, %c_35 : tensor<2xui32>
          %2256 = stablehlo.or %2254, %2255 : tensor<2xui32>
          %2257 = stablehlo.xor %2253, %2256 : tensor<2xui32>
          %2258 = stablehlo.add %2253, %2047 : tensor<2xui32>
          %2259 = stablehlo.add %2257, %2167 : tensor<2xui32>
          %2260 = stablehlo.add %2259, %c_71 : tensor<2xui32>
          %2261 = stablehlo.add %2258, %2260 : tensor<2xui32>
          %2262 = stablehlo.shift_left %2260, %c_82 : tensor<2xui32>
          %2263 = stablehlo.shift_right_logical %2260, %c_36 : tensor<2xui32>
          %2264 = stablehlo.or %2262, %2263 : tensor<2xui32>
          %2265 = stablehlo.xor %2261, %2264 : tensor<2xui32>
          %2266 = stablehlo.add %2261, %2265 : tensor<2xui32>
          %2267 = stablehlo.shift_left %2265, %c_81 : tensor<2xui32>
          %2268 = stablehlo.shift_right_logical %2265, %c_77 : tensor<2xui32>
          %2269 = stablehlo.or %2267, %2268 : tensor<2xui32>
          %2270 = stablehlo.xor %2266, %2269 : tensor<2xui32>
          %2271 = stablehlo.add %2266, %2270 : tensor<2xui32>
          %2272 = stablehlo.shift_left %2270, %c_80 : tensor<2xui32>
          %2273 = stablehlo.shift_right_logical %2270, %c_79 : tensor<2xui32>
          %2274 = stablehlo.or %2272, %2273 : tensor<2xui32>
          %2275 = stablehlo.xor %2271, %2274 : tensor<2xui32>
          %2276 = stablehlo.add %2271, %2275 : tensor<2xui32>
          %2277 = stablehlo.shift_left %2275, %c_79 : tensor<2xui32>
          %2278 = stablehlo.shift_right_logical %2275, %c_80 : tensor<2xui32>
          %2279 = stablehlo.or %2277, %2278 : tensor<2xui32>
          %2280 = stablehlo.xor %2276, %2279 : tensor<2xui32>
          %2281 = stablehlo.add %2276, %2167 : tensor<2xui32>
          %2282 = stablehlo.add %2280, %2046 : tensor<2xui32>
          %2283 = stablehlo.add %2282, %c_70 : tensor<2xui32>
          %2284 = stablehlo.slice %2163 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
          %2285 = stablehlo.slice %2165 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
          %2286 = stablehlo.slice %2281 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
          %2287 = stablehlo.slice %2283 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
          %2288 = stablehlo.concatenate %2284, %2285, %2286, %2287, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
          %2289 = stablehlo.reshape %2288 : (tensor<4xui32>) -> tensor<2x2xui32>
          %2290 = stablehlo.bitcast_convert %2289 : (tensor<2x2xui32>) -> tensor<2xui64>
          %2291 = stablehlo.slice %2163 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
          %2292 = stablehlo.slice %2165 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
          %2293 = stablehlo.slice %2281 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
          %2294 = stablehlo.slice %2283 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
          %2295 = stablehlo.concatenate %2291, %2292, %2293, %2294, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
          %2296 = stablehlo.reshape %2295 : (tensor<4xui32>) -> tensor<2x2xui32>
          %2297 = stablehlo.bitcast_convert %2296 : (tensor<2x2xui32>) -> tensor<2xui64>
          %2298 = stablehlo.negate %arg2 : tensor<f64>
          %2299 = stablehlo.select %1733, %arg2, %2298 : tensor<i1>, tensor<f64>
          %2300 = stablehlo.broadcast_in_dim %2299, dims = [] : (tensor<f64>) -> tensor<1x3xf64>
          %2301 = stablehlo.multiply %2299, %cst_88 : tensor<f64>
          %2302 = stablehlo.broadcast_in_dim %2301, dims = [] : (tensor<f64>) -> tensor<1x3xf64>
          %2303 = stablehlo.multiply %2302, %2037 : tensor<1x3xf64>
          %2304 = stablehlo.subtract %2036, %2303 : tensor<1x3xf64>
          %2305 = stablehlo.dot_general %2304, %arg3, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
          %2306 = stablehlo.multiply %2300, %2305 : tensor<1x3xf64>
          %2307 = stablehlo.add %2035, %2306 : tensor<1x3xf64>
          %2308 = stablehlo.reshape %2307 : (tensor<1x3xf64>) -> tensor<3xf64>
          %2309 = stablehlo.slice %2308 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
          %2310 = stablehlo.reshape %2309 : (tensor<1xf64>) -> tensor<f64>
          %2311 = stablehlo.reshape %2309 : (tensor<1xf64>) -> tensor<1x1xf64>
          %2312 = stablehlo.slice %2308 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
          %2313 = stablehlo.reshape %2312 : (tensor<1xf64>) -> tensor<f64>
          %2314 = stablehlo.reshape %2312 : (tensor<1xf64>) -> tensor<1x1xf64>
          %2315 = stablehlo.slice %2308 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
          %2316 = stablehlo.reshape %2315 : (tensor<1xf64>) -> tensor<1x1xf64>
          %2317 = stablehlo.reshape %2315 : (tensor<1xf64>) -> tensor<f64>
          %2318 = stablehlo.exponential %2310 : tensor<f64>
          %2319 = stablehlo.reshape %2318 : (tensor<f64>) -> tensor<1x1xf64>
          %2320 = stablehlo.exponential %2313 : tensor<f64>
          %2321 = stablehlo.reshape %2320 : (tensor<f64>) -> tensor<1x1xf64>
          %2322 = stablehlo.exponential %2317 : tensor<f64>
          %2323 = stablehlo.reshape %2322 : (tensor<f64>) -> tensor<1x1xf64>
          %2324 = stablehlo.add %cst_41, %2317 : tensor<f64>
          %2325 = stablehlo.multiply %cst_42, %2324 : tensor<f64>
          %2326 = stablehlo.multiply %cst_43, %2318 : tensor<f64>
          %2327 = stablehlo.broadcast_in_dim %2326, dims = [] : (tensor<f64>) -> tensor<20x20xf64>
          %2328 = stablehlo.multiply %2327, %1 : tensor<20x20xf64>
          %2329 = stablehlo.add %cst_47, %2328 : tensor<20x20xf64>
          %2330:2 = stablehlo.custom_call @enzymexla_compile_cpu(%c_4, %c_91, %2329, %c_91, %c) {api_version = 3 : i32, backend_config = "\00p~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i8>, tensor<i64>, tensor<20x20xf64>, tensor<i64>, tensor<i64>) -> (tensor<20x20xf64>, tensor<i64>)
          %2331 = stablehlo.iota dim = 0 : tensor<20x20xi64>
          %2332 = stablehlo.iota dim = 1 : tensor<20x20xi64>
          %2333 = stablehlo.compare LE, %2331, %2332 : (tensor<20x20xi64>, tensor<20x20xi64>) -> tensor<20x20xi1>
          %2334 = stablehlo.select %2333, %2330#0, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2335 = stablehlo.multiply %cst_43, %2320 : tensor<f64>
          %2336 = stablehlo.broadcast_in_dim %2335, dims = [] : (tensor<f64>) -> tensor<20x1xf64>
          %2337 = stablehlo.multiply %2336, %839 : tensor<20x1xf64>
          %2338 = stablehlo.subtract %2, %2337 : tensor<20x1xf64>
          %2339 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2338, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2340 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2339, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2341 = stablehlo.dot_general %arg5, %2340, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2342 = stablehlo.multiply %2336, %2341 : tensor<20x1xf64>
          %2343 = stablehlo.subtract %2340, %2342 : tensor<20x1xf64>
          %2344 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2343, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2345 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2344, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2346 = "stablehlo.gather"(%2345, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
          %2347 = stablehlo.subtract %853, %2346 : tensor<8xf64>
          %2348 = stablehlo.dot_general %2347, %2347, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
          %2349 = stablehlo.multiply %cst_88, %2348 : tensor<f64>
          %2350 = stablehlo.multiply %2322, %2322 : tensor<f64>
          %2351 = stablehlo.divide %2349, %2350 : tensor<f64>
          %2352 = stablehlo.reshape %2351 : (tensor<f64>) -> tensor<1x1xf64>
          %2353 = stablehlo.subtract %2325, %2351 : tensor<f64>
          %2354 = stablehlo.dot_general %arg5, %2345, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2355 = stablehlo.multiply %2336, %2354 : tensor<20x1xf64>
          %2356 = stablehlo.subtract %2345, %2355 : tensor<20x1xf64>
          %2357 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2356, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2358 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2357, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2359 = stablehlo.dot_general %arg5, %2358, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2360 = stablehlo.multiply %2336, %2359 : tensor<20x1xf64>
          %2361 = stablehlo.subtract %2358, %2360 : tensor<20x1xf64>
          %2362 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2361, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2363 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2362, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2364 = "stablehlo.gather"(%2363, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
          %2365 = stablehlo.subtract %873, %2364 : tensor<8xf64>
          %2366 = stablehlo.dot_general %2365, %2365, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
          %2367 = stablehlo.multiply %cst_88, %2366 : tensor<f64>
          %2368 = stablehlo.divide %2367, %2350 : tensor<f64>
          %2369 = stablehlo.reshape %2368 : (tensor<f64>) -> tensor<1x1xf64>
          %2370 = stablehlo.subtract %2353, %2368 : tensor<f64>
          %2371 = stablehlo.dot_general %arg5, %2363, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2372 = stablehlo.multiply %2336, %2371 : tensor<20x1xf64>
          %2373 = stablehlo.subtract %2363, %2372 : tensor<20x1xf64>
          %2374 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2373, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2375 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2374, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2376 = stablehlo.dot_general %arg5, %2375, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2377 = stablehlo.multiply %2336, %2376 : tensor<20x1xf64>
          %2378 = stablehlo.subtract %2375, %2377 : tensor<20x1xf64>
          %2379 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2378, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2380 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2379, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2381 = "stablehlo.gather"(%2380, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
          %2382 = stablehlo.subtract %892, %2381 : tensor<8xf64>
          %2383 = stablehlo.dot_general %2382, %2382, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
          %2384 = stablehlo.multiply %cst_88, %2383 : tensor<f64>
          %2385 = stablehlo.divide %2384, %2350 : tensor<f64>
          %2386 = stablehlo.reshape %2385 : (tensor<f64>) -> tensor<1x1xf64>
          %2387 = stablehlo.subtract %2370, %2385 : tensor<f64>
          %2388 = stablehlo.dot_general %arg5, %2380, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2389 = stablehlo.multiply %2336, %2388 : tensor<20x1xf64>
          %2390 = stablehlo.subtract %2380, %2389 : tensor<20x1xf64>
          %2391 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2390, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2392 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2391, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2393 = stablehlo.dot_general %arg5, %2392, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2394 = stablehlo.multiply %2336, %2393 : tensor<20x1xf64>
          %2395 = stablehlo.subtract %2392, %2394 : tensor<20x1xf64>
          %2396 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2395, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2397 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2396, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2398 = "stablehlo.gather"(%2397, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
          %2399 = stablehlo.subtract %911, %2398 : tensor<8xf64>
          %2400 = stablehlo.dot_general %2399, %2399, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
          %2401 = stablehlo.multiply %cst_88, %2400 : tensor<f64>
          %2402 = stablehlo.divide %2401, %2350 : tensor<f64>
          %2403 = stablehlo.reshape %2402 : (tensor<f64>) -> tensor<1x1xf64>
          %2404 = stablehlo.subtract %2387, %2402 : tensor<f64>
          %2405 = stablehlo.dot_general %arg5, %2397, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2406 = stablehlo.multiply %2336, %2405 : tensor<20x1xf64>
          %2407 = stablehlo.subtract %2397, %2406 : tensor<20x1xf64>
          %2408 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2407, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2409 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2408, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2410 = stablehlo.dot_general %arg5, %2409, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<20x20xf64>, tensor<20x1xf64>) -> tensor<20x1xf64>
          %2411 = stablehlo.multiply %2336, %2410 : tensor<20x1xf64>
          %2412 = stablehlo.subtract %2409, %2411 : tensor<20x1xf64>
          %2413 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_0, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2412, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2414 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2413, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2415 = "stablehlo.gather"(%2414, %850) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<20x1xf64>, tensor<8x1xi64>) -> tensor<8xf64>
          %2416 = stablehlo.subtract %930, %2415 : tensor<8xf64>
          %2417 = stablehlo.dot_general %2416, %2416, contracting_dims = [0] x [0] : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
          %2418 = stablehlo.multiply %cst_88, %2417 : tensor<f64>
          %2419 = stablehlo.divide %2418, %2350 : tensor<f64>
          %2420 = stablehlo.reshape %2419 : (tensor<f64>) -> tensor<1x1xf64>
          %2421 = stablehlo.subtract %2404, %2419 : tensor<f64>
          %2422 = stablehlo.multiply %2310, %2310 : tensor<f64>
          %2423 = stablehlo.multiply %2313, %2313 : tensor<f64>
          %2424 = stablehlo.multiply %2317, %2317 : tensor<f64>
          %2425 = stablehlo.add %2422, %2423 : tensor<f64>
          %2426 = stablehlo.add %2425, %2424 : tensor<f64>
          %2427 = stablehlo.multiply %cst_45, %2426 : tensor<f64>
          %2428 = stablehlo.subtract %2427, %cst_46 : tensor<f64>
          %2429 = stablehlo.add %2428, %2421 : tensor<f64>
          %2430 = stablehlo.negate %2429 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
          %2431 = stablehlo.divide %cst_95, %2350 : tensor<f64>
          %2432 = stablehlo.reshape %2431 : (tensor<f64>) -> tensor<1x1xf64>
          %2433 = stablehlo.multiply %2432, %2420 : tensor<1x1xf64>
          %2434 = stablehlo.negate %2433 : tensor<1x1xf64>
          %2435 = stablehlo.multiply %2431, %cst_88 : tensor<f64>
          %2436 = stablehlo.dot_general %2435, %2416, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
          %2437 = stablehlo.add %2436, %2436 : tensor<8xf64>
          %2438 = stablehlo.negate %2437 : tensor<8xf64>
          %2439 = "stablehlo.scatter"(%cst_39, %850, %2438) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
          ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
            %2686 = stablehlo.add %arg8, %arg9 : tensor<f64>
            stablehlo.return %2686 : tensor<f64>
          }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
          %2440 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2439, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2441 = stablehlo.negate %2440 : tensor<20x1xf64>
          %2442 = stablehlo.dot_general %2441, %2414, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2443 = stablehlo.select %c_29, %2442, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2444 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2440, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2445 = stablehlo.negate %2413 : tensor<20x1xf64>
          %2446 = stablehlo.dot_general %2445, %2444, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2447 = stablehlo.select %c_29, %2446, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2448 = stablehlo.add %2443, %2447 : tensor<20x20xf64>
          %2449 = stablehlo.negate %2444 : tensor<20x1xf64>
          %2450 = stablehlo.multiply %2449, %2410 : tensor<20x1xf64>
          %2451 = stablehlo.dot_general %2449, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2452 = stablehlo.reshape %2451 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2453 = stablehlo.multiply %2336, %2452 : tensor<20x1xf64>
          %2454 = stablehlo.add %2444, %2453 : tensor<20x1xf64>
          %2455 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2454, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2456 = stablehlo.negate %2455 : tensor<20x1xf64>
          %2457 = stablehlo.dot_general %2456, %2409, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2458 = stablehlo.select %c_29, %2457, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2459 = stablehlo.add %2448, %2458 : tensor<20x20xf64>
          %2460 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2455, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2461 = stablehlo.negate %2408 : tensor<20x1xf64>
          %2462 = stablehlo.dot_general %2461, %2460, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2463 = stablehlo.select %c_29, %2462, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2464 = stablehlo.add %2459, %2463 : tensor<20x20xf64>
          %2465 = stablehlo.negate %2460 : tensor<20x1xf64>
          %2466 = stablehlo.multiply %2465, %2405 : tensor<20x1xf64>
          %2467 = stablehlo.add %2450, %2466 : tensor<20x1xf64>
          %2468 = stablehlo.dot_general %2465, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2469 = stablehlo.reshape %2468 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2470 = stablehlo.multiply %2336, %2469 : tensor<20x1xf64>
          %2471 = stablehlo.add %2460, %2470 : tensor<20x1xf64>
          %2472 = stablehlo.multiply %2432, %2403 : tensor<1x1xf64>
          %2473 = stablehlo.subtract %2434, %2472 : tensor<1x1xf64>
          %2474 = stablehlo.dot_general %2435, %2399, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
          %2475 = stablehlo.add %2474, %2474 : tensor<8xf64>
          %2476 = stablehlo.negate %2475 : tensor<8xf64>
          %2477 = "stablehlo.scatter"(%cst_39, %850, %2476) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
          ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
            %2686 = stablehlo.add %arg8, %arg9 : tensor<f64>
            stablehlo.return %2686 : tensor<f64>
          }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
          %2478 = stablehlo.add %2471, %2477 : tensor<20x1xf64>
          %2479 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2478, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2480 = stablehlo.negate %2479 : tensor<20x1xf64>
          %2481 = stablehlo.dot_general %2480, %2397, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2482 = stablehlo.select %c_29, %2481, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2483 = stablehlo.add %2464, %2482 : tensor<20x20xf64>
          %2484 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2479, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2485 = stablehlo.negate %2396 : tensor<20x1xf64>
          %2486 = stablehlo.dot_general %2485, %2484, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2487 = stablehlo.select %c_29, %2486, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2488 = stablehlo.add %2483, %2487 : tensor<20x20xf64>
          %2489 = stablehlo.negate %2484 : tensor<20x1xf64>
          %2490 = stablehlo.multiply %2489, %2393 : tensor<20x1xf64>
          %2491 = stablehlo.add %2467, %2490 : tensor<20x1xf64>
          %2492 = stablehlo.dot_general %2489, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2493 = stablehlo.reshape %2492 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2494 = stablehlo.multiply %2336, %2493 : tensor<20x1xf64>
          %2495 = stablehlo.add %2484, %2494 : tensor<20x1xf64>
          %2496 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2495, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2497 = stablehlo.negate %2496 : tensor<20x1xf64>
          %2498 = stablehlo.dot_general %2497, %2392, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2499 = stablehlo.select %c_29, %2498, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2500 = stablehlo.add %2488, %2499 : tensor<20x20xf64>
          %2501 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2496, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2502 = stablehlo.negate %2391 : tensor<20x1xf64>
          %2503 = stablehlo.dot_general %2502, %2501, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2504 = stablehlo.select %c_29, %2503, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2505 = stablehlo.add %2500, %2504 : tensor<20x20xf64>
          %2506 = stablehlo.negate %2501 : tensor<20x1xf64>
          %2507 = stablehlo.multiply %2506, %2388 : tensor<20x1xf64>
          %2508 = stablehlo.add %2491, %2507 : tensor<20x1xf64>
          %2509 = stablehlo.dot_general %2506, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2510 = stablehlo.reshape %2509 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2511 = stablehlo.multiply %2336, %2510 : tensor<20x1xf64>
          %2512 = stablehlo.add %2501, %2511 : tensor<20x1xf64>
          %2513 = stablehlo.multiply %2432, %2386 : tensor<1x1xf64>
          %2514 = stablehlo.subtract %2473, %2513 : tensor<1x1xf64>
          %2515 = stablehlo.dot_general %2435, %2382, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
          %2516 = stablehlo.add %2515, %2515 : tensor<8xf64>
          %2517 = stablehlo.negate %2516 : tensor<8xf64>
          %2518 = "stablehlo.scatter"(%cst_39, %850, %2517) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
          ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
            %2686 = stablehlo.add %arg8, %arg9 : tensor<f64>
            stablehlo.return %2686 : tensor<f64>
          }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
          %2519 = stablehlo.add %2512, %2518 : tensor<20x1xf64>
          %2520 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2519, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2521 = stablehlo.negate %2520 : tensor<20x1xf64>
          %2522 = stablehlo.dot_general %2521, %2380, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2523 = stablehlo.select %c_29, %2522, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2524 = stablehlo.add %2505, %2523 : tensor<20x20xf64>
          %2525 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2520, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2526 = stablehlo.negate %2379 : tensor<20x1xf64>
          %2527 = stablehlo.dot_general %2526, %2525, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2528 = stablehlo.select %c_29, %2527, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2529 = stablehlo.add %2524, %2528 : tensor<20x20xf64>
          %2530 = stablehlo.negate %2525 : tensor<20x1xf64>
          %2531 = stablehlo.multiply %2530, %2376 : tensor<20x1xf64>
          %2532 = stablehlo.add %2508, %2531 : tensor<20x1xf64>
          %2533 = stablehlo.dot_general %2530, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2534 = stablehlo.reshape %2533 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2535 = stablehlo.multiply %2336, %2534 : tensor<20x1xf64>
          %2536 = stablehlo.add %2525, %2535 : tensor<20x1xf64>
          %2537 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2536, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2538 = stablehlo.negate %2537 : tensor<20x1xf64>
          %2539 = stablehlo.dot_general %2538, %2375, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2540 = stablehlo.select %c_29, %2539, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2541 = stablehlo.add %2529, %2540 : tensor<20x20xf64>
          %2542 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2537, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2543 = stablehlo.negate %2374 : tensor<20x1xf64>
          %2544 = stablehlo.dot_general %2543, %2542, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2545 = stablehlo.select %c_29, %2544, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2546 = stablehlo.add %2541, %2545 : tensor<20x20xf64>
          %2547 = stablehlo.negate %2542 : tensor<20x1xf64>
          %2548 = stablehlo.multiply %2547, %2371 : tensor<20x1xf64>
          %2549 = stablehlo.add %2532, %2548 : tensor<20x1xf64>
          %2550 = stablehlo.dot_general %2547, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2551 = stablehlo.reshape %2550 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2552 = stablehlo.multiply %2336, %2551 : tensor<20x1xf64>
          %2553 = stablehlo.add %2542, %2552 : tensor<20x1xf64>
          %2554 = stablehlo.multiply %2432, %2369 : tensor<1x1xf64>
          %2555 = stablehlo.subtract %2514, %2554 : tensor<1x1xf64>
          %2556 = stablehlo.dot_general %2435, %2365, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
          %2557 = stablehlo.add %2556, %2556 : tensor<8xf64>
          %2558 = stablehlo.negate %2557 : tensor<8xf64>
          %2559 = "stablehlo.scatter"(%cst_39, %850, %2558) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
          ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
            %2686 = stablehlo.add %arg8, %arg9 : tensor<f64>
            stablehlo.return %2686 : tensor<f64>
          }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
          %2560 = stablehlo.add %2553, %2559 : tensor<20x1xf64>
          %2561 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2560, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2562 = stablehlo.negate %2561 : tensor<20x1xf64>
          %2563 = stablehlo.dot_general %2562, %2363, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2564 = stablehlo.select %c_29, %2563, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2565 = stablehlo.add %2546, %2564 : tensor<20x20xf64>
          %2566 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2561, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2567 = stablehlo.negate %2362 : tensor<20x1xf64>
          %2568 = stablehlo.dot_general %2567, %2566, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2569 = stablehlo.select %c_29, %2568, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2570 = stablehlo.add %2565, %2569 : tensor<20x20xf64>
          %2571 = stablehlo.negate %2566 : tensor<20x1xf64>
          %2572 = stablehlo.multiply %2571, %2359 : tensor<20x1xf64>
          %2573 = stablehlo.add %2549, %2572 : tensor<20x1xf64>
          %2574 = stablehlo.dot_general %2571, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2575 = stablehlo.reshape %2574 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2576 = stablehlo.multiply %2336, %2575 : tensor<20x1xf64>
          %2577 = stablehlo.add %2566, %2576 : tensor<20x1xf64>
          %2578 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2577, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2579 = stablehlo.negate %2578 : tensor<20x1xf64>
          %2580 = stablehlo.dot_general %2579, %2358, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2581 = stablehlo.select %c_29, %2580, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2582 = stablehlo.add %2570, %2581 : tensor<20x20xf64>
          %2583 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2578, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2584 = stablehlo.negate %2357 : tensor<20x1xf64>
          %2585 = stablehlo.dot_general %2584, %2583, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2586 = stablehlo.select %c_29, %2585, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2587 = stablehlo.add %2582, %2586 : tensor<20x20xf64>
          %2588 = stablehlo.negate %2583 : tensor<20x1xf64>
          %2589 = stablehlo.multiply %2588, %2354 : tensor<20x1xf64>
          %2590 = stablehlo.add %2573, %2589 : tensor<20x1xf64>
          %2591 = stablehlo.dot_general %2588, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2592 = stablehlo.reshape %2591 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2593 = stablehlo.multiply %2336, %2592 : tensor<20x1xf64>
          %2594 = stablehlo.add %2583, %2593 : tensor<20x1xf64>
          %2595 = stablehlo.multiply %2432, %2352 : tensor<1x1xf64>
          %2596 = stablehlo.subtract %2555, %2595 : tensor<1x1xf64>
          %2597 = stablehlo.add %2596, %2596 : tensor<1x1xf64>
          %2598 = stablehlo.dot_general %2435, %2347, contracting_dims = [] x [] : (tensor<f64>, tensor<8xf64>) -> tensor<8xf64>
          %2599 = stablehlo.add %2598, %2598 : tensor<8xf64>
          %2600 = stablehlo.negate %2599 : tensor<8xf64>
          %2601 = "stablehlo.scatter"(%cst_39, %850, %2600) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
          ^bb0(%arg8: tensor<f64>, %arg9: tensor<f64>):
            %2686 = stablehlo.add %arg8, %arg9 : tensor<f64>
            stablehlo.return %2686 : tensor<f64>
          }) : (tensor<20x1xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<20x1xf64>
          %2602 = stablehlo.add %2594, %2601 : tensor<20x1xf64>
          %2603 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2602, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2604 = stablehlo.negate %2603 : tensor<20x1xf64>
          %2605 = stablehlo.dot_general %2604, %2345, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2606 = stablehlo.select %c_29, %2605, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2607 = stablehlo.add %2587, %2606 : tensor<20x20xf64>
          %2608 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2603, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2609 = stablehlo.negate %2344 : tensor<20x1xf64>
          %2610 = stablehlo.dot_general %2609, %2608, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2611 = stablehlo.select %c_29, %2610, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2612 = stablehlo.add %2607, %2611 : tensor<20x20xf64>
          %2613 = stablehlo.negate %2608 : tensor<20x1xf64>
          %2614 = stablehlo.multiply %2613, %2341 : tensor<20x1xf64>
          %2615 = stablehlo.add %2590, %2614 : tensor<20x1xf64>
          %2616 = stablehlo.dot_general %2613, %arg5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<20x1xf64>, tensor<20x20xf64>) -> tensor<1x20xf64>
          %2617 = stablehlo.reshape %2616 : (tensor<1x20xf64>) -> tensor<20x1xf64>
          %2618 = stablehlo.multiply %2336, %2617 : tensor<20x1xf64>
          %2619 = stablehlo.add %2608, %2618 : tensor<20x1xf64>
          %2620 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_3, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2619, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2621 = stablehlo.negate %2620 : tensor<20x1xf64>
          %2622 = stablehlo.dot_general %2621, %2340, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2623 = stablehlo.select %c_29, %2622, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2624 = stablehlo.add %2612, %2623 : tensor<20x20xf64>
          %2625 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_92, %cst_95, %2334, %c_91, %2620, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x1xf64>, tensor<i64>) -> tensor<20x1xf64>
          %2626 = stablehlo.negate %2339 : tensor<20x1xf64>
          %2627 = stablehlo.dot_general %2626, %2625, contracting_dims = [1] x [1] : (tensor<20x1xf64>, tensor<20x1xf64>) -> tensor<20x20xf64>
          %2628 = stablehlo.select %c_29, %2627, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2629 = stablehlo.add %2624, %2628 : tensor<20x20xf64>
          %2630 = stablehlo.negate %2625 : tensor<20x1xf64>
          %2631 = stablehlo.multiply %2630, %839 : tensor<20x1xf64>
          %2632 = stablehlo.add %2615, %2631 : tensor<20x1xf64>
          %2633 = stablehlo.reduce(%2632 init: %cst_90) applies stablehlo.add across dimensions = [0, 1] : (tensor<20x1xf64>, tensor<f64>) -> tensor<f64>
          %2634 = stablehlo.reshape %2633 : (tensor<f64>) -> tensor<1x1xf64>
          %2635 = stablehlo.dot_general %2334, %2629, contracting_dims = [1] x [1] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
          %2636 = stablehlo.select %c_29, %cst_40, %2635 : tensor<20x20xi1>, tensor<20x20xf64>
          %2637 = stablehlo.multiply %2635, %cst_38 : tensor<20x20xf64>
          %2638 = stablehlo.select %c_28, %2637, %cst_40 : tensor<20x20xi1>, tensor<20x20xf64>
          %2639 = stablehlo.add %2636, %2638 : tensor<20x20xf64>
          %2640 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_91, %cst_95, %2334, %c_91, %2639, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x20xf64>, tensor<i64>) -> tensor<20x20xf64>
          %2641 = stablehlo.transpose %2640, dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
          %2642 = stablehlo.custom_call @enzymexla_compile_cpu(%c_5, %c_4, %c_2, %c_2, %c_91, %c_91, %cst_95, %2334, %c_91, %2641, %c_91) {api_version = 3 : i32, backend_config = "\00P~*\F2q\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i8>, tensor<i8>, tensor<i8>, tensor<i8>, tensor<i64>, tensor<i64>, tensor<f64>, tensor<20x20xf64>, tensor<i64>, tensor<20x20xf64>, tensor<i64>) -> tensor<20x20xf64>
          %2643 = stablehlo.transpose %2642, dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
          %2644 = stablehlo.add %2643, %2642 : tensor<20x20xf64>
          %2645 = stablehlo.dot_general %2644, %arg4, contracting_dims = [0, 1] x [1, 0] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<f64>
          %2646 = stablehlo.reshape %2645 : (tensor<f64>) -> tensor<1x1xf64>
          %2647 = stablehlo.multiply %cst_89, %2646 : tensor<1x1xf64>
          %2648 = stablehlo.add %2316, %cst_27 : tensor<1x1xf64>
          %2649 = stablehlo.concatenate %2311, %2314, %2648, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x3xf64>
          %2650 = stablehlo.concatenate %2647, %2634, %2323, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x3xf64>
          %2651 = stablehlo.pad %2597, %cst_43, low = [0, 2], high = [0, 0], interior = [0, 0] : (tensor<1x1xf64>, tensor<f64>) -> tensor<1x3xf64>
          %2652 = stablehlo.multiply %2650, %2651 : tensor<1x3xf64>
          %2653 = stablehlo.concatenate %2319, %2321, %2323, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x3xf64>
          %2654 = stablehlo.multiply %2652, %2653 : tensor<1x3xf64>
          %2655 = stablehlo.add %2649, %2654 : tensor<1x3xf64>
          %2656 = stablehlo.multiply %2302, %2655 : tensor<1x3xf64>
          %2657 = stablehlo.subtract %2304, %2656 : tensor<1x3xf64>
          %2658 = stablehlo.dot_general %2657, %arg3, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
          %2659 = stablehlo.dot_general %2657, %2658, contracting_dims = [0, 1] x [0, 1] {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<f64>
          %2660 = stablehlo.multiply %2659, %cst_88 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
          %2661 = stablehlo.add %2430, %2660 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
          %2662 = stablehlo.subtract %2661, %1454 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
          %2663 = stablehlo.compare NE, %2662, %2662, FLOAT {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %2664 = stablehlo.select %2663, %cst_84, %2662 {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<i1>, tensor<f64>
          %2665 = stablehlo.negate %2664 {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
          %2666 = stablehlo.compare GT, %2664, %cst_98, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %2667 = stablehlo.exponential %2665 : tensor<f64>
          %2668 = stablehlo.minimum %2667, %cst_95 : tensor<f64>
          %2669 = stablehlo.compare EQ, %iterArg_142, %c_93, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
          %2670:16 = "stablehlo.if"(%2669) ({
            stablehlo.return %2307, %2657, %2655, %2307, %2657, %2655, %2307, %2655, %2430, %2661, %c_93, %2665, %2666, %2668, %c_92, %2657 : tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x3xf64>
          }, {
            %2686 = stablehlo.select %1733, %iterArg_127, %2307 : tensor<i1>, tensor<1x3xf64>
            %2687 = stablehlo.select %1733, %iterArg_128, %2657 : tensor<i1>, tensor<1x3xf64>
            %2688 = stablehlo.select %1733, %iterArg_129, %2655 : tensor<i1>, tensor<1x3xf64>
            %2689 = stablehlo.select %1733, %2307, %iterArg_130 : tensor<i1>, tensor<1x3xf64>
            %2690 = stablehlo.select %1733, %2657, %iterArg_131 : tensor<i1>, tensor<1x3xf64>
            %2691 = stablehlo.select %1733, %2655, %iterArg_132 : tensor<i1>, tensor<1x3xf64>
            %2692 = stablehlo.maximum %iterArg_138, %2665 : tensor<f64>
            %2693 = stablehlo.subtract %iterArg_138, %2665 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
            %2694 = stablehlo.compare NE, %2693, %2693 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<f64>, tensor<f64>) -> tensor<i1>
            %2695 = stablehlo.add %iterArg_138, %2665 : tensor<f64>
            %2696 = stablehlo.abs %2693 : tensor<f64>
            %2697 = stablehlo.negate %2696 : tensor<f64>
            %2698 = stablehlo.exponential %2697 : tensor<f64>
            %2699 = stablehlo.log_plus_one %2698 : tensor<f64>
            %2700 = stablehlo.add %2692, %2699 : tensor<f64>
            %2701 = stablehlo.select %2694, %2695, %2700 : tensor<i1>, tensor<f64>
            %2702 = stablehlo.subtract %2665, %iterArg_138 : tensor<f64>
            %2703 = stablehlo.logistic %2702 : tensor<f64>
            %output_state_148, %output_149 = stablehlo.rng_bit_generator %2297, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
            %2704 = stablehlo.shift_right_logical %output_149, %c_50 : tensor<ui64>
            %2705 = stablehlo.or %2704, %c_49 : tensor<ui64>
            %2706 = stablehlo.bitcast_convert %2705 : (tensor<ui64>) -> tensor<f64>
            %2707 = stablehlo.subtract %2706, %cst_95 : tensor<f64>
            %2708 = stablehlo.compare LT, %2707, %2703, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
            %2709 = stablehlo.select %2708, %2307, %iterArg_133 : tensor<i1>, tensor<1x3xf64>
            %2710 = stablehlo.select %2708, %2655, %iterArg_134 : tensor<i1>, tensor<1x3xf64>
            %2711 = stablehlo.select %2708, %2430, %iterArg_135 : tensor<i1>, tensor<f64>
            %2712 = stablehlo.select %2708, %2661, %iterArg_136 : tensor<i1>, tensor<f64>
            %2713 = stablehlo.add %iterArg_137, %c_92 : tensor<i64>
            %2714 = stablehlo.or %iterArg_140, %2666 : tensor<i1>
            %2715 = stablehlo.add %iterArg_141, %2668 : tensor<f64>
            %2716 = stablehlo.add %iterArg_142, %c_92 : tensor<i64>
            %2717 = stablehlo.add %iterArg_143, %2657 : tensor<1x3xf64>
            stablehlo.return %2686, %2687, %2688, %2689, %2690, %2691, %2709, %2710, %2711, %2712, %2713, %2701, %2714, %2715, %2716, %2717 : tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x3xf64>
          }) : (tensor<i1>) -> (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x3xf64>)
          %2671 = stablehlo.shift_right_logical %iterArg_147, %c_92 : tensor<i64>
          %2672 = stablehlo.popcnt %2671 : tensor<i64>
          %2673 = stablehlo.add %iterArg_147, %c_92 : tensor<i64>
          %2674 = stablehlo.not %iterArg_147 : tensor<i64>
          %2675 = stablehlo.and %2674, %2673 : tensor<i64>
          %2676 = stablehlo.subtract %2675, %c_92 : tensor<i64>
          %2677 = stablehlo.popcnt %2676 : tensor<i64>
          %2678 = stablehlo.subtract %2672, %2677 : tensor<i64>
          %2679 = stablehlo.add %2678, %c_92 : tensor<i64>
          %2680 = stablehlo.and %iterArg_147, %c_92 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<i64>
          %2681 = stablehlo.compare EQ, %2680, %c_93, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
          %2682:2 = "stablehlo.if"(%2681) ({
            %2686 = stablehlo.dynamic_update_slice %iterArg_145, %2657, %2672, %c_93 : (tensor<5x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<5x3xf64>
            %2687 = stablehlo.dynamic_update_slice %iterArg_146, %2670#15, %2672, %c_93 : (tensor<5x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<5x3xf64>
            stablehlo.return %2686, %2687 : tensor<5x3xf64>, tensor<5x3xf64>
          }, {
            stablehlo.return %iterArg_145, %iterArg_146 : tensor<5x3xf64>, tensor<5x3xf64>
          }) : (tensor<i1>) -> (tensor<5x3xf64>, tensor<5x3xf64>)
          %2683:2 = stablehlo.while(%iterArg_148 = %2672, %iterArg_149 = %c_87) : tensor<i64>, tensor<i1>
          cond {
            %2686 = stablehlo.compare GE, %iterArg_148, %2679, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
            %2687 = stablehlo.not %iterArg_149 : tensor<i1>
            %2688 = stablehlo.and %2686, %2687 : tensor<i1>
            stablehlo.return %2688 : tensor<i1>
          } do {
            %2686 = stablehlo.dynamic_slice %2682#0, %iterArg_148, %c_93, sizes = [1, 3] : (tensor<5x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
            %2687 = stablehlo.dynamic_slice %2682#1, %iterArg_148, %c_93, sizes = [1, 3] : (tensor<5x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
            %2688 = stablehlo.subtract %2670#15, %2687 : tensor<1x3xf64>
            %2689 = stablehlo.add %2688, %2686 : tensor<1x3xf64>
            %2690 = stablehlo.dot_general %2686, %arg3, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
            %2691 = stablehlo.add %2686, %2657 : tensor<1x3xf64>
            %2692 = stablehlo.multiply %cst_22, %2691 : tensor<1x3xf64>
            %2693 = stablehlo.add %2689, %2692 : tensor<1x3xf64>
            %2694 = stablehlo.dot_general %2690, %2693, contracting_dims = [0, 1] x [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<f64>
            %2695 = stablehlo.dot_general %2658, %2693, contracting_dims = [0, 1] x [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<f64>
            %2696 = stablehlo.compare LE, %2694, %cst_90, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
            %2697 = stablehlo.compare LE, %2695, %cst_90, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
            %2698 = stablehlo.or %2696, %2697 : tensor<i1>
            %2699 = stablehlo.subtract %iterArg_148, %c_92 : tensor<i64>
            stablehlo.return %2699, %2698 : tensor<i64>, tensor<i1>
          }
          %2684 = stablehlo.not %2669 : tensor<i1>
          %2685 = stablehlo.and %2684, %2683#1 : tensor<i1>
          stablehlo.return %2670#0, %2670#1, %2670#2, %2670#3, %2670#4, %2670#5, %2670#6, %2670#7, %2670#8, %2670#9, %2670#10, %2670#11, %2685, %2670#12, %2670#13, %2670#14, %2670#15, %2290, %2682#0, %2682#1, %2673 : tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x3xf64>, tensor<2xui64>, tensor<5x3xf64>, tensor<5x3xf64>, tensor<i64>
        }
        %1990 = stablehlo.select %1733, %iterArg_106, %1989#0 : tensor<i1>, tensor<1x3xf64>
        %1991 = stablehlo.select %1733, %iterArg_107, %1989#1 : tensor<i1>, tensor<1x3xf64>
        %1992 = stablehlo.select %1733, %iterArg_108, %1989#2 : tensor<i1>, tensor<1x3xf64>
        %1993 = stablehlo.select %1733, %1989#3, %iterArg_109 : tensor<i1>, tensor<1x3xf64>
        %1994 = stablehlo.select %1733, %1989#4, %iterArg_110 : tensor<i1>, tensor<1x3xf64>
        %1995 = stablehlo.select %1733, %1989#5, %iterArg_111 : tensor<i1>, tensor<1x3xf64>
        %1996 = stablehlo.maximum %iterArg_117, %1989#11 : tensor<f64>
        %1997 = stablehlo.subtract %iterArg_117, %1989#11 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<f64>
        %1998 = stablehlo.compare NE, %1997, %1997 {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %1999 = stablehlo.add %iterArg_117, %1989#11 : tensor<f64>
        %2000 = stablehlo.abs %1997 : tensor<f64>
        %2001 = stablehlo.negate %2000 : tensor<f64>
        %2002 = stablehlo.exponential %2001 : tensor<f64>
        %2003 = stablehlo.log_plus_one %2002 : tensor<f64>
        %2004 = stablehlo.add %1996, %2003 : tensor<f64>
        %2005 = stablehlo.select %1998, %1999, %2004 : tensor<i1>, tensor<f64>
        %2006 = stablehlo.subtract %1989#11, %iterArg_117 : tensor<f64>
        %2007 = stablehlo.exponential %2006 : tensor<f64>
        %2008 = stablehlo.minimum %2007, %cst_95 : tensor<f64>
        %2009 = stablehlo.or %1989#12, %1989#13 : tensor<i1>
        %2010 = stablehlo.select %2009, %cst_90, %2008 : tensor<i1>, tensor<f64>
        %output_state_125, %output_126 = stablehlo.rng_bit_generator %1987, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
        %2011 = stablehlo.shift_right_logical %output_126, %c_50 : tensor<ui64>
        %2012 = stablehlo.or %2011, %c_49 : tensor<ui64>
        %2013 = stablehlo.bitcast_convert %2012 : (tensor<ui64>) -> tensor<f64>
        %2014 = stablehlo.subtract %2013, %cst_95 : tensor<f64>
        %2015 = stablehlo.compare LT, %2014, %2010, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %2016 = stablehlo.select %2015, %1989#6, %iterArg_112 : tensor<i1>, tensor<1x3xf64>
        %2017 = stablehlo.select %2015, %1989#7, %iterArg_113 : tensor<i1>, tensor<1x3xf64>
        %2018 = stablehlo.select %2015, %1989#8, %iterArg_114 : tensor<i1>, tensor<f64>
        %2019 = stablehlo.select %2015, %1989#9, %iterArg_115 : tensor<i1>, tensor<f64>
        %2020 = stablehlo.add %iterArg_116, %c_92 : tensor<i64>
        %2021 = stablehlo.add %iterArg_121, %1989#16 : tensor<1x3xf64>
        %2022 = stablehlo.dot_general %1991, %arg3, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
        %2023 = stablehlo.dot_general %1994, %arg3, contracting_dims = [1] x [1] : (tensor<1x3xf64>, tensor<3x3xf64>) -> tensor<1x3xf64>
        %2024 = stablehlo.add %1991, %1994 : tensor<1x3xf64>
        %2025 = stablehlo.multiply %cst_22, %2024 : tensor<1x3xf64>
        %2026 = stablehlo.add %2021, %2025 : tensor<1x3xf64>
        %2027 = stablehlo.dot_general %2022, %2026, contracting_dims = [0, 1] x [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<f64>
        %2028 = stablehlo.dot_general %2023, %2026, contracting_dims = [0, 1] x [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<f64>
        %2029 = stablehlo.compare LE, %2027, %cst_90, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %2030 = stablehlo.compare LE, %2028, %cst_90, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %2031 = stablehlo.or %2029, %2030 : tensor<i1>
        %2032 = stablehlo.or %1989#12, %2031 : tensor<i1>
        %2033 = stablehlo.or %iterArg_119, %1989#13 : tensor<i1>
        %2034 = stablehlo.add %iterArg_120, %1989#14 : tensor<f64>
        stablehlo.return %1990, %1991, %1992, %1993, %1994, %1995, %2016, %2017, %2018, %2019, %2020, %2005, %2032, %2033, %2034, %2021, %1717 : tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x3xf64>, tensor<2xui64>
      }
      %1456 = stablehlo.reshape %1455#8 : (tensor<f64>) -> tensor<1xf64>
      %1457 = stablehlo.reshape %1455#13 : (tensor<i1>) -> tensor<1x1xi1>
      %1458 = stablehlo.negate %1456 : tensor<1xf64>
      %1459 = stablehlo.dynamic_update_slice %iterArg_105, %1458, %iterArg : (tensor<20xf64>, tensor<1xf64>, tensor<i64>) -> tensor<20xf64>
      %1460 = stablehlo.add %iterArg, %c_92 {enzymexla.bounds = [[1, 20]]} : tensor<i64>
      %1461 = stablehlo.reshape %1455#6 : (tensor<1x3xf64>) -> tensor<3x1xf64>
      %1462 = stablehlo.dynamic_update_slice %iterArg_103, %1461, %c_93, %iterArg : (tensor<3x20xf64>, tensor<3x1xf64>, tensor<i64>, tensor<i64>) -> tensor<3x20xf64>
      %1463 = stablehlo.pad %1457, %c_23, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<1x1xi1>, tensor<i1>) -> tensor<2x1xi1>
      %1464 = stablehlo.dynamic_update_slice %iterArg_104, %1463, %c_93, %iterArg : (tensor<2x20xi1>, tensor<2x1xi1>, tensor<i64>, tensor<i64>) -> tensor<2x20xi1>
      stablehlo.return %1460, %1455#6, %1455#7, %1455#8, %1426, %1462, %1464, %1459 : tensor<i64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<3x20xf64>, tensor<2x20xi1>, tensor<20xf64>
    }
    %1172 = stablehlo.reshape %1171#1 : (tensor<1x3xf64>) -> tensor<3x1xf64>
    %1173 = stablehlo.reshape %1171#2 : (tensor<1x3xf64>) -> tensor<3x1xf64>
    return %1171#5, %1171#6, %1171#7, %1171#4, %1172, %1173, %1171#3, %arg2, %arg3, %1171#4, %1171#4 : tensor<3x20xf64>, tensor<2x20xi1>, tensor<20xf64>, tensor<2xui64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<f64>, tensor<f64>, tensor<3x3xf64>, tensor<2xui64>, tensor<2xui64>
  }
}