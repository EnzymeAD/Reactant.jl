// Structurally minimized from brusselator-k8-n4096-batched.mlir while keeping
// the production K=8, N=4096 tensor sizes and all eight batch partitions.
//
// The slices form the same complete partition of the leading dimension for
// both results. `--enzyme-hlo-opt` currently leaves the full-width elementwise
// DAG in place, including the shared 8x4096x4096 value.
//
// The desired form has no concatenate or slice operations. For each lane i it
// computes shared_i = lhs_i * rhs_i, lhs_i + shared_i, and rhs_i - shared_i.
module {
  func.func @main(
      %lhs0: tensor<1x4096x4096xf64>, %lhs1: tensor<1x4096x4096xf64>,
      %lhs2: tensor<1x4096x4096xf64>, %lhs3: tensor<1x4096x4096xf64>,
      %lhs4: tensor<1x4096x4096xf64>, %lhs5: tensor<1x4096x4096xf64>,
      %lhs6: tensor<1x4096x4096xf64>, %lhs7: tensor<1x4096x4096xf64>,
      %rhs0: tensor<1x4096x4096xf64>, %rhs1: tensor<1x4096x4096xf64>,
      %rhs2: tensor<1x4096x4096xf64>, %rhs3: tensor<1x4096x4096xf64>,
      %rhs4: tensor<1x4096x4096xf64>, %rhs5: tensor<1x4096x4096xf64>,
      %rhs6: tensor<1x4096x4096xf64>, %rhs7: tensor<1x4096x4096xf64>)
      -> (tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>) {
    %lhs = stablehlo.concatenate
        %lhs0, %lhs1, %lhs2, %lhs3, %lhs4, %lhs5, %lhs6, %lhs7, dim = 0
        : (tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
           tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
           tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
           tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>)
        -> tensor<8x4096x4096xf64>
    %rhs = stablehlo.concatenate
        %rhs0, %rhs1, %rhs2, %rhs3, %rhs4, %rhs5, %rhs6, %rhs7, dim = 0
        : (tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
           tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
           tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
           tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>)
        -> tensor<8x4096x4096xf64>

    // This is the minimized analogue of the shared nonlinear tangent %1033.
    %shared = stablehlo.multiply %lhs, %rhs : tensor<8x4096x4096xf64>
    %sum = stablehlo.add %lhs, %shared : tensor<8x4096x4096xf64>
    %difference = stablehlo.subtract %rhs, %shared
        : tensor<8x4096x4096xf64>

    %sum0 = stablehlo.slice %sum [0:1, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum1 = stablehlo.slice %sum [1:2, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum2 = stablehlo.slice %sum [2:3, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum3 = stablehlo.slice %sum [3:4, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum4 = stablehlo.slice %sum [4:5, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum5 = stablehlo.slice %sum [5:6, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum6 = stablehlo.slice %sum [6:7, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %sum7 = stablehlo.slice %sum [7:8, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>

    %difference0 = stablehlo.slice %difference [0:1, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference1 = stablehlo.slice %difference [1:2, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference2 = stablehlo.slice %difference [2:3, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference3 = stablehlo.slice %difference [3:4, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference4 = stablehlo.slice %difference [4:5, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference5 = stablehlo.slice %difference [5:6, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference6 = stablehlo.slice %difference [6:7, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>
    %difference7 = stablehlo.slice %difference [7:8, 0:4096, 0:4096]
        : (tensor<8x4096x4096xf64>) -> tensor<1x4096x4096xf64>

    return %sum0, %sum1, %sum2, %sum3, %sum4, %sum5, %sum6, %sum7,
        %difference0, %difference1, %difference2, %difference3,
        %difference4, %difference5, %difference6, %difference7
        : tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>,
          tensor<1x4096x4096xf64>, tensor<1x4096x4096xf64>
  }
}
