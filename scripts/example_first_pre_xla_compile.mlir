module @example_first attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=1]>
  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>, tf.aliasing_output = 0 : i32}, %arg2: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 1 : i32}) -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
    %0 = stablehlo.add %arg0, %arg2 : tensor<4xf32>
    return %arg1, %0 : tensor<f32>, tensor<4xf32>
  }
}
