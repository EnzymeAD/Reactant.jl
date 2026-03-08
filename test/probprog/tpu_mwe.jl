using Reactant, Test
using Reactant: MLIR, XLA

# MWE for XLA TPU crash in HloReplicationAnalysis::ComputeHloReplicationOnComputation
#
# The crash occurs because:
# 1. NUTS generates 3 nested while loops (outer MCMC + buildTree + buildIterativeSubtree)
# 2. Our RNG state is tensor<2xui64>, which TPU decomposes to u32[2,2]
# 3. HloReplicationAnalysis processes GetTupleElement on the decomposed shape,
#    calling ShapeUtil::GetSubshape(u32[2,2], {0}) which fails because u32[2,2] is not a tuple
#
# Stack trace from CI:
#   F0306 shape_util.cc:1214 Check failed: return_shape->IsTuple()
#     Invalid index {0} for shape u32[2,2]{1,0}
#   in xla::HloReplicationAnalysis::ComputeHloReplicationOnComputation
#   called from xla::AllReduceSimplifier::RunImpl
#   inside xla::jellyfish::HloOptimizeThroughLayoutAssignment

# Minimal StableHLO module with 3 nested while loops + rng_bit_generator using ui64 state.
# This is the exact pattern that crashes on TPU.
const NESTED_WHILE_RNG_MODULE = """
module @nuts_mwe attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%rng_state: tensor<2xui64>) -> tensor<f64> {
    %init_q = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %zero = stablehlo.constant dense<0> : tensor<i32>

    // outer while: MCMC iterations
    %outer:3 = stablehlo.while(%arg0 = %rng_state, %arg1 = %init_q, %arg2 = %zero)
      : tensor<2xui64>, tensor<f64>, tensor<i32>
      cond {
      %max_iter = stablehlo.constant dense<5> : tensor<i32>
      %pred = stablehlo.compare LT, %arg2, %max_iter : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
    } do {
      %zero_m = stablehlo.constant dense<0> : tensor<i32>

      // middle while: tree building
      %mid:3 = stablehlo.while(%m0 = %arg0, %m1 = %arg1, %m2 = %zero_m)
        : tensor<2xui64>, tensor<f64>, tensor<i32>
        cond {
        %max_depth = stablehlo.constant dense<3> : tensor<i32>
        %pred_m = stablehlo.compare LT, %m2, %max_depth : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %pred_m : tensor<i1>
      } do {
        %new_state_m, %_random_m = stablehlo.rng_bit_generator %m0, algorithm = DEFAULT
          : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
        %zero_i = stablehlo.constant dense<0> : tensor<i32>

        // inner while: iterative subtree building
        %inn:3 = stablehlo.while(%i0 = %new_state_m, %i1 = %m1, %i2 = %zero_i)
          : tensor<2xui64>, tensor<f64>, tensor<i32>
          cond {
          %limit = stablehlo.constant dense<4> : tensor<i32>
          %pred_i = stablehlo.compare LT, %i2, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %pred_i : tensor<i1>
        } do {
          %new_state_i, %_random_i = stablehlo.rng_bit_generator %i0, algorithm = DEFAULT
            : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
          %step = stablehlo.constant dense<1.000000e-01> : tensor<f64>
          %new_q = stablehlo.add %i1, %step : tensor<f64>
          %one_i = stablehlo.constant dense<1> : tensor<i32>
          %new_j = stablehlo.add %i2, %one_i : tensor<i32>
          stablehlo.return %new_state_i, %new_q, %new_j : tensor<2xui64>, tensor<f64>, tensor<i32>
        }

        %one_m = stablehlo.constant dense<1> : tensor<i32>
        %new_depth = stablehlo.add %m2, %one_m : tensor<i32>
        stablehlo.return %inn#0, %inn#1, %new_depth : tensor<2xui64>, tensor<f64>, tensor<i32>
      }

      %one_o = stablehlo.constant dense<1> : tensor<i32>
      %new_iter = stablehlo.add %arg2, %one_o : tensor<i32>
      stablehlo.return %mid#0, %mid#1, %new_iter : tensor<2xui64>, tensor<f64>, tensor<i32>
    }

    return %outer#1 : tensor<f64>
  }
}
"""

client = XLA.default_backend()
device = XLA.default_device()
platform = lowercase(XLA.platform_name(client))
@info "Running on platform: $platform"

@testset "TPU MWE - nested while + rng_bit_generator u64 crash" begin
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)

    mod = parse(MLIR.IR.Module, NESTED_WHILE_RNG_MODULE)

    compile_options = XLA.make_compile_options(;
        device_id=Int64(XLA.device_ordinal(device)),
    )

    # The crash happens HERE during XLA compilation on TPU:
    # AllReduceSimplifier -> HloReplicationAnalysis -> GetSubshape(u32[2,2], {0})
    exec = XLA.compile(
        client,
        mod;
        compile_options,
        num_parameters=Int64(1),
        num_outputs=Int64(1),
        is_sharded=false,
        num_replicas=Int64(1),
        num_partitions=Int64(1),
    )

    @test true  # If we get here, the crash is fixed

    MLIR.IR.deactivate(ctx)
    MLIR.IR.dispose(ctx)
end
