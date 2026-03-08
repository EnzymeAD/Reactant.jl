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

# Load the StableHLO module that triggers the crash (dumped from compile_xla on CPU)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_stablehlo.mlir"), String)

client = XLA.default_backend()
device = XLA.default_device()
platform = lowercase(XLA.platform_name(client))
@info "Running on platform: $platform"

@testset "TPU MWE - StableHLO compilation crash" begin
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)

    mod = parse(MLIR.IR.Module, MWE_MODULE)

    compile_options = XLA.make_compile_options(;
        device_id=Int64(XLA.device_ordinal(device)),
    )

    # The crash happens HERE during XLA compilation on TPU:
    # AllReduceSimplifier -> HloReplicationAnalysis -> GetSubshape(u32[2,2], {0})
    # @main takes 4 args: tensor<2xui64>, tensor<2x1xf64>, tensor<f64>, tensor<2x2xf64>
    # and returns 2 results: tensor<2x1xf64>, tensor<2xui64>
    exec = XLA.compile(
        client,
        mod;
        compile_options,
        num_parameters=Int64(4),
        num_outputs=Int64(2),
        is_sharded=false,
        num_replicas=Int64(1),
        num_partitions=Int64(1),
    )

    @test true  # If we get here, the crash is fixed

    MLIR.IR.deactivate(ctx)
    MLIR.IR.dispose(ctx)
end
