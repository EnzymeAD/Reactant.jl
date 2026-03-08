using Reactant, Test
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

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

function standard_normal_logpdf(x)
    return -0.5 * sum(x .^ 2)
end

function nuts_mwe(rng, logpdf_fn, initial_position, step_size, inverse_mass_matrix)
    samples, diagnostics, rng = ProbProg.mcmc_logpdf(
        rng,
        logpdf_fn,
        initial_position;
        algorithm=:NUTS,
        step_size,
        inverse_mass_matrix,
        max_tree_depth=3,
        num_warmup=0,
        num_samples=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
    )
    return samples
end

@testset "TPU NUTS MWE - XLA HloReplicationAnalysis crash" begin
    seed = Reactant.to_rarray(UInt64[1, 5])
    rng = ReactantRNG(seed)
    initial_position = Reactant.to_rarray(reshape([0.5, -0.5], 1, 2))
    step_size = ConcreteRNumber(0.1)
    inverse_mass_matrix = ConcreteRArray([1.0 0.0; 0.0 1.0])

    compiled = @compile optimize = :probprog nuts_mwe(
        rng, standard_normal_logpdf, initial_position, step_size, inverse_mass_matrix
    )
    result = compiled(
        rng, standard_normal_logpdf, initial_position, step_size, inverse_mass_matrix
    )
    @test size(Array(result)) == (1, 2)
end
