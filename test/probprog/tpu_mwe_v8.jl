using Reactant, Test
using Reactant: Ops

# v8: many args + Threefry + rng, NO stablehlo.if (131 lines)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_v8.mlir"), String)

@testset "TPU MWE v8 - many args + Threefry + rng, no if" begin
    Reactant.@jit(Ops.hlo_call(MWE_MODULE; dummy_arguments=true))
    @test true
end
