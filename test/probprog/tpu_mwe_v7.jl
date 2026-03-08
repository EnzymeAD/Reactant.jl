using Reactant, Test
using Reactant: Ops

# v7: many args + Threefry + stablehlo.if(16 results) + rng in else (145 lines)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_v7.mlir"), String)

@testset "TPU MWE v7 - many args + Threefry + if + rng" begin
    Reactant.@jit(Ops.hlo_call(MWE_MODULE; dummy_arguments=true))
    @test true
end
