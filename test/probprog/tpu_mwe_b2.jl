using Reactant, Test
using Reactant: Ops

# b2: Full Threefry in nested while, inner while removed (472 lines)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_b2.mlir"), String)

@testset "TPU MWE b2 - Full Threefry, no inner while" begin
    Reactant.@jit(Ops.hlo_call(MWE_MODULE; dummy_arguments=true))
    @test true
end
