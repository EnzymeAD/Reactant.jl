using Reactant, Test
using Reactant: Ops

# b2_no_leapfrog: full 5-round Threefry + many args + if(16), NO dot_general (313 lines)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_b2_no_leapfrog.mlir"), String)

@testset "TPU MWE b2_no_leapfrog" begin
    Reactant.@jit(Ops.hlo_call(MWE_MODULE; dummy_arguments=true))
    @test true
end
