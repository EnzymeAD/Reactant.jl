using Reactant, Test
using Reactant: MLIR, XLA

# b2: Full Threefry in nested while, inner while removed (472 lines)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_b2.mlir"), String)

client = XLA.default_backend()
device = XLA.default_device()
@info "b2 test on $(lowercase(XLA.platform_name(client)))"

@testset "TPU MWE b2 - Full Threefry, no inner while" begin
    ctx = Reactant.ReactantContext()
    MLIR.IR.activate(ctx)
    mod = parse(MLIR.IR.Module, MWE_MODULE)
    compile_options = XLA.make_compile_options(; device_id=Int64(XLA.device_ordinal(device)))
    exec = XLA.compile(client, mod; compile_options,
        num_parameters=Int64(4), num_outputs=Int64(2),
        is_sharded=false, num_replicas=Int64(1), num_partitions=Int64(1))
    @test true
    MLIR.IR.deactivate(ctx)
    MLIR.IR.dispose(ctx)
end
