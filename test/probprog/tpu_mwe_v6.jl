using Reactant, Test
using Reactant: MLIR, XLA

# v6: Few while args + Threefry + stablehlo.if with 8 results (81 lines)
const MWE_MODULE = read(joinpath(@__DIR__, "mwe_v6.mlir"), String)

client = XLA.default_backend()
device = XLA.default_device()
@info "v6 test on $(lowercase(XLA.platform_name(client)))"

@testset "TPU MWE v6 - few args + Threefry + stablehlo.if" begin
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
