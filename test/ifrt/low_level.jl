# Testing manual IFRT buffer creation + compilation + execution
using Reactant, Test
using Reactant: XLA, ConcretePJRTArray
using Reactant.XLA: IFRT, PJRT

fn_test1(x, y) = x .+ y
fn_test2(x, y) = x .* y
fn_test3(x, y) = x .+ y' .- x

@testset "IFRT Low-level API" begin
    x = reshape(collect(Float32, 1:64), 8, 8)
    y = collect((x .+ 64)')

    pjrt_client = Reactant.XLA.default_backend()
    platform_name = lowercase(XLA.platform_name(pjrt_client))

    ifrt_client, pjrt_client = if platform_name == "cpu"
        IFRT.CPUClient(; checkcount=false), PJRT.CPUClient(; checkcount=false)
    elseif platform_name == "gpu" || platform_name == "cuda"
        IFRT.GPUClient(; checkcount=false), PJRT.GPUClient(; checkcount=false)
    elseif platform_name == "tpu"
        IFRT.TPUClient(; checkcount=false), PJRT.TPUClient(; checkcount=false)
    else
        error("Unsupported platform: $(platform_name)")
    end

    pjrt_x = ConcretePJRTArray(x; client=pjrt_client)
    pjrt_y = ConcretePJRTArray(y; client=pjrt_client)

    ifrt_x = IFRT.Array(ifrt_client, x)
    ifrt_y = IFRT.Array(ifrt_client, y)

    @testset for fn in (fn_test1, fn_test2, fn_test3)
        pjrt_result = @jit fn(pjrt_x, pjrt_y)

        mlir_mod, mlir_fn_res = Reactant.Compiler.compile_mlir(fn, (pjrt_x, pjrt_y))

        ifrt_loaded_executable = XLA.compile(
            ifrt_client,
            XLA.default_device(ifrt_client),
            mlir_mod;
            num_outputs=length(mlir_fn_res.linear_results),
            num_parameters=length(mlir_fn_res.linear_args),
            mlir_fn_res.is_sharded,
            global_device_ids=Int64[],
            num_replicas=1,
            num_partitions=1,
        )

        ifrt_result = XLA.execute(
            ifrt_loaded_executable, (ifrt_x.buffer, ifrt_y.buffer), UInt8.((0, 0)), Val(1)
        )

        @test convert(Array, only(ifrt_result)) ≈ Array(pjrt_result)
    end
end
