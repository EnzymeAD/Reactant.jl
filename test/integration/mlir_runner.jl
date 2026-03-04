using Reactant, Test
using Reactant.MLIR
using Reactant.MLIR.IR

const MLIRRunner = Reactant.Serialization.MLIRRunner

# Example MLIR modules for testing (matching scripts/example_*.mlir)
const EXAMPLE_FIRST_MLIR = """
module @example_first attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=1]>
  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>, tf.aliasing_output = 0 : i32}, %arg2: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 1 : i32}) -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
    %0 = stablehlo.add %arg0, %arg2 : tensor<4xf32>
    return %arg1, %0 : tensor<f32>, tensor<4xf32>
  }
}
"""

const EXAMPLE_LOOP_MLIR = """
module @example_loop attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=1]>
  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>, tf.aliasing_output = 0 : i32}, %arg2: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, tf.aliasing_output = 1 : i32}, %arg3: tensor<i64> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
    %0 = stablehlo.add %arg0, %arg2 : tensor<4xf32>
    return %arg1, %0 : tensor<f32>, tensor<4xf32>
  }
}
"""

# Simple module with no aliasing
const SIMPLE_MLIR = """
module @simple attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=1]>
  func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""

# Multi-arg module (matmul-like)
const MULTI_ARG_MLIR = """
module @multi_arg attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  sdy.mesh @mesh = <["x"=1]>
  func.func @main(%arg0: tensor<3x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<4x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}
"""

@testset "MLIRRunner" begin
    @testset "analyze_module — single module, no aliasing" begin
        ctx = Reactant.ReactantContext()
        IR.activate(ctx)

        info = MLIRRunner.analyze_module(SIMPLE_MLIR)

        IR.deactivate(ctx)

        @test length(info.inputs) == 1
        @test length(info.outputs) == 1
        @test info.inputs[1].eltype == "Float32"
        @test info.inputs[1].mlir_shape == [4, 4]
        @test info.outputs[1].eltype == "Float32"
        @test info.outputs[1].mlir_shape == [4, 4]
        @test isempty(info.alias_map)
        @test info.num_partitions == 1
        @test info.num_replicas == 1
        @test info.mesh_axes == [:x]
        @test info.mesh_sizes == [1]
    end

    @testset "analyze_module — multi-arg" begin
        ctx = Reactant.ReactantContext()
        IR.activate(ctx)

        info = MLIRRunner.analyze_module(MULTI_ARG_MLIR)

        IR.deactivate(ctx)

        @test length(info.inputs) == 2
        @test info.inputs[1].eltype == "Float32"
        @test info.inputs[1].mlir_shape == [3, 4]
        @test info.inputs[2].eltype == "Float32"
        @test info.inputs[2].mlir_shape == [4, 5]
        @test length(info.outputs) == 1
        @test info.outputs[1].mlir_shape == [3, 5]
        @test isempty(info.alias_map)
    end

    @testset "analyze_module — aliasing (first_time_step)" begin
        ctx = Reactant.ReactantContext()
        IR.activate(ctx)

        info = MLIRRunner.analyze_module(EXAMPLE_FIRST_MLIR)

        IR.deactivate(ctx)

        @test length(info.inputs) == 3
        @test length(info.outputs) == 2

        # arg0: grid (no alias) — constant
        @test info.inputs[1].eltype == "Float32"
        @test info.inputs[1].mlir_shape == [4]

        # arg1: dt, tf.aliasing_output=0 → output 1 maps to input 2
        @test info.inputs[2].eltype == "Float32"
        @test info.inputs[2].mlir_shape == Int[]  # scalar

        # arg2: state, tf.aliasing_output=1 → output 2 maps to input 3
        @test info.inputs[3].eltype == "Float32"
        @test info.inputs[3].mlir_shape == [4]

        # Alias map: output 1 (0-based 0) → input 2, output 2 (0-based 1) → input 3
        @test info.alias_map[1] == 2  # output 1 → input 2 (dt)
        @test info.alias_map[2] == 3  # output 2 → input 3 (state)
    end

    @testset "analyze_module — loop module with extra arg" begin
        ctx = Reactant.ReactantContext()
        IR.activate(ctx)

        info = MLIRRunner.analyze_module(EXAMPLE_LOOP_MLIR)

        IR.deactivate(ctx)

        @test length(info.inputs) == 4
        @test length(info.outputs) == 2

        # arg3: ninner (Int64 scalar, no alias)
        @test info.inputs[4].eltype == "Int64"
        @test info.inputs[4].mlir_shape == Int[]

        # Same alias pattern as first
        @test info.alias_map[1] == 2
        @test info.alias_map[2] == 3
    end

    @testset "generate_mlir_runner — single module" begin
        mktempdir() do dir
            mlir_path = joinpath(dir, "simple.mlir")
            write(mlir_path, SIMPLE_MLIR)
            out_path = joinpath(dir, "execute.jl")

            result = Reactant.Serialization.generate_mlir_runner(
                [mlir_path]; output_path=out_path
            )

            @test isfile(out_path)
            @test result == out_path

            script = read(out_path, String)
            @test contains(script, "ConcreteRArray")
            @test contains(script, "compile_module")
            @test contains(script, "MLIR_PATH_1")
            @test contains(script, "N_IN_1")
            @test contains(script, "N_OUT_1")
            @test contains(script, "main()")
            # Single module: no marshaling
            @test !contains(script, "MLIR_PATH_2")
        end
    end

    @testset "generate_mlir_runner — two modules with aliasing" begin
        mktempdir() do dir
            first_path = joinpath(dir, "first.mlir")
            loop_path = joinpath(dir, "loop.mlir")
            write(first_path, EXAMPLE_FIRST_MLIR)
            write(loop_path, EXAMPLE_LOOP_MLIR)
            out_path = joinpath(dir, "execute.jl")

            Reactant.Serialization.generate_mlir_runner(
                [first_path, loop_path]; output_path=out_path
            )

            @test isfile(out_path)

            script = read(out_path, String)

            # Both modules referenced
            @test contains(script, "MLIR_PATH_1")
            @test contains(script, "MLIR_PATH_2")
            @test contains(script, "N_IN_1")
            @test contains(script, "N_IN_2")

            # Alias maps present
            @test contains(script, "ALIAS_MAP_1")
            @test contains(script, "ALIAS_MAP_2")

            # Marshaling code
            @test contains(script, "marshal_next_inputs")
            @test contains(script, "CONST_INDICES")

            # Extra inputs for loop (ninner)
            @test contains(script, "create_extra_inputs_2")

            # Not hardcoded n_grid_const
            @test !contains(script, "N_GRID_CONST")
        end
    end

    @testset "generate_mlir_runner — vararg (3 modules)" begin
        mktempdir() do dir
            # Use same module 3 times
            for name in ["mod1.mlir", "mod2.mlir", "mod3.mlir"]
                write(joinpath(dir, name), EXAMPLE_FIRST_MLIR)
            end
            out_path = joinpath(dir, "execute.jl")

            Reactant.Serialization.generate_mlir_runner(
                [joinpath(dir, "mod$i.mlir") for i in 1:3];
                output_path=out_path
            )

            script = read(out_path, String)
            @test contains(script, "MLIR_PATH_3")
            @test contains(script, "N_IN_3")
            @test contains(script, "exec_3")
            @test contains(script, "Module 3")
        end
    end

    @testset "generate_mlir_runner — error on empty input" begin
        @test_throws ErrorException Reactant.Serialization.generate_mlir_runner(
            String[]; output_path="/tmp/dummy.jl"
        )
    end

    @testset "julia_shape_str / julia_sharding_expr helpers" begin
        sig_scalar = MLIRRunner.TensorSig("Float32", Int[], Symbol[])
        @test MLIRRunner.julia_shape_str(sig_scalar) == "()"
        @test MLIRRunner.julia_sharding_expr(sig_scalar) == "Sharding.Replicated(mesh)"

        sig_vec = MLIRRunner.TensorSig("Float32", [4], Symbol[:_none])
        @test MLIRRunner.julia_shape_str(sig_vec) == "(4,)"

        sig_mat = MLIRRunner.TensorSig("Float64", [3, 5], Symbol[:_none, :_none])
        @test MLIRRunner.julia_shape_str(sig_mat) == "(5, 3)"
        @test MLIRRunner.julia_sharding_expr(sig_mat) == "Sharding.Replicated(mesh)"

        sig_sharded = MLIRRunner.TensorSig("Float32", [4, 8], Symbol[:x, :_none])
        @test MLIRRunner.julia_sharding_expr(sig_sharded) == "Sharding.NamedSharding(mesh, (nothing, :x,))"
    end

    @testset "end-to-end with example MLIR files" begin
        scripts_dir = joinpath(dirname(dirname(@__DIR__)), "scripts")
        first_mlir = joinpath(scripts_dir, "example_first_pre_xla_compile.mlir")
        loop_mlir = joinpath(scripts_dir, "example_loop_pre_xla_compile.mlir")

        if isfile(first_mlir) && isfile(loop_mlir)
            mktempdir() do dir
                out_path = joinpath(dir, "execute.jl")
                Reactant.Serialization.generate_mlir_runner(
                    [first_mlir, loop_mlir]; output_path=out_path
                )

                @test isfile(out_path)
                script = read(out_path, String)

                # Verify it references both files
                @test contains(script, "example_first_pre_xla_compile.mlir")
                @test contains(script, "example_loop_pre_xla_compile.mlir")

                # Verify correct structure
                @test contains(script, "N_IN_1  = 3")
                @test contains(script, "N_OUT_1 = 2")
                @test contains(script, "N_IN_2  = 4")
                @test contains(script, "N_OUT_2 = 2")
            end
        else
            @warn "Skipping end-to-end test: example MLIR files not found at $scripts_dir"
        end
    end
end
