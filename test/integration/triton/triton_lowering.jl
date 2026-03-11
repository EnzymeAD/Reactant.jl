using Reactant, Test, StableRNGs

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

const vector_add_kernel = """
module {
  func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c = stablehlo.constant dense<16> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %0 = enzymexla_tt_ext.call @triton_module::@triton_module_inner::@add_kernel clusters in(%c_0, %c_0, %c_0) blocks in(%c, %c_0, %c_0) (%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  enzymexla_tt_ext.module @triton_module {
    builtin.module @triton_module_inner attributes {enzymexla.num_stages = 3 : i32, enzymexla.num_warps = 4 : i32} {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
        %cst = arith.constant dense<98432> : tensor<1024xi32>
        %c1024_i32 = arith.constant 1024 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c1024_i32 : i32
        %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        %3 = tt.splat %1 : i32 -> tensor<1024xi32>
        %4 = arith.addi %3, %2 : tensor<1024xi32>
        %5 = arith.cmpi slt, %4, %cst : tensor<1024xi32>
        %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %8 = tt.load %7, %5 : tensor<1024x!tt.ptr<f32>>
        %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %10 = tt.addptr %9, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %11 = tt.load %10, %5 : tensor<1024x!tt.ptr<f32>>
        %12 = arith.addf %8, %11 : tensor<1024xf32>
        %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %14 = tt.addptr %13, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        tt.store %14, %12, %5 : tensor<1024x!tt.ptr<f32>>
        tt.return
      }
    }
  }
}
"""

vector_add_triton(x, y) = Reactant.Ops.hlo_call(vector_add_kernel, x, y)[1]

@testset "TTIR MLIR Lowering" begin
    x_ra = Reactant.to_rarray(rand(StableRNG(0), Float32, 1024))
    y_ra = Reactant.to_rarray(rand(StableRNG(4), Float32, 1024))

    # TODO: ROCM should also work, but needs testing
    if RunningOnCUDA
        props = Reactant.XLA.device_properties(
            Reactant.XLA.default_device(Reactant.XLA.client(Reactant.devices()[1]))
        )

        if props.major ≥ 8
            @test @jit(vector_add_triton(x_ra, y_ra)) ≈ .+(x_ra, y_ra)
        else
            @warn "TTIR lowering test skipped: requires NVIDIA Ampere (SM 8.0) or later, \
                   found SM $(props.major).$(props.minor)"
        end
    end
end
