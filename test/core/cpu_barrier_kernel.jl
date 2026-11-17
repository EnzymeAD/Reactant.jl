using Reactant, Test

const _NVVM_BARRIER_NOOP_IR = """
module {
  llvm.func ptx_kernelcc @barrier(%ptr: !llvm.ptr<1>) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    nvvm.barrier id = %zero
    llvm.return
  }
  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %result = enzymexla.kernel_call @barrier
      blocks in(%one, %one, %one) threads in(%two, %one, %one)
      shmem = %zero (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}>
      : (tensor<2xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
"""

const _NVVM_BARRIER_WRITE_IR = """
module {
  llvm.func ptx_kernelcc @barrier_write(%ptr: !llvm.ptr<1>) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    nvvm.barrier id = %zero
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %tid64 = llvm.sext %tid : i32 to i64
    %gep = llvm.getelementptr %ptr[%tid64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %one = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %tidf = llvm.sitofp %tid : i32 to f32
    %val = llvm.fadd %tidf, %one : f32
    llvm.store %val, %gep : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %one = stablehlo.constant dense<1> : tensor<i64>
    %four = stablehlo.constant dense<4> : tensor<i64>
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %result = enzymexla.kernel_call @barrier_write
      blocks in(%one, %one, %one) threads in(%four, %one, %one)
      shmem = %zero (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>],
       xla_side_effect_free}>
      : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
"""

const _GPU_BARRIER_VARIANTS_IR = """
module {
  llvm.func ptx_kernelcc @barrier_variants(%ptr: !llvm.ptr<1>) {
    nvvm.barrier
    gpu.barrier
    rocdl.s.barrier
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %tid64 = llvm.sext %tid : i32 to i64
    %gep = llvm.getelementptr %ptr[%tid64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %two = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %tidf = llvm.sitofp %tid : i32 to f32
    %val = llvm.fadd %tidf, %two : f32
    llvm.store %val, %gep : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%x: tensor<3xf32>) -> tensor<3xf32> {
    %one = stablehlo.constant dense<1> : tensor<i64>
    %three = stablehlo.constant dense<3> : tensor<i64>
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %result = enzymexla.kernel_call @barrier_variants
      blocks in(%one, %one, %one) threads in(%three, %one, %one)
      shmem = %zero (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>],
       xla_side_effect_free}>
      : (tensor<3xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }
}
"""

const _MULTIBLOCK_STATIC_SHMEM_IR = """
module {
  llvm.mlir.global internal @shmem() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
  llvm.func ptx_kernelcc @barrier_shmem(%ptr: !llvm.ptr<1>) {
    %bid = nvvm.read.ptx.sreg.ctaid.x : i32
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %bid64 = llvm.sext %bid : i32 to i64
    %tid64 = llvm.sext %tid : i32 to i64
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %base = llvm.mul %bid64, %c2 : i64
    %gidx = llvm.add %base, %tid64 : i64
    %sh = llvm.mlir.addressof @shmem : !llvm.ptr<3>
    %gep_sh_w = llvm.getelementptr %sh[%c0, %tid64] : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>, !llvm.array<2 x f32>
    %gep_in = llvm.getelementptr %ptr[%gidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %val = llvm.load %gep_in : !llvm.ptr<1> -> f32
    llvm.store %val, %gep_sh_w : f32, !llvm.ptr<3>
    %zero = llvm.mlir.constant(0 : i32) : i32
    nvvm.barrier id = %zero
    %other = llvm.sub %c1, %tid64 : i64
    %gep_sh_r = llvm.getelementptr %sh[%c0, %other] : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>, !llvm.array<2 x f32>
    %swapped = llvm.load %gep_sh_r : !llvm.ptr<3> -> f32
    llvm.store %swapped, %gep_in : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%x: tensor<6xf32>) -> tensor<6xf32> {
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %three = stablehlo.constant dense<3> : tensor<i64>
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %result = enzymexla.kernel_call @barrier_shmem
      blocks in(%three, %one, %one) threads in(%two, %one, %one)
      shmem = %zero (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>],
       xla_side_effect_free}>
      : (tensor<6xf32>) -> tensor<6xf32>
    return %result : tensor<6xf32>
  }
}
"""

const _GRID2D_DYNAMIC_SHMEM_IR = """
module {
  llvm.mlir.global external @dyn_shmem() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<0 x f32>
  llvm.func ptx_kernelcc @barrier_dyn_shmem(%ptr: !llvm.ptr<1>) {
    %bx = nvvm.read.ptx.sreg.ctaid.x : i32
    %by = nvvm.read.ptx.sreg.ctaid.y : i32
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %bx64 = llvm.sext %bx : i32 to i64
    %by64 = llvm.sext %by : i32 to i64
    %tid64 = llvm.sext %tid : i32 to i64
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %bxy = llvm.mul %bx64, %c2 : i64
    %bid = llvm.add %bxy, %by64 : i64
    %base = llvm.mul %bid, %c2 : i64
    %gidx = llvm.add %base, %tid64 : i64
    %sh = llvm.mlir.addressof @dyn_shmem : !llvm.ptr<3>
    %gep_sh_w = llvm.getelementptr %sh[%c0, %tid64] : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>, !llvm.array<0 x f32>
    %gep_in = llvm.getelementptr %ptr[%gidx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %val = llvm.load %gep_in : !llvm.ptr<1> -> f32
    llvm.store %val, %gep_sh_w : f32, !llvm.ptr<3>
    %zero = llvm.mlir.constant(0 : i32) : i32
    nvvm.barrier id = %zero
    %other = llvm.sub %c1, %tid64 : i64
    %gep_sh_r = llvm.getelementptr %sh[%c0, %other] : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>, !llvm.array<0 x f32>
    %swapped = llvm.load %gep_sh_r : !llvm.ptr<3> -> f32
    llvm.store %swapped, %gep_in : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %eight = stablehlo.constant dense<8> : tensor<i64>
    %result = enzymexla.kernel_call @barrier_dyn_shmem
      blocks in(%two, %two, %one) threads in(%two, %one, %one)
      shmem = %eight (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>],
       xla_side_effect_free}>
      : (tensor<8xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
"""

run_nvvm_barrier_noop(x) = only(Reactant.Ops.hlo_call(_NVVM_BARRIER_NOOP_IR, x))
run_nvvm_barrier_write(x) = only(Reactant.Ops.hlo_call(_NVVM_BARRIER_WRITE_IR, x))
run_gpu_barrier_variants(x) = only(Reactant.Ops.hlo_call(_GPU_BARRIER_VARIANTS_IR, x))
run_multiblock_static_shmem(x) = only(Reactant.Ops.hlo_call(_MULTIBLOCK_STATIC_SHMEM_IR, x))
run_grid2d_dynamic_shmem(x) = only(Reactant.Ops.hlo_call(_GRID2D_DYNAMIC_SHMEM_IR, x))

@testset "CPU GPU-barrier and shared-memory kernel lowering (#3284)" begin
    cpu_client = Reactant.XLA.client("cpu")

    @testset "nvvm.barrier reproducer (#3284)" begin
        x0 = Reactant.to_rarray(Float32[3.0, 5.0]; client=cpu_client)
        hlo_unopt = @code_hlo client = cpu_client optimize = false run_nvvm_barrier_noop(x0)
        @test occursin("nvvm.barrier", repr(hlo_unopt))

        res0 = @jit client = cpu_client run_nvvm_barrier_noop(x0)
        @test Array(res0) == Float32[3.0, 5.0]

        x1 = Reactant.to_rarray(zeros(Float32, 4); client=cpu_client)
        res1 = @jit client = cpu_client run_nvvm_barrier_write(x1)
        @test Array(res1) == Float32[1.0, 2.0, 3.0, 4.0]

        x1_noopt = Reactant.to_rarray(zeros(Float32, 4); client=cpu_client)
        res1_noopt = @jit client = cpu_client optimize = :noopt run_nvvm_barrier_write(
            x1_noopt
        )
        @test Array(res1_noopt) == Float32[1.0, 2.0, 3.0, 4.0]

        x1_raise = Reactant.to_rarray(zeros(Float32, 4); client=cpu_client)
        hlo_raise = @code_hlo client = cpu_client raise = true run_nvvm_barrier_write(
            x1_raise
        )
        @test !occursin("nvvm.barrier", repr(hlo_raise))
        @test !occursin("jit_call", repr(hlo_raise))
        res1_raise = @jit client = cpu_client raise = true run_nvvm_barrier_write(x1_raise)
        @test Array(res1_raise) == Float32[1.0, 2.0, 3.0, 4.0]
    end

    @testset "nvvm.barrier0, gpu.barrier, and rocdl.s.barrier variants" begin
        x = Reactant.to_rarray(zeros(Float32, 3); client=cpu_client)
        res = @jit client = cpu_client run_gpu_barrier_variants(x)
        @test Array(res) == Float32[2.0, 3.0, 4.0]
    end

    @testset "Multi-block static and dynamic shared memory + barrier on CPU" begin
        saved_omp = Reactant.Compiler.OpenMP[]
        try
            for omp_enabled in (true, false)
                Reactant.Compiler.OpenMP[] = omp_enabled

                x_static = Reactant.to_rarray(
                    Float32[10, 20, 30, 40, 50, 60]; client=cpu_client
                )
                res_static = @jit client = cpu_client run_multiblock_static_shmem(x_static)
                @test Array(res_static) == Float32[20, 10, 40, 30, 60, 50]

                x_dyn = Reactant.to_rarray(
                    Float32[1, 2, 3, 4, 5, 6, 7, 8]; client=cpu_client
                )
                res_dyn = @jit client = cpu_client run_grid2d_dynamic_shmem(x_dyn)
                @test Array(res_dyn) == Float32[2, 1, 4, 3, 6, 5, 8, 7]
            end

            x_static_raise = Reactant.to_rarray(
                Float32[10, 20, 30, 40, 50, 60]; client=cpu_client
            )
            res_static_raise = @jit client = cpu_client raise = true run_multiblock_static_shmem(
                x_static_raise
            )
            @test Array(res_static_raise) == Float32[20, 10, 40, 30, 60, 50]
        finally
            Reactant.Compiler.OpenMP[] = saved_omp
        end
    end
end
