using Reactant: MLIR

@testset "inject" begin
    mod = MLIR.IR.with_context() do ctx
        mod = MLIR.IR.Module()

        MLIR.IR.mmodule!(mod) do
            MLIR.IR.inject!(
                "MPI_COMM_WORLD",
                "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr",
            )

            MLIR.IR.inject!(
                "MPI_Comm_rank", "llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32"
            )

            MLIR.IR.inject!(
                "wrapper_function",
                """
                llvm.func @wrapper_function(%rank_ptr : !llvm.ptr) -> () {
                    %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
                    %errcode = llvm.call @MPI_Comm_rank(%comm, %rank_ptr) : (!llvm.ptr, !llvm.ptr) -> (i32)
                    llvm.return
                }
                """,
            )
        end

        return mod
    end

    mod_op = MLIR.IR.Operation(mod)
    @test MLIR.API.mlirOperationVerify(mod_op)
end
