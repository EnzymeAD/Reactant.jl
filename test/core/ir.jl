using Reactant: Reactant, MLIR

@testset "inject" begin
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() mod = MLIR.IR.Module(
        MLIR.IR.Location(; context=ctx)
    ) begin
        MLIR.IR.@scope ctx mod begin
            MLIR.IR.inject!(
                "MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr"
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

            mod_op = MLIR.IR.Operation(mod)
            @test MLIR.API.mlirOperationVerify(mod_op)
        end
    end
end
