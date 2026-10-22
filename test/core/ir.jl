using Reactant: MLIR

@testset "inject" begin
    MLIR.IR.@with_context Reactant.ReactantContext() begin
        mod = MLIR.IR.Module()

        MLIR.IR.@with_module mod begin
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
        end

        mod_op = MLIR.IR.Operation(mod)
        @test MLIR.API.mlirOperationVerify(mod_op)

        MLIR.IR.dispose(mod)
    end
end

@testset "location of a disposed context" begin
    ctx = Reactant.ReactantContext()
    loc = MLIR.IR.@with_context ctx MLIR.IR.Location("a.jl", 3, 4)
    @test occursin("a.jl", sprint(show, loc))
    MLIR.IR.dispose(ctx)
    @test occursin("disposed", sprint(show, loc))

    # an error that carries a location, shown after the compilation it came
    # from has disposed of its context
    function bad_fill(x)
        loc = MLIR.IR.Location()
        return Reactant.Ops.fill(1.0, (Base.OneTo(2), "not a size"); location=loc) .+ x
    end
    err = try
        @jit bad_fill(Reactant.to_rarray([1.0, 2.0]))
        nothing
    catch e
        e
    end
    @test err isa MethodError
    @test occursin("location", sprint(showerror, err))
    @test occursin("disposed", sprint(show, err.args))
end
