using MPI
using Reactant
using Reactant: Ops

                    # operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
                    # output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>],
                    # result_layouts = [dense<[1, 0]>

MPI.Init()

comm = MPI.COMM_WORLD

function comm_rank(x)
    Ops.hlo_call("""
        module {
            func.func @main(%comm : tensor<i64>) -> tensor<i64> {
                %0 = stablehlo.custom_call @enzymexla_ffi_mpi_comm_rank(%comm) {
                    api_version = 4 : i32,
                } : (tensor<i64>) -> tensor<i64>
                return %0 : tensor<i64>
            }
        }
        """, x)
end

@info "naive" rank = comm_rank(comm)

rank = @jit sync=true comm_rank(comm)
@info "reactant" rank = rank

code = @code_hlo comm_rank(comm)
println(code)

MPI.Finalize()
