using Test
using MPI
using NCCL
using CUDA
using Reactant

using InteractiveUtils

# Reactant.set_default_backend("cpu") # sanity check: see if it works on cpu
# Reactant.set_default_backend("gpu")

const ReactantNCCLExt = Base.get_extension(Reactant, :ReactantNCCLExt)
ReactantNCCLExt === nothing && error("ReactantNCCLExt is not loaded")

const MPICOMM = MPI.COMM_WORLD

MPI.Init()

ncclComm = ReactantNCCLExt.init_default_comm()

# # *debug*
# println("MPI rank $(MPI.Comm_rank(MPICOMM)) NCCL comm = ", ncclComm)
# println("MPI rank $(MPI.Comm_rank(MPICOMM)) XLA device = ",
#         Reactant.XLA.default_device(Reactant.XLA.default_backend()))

rank = MPI.Comm_rank(MPICOMM)


# MPI.Allreduce
if true
    # NOTE:
    #   for size 1: doesn't hang, and returns without crashing, but returns the wrong result, ie just returns zeros
    #   for a bigger size, ie 5: non-deterministically either hangs, or crashes with some sort of heap corruption error
    # Dealers choice going forward in terms of which one to target for debugging.
    send_buf = fill(Float64(rank + 1), 1)
    recv_buf = zeros(Float64, 1)
    rsend_buf = ConcreteRArray(send_buf)
    rrecv_buf = ConcreteRArray(recv_buf)

    rank==0 && println("\ncode_hlo optimize=false:\n",
            @code_hlo optimize=false MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM))
    rank==0 && println("\ncode_hlo optimize=\"lower-enzymexla-mpi{backend=cuda ncclCommPtr=1}\":\n",
            @code_hlo optimize="lower-enzymexla-mpi{backend=cuda ncclCommPtr=1}" MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM))
    rank==0 && println("\ncode_hlo sync=true:\n",
            @code_hlo MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM))

    result = @jit sync=true MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM)
    # Reactant.synchronize(result) # optionally use this instead of sync=true?
    println("MPI rank $(MPI.Comm_rank(MPICOMM)) send_buf = $(send_buf) result = $(Array(result)) rrecv_buf = $(Array(rrecv_buf))", )

    # # sanity check: make sure NCCL.jl works
    # recv_buf = NCCL.Allreduce!(CuArray(send_buf), CuArray(recv_buf), +, ncclComm)
    # CUDA.synchronize()
    # println(Array(recv_buf))

end

ReactantNCCLExt.destroy_default_comm()
MPI.Finalize()
