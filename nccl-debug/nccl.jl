using Test
using MPI
using NCCL
using CUDA
using Reactant

using InteractiveUtils

# Reactant.set_default_backend("cpu")
# Reactant.set_default_backend("gpu")

const ReactantNCCLExt = Base.get_extension(Reactant, :ReactantNCCLExt)
ReactantNCCLExt === nothing && error("ReactantNCCLExt is not loaded")

const MPICOMM = MPI.COMM_WORLD

MPI.Init()
ncclComm = ReactantNCCLExt.init_default_comm()
println("MPI rank $(MPI.Comm_rank(MPICOMM)) NCCL comm = ", ncclComm)
println("MPI rank $(MPI.Comm_rank(MPICOMM)) XLA device = ",
        Reactant.XLA.default_device(Reactant.XLA.default_backend()))


# MPI.Allreduce
if true
    send_buf = ones(Float64, 5)
    recv_buf = ones(Float64, 5)
    rsend_buf = ConcreteRArray(send_buf)
    rrecv_buf = ConcreteRArray(recv_buf)

    # println("\ncode_hlo optimize=false:\n",
    #         @code_hlo optimize=false MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM))
    # println("\ncode_hlo optimize=\"lower-enzymexla-mpi{backend=cuda ncclCommPtr=1}\":\n",
    #         @code_hlo optimize="lower-enzymexla-mpi{backend=cuda ncclCommPtr=1}" MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM))
    println("\ncode_hlo:\n",
            @code_hlo MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM))

    # println(MPI.Allreduce!(send_buf, ConcreteRArray(recv_buf), MPI.SUM, MPICOMM))
    result = @jit MPI.Allreduce!(rsend_buf, rrecv_buf, MPI.SUM, MPICOMM)
    Reactant.synchronize(result)
    println("MPI rank $(MPI.Comm_rank(MPICOMM)) result = ", Array(result))


    # recv_buf = NCCL.Allreduce!(CuArray(send_buf), CuArray(recv_buf), +, ncclComm)
    # CUDA.synchronize()
    # println(Array(recv_buf))

end

ReactantNCCLExt.destroy_default_comm()
MPI.Finalize()
