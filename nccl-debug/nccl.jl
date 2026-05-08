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


# MPI.Allreduce
if true
    send_buf = ones(Float64, 5)
    recv_buf = ones(Float64, 5)

    println("\ncode_hlo optimize=false:\n",
            @code_hlo optimize=false MPI.Allreduce!(ConcreteRArray(send_buf), ConcreteRArray(recv_buf), MPI.SUM, MPICOMM))
    println("\ncode_hlo optimize=\"lower-enzymexla-mpi{backend=cuda ncclCommPtr=1}\":\n",
            @code_hlo optimize="lower-enzymexla-mpi{backend=cuda ncclCommPtr=1}" MPI.Allreduce!(ConcreteRArray(send_buf), ConcreteRArray(recv_buf), MPI.SUM, MPICOMM))
    println("\ncode_hlo:\n",
            @code_hlo MPI.Allreduce!(ConcreteRArray(send_buf), ConcreteRArray(recv_buf), MPI.SUM, MPICOMM))

    # println(MPI.Allreduce!(send_buf, ConcreteRArray(recv_buf), MPI.SUM, MPICOMM))
    println(@jit MPI.Allreduce!(ConcreteRArray(send_buf), ConcreteRArray(recv_buf), MPI.SUM, MPICOMM))
    TODO might need a Reactant CUDA Synchronize here before we inspect the output?


    # recv_buf = NCCL.Allreduce!(CuArray(send_buf), CuArray(recv_buf), +, ncclComm)
    # CUDA.synchronize()
    # println(Array(recv_buf))

end

ReactantNCCLExt.destroy_default_comm()
MPI.Finalize()
