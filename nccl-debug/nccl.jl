using Test
using MPI
using NCCL
using Reactant

const ReactantNCCLExt = Base.get_extension(Reactant, :ReactantNCCLExt)
ReactantNCCLExt === nothing && error("ReactantNCCLExt is not loaded")

const COMM = MPI.COMM_WORLD

MPI.Init()
ReactantNCCLExt.init_default_comm()

println("\ncode_hlo optimize=false:\n",
        @code_hlo optimize=false MPI.Comm_rank(COMM))

ReactantNCCLExt.destroy_default_comm()
MPI.Finalize()
