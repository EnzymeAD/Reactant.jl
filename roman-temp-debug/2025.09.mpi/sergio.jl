using Reactant
using MPI
using Libdl

Reactant.set_default_backend("cpu")

tag = 43
comm = MPI.COMM_WORLD
source = 1

println("Here we go!")

MPI.Init()

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    buffer = Reactant.to_rarray(zeros(Int32, 8))
    println("[$(MPI.Comm_rank(MPI.COMM_WORLD))] before - $buffer")
    @jit MPI.Recv!(buffer, source, tag, comm)
    println("[$(MPI.Comm_rank(MPI.COMM_WORLD))] after - $buffer")
    println(isapprox(buffer, ones(8)))
else
    buffer = ones(Int32, 8)
    destination = 0
    println("[$(MPI.Comm_rank(MPI.COMM_WORLD))] sending - $buffer")
    MPI.Send(buffer, destination, tag, comm)
    println("[$(MPI.Comm_rank(MPI.COMM_WORLD))] sent!")
end

MPI.Finalize()
