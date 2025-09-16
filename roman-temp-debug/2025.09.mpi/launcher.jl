# launcher.jl
# usage e.g.: julia launcher.jl 1 test-isend-irecv.jl
using MPI
run(`$(MPI.mpiexec()) -n $(ARGS[1]) julia --project $(ARGS[2])`)
