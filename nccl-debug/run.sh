mpiexecjl -n 2 julia --color=yes --project nccl.jl
gdb -batch -ex run -ex bt --args bash -lc 'XLA_REACTANT_GPU_MEM_FRACTION=0.5 XLA_REACTANT_GPU_PREALLOCATE=false julia --project nccl.jl'
gdb --batch -ex 'set env XLA_REACTANT_GPU_MEM_FRACTION 0.5' -ex 'set env XLA_REACTANT_GPU_PREALLOCATE false' -ex run -ex bt --args julia --project nccl.jl
