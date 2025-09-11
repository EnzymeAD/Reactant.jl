# -------------------
# perlmutter
# -------------------
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=nstaff

# Flags from https://github.com/PRONTOLab/GB-25/blob/main/sharding/perlmutter_scaling_test.jl
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_CUDA_USE_COMPAT=false

# Flag from: https://github.com/PRONTOLab/GB-25/blob/main/sharding/common_submission_generator.jl
export XLA_REACTANT_GPU_MEM_FRACTION=0.9

srun -n 2 julia --project ./mpi.jl

# Then added this flag to srun
srun -n 2 --gpus-per-task=1 julia --project ./mpi.jl


# -------------------
# local laptop
# -------------------
mpiexec -n 2 julia --project mpi.jl
