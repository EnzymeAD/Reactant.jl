# This file contains variables, configuration and functions related to the MPI extension that need to be accessed
# either by users or by the main Reactant module.

# controls whether calls like `MPI.Comm_rank(MPI.COMM_WORLD)` should be traced
const MPI_OVERLAY_NATIVE_CALLS = Ref(false)

# bitwidth of `MPI_Request`, which is implementation-defined (e.g. `int` in MPICH, a pointer in Open MPI)
# set by `ReactantMPIExt` to match the loaded MPI library
const MPI_REQUEST_WIDTH = Ref(64)

function initialize_nccl_comm! end
