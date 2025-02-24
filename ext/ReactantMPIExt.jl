module ReactantMPIExt

using Reactant: Reactant, Distributed
using MPI: MPI

# https://github.com/jax-ml/jax/blob/b0117366686ab084d38ad2657d9a2ae3a581ca7e/jax/_src/clusters/mpi4py_cluster.py
Distributed.is_env_present(::Distributed.MPIEnvDetector) = MPI.Initialized()

function Distributed.get_coordinator_address(
    ::Distributed.MPIEnvDetector, timeout_in_seconds::Integer
)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        hostname = gethostname()
        port_id = hash(hostname) % 2^12 + (65535 - 2^12 + 1)
        hostname = "$(hostname):$(port_id)"
    else
        hostname = nothing
    end

    return MPI.bcast(hostname, MPI.COMM_WORLD; root=0)
end

function Distributed.get_process_count(::Distributed.MPIEnvDetector)
    return Int(MPI.Comm_size(MPI.COMM_WORLD))
end

function Distributed.get_process_id(::Distributed.MPIEnvDetector)
    return Int(MPI.Comm_rank(MPI.COMM_WORLD))
end

function Distributed.get_local_process_id(::Distributed.MPIEnvDetector)
    new_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
    return Int(MPI.Comm_rank(new_comm))
end

end
