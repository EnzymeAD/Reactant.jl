const DEFAULT_COMM = Ref{Union{Nothing,NCCL.Communicator}}(nothing)
const DEFAULT_COMM_HANDLE = Ref{UInt}(0)

function local_rank(comm::MPI.Comm)
    shared = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, 0)
    try
        return MPI.Comm_rank(shared)
    finally
        MPI.free(shared)
    end
end

function choose_device(lrank::Integer)
    devices = collect(NCCL.CUDA.devices())
    isempty(devices) && error("No CUDA devices available for NCCL communicator setup")
    return devices[mod(lrank, length(devices)) + 1]
end

function init_default_comm(; comm::MPI.Comm=MPI.COMM_WORLD)
    if DEFAULT_COMM[] !== nothing
        return DEFAULT_COMM[]
    end

    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    lrank = local_rank(comm)
    device = choose_device(lrank)

    NCCL.CUDA.device!(device)

    unique_id = if rank == 0
        NCCL.UniqueID()
    else
        nothing
    end
    unique_id = MPI.bcast(unique_id, comm; root=0)

    nccl_comm = NCCL.Communicator(nranks, rank; unique_id)
    DEFAULT_COMM[] = nccl_comm
    DEFAULT_COMM_HANDLE[] = Base.reinterpret(UInt, nccl_comm.handle)

    return nccl_comm
end

function destroy_default_comm()
    if DEFAULT_COMM[] !== nothing
        NCCL.destroy(DEFAULT_COMM[]::NCCL.Communicator)
        DEFAULT_COMM[] = nothing
        DEFAULT_COMM_HANDLE[] = 0
    end
    return nothing
end

function default_comm()
    comm = DEFAULT_COMM[]
    comm === nothing && error("Default NCCL communicator has not been initialized")
    return comm
end

function default_comm_handle()
    DEFAULT_COMM[] === nothing &&
        error("Default NCCL communicator has not been initialized")
    return DEFAULT_COMM_HANDLE[]
end
