const DEFAULT_COMM = Ref{Union{Nothing,NCCL.Communicator}}(nothing)
const DEFAULT_COMM_HANDLE = Ref{UInt}(0)
const DEFAULT_XLA_DEVICE = Ref{Union{Nothing,Reactant.XLA.AbstractDevice}}(nothing)

function default_xla_device()
    return DEFAULT_XLA_DEVICE[]
end

function require_default_xla_device(client)
    override = default_xla_device()
    override === nothing &&
        error("ReactantNCCLExt default XLA device has not been initialized")
    Reactant.XLA.client(override) == client ||
        error("ReactantNCCLExt default XLA device does not belong to the active client")
    return override
end

function Reactant.XLA.default_device(client::Reactant.XLA.PJRT.Client)
    return require_default_xla_device(client)
end

function Reactant.XLA.default_device(client::Reactant.XLA.IFRT.Client)
    return require_default_xla_device(client)
end

function local_rank(comm::MPI.Comm)
    shared = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, 0)
    try
        return MPI.Comm_rank(shared)
    finally
        MPI.free(shared)
    end
end

function choose_xla_device(lrank::Integer)
    client = Reactant.XLA.default_backend()
    nd = Int(Reactant.XLA.num_addressable_devices(client))
    lrank < nd ||
        error("MPI local rank $lrank exceeds Reactant addressable device count $nd")
    return Reactant.XLA.get_addressable_device(client, lrank)
end

function choose_nccl_device(xla_device::Reactant.XLA.AbstractDevice)
    devices = collect(NCCL.CUDA.devices())
    isempty(devices) && error("No CUDA devices available for NCCL communicator setup")
    hardware_id = Int(Reactant.XLA.get_local_hardware_id(xla_device))
    0 <= hardware_id < length(devices) ||
        error("Reactant XLA device hardware id $hardware_id is out of range for $(length(devices)) CUDA devices")
    return devices[hardware_id + 1]
end

function init_default_comm(; comm::MPI.Comm=MPI.COMM_WORLD)
    if DEFAULT_COMM[] !== nothing
        return DEFAULT_COMM[]
    end

    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    lrank = local_rank(comm)

    # pick xla device first, then make nccl follow it
    # ensures nccl and xla are in agreement process to device mapping
    xla_device = choose_xla_device(lrank)
    device = choose_nccl_device(xla_device)

    NCCL.CUDA.device!(device)
    DEFAULT_XLA_DEVICE[] = xla_device

    # setup the nccl communicator
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
    DEFAULT_XLA_DEVICE[] = nothing
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
