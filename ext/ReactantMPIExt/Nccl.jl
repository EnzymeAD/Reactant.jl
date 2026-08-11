const NcclComm_t = Ptr{Cvoid}

struct NcclUniqueId
    internal::NTuple{128,UInt8}
end

const DEFAULT_COMM = Ref{NcclComm_t}(C_NULL)
const DEFAULT_COMM_HANDLE = Ref{UInt}(0)
const DEFAULT_XLA_DEVICE = Ref{Union{Nothing,Reactant.XLA.AbstractDevice}}(nothing)

function nccl_symbol(name::Symbol)
    Reactant_jll.is_available() ||
        error("Reactant_jll is not available; bundled NCCL cannot be used")
    return Libdl.dlsym(Reactant_jll.libReactantExtra_handle, name)
end

function check_nccl(status::Cint, operation::AbstractString)
    status == 0 || error("$operation failed with NCCL status $status")
    return nothing
end

function nccl_unique_id()
    id = Ref{NcclUniqueId}()
    ptr = nccl_symbol(:ncclGetUniqueId)
    status = @ccall $ptr(id::Ref{NcclUniqueId})::Cint
    check_nccl(status, "ncclGetUniqueId")
    return id[]
end

function nccl_comm_init(nranks::Integer, rank::Integer, id::NcclUniqueId)
    comm = Ref{NcclComm_t}(C_NULL)
    ptr = nccl_symbol(:ncclCommInitRank)
    status = @ccall $ptr(
        comm::Ref{NcclComm_t}, nranks::Cint, id::NcclUniqueId, rank::Cint
    )::Cint
    check_nccl(status, "ncclCommInitRank")
    return comm[]
end

function nccl_comm_destroy(comm::NcclComm_t)
    ptr = nccl_symbol(:ncclCommDestroy)
    status = @ccall $ptr(comm::NcclComm_t)::Cint
    check_nccl(status, "ncclCommDestroy")
    return nothing
end

function default_xla_device()
    return DEFAULT_XLA_DEVICE[]
end

function require_default_xla_device(client)
    override = default_xla_device()
    override === nothing &&
        error("ReactantMPIExt default XLA device has not been initialized")
    Reactant.XLA.client(override) == client ||
        error("ReactantMPIExt default XLA device does not belong to the active client")
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

function get_hardware_id(xla_device::Reactant.XLA.AbstractDevice)
    hardware_id = Int(Reactant.XLA.get_local_hardware_id(xla_device))
    hardware_id ≥ 0 || error("Reactant XLA device has invalid hardware id $hardware_id")
    return hardware_id
end

function init_default_comm(; comm::MPI.Comm=MPI.COMM_WORLD)
    DEFAULT_COMM[] != C_NULL && return DEFAULT_COMM[]

    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    lrank = local_rank(comm)

    # make sure nccl and xla are agree on process-to-device mapping
    xla_device = choose_xla_device(lrank)
    Reactant.set_nccl_device!(get_hardware_id(xla_device))
    DEFAULT_XLA_DEVICE[] = xla_device

    unique_id = rank == 0 ? nccl_unique_id() : NcclUniqueId(ntuple(_ -> UInt8(0), 128))
    unique_id_bytes = collect(unique_id.internal)
    MPI.Bcast!(unique_id_bytes, 0, comm)
    nccl_comm = nccl_comm_init(nranks, rank, NcclUniqueId(Tuple(unique_id_bytes)))

    DEFAULT_COMM[] = nccl_comm
    DEFAULT_COMM_HANDLE[] = UInt(nccl_comm)
    return nccl_comm
end

function destroy_default_comm()
    if DEFAULT_COMM[] != C_NULL
        nccl_comm_destroy(DEFAULT_COMM[])
        DEFAULT_COMM[] = C_NULL
        DEFAULT_COMM_HANDLE[] = 0
    end
    DEFAULT_XLA_DEVICE[] = nothing
    return nothing
end

function default_comm()
    comm = DEFAULT_COMM[]
    comm == C_NULL && error("Default NCCL communicator has not been initialized")
    return comm
end

function default_comm_handle()
    DEFAULT_COMM[] == C_NULL && error("Default NCCL communicator has not been initialized")
    return DEFAULT_COMM_HANDLE[]
end
