const NcclComm_t = Ptr{Cvoid}

struct NcclUniqueId
    internal::NTuple{128,UInt8}
end

const DEFAULT_COMM = Ref{NcclComm_t}(C_NULL)
const DEFAULT_COMM_HANDLE = Ref{UInt}(0)

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

function local_rank(comm::MPI.Comm)
    shared = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, 0)
    try
        return MPI.Comm_rank(shared)
    finally
        MPI.free(shared)
    end
end

function get_hardware_id(xla_device::Reactant.XLA.AbstractDevice)
    hardware_id = Int(Reactant.XLA.get_local_hardware_id(xla_device))
    hardware_id ≥ 0 || error("Reactant XLA device has invalid hardware id $hardware_id")
    return hardware_id
end

"""
    initialize!(comm::MPI.Comm)

Initialize MPI GPU support for `MPI.COMM_WORLD`. Each local MPI rank is assigned
one GPU, which is the only GPU exposed to the XLA client and to NCCL.
"""
function initialize!(comm::MPI.Comm)
    MPI.Initialized() || MPI.Init()
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"

    Reactant.XLA.has_initialized_client("gpu") && error(
        "A GPU XLA client already exists - ReactantMPIExt.initialize! must be called before \
        any GPU XLA client is created.",
    )

    Reactant.XLA.claim_gpu_device_mapping!(:mpi)
    lrank = local_rank(comm)
    Reactant.XLA.global_state.local_gpu_device_ids = [lrank]

    client = Reactant.XLA.client("gpu")
    xla_devices = Reactant.XLA.addressable_devices(client)
    @assert length(xla_devices) == 1 "GPU MPI requires exactly one addressable \
        GPU per local MPI rank; \
        XLA exposed $(length(xla_devices)) devices for local rank $lrank",
    Reactant.set_nccl_device!(get_hardware_id(only(xla_devices)))

    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    unique_id = rank == 0 ? nccl_unique_id() : NcclUniqueId(ntuple(_ -> UInt8(0), 128))
    unique_id_bytes = collect(unique_id.internal)
    MPI.Bcast!(unique_id_bytes, 0, comm)
    nccl_comm = nccl_comm_init(nranks, rank, NcclUniqueId(Tuple(unique_id_bytes)))

    DEFAULT_COMM[] = nccl_comm
    DEFAULT_COMM_HANDLE[] = UInt(nccl_comm)

    return nothing
end

function destroy_default_comm()
    if DEFAULT_COMM[] != C_NULL
        nccl_comm_destroy(DEFAULT_COMM[])
        DEFAULT_COMM[] = C_NULL
        DEFAULT_COMM_HANDLE[] = 0
    end
    return nothing
end

function Reactant.default_nccl_comm_handle()
    DEFAULT_COMM[] == C_NULL && error("Default NCCL communicator has not been initialized")
    return DEFAULT_COMM_HANDLE[]
end
