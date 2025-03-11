module ReactantMPIExt

using Reactant
using Reactant: Reactant, Distributed, MLIR
using MPI: MPI
using Libdl

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

function __init__()
    # TODO maybe it's more efficient if we use `RTLD_NOW` instead of `RTLD_LAZY`?
    libmpi_handle = Libdl.dlopen(MPI.API.libmpi, RTLD_LAZY | RTLD_GLOBAL)

    # register MPI routines
    for name in [
        :MPI_Init,
        :MPI_Finalize,
        :MPI_Comm_rank,
        :MPI_Comm_size,
        :MPI_Send,
        :MPI_Recv,
        :MPI_Isend,
        :MPI_Irecv,
        :MPI_Barrier,
        :MPI_Wait,
        :MPI_Request_free,
    ]
        sym = Libdl.dlsym(libmpi_handle, name)
        @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end

    # register MPI constants
    # NOTE these symbols are not ABI-stable until MPI 5.0, but in practice, they are represented as word-size values (i.e. `int` or ptr)
    for name in [
        # communicators
        :MPI_COMM_WORLD,
        :MPI_COMM_SELF,
        :MPI_COMM_NULL,
        :MPI_COMM_TYPE_SHARED,
        # datatypes
        :MPI_DATATYPE_NULL,
        :MPI_BYTE,
        :MPI_PACKED,
        :MPI_CHAR,
        :MPI_SHORT,
        :MPI_INT,
        :MPI_LONG,
        :MPI_FLOAT,
        :MPI_DOUBLE,
        :MPI_UNSIGNED_CHAR,
        :MPI_SIGNED_CHAR,
        :MPI_UNSIGNED_SHORT,
        :MPI_UNSIGNED_LONG,
        :MPI_UNSIGNED,
        :MPI_FLOAT_INT,
        :MPI_DOUBLE_INT,
        :MPI_LONG_DOUBLE_INT,
        :MPI_LONG_INT,
        :MPI_SHORT_INT,
        # :MPI_2INT,
        :MPI_UB,
        :MPI_LB,
        :MPI_WCHAR,
        :MPI_LONG_LONG_INT,
        :MPI_UNSIGNED_LONG_LONG,
        # :MPI_2COMPLEX,
        # :MPI_2DOUBLE_COMPLEX,
        :MPI_INT8_T,
        :MPI_UINT8_T,
        :MPI_INT16_T,
        :MPI_UINT16_T,
        :MPI_INT32_T,
        :MPI_UINT32_T,
        :MPI_INT64_T,
        :MPI_UINT64_T,
        :MPI_AINT,
        :MPI_OFFSET,
        :MPI_C_BOOL,
        :MPI_C_FLOAT_COMPLEX,
        :MPI_C_DOUBLE_COMPLEX,
        # :MPI_C_LONG_DOUBLE_COMPLEX,
        :MPI_COUNT,
        # ops
        :MPI_OP_NULL,
        :MPI_MAX,
        :MPI_MIN,
        :MPI_SUM,
        :MPI_PROD,
        :MPI_LAND,
        :MPI_BAND,
        :MPI_LOR,
        :MPI_BOR,
        :MPI_LXOR,
        :MPI_BXOR,
        :MPI_MINLOC,
        :MPI_MAXLOC,
        :MPI_REPLACE,
        :MPI_NO_OP,
        # request
        :MPI_REQUEST_NULL,
        # status
        :MPI_STATUS_IGNORE,
        :MPI_STATUSES_IGNORE,
        # error
        :MPI_SUCCESS,
        :MPI_ERR_BUFFER,
        :MPI_ERR_COUNT,
        :MPI_ERR_TYPE,
        :MPI_ERR_TAG,
        :MPI_ERR_COMM,
        :MPI_ERR_RANK,
        :MPI_ERR_REQUEST,
        :MPI_ERR_ROOT,
        :MPI_ERR_GROUP,
        :MPI_ERR_OP,
        :MPI_ERR_TOPOLOGY,
        :MPI_ERR_DIMS,
        :MPI_ERR_ARG,
        :MPI_ERR_UNKNOWN,
        :MPI_ERR_TRUNCATE,
        :MPI_ERR_OTHER,
        :MPI_ERR_INTERN,
        :MPI_ERR_IN_STATUS,
        :MPI_ERR_PENDING,
        :MPI_ERR_ACCESS,
        :MPI_ERR_AMODE,
        :MPI_ERR_ASSERT,
        :MPI_ERR_BAD_FILE,
        :MPI_ERR_BASE,
        :MPI_ERR_CONVERSION,
        :MPI_ERR_DISP,
        :MPI_ERR_DUP_DATAREP,
        :MPI_ERR_FILE_EXISTS,
        :MPI_ERR_FILE_IN_USE,
        :MPI_ERR_FILE,
        :MPI_ERR_INFO_KEY,
        :MPI_ERR_INFO_NOKEY,
        :MPI_ERR_INFO_VALUE,
        :MPI_ERR_INFO,
        :MPI_ERR_IO,
        :MPI_ERR_KEYVAL,
        :MPI_ERR_LOCKTYPE,
        :MPI_ERR_NAME,
        :MPI_ERR_NO_MEM,
        :MPI_ERR_NOT_SAME,
        :MPI_ERR_NO_SPACE,
        :MPI_ERR_NO_SUCH_FILE,
        :MPI_ERR_PORT,
        :MPI_ERR_QUOTA,
        :MPI_ERR_READ_ONLY,
        :MPI_ERR_RMA_CONFLICT,
        :MPI_ERR_RMA_SYNC,
        :MPI_ERR_SERVICE,
        :MPI_ERR_SIZE,
        :MPI_ERR_SPAWN,
        :MPI_ERR_UNSUPPORTED_DATAREP,
        :MPI_ERR_UNSUPPORTED_OPERATION,
        :MPI_ERR_WIN,
        # :MPI_T_ERR_MEMORY,
        # :MPI_T_ERR_NOT_INITIALIZED,
        # :MPI_T_ERR_CANNOT_INIT,
        # :MPI_T_ERR_INVALID_INDEX,
        # :MPI_T_ERR_INVALID_ITEM,
        # :MPI_T_ERR_INVALID_HANDLE,
        # :MPI_T_ERR_OUT_OF_HANDLES,
        # :MPI_T_ERR_OUT_OF_SESSIONS,
        # :MPI_T_ERR_INVALID_SESSION,
        # :MPI_T_ERR_CVAR_SET_NOT_NOW,
        # :MPI_T_ERR_CVAR_SET_NEVER,
        # :MPI_T_ERR_PVAR_NO_STARTSTOP,
        # :MPI_T_ERR_PVAR_NO_WRITE,
        # :MPI_T_ERR_PVAR_NO_ATOMIC,
        :MPI_ERR_RMA_RANGE,
        :MPI_ERR_RMA_ATTACH,
        :MPI_ERR_RMA_FLAVOR,
        :MPI_ERR_RMA_SHARED,
        # :MPI_T_ERR_INVALID,
        # :MPI_T_ERR_INVALID_NAME,
        # :MPI_ERR_PROC_ABORTED,
        # :MPI_ERR_PROC_FAILED,
        # :MPI_ERR_PROC_FAILED_PENDING,
        # :MPI_ERR_REVOKED,
    ]
        value = getproperty(MPI.API, name)
        if value isa Base.RefValue
            value = value[]
        end
        value = convert(Int, value)
        @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, value::Int)::Cvoid
    end
end

struct TracedRequest <: MPI.AbstractRequest
    mlir_data::Union{Nothing,Reactant.MLIR.IR.Value}
end

include("Ops.jl")
include("Overrides.jl")

end # module
