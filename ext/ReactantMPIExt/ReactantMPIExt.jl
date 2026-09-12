module ReactantMPIExt

using Reactant
using Reactant: Reactant, Distributed, MLIR
using Reactant_jll
using MPI: MPI
using Libdl: Libdl

include("Types.jl")
include("Ops.jl")
include("Overrides.jl")

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
    mpi_handle = MPI.API.libmpi_handle

    # register MPI routines
    #! explicit-imports: off
    for name in [
        :MPI_Init,
        :MPI_Finalize,
        :MPI_Comm_rank,
        :MPI_Comm_size,
        :MPI_Comm_split,
        :MPI_Send,
        :MPI_Isend,
        :MPI_Recv,
        :MPI_Irecv,
        :MPI_Barrier,
        :MPI_Wait,
        :MPI_Waitall,
        :MPI_Request_free,
        :MPI_Allreduce,
        :MPI_Bcast,
        :MPI_Error_string,
    ]
        MLIR.API.EnzymeJaXMapSymbol(name, Libdl.dlsym(mpi_handle, name))
    end
    #! explicit-imports: on

    # register MPI constants
    # NOTE these symbols are not ABI-stable until MPI 5.0, but in practice, they are represented as word-size values (i.e. `int` or ptr)
    for name in [
        # communicators
        :MPI_COMM_WORLD,
        :MPI_COMM_SELF,
        :MPI_COMM_NULL,
        # communicator types
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
        :MPI_STATUS_SIZE,
        # error
        :MPI_SUCCESS,
        # other
        :MPI_MAX_ERROR_STRING,
    ]
        !isdefined(MPI.API, name) && continue
        value = getproperty(MPI.API, name)
        if value isa Base.RefValue
            value = value[]
        end
        MLIR.API.EnzymeJaXMapSymbol(name, convert(Int64, value))
    end

    # register MPI_STATUS_SIZE constant (which is not directly present in MPI.jl)
    # MLIR.API.EnzymeJaXMapSymbol(:MPI_STATUS_SIZE, convert(Int64, status_size))

    return nothing
end

end # module
