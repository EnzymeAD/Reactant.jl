module mpi
using ...IR
import ...IR:
    NamedAttribute,
    Value,
    Location,
    Block,
    Region,
    Attribute,
    create_operation,
    context,
    IndexType
import ..Dialects: namedattribute, operandsegmentsizes, c
import ...API
using EnumX

"""
`MPI_OpClassEnum`
MPI operation class
"""
@enumx MPI_OpClassEnum MPI_OP_NULL MPI_MAX MPI_MIN MPI_SUM MPI_PROD MPI_LAND MPI_BAND MPI_LOR MPI_BOR MPI_LXOR MPI_BXOR MPI_MINLOC MPI_MAXLOC MPI_REPLACE
MPI_OpClassEnumStorage = [
    "MPI_OP_NULL",
    "MPI_MAX",
    "MPI_MIN",
    "MPI_SUM",
    "MPI_PROD",
    "MPI_LAND",
    "MPI_BAND",
    "MPI_LOR",
    "MPI_BOR",
    "MPI_LXOR",
    "MPI_BXOR",
    "MPI_MINLOC",
    "MPI_MAXLOC",
    "MPI_REPLACE",
]

function IR.Attribute(e::MPI_OpClassEnum.T)
    return parse(Attribute, "#mpi<opclass <$(MPI_OpClassEnumStorage[Int(e)+1])>>")
end

"""
`MPI_ErrorClassEnum`
MPI error class name
"""
@enumx MPI_ErrorClassEnum MPI_SUCCESS MPI_ERR_ACCESS MPI_ERR_AMODE MPI_ERR_ARG MPI_ERR_ASSERT MPI_ERR_BAD_FILE MPI_ERR_BASE MPI_ERR_BUFFER MPI_ERR_COMM MPI_ERR_CONVERSION MPI_ERR_COUNT MPI_ERR_DIMS MPI_ERR_DISP MPI_ERR_DUP_DATAREP MPI_ERR_ERRHANDLER MPI_ERR_FILE MPI_ERR_FILE_EXISTS MPI_ERR_FILE_IN_USE MPI_ERR_GROUP MPI_ERR_INFO MPI_ERR_INFO_KEY MPI_ERR_INFO_NOKEY MPI_ERR_INFO_VALUE MPI_ERR_IN_STATUS MPI_ERR_INTERN MPI_ERR_IO MPI_ERR_KEYVAL MPI_ERR_LOCKTYPE MPI_ERR_NAME MPI_ERR_NO_MEM MPI_ERR_NO_SPACE MPI_ERR_NO_SUCH_FILE MPI_ERR_NOT_SAME MPI_ERR_OP MPI_ERR_OTHER MPI_ERR_PENDING MPI_ERR_PORT MPI_ERR_PROC_ABORTED MPI_ERR_QUOTA MPI_ERR_RANK MPI_ERR_READ_ONLY MPI_ERR_REQUEST MPI_ERR_RMA_ATTACH MPI_ERR_RMA_CONFLICT MPI_ERR_RMA_FLAVOR MPI_ERR_RMA_RANGE MPI_ERR_RMA_SHARED MPI_ERR_RMA_SYNC MPI_ERR_ROOT MPI_ERR_SERVICE MPI_ERR_SESSION MPI_ERR_SIZE MPI_ERR_SPAWN MPI_ERR_TAG MPI_ERR_TOPOLOGY MPI_ERR_TRUNCATE MPI_ERR_TYPE MPI_ERR_UNKNOWN MPI_ERR_UNSUPPORTED_DATAREP MPI_ERR_UNSUPPORTED_OPERATION MPI_ERR_VALUE_TOO_LARGE MPI_ERR_WIN MPI_ERR_LASTCODE
MPI_ErrorClassEnumStorage = [
    "MPI_SUCCESS",
    "MPI_ERR_ACCESS",
    "MPI_ERR_AMODE",
    "MPI_ERR_ARG",
    "MPI_ERR_ASSERT",
    "MPI_ERR_BAD_FILE",
    "MPI_ERR_BASE",
    "MPI_ERR_BUFFER",
    "MPI_ERR_COMM",
    "MPI_ERR_CONVERSION",
    "MPI_ERR_COUNT",
    "MPI_ERR_DIMS",
    "MPI_ERR_DISP",
    "MPI_ERR_DUP_DATAREP",
    "MPI_ERR_ERRHANDLER",
    "MPI_ERR_FILE",
    "MPI_ERR_FILE_EXISTS",
    "MPI_ERR_FILE_IN_USE",
    "MPI_ERR_GROUP",
    "MPI_ERR_INFO",
    "MPI_ERR_INFO_KEY",
    "MPI_ERR_INFO_NOKEY",
    "MPI_ERR_INFO_VALUE",
    "MPI_ERR_IN_STATUS",
    "MPI_ERR_INTERN",
    "MPI_ERR_IO",
    "MPI_ERR_KEYVAL",
    "MPI_ERR_LOCKTYPE",
    "MPI_ERR_NAME",
    "MPI_ERR_NO_MEM",
    "MPI_ERR_NO_SPACE",
    "MPI_ERR_NO_SUCH_FILE",
    "MPI_ERR_NOT_SAME",
    "MPI_ERR_OP",
    "MPI_ERR_OTHER",
    "MPI_ERR_PENDING",
    "MPI_ERR_PORT",
    "MPI_ERR_PROC_ABORTED",
    "MPI_ERR_QUOTA",
    "MPI_ERR_RANK",
    "MPI_ERR_READ_ONLY",
    "MPI_ERR_REQUEST",
    "MPI_ERR_RMA_ATTACH",
    "MPI_ERR_RMA_CONFLICT",
    "MPI_ERR_RMA_FLAVOR",
    "MPI_ERR_RMA_RANGE",
    "MPI_ERR_RMA_SHARED",
    "MPI_ERR_RMA_SYNC",
    "MPI_ERR_ROOT",
    "MPI_ERR_SERVICE",
    "MPI_ERR_SESSION",
    "MPI_ERR_SIZE",
    "MPI_ERR_SPAWN",
    "MPI_ERR_TAG",
    "MPI_ERR_TOPOLOGY",
    "MPI_ERR_TRUNCATE",
    "MPI_ERR_TYPE",
    "MPI_ERR_UNKNOWN",
    "MPI_ERR_UNSUPPORTED_DATAREP",
    "MPI_ERR_UNSUPPORTED_OPERATION",
    "MPI_ERR_VALUE_TOO_LARGE",
    "MPI_ERR_WIN",
    "MPI_ERR_LASTCODE",
]

function IR.Attribute(e::MPI_ErrorClassEnum.T)
    return parse(Attribute, "#mpi<errclass <$(MPI_ErrorClassEnumStorage[Int(e)+1])>>")
end

"""
`allreduce`

MPI_Allreduce performs a reduction operation on the values in the sendbuf
array and stores the result in the recvbuf array. The operation is 
performed across all processes in the communicator.

The `op` attribute specifies the reduction operation to be performed.
Currently only the `MPI_Op` predefined in the standard (e.g. `MPI_SUM`) are
supported.

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function allreduce(
    sendbuf::Value,
    recvbuf::Value;
    retval::Union{Nothing,IR.Type}=nothing,
    op::MPI_OpClassEnum.T,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[sendbuf, recvbuf]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op", op),]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.allreduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`barrier`

MPI_Barrier blocks execution until all processes in the communicator have
reached this routine.

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function barrier(; retval::Union{Nothing,IR.Type}=nothing, location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`comm_rank`

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_rank(;
    retval::Union{Nothing,IR.Type}=nothing, rank::IR.Type, location::Location=Location()
)
    op_ty_results = IR.Type[rank,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.comm_rank",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`comm_size`

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_size(;
    retval::Union{Nothing,IR.Type}=nothing, size::IR.Type, location::Location=Location()
)
    op_ty_results = IR.Type[size,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.comm_size",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`error_class`

`MPI_Error_class` maps return values from MPI calls to a set of well-known
MPI error classes.
"""
function error_class(val::Value; errclass::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[errclass,]
    operands = Value[val,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mpi.error_class",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`finalize`

This function cleans up the MPI state. Afterwards, no MPI methods may 
be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized).
Notably, MPI_Init cannot be called again in the same program.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function finalize(; retval::Union{Nothing,IR.Type}=nothing, location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.finalize",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`irecv`

MPI_Irecv begins a non-blocking receive of `size` elements of type `dtype` 
from rank `dest`. The `tag` value and communicator enables the library to 
determine the matching of multiple sends and receives between the same 
ranks.

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function irecv(
    ref::Value,
    tag::Value,
    rank::Value;
    retval::Union{Nothing,IR.Type}=nothing,
    req::IR.Type,
    location::Location=Location(),
)
    op_ty_results = IR.Type[req,]
    operands = Value[ref, tag, rank]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.irecv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`isend`

MPI_Isend begins a non-blocking send of `size` elements of type `dtype` to
rank `dest`. The `tag` value and communicator enables the library to
determine the matching of multiple sends and receives between the same
ranks.

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function isend(
    ref::Value,
    tag::Value,
    rank::Value;
    retval::Union{Nothing,IR.Type}=nothing,
    req::IR.Type,
    location::Location=Location(),
)
    op_ty_results = IR.Type[req,]
    operands = Value[ref, tag, rank]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.isend",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`init`

This operation must preceed most MPI calls (except for very few exceptions,
please consult with the MPI specification on these).

Passing &argc, &argv is not supported currently.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function init(; retval::Union{Nothing,IR.Type}=nothing, location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.init",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`recv`

MPI_Recv performs a blocking receive of `size` elements of type `dtype` 
from rank `dest`. The `tag` value and communicator enables the library to 
determine the matching of multiple sends and receives between the same 
ranks.

Communicators other than `MPI_COMM_WORLD` are not supported for now.
The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function recv(
    ref::Value,
    tag::Value,
    rank::Value;
    retval::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ref, tag, rank]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.recv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`retval_check`

This operation compares MPI status codes to known error class
constants such as `MPI_SUCCESS`, or `MPI_ERR_COMM`.
"""
function retval_check(
    val::Value; res::IR.Type, errclass::MPI_ErrorClassEnum.T, location::Location=Location()
)
    op_ty_results = IR.Type[res,]
    operands = Value[val,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("errclass", errclass),]

    return create_operation(
        "mpi.retval_check",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`send`

MPI_Send performs a blocking send of `size` elements of type `dtype` to rank
`dest`. The `tag` value and communicator enables the library to determine 
the matching of multiple sends and receives between the same ranks.

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function send(
    ref::Value,
    tag::Value,
    rank::Value;
    retval::Union{Nothing,IR.Type}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ref, tag, rank]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.send",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`wait`

MPI_Wait blocks execution until the request has completed.

The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function wait(
    req::Value; retval::Union{Nothing,IR.Type}=nothing, location::Location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[req,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.wait",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # mpi
