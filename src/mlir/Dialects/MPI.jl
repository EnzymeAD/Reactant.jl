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
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

"""
`allreduce`

MPI_Allreduce performs a reduction operation on the values in the sendbuf
array and stores the result in the recvbuf array. The operation is 
performed across all processes in the communicator.

The `op` attribute specifies the reduction operation to be performed.
Currently only the `MPI_Op` predefined in the standard (e.g. `MPI_SUM`) are
supported.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function allreduce(
    sendbuf::Value,
    recvbuf::Value,
    comm::Value;
    retval=nothing::Union{Nothing,IR.Type},
    op,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[sendbuf, recvbuf, comm]
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

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function barrier(comm::Value; retval=nothing::Union{Nothing,IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[comm,]
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

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_rank(
    comm::Value; retval=nothing::Union{Nothing,IR.Type}, rank::IR.Type, location=Location()
)
    op_ty_results = IR.Type[rank,]
    operands = Value[comm,]
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

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_size(
    comm::Value; retval=nothing::Union{Nothing,IR.Type}, size::IR.Type, location=Location()
)
    op_ty_results = IR.Type[size,]
    operands = Value[comm,]
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
`comm_split`

This operation splits the communicator into multiple sub-communicators.
The color value determines the group of processes that will be part of the
new communicator. The key value determines the rank of the calling process
in the new communicator.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_split(
    comm::Value,
    color::Value,
    key::Value;
    retval=nothing::Union{Nothing,IR.Type},
    newcomm::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[newcomm,]
    operands = Value[comm, color, key]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(retval) && push!(op_ty_results, retval)

    return create_operation(
        "mpi.comm_split",
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
`comm_world`

This operation returns the predefined MPI_COMM_WORLD communicator.
"""
function comm_world(; comm::IR.Type, location=Location())
    op_ty_results = IR.Type[comm,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mpi.comm_world",
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
function error_class(val::Value; errclass::IR.Type, location=Location())
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
function finalize(; retval=nothing::Union{Nothing,IR.Type}, location=Location())
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
from rank `source`. The `tag` value and communicator enables the library to
determine the matching of multiple sends and receives between the same 
ranks.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function irecv(
    ref::Value,
    tag::Value,
    source::Value,
    comm::Value;
    retval=nothing::Union{Nothing,IR.Type},
    req::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[req,]
    operands = Value[ref, tag, source, comm]
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

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function isend(
    ref::Value,
    tag::Value,
    dest::Value,
    comm::Value;
    retval=nothing::Union{Nothing,IR.Type},
    req::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[req,]
    operands = Value[ref, tag, dest, comm]
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
function init(; retval=nothing::Union{Nothing,IR.Type}, location=Location())
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
from rank `source`. The `tag` value and communicator enables the library to
determine the matching of multiple sends and receives between the same 
ranks.

The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function recv(
    ref::Value,
    tag::Value,
    source::Value,
    comm::Value;
    retval=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ref, tag, source, comm]
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
function retval_check(val::Value; res::IR.Type, errclass, location=Location())
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

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function send(
    ref::Value,
    tag::Value,
    dest::Value,
    comm::Value;
    retval=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ref, tag, dest, comm]
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
function wait(req::Value; retval=nothing::Union{Nothing,IR.Type}, location=Location())
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
