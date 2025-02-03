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
`comm_rank`

Communicators other than `MPI_COMM_WORLD` are not supported for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function comm_rank(;
    retval=nothing::Union{Nothing,IR.Type}, rank::IR.Type, location=Location()
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
from rank `dest`. The `tag` value and communicator enables the library to 
determine the matching of multiple sends and receives between the same 
ranks.

Communicators other than `MPI_COMM_WORLD` are not supprted for now.
The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function recv(
    ref::Value,
    tag::Value,
    rank::Value;
    retval=nothing::Union{Nothing,IR.Type},
    location=Location(),
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

Communicators other than `MPI_COMM_WORLD` are not supprted for now.

This operation can optionally return an `!mpi.retval` value that can be used
to check for errors.
"""
function send(
    ref::Value,
    tag::Value,
    rank::Value;
    retval=nothing::Union{Nothing,IR.Type},
    location=Location(),
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

end # mpi
