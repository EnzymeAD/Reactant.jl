module comm
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
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API

function mpi_allreduce(
    sendbuf::Value, comm::Value; recvbuf::IR.Type, reduceOp, location=Location()
)
    op_ty_results = IR.Type[recvbuf,]
    operands = Value[sendbuf, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("reduceOp", reduceOp),]

    return create_operation(
        "comm.mpi.allreduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_barrier(comm::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_bcast(
    inBuffer::Value, root::Value, comm::Value; outBuffer::IR.Type, location=Location()
)
    op_ty_results = IR.Type[outBuffer,]
    operands = Value[inBuffer, root, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.bcast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_comm_rank(comm::Value; rank::IR.Type, location=Location())
    op_ty_results = IR.Type[rank,]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.comm_rank",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_comm_size(comm::Value; size::IR.Type, location=Location())
    op_ty_results = IR.Type[size,]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.comm_size",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_comm_split(
    comm::Value, color::Value, key::Value; newcomm::IR.Type, location=Location()
)
    op_ty_results = IR.Type[newcomm,]
    operands = Value[comm, color, key]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.comm_split",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_constant(; result::IR.Type, value, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value),]

    return create_operation(
        "comm.mpi.constant",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_irecv(
    source::Value,
    tag::Value,
    comm::Value;
    buffer::IR.Type,
    request::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[buffer, request]
    operands = Value[source, tag, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.irecv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_isend(
    buffer::Value,
    dest::Value,
    tag::Value,
    comm::Value;
    request::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[request,]
    operands = Value[buffer, dest, tag, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.isend",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_recv(
    source::Value, tag::Value, comm::Value; buffer::IR.Type, location=Location()
)
    op_ty_results = IR.Type[buffer,]
    operands = Value[source, tag, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.recv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_send(buffer::Value, dest::Value, tag::Value, comm::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[buffer, dest, tag, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.send",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_wait(request::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[request,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.wait",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mpi_waitall(requests::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[requests...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.mpi.waitall",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_all_reduce(
    sendbuf::Value, comm::Value; recvbuf::IR.Type, reduceOp, location=Location()
)
    op_ty_results = IR.Type[recvbuf,]
    operands = Value[sendbuf, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("reduceOp", reduceOp),]

    return create_operation(
        "comm.nccl.all_reduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_broadcast(
    sendbuff::Value, root::Value, comm::Value; recvbuff::IR.Type, location=Location()
)
    op_ty_results = IR.Type[recvbuff,]
    operands = Value[sendbuff, root, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.broadcast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_abort(comm::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_abort",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_count(comm::Value; count::IR.Type, location=Location())
    op_ty_results = IR.Type[count,]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_count",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_cudevice(comm::Value; cu_device::IR.Type, location=Location())
    op_ty_results = IR.Type[cu_device,]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_cudevice",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_destroy(comm::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_destroy",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_finalize(comm::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_finalize",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_split(
    comm::Value, color::Value, key::Value; new_comm::IR.Type, location=Location()
)
    op_ty_results = IR.Type[new_comm,]
    operands = Value[comm, color, key]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_split",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_comm_user_rank(comm::Value; user_rank::IR.Type, location=Location())
    op_ty_results = IR.Type[user_rank,]
    operands = Value[comm,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.comm_user_rank",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_recv(peer::Value, comm::Value; recvbuff::IR.Type, location=Location())
    op_ty_results = IR.Type[recvbuff,]
    operands = Value[peer, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.recv",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function nccl_send(sendbuff::Value, peer::Value, comm::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[sendbuff, peer, comm]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "comm.nccl.send",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # comm
