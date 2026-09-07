module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant.TracedUtils: get_mlir_data
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.MLIR.Dialects: comm
using Reactant.Ops: mlir_stacktrace, mlir_type
using MPI: MPI

@noinline function constant(comm::MPI.Comm; location=mlir_stacktrace("comm.mpi.constant", @__FILE__, @__LINE__))
    type_result = mlir_type(TracedCommunicator)
    op = comm.mpi_constant(comm; result = type_result, location)
    return TracedCommunicator((), IR.result(op))
end

@noinline function comm_rank(comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.comm_rank", @__FILE__, @__LINE__)
)
    op = comm.mpi_comm_rank(get_mlir_data(comm); rank = type_rank, location)
    return TracedRNumber{Int32}((), IR.result(op))
end

@noinline function comm_size(;
    location=mlir_stacktrace("comm.mpi.comm_size", @__FILE__, @__LINE__)
)
    type_size = mlir_type(TracedRArray{Int32,0}, ())
    op = comm.mpi_comm_size(; size = type_size, location)
    return TracedRNumber{Int32}((), IR.result(op))
end

@noinline function comm_split(
    comm::TracedCommunicator,
    color::TracedRNumber,
    key::TracedRNumber;
    location=mlir_stacktrace("comm.mpi.comm_split", @__FILE__, @__LINE__)
)
    type_newcomm = mlir_type(TracedCommunicator)
    op = comm.mpi_comm_split(get_mlir_data(comm), get_mlir_data(color), get_mlir_data(key); newcomm = type_newcomm, location)
    return TracedCommunicator((), IR.result(op))
end

@noinline function barrier(comm::TracedCommunicator; location=mlir_stacktrace("comm.mpi.barrier", @__FILE__, @__LINE__))
    comm.mpi_barrier(get_mlir_data(comm); location)
    return nothing
end

@noinline function send(
    buf::TracedRArray,
    dest::TracedRNumber,
    tag::TracedRNumber,
    comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.send", @__FILE__, @__LINE__),
)
    comm.mpi_send(get_mlir_data(buf), get_mlir_data(dest), get_mlir_data(tag), get_mlir_data(comm); location)
    return nothing
end

@noinline function isend(
    buf::TracedRArray,
    dest::TracedRNumber,
    tag::TracedRNumber,
    comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.isend", @__FILE__, @__LINE__),
)
    type_result = mlir_type(TracedRequest)
    op = comm.mpi_send(get_mlir_data(buf), get_mlir_data(dest), get_mlir_data(tag), get_mlir_data(comm); request = type_result, location)
    return TracedRequest((), IR.result(op))
end

@noinline function recv!(
    buf::TracedRArray,
    src::TracedRNumber,
    tag::TracedRNumber,
    comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.recv", @__FILE__, @__LINE__),
)
    op = comm.mpi_recv(get_mlir_data(buf), get_mlir_data(src), get_mlir_data(tag), get_mlir_data(comm); outbuf = mlir_type(buf), location)
    set_mlir_data!(buf, IR.result(op))
    return buf
end

@noinline function irecv!(
    buf::TracedRArray,
    src::TracedRNumber,
    tag::TracedRNumber,
    comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.irecv", @__FILE__, @__LINE__),
)
    op = comm.mpi_irecv(get_mlir_data(buf), get_mlir_data(src), get_mlir_data(tag), get_mlir_data(comm); outbuf = mlir_type(buf), request = mlir_type(TracedRequest), location)
    set_mlir_data!(buf, IR.result(op, 1))

    # return only request?
    return buf, TracedRequest((), IR.result(op, 2))
end

@noinline function wait(
    request::TracedRequest; location=mlir_stacktrace("comm.mpi.wait", @__FILE__, @__LINE__)
)
    comm.mpi_wait(get_mlir_data(request); location)
    return nothing
end

@noinline function waitall(
    requests::Vector{TracedRequest}; location=mlir_stacktrace("comm.mpi.waitall", @__FILE__, @__LINE__)
)
    comm.mpi_waitall(get_mlir_data.(requests); location)
    return nothing
end

# TODO inplace and outplace versions?
@noinline function allreduce(
    mpi_op::MPI.Op,
    sendbuff::TracedRArray,
    recvbuff::TracedRArray,
    comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.allreduce", @__FILE__, @__LINE__),
)
    mapped_mpi_op = MPI_OP_MAP[mpi_op]
    mpi_op_attr = MLIR.API.enzymexlaCommMpiOpAttrGet(IR.current_context(), mapped_mpi_op)
    type_recvbuf = mlir_type(recvbuff)
    op = comm.mpi_allreduce(get_mlir_data(sendbuff), get_mlir_data(comm); recvbuf = type_recvbuf, reduceOp = mpi_op_attr, location)
    set_mlir_data!(recvbuff, IR.result(op))
    return recvbuff
end

# TODO inplace and outplace versions?
@noinline function bcast!(
    buf::TracedRArray,
    root::TracedRNumber,
    comm::TracedCommunicator;
    location=mlir_stacktrace("comm.mpi.bcast", @__FILE__, @__LINE__),
)
    op = comm.mpi_bcast(get_mlir_data(buf), get_mlir_data(root), get_mlir_data(comm); outbuf = mlir_type(buf), location)
    set_mlir_data!(buf, IR.result(op))
    return buf
end

end # module
