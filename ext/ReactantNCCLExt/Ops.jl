using NCCL
using Reactant: promote_to
using Reactant.Ops: mlir_stacktrace, mlir_type
using Reactant.MLIR: IR
using Reactant.MLIR.Dialects: comm

@noinline function comm_count(communicator::TracedCommunicator; location=mlir_stacktrace("comm.nccl.comm_count", @__FILE__, @__LINE__))
    count = mlir_type(TracedRArray{Int32,0}, ())
    res = IR.result(comm.nccl_comm_count(communicator; count, location))
    return TracedRNumber{Int32}((), res)
end

@noinline function comm_user_rank(communicator::TracedCommunicator; location=mlir_stacktrace("comm.nccl.comm_user_rank", @__FILE__, @__LINE__))
    user_rank = mlir_type(TracedRArray{Int32,0}, ())
    res = IR.result(comm.nccl_comm_user_rank(communicator; user_rank, location))
    return TracedRNumber{Int32}((), res)
end

@noinline function comm_split(communicator::TracedCommunicator, color, key; location=mlir_stacktrace("comm.nccl.comm_split", @__FILE__, @__LINE__))
    comm_type = mlir_type(TracedCommunicator)
    tcolor = promote_to(TracedRNumber{Int32}, color)
    tkey = promote_to(TracedRNumber{Int32}, key)
    res = IR.result(comm.nccl_comm_split(communicator, tcolor, tkey; new_comm=comm_type, location))
    return TracedCommunicator(res)
end

@noinline function send(buf, peer, communicator::TracedCommunicator, stream; location=mlir_stacktrace("comm.nccl.send", @__FILE__, @__LINE__))
    tbuf = Reactant.promote_to(TracedRArray, buf)
    tpeer = Reactant.promote_to(TracedRNumber{Int32}, peer)
    # TODO check stream is default stream
    IR.result(comm.nccl_send(tbuf, communicator, tpeer; location))
    return nothing
end

@noinline function recv(peer, communicator::TracedCommunicator; location=mlir_stacktrace("comm.nccl.recv", @__FILE__, @__LINE__))
    tpeer = Reactant.promote_to(TracedRNumber{Int32}, peer)
    # TODO probably check stream is default stream
    res = IR.result(comm.nccl_recv(tpeer, communicator; location))
    # TODO retrieve type and shape from res
    T = ...
    N = ...
    return TracedRArray{,}((), res, ...)
end

@noinline function all_reduce(buf, reduce_op, communicator::TracedCommunicator; location=mlir_stacktrace("comm.nccl.all_reduce", @__FILE__, @__LINE__))
    tbuf = Reactant.promote_to(TracedRArray, buf)
    # TODO convert reduce_op to attribute
    reduce_op_attr = ...
    res = IR.result(comm.nccl_all_reduce(tbuf, reduce_op_attr, communicator; location))
    return TracedRArray{unwrapped_eltype(buf),ndims(buf)}((), res, size(buf))
end

@noinline function broadcast(buf, root, communicator::TracedCommunicator; location=mlir_stacktrace("comm.nccl.all_reduce", @__FILE__, @__LINE__))
    tbuf = Reactant.promote_to(TracedRArray, buf)
    troot = Reactant.promote_to(TracedRNumber, root)
    res = IR.result(comm.nccl_broadcast(tbuf, troot, communicator; location))
end
