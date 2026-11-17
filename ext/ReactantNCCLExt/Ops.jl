module Ops
using Reactant: Reactant, MLIR, TracedRArray, TracedRNumber, promote_to
using Reactant.TracedUtils: get_mlir_data, set_mlir_data!
using Reactant.MLIR: IR
using Reactant.MLIR.Dialects: comm
using Reactant.Ops: mlir_stacktrace, mlir_type
using NCCL: NCCL
using ..ReactantNCCLExt: TracedCommunicator

nccl_red_op(::typeof(+)) = MLIR.API.ENZYMEXLA_COMM_NCCL_RED_OP_SUM
nccl_red_op(::typeof(*)) = MLIR.API.ENZYMEXLA_COMM_NCCL_RED_OP_PROD
nccl_red_op(::typeof(min)) = MLIR.API.ENZYMEXLA_COMM_NCCL_RED_OP_MIN
nccl_red_op(::typeof(max)) = MLIR.API.ENZYMEXLA_COMM_NCCL_RED_OP_MAX
function nccl_red_op(op)
    op === NCCL.avg && return MLIR.API.ENZYMEXLA_COMM_NCCL_RED_OP_AVG
    throw(ArgumentError("Unsupported NCCL reduction operator $op"))
end

@noinline function comm_count(
    c::TracedCommunicator;
    location=mlir_stacktrace("comm.nccl.comm_count", @__FILE__, @__LINE__),
)
    type_count = mlir_type(TracedRArray{Int32,0}, ())
    op = comm.nccl_comm_count(get_mlir_data(c); count=type_count, location)
    return TracedRNumber{Int32}((), IR.result(op))
end

@noinline function comm_user_rank(
    c::TracedCommunicator;
    location=mlir_stacktrace("comm.nccl.comm_user_rank", @__FILE__, @__LINE__),
)
    type_user_rank = mlir_type(TracedRArray{Int32,0}, ())
    op = comm.nccl_comm_user_rank(get_mlir_data(c); user_rank=type_user_rank, location)
    return TracedRNumber{Int32}((), IR.result(op))
end

@noinline function comm_split(
    c::TracedCommunicator,
    color,
    key;
    location=mlir_stacktrace("comm.nccl.comm_split", @__FILE__, @__LINE__),
)
    tcolor = promote_to(TracedRNumber{Int32}, color)
    tkey = promote_to(TracedRNumber{Int32}, key)
    op = comm.nccl_comm_split(
        get_mlir_data(c),
        get_mlir_data(tcolor),
        get_mlir_data(tkey);
        new_comm=mlir_type(TracedCommunicator),
        location,
    )
    return TracedCommunicator((), IR.result(op))
end

@noinline function send(
    buf::TracedRArray,
    peer,
    c::TracedCommunicator;
    location=mlir_stacktrace("comm.nccl.send", @__FILE__, @__LINE__),
)
    tpeer = promote_to(TracedRNumber{Int32}, peer)
    comm.nccl_send(get_mlir_data(buf), get_mlir_data(tpeer), get_mlir_data(c); location)
    return nothing
end

@noinline function recv!(
    buf::TracedRArray,
    peer,
    c::TracedCommunicator;
    location=mlir_stacktrace("comm.nccl.recv", @__FILE__, @__LINE__),
)
    tpeer = promote_to(TracedRNumber{Int32}, peer)
    op = comm.nccl_recv(
        get_mlir_data(tpeer), get_mlir_data(c); recvbuff=mlir_type(buf), location
    )
    set_mlir_data!(buf, IR.result(op))
    return buf
end

@noinline function all_reduce(
    buf::TracedRArray{T,N},
    reduce_op,
    c::TracedCommunicator;
    location=mlir_stacktrace("comm.nccl.all_reduce", @__FILE__, @__LINE__),
) where {T,N}
    reduce_op_attr = IR.Attribute(
        MLIR.API.enzymexlaCommNcclRedOpAttrGet(IR.current_context(), nccl_red_op(reduce_op))
    )
    op = comm.nccl_all_reduce(
        get_mlir_data(buf),
        get_mlir_data(c);
        recvbuf=mlir_type(buf),
        reduceOp=reduce_op_attr,
        location,
    )
    return TracedRArray{T,N}((), IR.result(op), size(buf))
end

@noinline function broadcast(
    buf::TracedRArray{T,N},
    root,
    c::TracedCommunicator;
    location=mlir_stacktrace("comm.nccl.broadcast", @__FILE__, @__LINE__),
) where {T,N}
    troot = promote_to(TracedRNumber{Int32}, root)
    op = comm.nccl_broadcast(
        get_mlir_data(buf),
        get_mlir_data(troot),
        get_mlir_data(c);
        recvbuff=mlir_type(buf),
        location,
    )
    return TracedRArray{T,N}((), IR.result(op), size(buf))
end

end # module
