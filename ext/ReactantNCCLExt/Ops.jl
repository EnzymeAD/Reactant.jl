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

@noinline function group(f; location=mlir_stacktrace("comm.nccl.group", @__FILE__, @__LINE__))
    block = IR.Block()
    IR.activate(block)
    Reactant.Ops.activate_constant_context!(block)
    res = try
        Reactant.call_with_reactant(f)
    finally
        Reactant.Ops.deactivate_constant_context!(block)
        IR.deactivate(block)
    end

    # values captured (and mutated) or returned by `f` that are defined inside the group
    # must escape through `comm.nccl.yield`
    seen = Reactant.OrderedIdDict()
    Reactant.make_tracer(seen, (f, res), (gensym(:nccl_group),), Reactant.NoStopTracedTrack)
    escaping = [
        x for x in values(seen) if x isa Union{TracedRArray,TracedRNumber,TracedCommunicator} &&
        IR.is_op_res(get_mlir_data(x)) &&
        IR.block(IR.op_owner(get_mlir_data(x))) == block
    ]
    if any(x -> x isa TracedCommunicator, escaping)
        error("NCCL communicators created inside `NCCL.group` cannot be used outside of it")
    end

    MLIR.IR.@with_block block comm.nccl_yield(IR.Value[get_mlir_data(x) for x in escaping]; location)

    region = IR.Region()
    push!(region, block)
    op = comm.nccl_group(; results=IR.Type[IR.type(get_mlir_data(x)) for x in escaping], region, location)

    for (i, x) in enumerate(escaping)
        set_mlir_data!(x, IR.result(op, i))
    end

    return res
end

end # module
