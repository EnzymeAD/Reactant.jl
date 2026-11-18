module ReactantNCCLExt

using Reactant: TracedRArray, @reactant_overlay
using Reactant.TracedUtils: set_mlir_data!
using NCCL: NCCL

include("Types.jl")
include("Ops.jl")

const NCCL_COMM_WORLD = Ref{Union{Nothing,NCCL.Communicator}}(nothing)

is_nccl_comm_world(comm) = !isnothing(comm) && comm == NCCL_COMM_WORLD[]

function nccl_comm_world()
    comm = NCCL_COMM_WORLD[]
    isnothing(comm) && error("NCCL_COMM_WORLD is not set!")
    return comm
end
set_nccl_comm_world!(comm::NCCL.Communicator) = NCCL_COMM_WORLD[] = comm

function materialize_nccl_comm(comm)
    if comm isa TracedCommunicator
        comm
    elseif is_nccl_comm_world(comm)
        nccl_comm_world()
    else
        error("Non-traced NCCL.Communicator not yet supported")
    end
end

NCCL.nranks(comm::TracedCommunicator) = Ops.comm_count(comm)

NCCL.rank(comm::TracedCommunicator) = Ops.comm_user_rank(comm)

@reactant_overlay function NCCL.split(comm, color, key=nothing; config=nothing)
    isnothing(config) || throw(ArgumentError("config argument is not supported"))
    comm = materialize_nccl_comm(comm)
    color = isnothing(color) ? NCCL.LibNCCL.NCCL_SPLIT_NOCOLOR : color
    key = isnothing(key) ? NCCL.rank(comm) : key
    return Ops.comm_split(comm, color, key)
end

@reactant_overlay function NCCL.Send(sendbuf, comm; dest::Integer, stream=nothing)
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    comm = materialize_nccl_comm(comm)
    sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
    return Ops.send(sendbuf_traced, dest, comm)
end

@reactant_overlay function NCCL.Recv!(recvbuf, comm; source::Integer, stream=nothing)
    recvbuf isa TracedRArray || error("`recvbuf` in NCCL.Recv! must be a traced array when tracing")
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    comm = materialize_nccl_comm(comm)
    return Ops.recv!(recvbuf, source, comm)
end

@reactant_overlay function NCCL.Allreduce!(sendbuf, recvbuf, op, comm; stream=nothing)
    recvbuf isa TracedRArray || error("`recvbuf` in NCCL.Allreduce! must be a traced array when tracing")
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    comm = materialize_nccl_comm(comm)
    sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
    res = Ops.allreduce(sendbuf_traced, op, comm)
    set_mlir_data!(recvbuf, get_mlir_data(res))
    return recvbuf
end

@reactant_overlay function NCCL.Allreduce!(sendrecvbuf, op, comm; stream=nothing)
    sendrecvbuf isa TracedRArray || error("`sendrecvbuf` in NCCL.Allreduce! must be a traced array when tracing")
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    comm = materialize_nccl_comm(comm)
    res = Ops.allreduce(sendrecvbuf, op, comm)
    set_mlir_data!(sendrecvbuf, get_mlir_data(res))
    return sendrecvbuf
end

@reactant_overlay function NCCL.Broadcast!(sendbuf, recvbuf, comm; root::Integer=0, stream=nothing)
    recvbuf isa TracedRArray || error("`recvbuf` in NCCL.Broadcast! must be a traced array when tracing")
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    comm = materialize_nccl_comm(comm)
    sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
    res = Ops.broadcast(sendbuf_traced, root, comm)
    set_mlir_data!(recvbuf, get_mlir_data(res))
    return recvbuf
end

@reactant_overlay function NCCL.Broadcast!(sendrecvbuf, comm; root::Integer=0, stream=nothing)
    sendrecvbuf isa TracedRArray || error("`sendrecvbuf` in NCCL.Broadcast! must be a traced array when tracing")
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    comm = materialize_nccl_comm(comm)
    res = Ops.broadcast(sendrecvbuf, root, comm)
    set_mlir_data!(sendrecvbuf, get_mlir_data(res))
    return sendrecvbuf
end

@reactant_overlay NCCL.group(f) = Ops.group(f)

end # module
