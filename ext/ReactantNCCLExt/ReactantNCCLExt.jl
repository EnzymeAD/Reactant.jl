module ReactantNCCLExt


using Reactant: TracedRArray, @reactant_overlay
using Reactant.TracedUtils: set_mlir_data!
using NCCL: NCCL

include("Types.jl")
include("Ops.jl")

NCCL.nranks(comm::TracedCommunicator) = Ops.comm_count(comm)

NCCL.rank(comm::TracedCommunicator) = Ops.comm_user_rank(comm)

# NOTE unlike NCCL.jl, ranks passing `nothing` as color get a (null) `TracedCommunicator` instead of `nothing`
function NCCL.split(comm::TracedCommunicator, color, key=NCCL.rank(comm); config=nothing)
    isnothing(config) || throw(ArgumentError("config argument is not supported"))
    color = isnothing(color) ? NCCL.LibNCCL.NCCL_SPLIT_NOCOLOR : color
    return Ops.comm_split(comm, color, key)
end

function NCCL.Send(sendbuf, comm::TracedCommunicator; dest::Integer, stream=nothing)
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
    return Ops.send(sendbuf_traced, dest, comm)
end

function NCCL.Recv!(recvbuf::TracedRArray, comm::TracedCommunicator; source::Integer, stream=nothing)
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    return Ops.recv!(recvbuf, source, comm)
end

function NCCL.Allreduce!(sendbuf, recvbuf::TracedRArray, op, comm::TracedCommunicator; stream=nothing)
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
    res = Ops.all_reduce(sendbuf_traced, op, comm)
    set_mlir_data!(recvbuf, get_mlir_data(res))
    return recvbuf
end

function NCCL.Allreduce!(sendrecvbuf::TracedRArray, op, comm::TracedCommunicator; stream=nothing)
    return NCCL.Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)
end

function NCCL.Broadcast!(
    sendbuf, recvbuf::TracedRArray, comm::TracedCommunicator; root::Integer=0, stream=nothing
)
    isnothing(stream) || throw(ArgumentError("stream argument is not supported"))
    sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
    res = Ops.broadcast(sendbuf_traced, root, comm)
    set_mlir_data!(recvbuf, get_mlir_data(res))
    return recvbuf
end

function NCCL.Broadcast!(sendrecvbuf::TracedRArray, comm::TracedCommunicator; root::Integer=0, stream=nothing)
    return NCCL.Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)
end

@reactant_overlay NCCL.group(f) = Ops.group(f)

end # module
