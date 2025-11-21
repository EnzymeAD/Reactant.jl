using Reactant: @reactant_overlay, TracedRArray, TracedRNumber

# @reactant_overlay @noinline function MPI.Init(; kwargs...)
#     if !isempty(kwargs)
#         @warn "Ignoring MPI.Init kwargs when tracing over MPI..." kwargs...
#     end
#     return Ops.init()
# end

# @reactant_overlay @noinline function MPI.Finalize(; kwargs...)
#     return Ops.finalize()
# end

@reactant_overlay @noinline function MPI.Comm_rank(comm::MPI.Comm)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.comm_rank()
end

@reactant_overlay @noinline function MPI.Comm_size(comm::MPI.Comm)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.comm_size()
end

@reactant_overlay @noinline function MPI.Barrier(comm::MPI.Comm)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.barrier()
end

# TODO status not supported yet
function MPI.Wait(req::TracedRequest)
    return Ops.wait(req)
end

# TODO use `make_tracer` to linearize arbitrary types? check out `MPI.Buffer`
function MPI.Send(buf::TracedRArray, dest::Integer, tag::Integer, comm::MPI.Comm)
    tag = Reactant.Ops.constant(tag)
    dest = Reactant.Ops.constant(dest)
    return MPI.Send(buf, dest, tag, comm)
end

# TODO use `make_tracer` to linearize arbitrary types? check out `MPI.Buffer`
function MPI.Send(
    buf::TracedRArray, dest::TracedRNumber, tag::TracedRNumber, comm::MPI.Comm
)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.send(buf, tag, dest)
end

# TODO should we error if other `AbstractRequest` types are passed in?
function MPI.Isend(
    buf::TracedRArray,
    dest::Integer,
    tag::Integer,
    comm::MPI.Comm,
    request::TracedRequest=TracedRequest((), nothing),
)
    dest = Reactant.Ops.constant(dest)
    tag = Reactant.Ops.constant(tag)

    gen_request = MPI.Isend(buf, dest, tag, comm)
    request.mlir_data = gen_request.mlir_data
    return request
end

# TODO use `make_tracer` to linearize arbitrary types? check out `MPI.Buffer`
function MPI.Isend(
    buf::TracedRArray, dest::TracedRNumber, tag::TracedRNumber, comm::MPI.Comm
)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"

    return Ops.isend(buf, tag, dest)
end

function MPI.Recv!(buf::TracedRArray, source::Integer, tag::Integer, comm::MPI.Comm)
    tag = Reactant.Ops.constant(tag)
    source = Reactant.Ops.constant(source)
    return MPI.Recv!(buf, source, tag, comm)
end

# TODO Do we need these?
# function MPI.Recv!(
#     buf::TracedRArray,
#     source::Integer,
#     tag::Integer,
#     comm::MPI.Comm,
#     ::Type{MPI.API.MPI_Status},
# )
#     return MPI.Recv!(buf, source, tag, comm)
# end

# function MPI.Recv!(
#     buf::TracedRArray, source::Integer, tag::Integer, comm::MPI.Comm, ::Nothing
# )
#     return MPI.Recv!(buf, source, tag, comm)
# end

# TODO use `make_tracer` to delinearize arbitrary types? check out `MPI.Buffer`
function MPI.Recv!(
    buf::TracedRArray, source::TracedRNumber, tag::TracedRNumber, comm::MPI.Comm
)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.recv!(buf, tag, source)
end

function MPI.Irecv!(
    buf::TracedRArray,
    source::Integer,
    tag::Integer,
    comm::MPI.Comm,
    request::TracedRequest=TracedRequest((), nothing),
)
    source = Reactant.Ops.constant(source)
    tag = Reactant.Ops.constant(tag)

    gen_request = MPI.Irecv!(buf, source, tag, comm)
    request.mlir_data = gen_request.mlir_data
    return request
end

# TODO use `make_tracer` to delinearize arbitrary types? check out `MPI.Buffer`
function MPI.Irecv!(
    buf::TracedRArray, source::TracedRNumber, tag::TracedRNumber, comm::MPI.Comm
)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"

    return Ops.irecv!(buf, tag, source)
end

function MPI.Allreduce!(sendbuf::TracedRArray, recvbuf::TracedRArray, op, comm::MPI.Comm)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.allreduce!(op, sendbuf, recvbuf)
end
