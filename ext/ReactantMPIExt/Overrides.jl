using Reactant: @reactant_overlay, TracedRArray, TracedRNumber

@reactant_overlay @noinline function MPI.Init(; kwargs...)
    if !isempty(kwargs)
        @warn "Ignoring MPI.Init kwargs when tracing over MPI..." kwargs...
    end
    return Ops.init()
end

@reactant_overlay @noinline function MPI.Finalize(; kwargs...)
    return Ops.finalize()
end

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

# TODO use `make_tracer` to linearize arbitrary types? check out `MPI.Buffer`
function MPI.Isend(
    buf::TracedRArray,
    dest::Union{T,TracedRNumber{T}},
    tag::Union{T,TracedRNumber{T}},
    comm::MPI.Comm,
) where {T<:Integer}
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"

    tag = if !(tag isa TracedRNumber)
        Reactant.Ops.constant(tag)
    end

    dest = if !(dest isa TracedRNumber)
        Reactant.Ops.constant(dest)
    end

    return Ops.isend(buf, tag, dest)
end

# TODO should we error if other `AbstractRequest` types are passed in?
function MPI.Isend(
    buf::TracedRArray, dest::Number, tag::Number, comm::MPI.Comm, req::TracedRequest
)
    gen_req = MPI.Isend(buf, dest, tag, comm)
    req.mlir_data = gen_req.mlir_data
    return req
end

# TODO use `make_tracer` to delinearize arbitrary types? check out `MPI.Buffer`
function MPI.Recv!(
    recvbuf::TracedRArray, source::Number, tag::Number, comm::MPI.Comm, status
)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    @assert isnothing(status) "Status not supported yet"

    tag = if !(tag isa TracedRNumber)
        Reactant.Ops.constant(tag)
    end

    source = if !(source isa TracedRNumber)
        Reactant.Ops.constant(source)
    end

    return Ops.recv(recvbuf, tag, source)
end

# TODO use `make_tracer` to delinearize arbitrary types? check out `MPI.Buffer`
function MPI.Irecv!(recvbuf::TracedRArray, source::Number, tag::Number, comm::MPI.Comm)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"

    tag = if !(tag isa TracedRNumber)
        Reactant.Ops.constant(tag)
    end

    source = if !(source isa TracedRNumber)
        Reactant.Ops.constant(source)
    end

    return Ops.irecv!(recvbuf, tag, source)
end

function MPI.Irecv!(
    recvbuf::TracedRArray, source::Number, tag::Number, comm::MPI.Comm, req::TracedRequest
)
    gen_req = MPI.Irecv!(recvbuf, source, tag, comm)
    req.mlir_data = gen_req.mlir_data
    return req
end

function MPI.Allreduce!(sendbuf::TracedRArray, recvbuf::TracedRArray, op, comm::MPI.Comm)
    @assert comm == MPI.COMM_WORLD "Only MPI.COMM_WORLD is supported currently"
    return Ops.allreduce!(op, sendbuf, recvbuf)
end
