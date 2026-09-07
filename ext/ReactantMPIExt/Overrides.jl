using Reactant: @reactant_overlay, TracedRArray
using MPI
using Reactant: call_with_native, call_with_reactant, use_overlayed_version

const OVERLAY_NATIVE_CALLS = Ref(false)

function MPI.Comm_rank(comm::TracedCommunicator)
    return Ops.comm_rank(comm)
end

@reactant_overlay function MPI.Comm_rank(comm::MPI.Comm)
    if OVERLAY_NATIVE_CALLS[]
        return Ops.comm_rank(Ops.constant(comm))
    else
        return call_with_native(MPI.Comm_rank, comm)
    end
end

function MPI.Comm_size(comm::TracedCommunicator)
    return Ops.comm_size(comm)
end

@reactant_overlay function MPI.Comm_size(comm::MPI.Comm)
    if OVERLAY_NATIVE_CALLS[]
        return Ops.comm_size(Ops.constant(comm))
    else
        return call_with_native(MPI.Comm_size, comm)
    end
end

function MPI.Barrier(comm::TracedCommunicator)
    return Ops.barrier(comm)
end

@reactant_overlay function MPI.Barrier(comm::MPI.Comm)
    if OVERLAY_NATIVE_CALLS[]
        return Ops.barrier(Ops.constant(comm))
    else
        return call_with_native(MPI.Barrier, comm)
    end
end

# TODO(#2241) status not supported yet
function MPI.Wait(req::TracedRequest)
    return Ops.wait(req)
end

# TODO(#2241) status not supported yet
function MPI.Waitall(req::AbstractVector{TracedRequest})
    return Ops.waitall(req)
end

# TODO(#2241) use `make_tracer` to linearize arbitrary types? check out `MPI.Buffer`
@reactant_overlay function MPI.Send(buf, dest, tag, comm)
    if !any(use_overlayed_version, (buf, dest, tag, comm)) && !OVERLAY_NATIVE_CALLS[]
        return call_with_native(MPI.Send, buf, dest, tag, comm)
    else
        buf_traced = buf isa TracedRArray ? buf : Reactant.Ops.constant(buf)
        dest_traced = dest isa TracedRNumber ? dest : Reactant.Ops.constant(dest)
        tag_traced = tag isa TracedRNumber ? tag : Reactant.Ops.constant(tag)
        comm_traced = comm isa TracedCommunicator ? comm : Ops.constant(comm)
        return Ops.send(buf_traced, dest_traced, tag_traced, comm_traced)
    end
end

# TODO(#2241) use `make_tracer` to linearize arbitrary types? check out `MPI.Buffer`
@reactant_overlay function MPI.Isend(buf, dest, tag, comm)
    if !any(use_overlayed_version, (buf, dest, tag, comm)) && !OVERLAY_NATIVE_CALLS[]
        return call_with_native(MPI.Isend, buf, dest, tag, comm)
    else
        buf_traced = buf isa TracedRArray ? buf : Reactant.Ops.constant(buf)
        dest_traced = dest isa TracedRNumber ? dest : Reactant.Ops.constant(dest)
        tag_traced = tag isa TracedRNumber ? tag : Reactant.Ops.constant(tag)
        comm_traced = comm isa TracedCommunicator ? comm : Ops.constant(comm)
        return Ops.isend(buf_traced, dest_traced, tag_traced, comm_traced)
    end
end

# TODO(#2241) use `make_tracer` to delinearize arbitrary types? check out `MPI.Buffer`
@reactant_overlay function MPI.Recv!(buf, source, tag, comm)
    if !any(use_overlayed_version, (buf, source, tag, comm)) && !OVERLAY_NATIVE_CALLS[]
        return call_with_native(MPI.Recv!, buf, source, tag, comm)
    else
        buf_traced = buf isa TracedRArray ? buf : Reactant.Ops.constant(buf)
        source_traced = source isa TracedRNumber ? source : Reactant.Ops.constant(source)
        tag_traced = tag isa TracedRNumber ? tag : Reactant.Ops.constant(tag)
        comm_traced = comm isa TracedCommunicator ? comm : Ops.constant(comm)
        return Ops.recv!(buf_traced, source_traced, tag_traced, comm_traced)
    end
end

# TODO(#2241) use `make_tracer` to delinearize arbitrary types? check out `MPI.Buffer`
@reactant_overlay function MPI.Irecv!(buf, source, tag, comm)
    if !any(use_overlayed_version, (buf, source, tag, comm)) && !OVERLAY_NATIVE_CALLS[]
        return call_with_native(MPI.Irecv!, buf, source, tag, comm)
    else
        buf_traced = buf isa TracedRArray ? buf : Reactant.Ops.constant(buf)
        source_traced = source isa TracedRNumber ? source : Reactant.Ops.constant(source)
        tag_traced = tag isa TracedRNumber ? tag : Reactant.Ops.constant(tag)
        comm_traced = comm isa TracedCommunicator ? comm : Ops.constant(comm)
        return Ops.irecv!(buf_traced, source_traced, tag_traced, comm_traced)
    end
end

@reactant_overlay function MPI.Allreduce!(sendbuf, recvbuf, op, comm)
    if !any(use_overlayed_version, (sendbuf, recvbuf, op, comm)) && !OVERLAY_NATIVE_CALLS[]
        return call_with_native(MPI.Allreduce!, sendbuf, recvbuf, op, comm)
    else
        sendbuf_traced = sendbuf isa TracedRArray ? sendbuf : Reactant.Ops.constant(sendbuf)
        recvbuf_traced = recvbuf isa TracedRArray ? recvbuf : Reactant.Ops.constant(recvbuf)
        comm_traced = comm isa TracedCommunicator ? comm : Ops.constant(comm)
        return Ops.allreduce!(sendbuf_traced, recvbuf_traced, op, comm_traced)
    end
end

@reactant_overlay function MPI.Bcast!(buf, root, comm)
    if !any(use_overlayed_version, (buf, root, comm)) && !OVERLAY_NATIVE_CALLS[]
        return call_with_native(MPI.Bcast!, buf, root, comm)
    else
        buf_traced = buf isa TracedRArray ? buf : Reactant.Ops.constant(buf)
        root_traced = root isa TracedRNumber ? root : Reactant.Ops.constant(root)
        comm_traced = comm isa TracedCommunicator ? comm : Ops.constant(comm)
        return Ops.bcast!(buf_traced, root_traced, comm_traced)
    end
end
