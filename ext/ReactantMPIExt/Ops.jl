module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.MLIR.IR: @mlir_str
using Reactant.MLIR.Dialects: mpi, func, llvm, enzymexla
using Reactant.Ops: mlir_stacktrace, mlir_type
using ..ReactantMPIExt: TracedRequest
using MPI: MPI

# TODO
# function init(; location=mlir_stacktrace("mpi.init", @__FILE__, @__LINE__))
#     return mpi.init(; location)
# end

# TODO
# function finalize(; location=mlir_stacktrace("mpi.finalize", @__FILE__, @__LINE__))
#     return mpi.finalize(; location)
# end

@noinline function comm_rank(; location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__))
    rank = Reactant.Ops.constant(Int32(-1))
    ret = enzymexla.comm_rank(rank.mlir_data; outrank=mlir_type(rank), location)
    rank.mlir_data = IR.result(ret)
    return rank
end

@noinline function comm_size(; location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__))
    size = Reactant.Ops.constant(Int32(-1))
    ret = enzymexla.comm_size(size.mlir_data; outsize=mlir_type(size), location)
    size.mlir_data = IR.result(ret)
    return size
end

@noinline function barrier(; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    enzymexla.mpi_barrier(; location)
    return nothing
end

@noinline function send(
    buf::TracedRArray,
    dest::TracedRNumber,
    tag::TracedRNumber;
    location=mlir_stacktrace("mpi.send", @__FILE__, @__LINE__),
)
    count = Reactant.Ops.constant(Int32(length(buf)))
    enzymexla.send(buf.mlir_data, count.mlir_data, dest.mlir_data, tag.mlir_data; location)
    return nothing
end

@noinline function isend(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.isend", @__FILE__, @__LINE__),
)
    count = Reactant.Ops.constant(Int32(length(buf)))
    request = Reactant.Ops.constant(Int64(-1))

    ret = enzymexla.isend(
        buf.mlir_data,
        count.mlir_data,
        dest.mlir_data,
        tag.mlir_data,
        request.mlir_data;
        outrequest=mlir_type(request),
        location,
    )

    request.mlir_data = IR.result(ret)
    return request # we return a TracedRNumber, converted to TracedRequest in Overrides.jl
end

@noinline function recv!(
   recvbuf::TracedRArray,
   src::TracedRNumber,
   tag::TracedRNumber;
   location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    count = Reactant.Ops.constant(Int32(length(recvbuf)))
    ret = enzymexla.recv(recvbuf.mlir_data, 
                         count.mlir_data, 
                         src.mlir_data, 
                         tag.mlir_data; 
                         outbuf=mlir_type(recvbuf), 
                         location)
    recvbuf.mlir_data = IR.result(ret)
    return recvbuf
end

@noinline function irecv!(
    buf::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.irecv", @__FILE__, @__LINE__),
)
    count = Reactant.Ops.constant(Int32(length(buf)))
    request = Reactant.Ops.constant(Int64(-1))

    ret = enzymexla.irecv(
        buf.mlir_data,
        count.mlir_data,
        src.mlir_data,
        tag.mlir_data,
        request.mlir_data;
        outbuf=mlir_type(buf),
        outrequest=mlir_type(request),
        location,
    )

   buf.mlir_data = IR.result(ret, 1)
   request.mlir_data = IR.result(ret, 2)
   return request # we return a TracedRNumber, converted to TracedRequest in Overrides.jl
end

@noinline function wait(
    req::TracedRequest; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    enzymexla.wait(req.mlir_data; location)
    return nothing
end

@noinline function allreduce!(
    op,
    sendbuf::TracedRArray,
    recvbuf::TracedRArray;
    location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    count = Reactant.Ops.constant(Int32(length(sendbuf)))
    ret = enzymexla.allreduce(
        sendbuf.mlir_data,
        recvbuf.mlir_data,
        count.mlir_data;
        outbuf=mlir_type(recvbuf),
        location
    )
    recvbuf.mlir_data = IR.result(ret)
    return recvbuf
end

end # module
