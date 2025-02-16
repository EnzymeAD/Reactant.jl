module Ops
using Reactant: TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR.Dialects: mpi
using ..ReactantMPIExt: TracedRequest

# TODO add communicators

function init(; location=mlir_stacktrace("mpi.init", @__FILE__, @__LINE__))
    return mpi.init(; location)
end

function finalize(; location=mlir_stacktrace("mpi.finalize", @__FILE__, @__LINE__))
    return mpi.finalize(; location)
end

function comm_rank(; location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__))
    res = MLIR.IR.result(mpi.comm_rank(; location))
    return TracedRNumber{Int}((), res)
end

function comm_size(; location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__))
    res = MLIR.IR.result(mpi.comm_size(; location))
    return TracedRNumber{Int}((), res)
end

# TODO should we emit `stablehlo.optimization_barrier` here too?
function barrier(; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    return mpi.barrier(; location)
end

function send(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.send", @__FILE__, @__LINE__),
)
    return mpi.send(buf.mlir_data, tag.mlir_data, dest.mlir_data; location)
end

# TODO need c-function for creating MLIR `mpi.request` type?
function isend(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.isend", @__FILE__, @__LINE__),
)
    return TracedRequest(
        MLIR.IR.result(mpi.isend(buf.mlir_data, tag.mlir_data, dest.mlir_data; location))
    )
end

function recv!(
    ref::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    return mpi.recv(ref.mlir_data, tag.mlir_data, src.mlir_data; location)
end

# TODO need c-function for creating MLIR `mpi.request` type?
function irecv!(
    ref::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.irecv", @__FILE__, @__LINE__),
)
    return TracedRequest(
        MLIR.IR.result(mpi.irecv(ref.mlir_data, tag.mlir_data, src.mlir_data; location))
    )
end

function wait(
    req::TracedRequest; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    return mpi.wait(req.mlir_data; location)
end

end # module
