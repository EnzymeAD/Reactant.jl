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
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = get_mpi_datatype_name(mpi_datatype)

    count = Reactant.Ops.constant(Int32(length(buf)))

    enzymexla.send(
        buf.mlir_data,
        count.mlir_data,
        dest.mlir_data,
        tag.mlir_data;
        datatype=mpi_datatype_name,
        location
    )

    return nothing
end

@noinline function isend(
    buf::TracedRArray,
    dest::TracedRNumber,
    tag::TracedRNumber;
    location=mlir_stacktrace("mpi.isend", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = get_mpi_datatype_name(mpi_datatype)

    count = Reactant.Ops.constant(Int32(length(buf)))
    request = Reactant.Ops.constant(Int64(-1))

    ret = enzymexla.isend(
        buf.mlir_data,
        count.mlir_data,
        dest.mlir_data,
        tag.mlir_data,
        request.mlir_data;
        outrequest=mlir_type(request),
        datatype=mpi_datatype_name,
        location,
    )

    request.mlir_data = IR.result(ret)
    return request # we return a TracedRNumber, converted to TracedRequest in Overrides.jl
end

@noinline function recv!(
   buf::TracedRArray,
   src::TracedRNumber,
   tag::TracedRNumber;
   location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = get_mpi_datatype_name(mpi_datatype)

    count = Reactant.Ops.constant(Int32(length(buf)))

    ret = enzymexla.recv(
        buf.mlir_data, 
        count.mlir_data, 
        src.mlir_data, 
        tag.mlir_data; 
        outbuf=mlir_type(buf), 
        datatype=mpi_datatype_name,
        location)

    buf.mlir_data = IR.result(ret)
    return buf
end

@noinline function irecv!(
    buf::TracedRArray,
    src::TracedRNumber,
    tag::TracedRNumber;
    location=mlir_stacktrace("mpi.irecv", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = get_mpi_datatype_name(mpi_datatype)

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
        datatype=mpi_datatype_name,
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
    mpi_op_name = get_mpi_op_name(op)

    T = Reactant.unwrapped_eltype(sendbuf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = get_mpi_datatype_name(mpi_datatype)

    count = Reactant.Ops.constant(Int32(length(sendbuf)))

    ret = enzymexla.allreduce(
        sendbuf.mlir_data,
        recvbuf.mlir_data,
        count.mlir_data;
        outbuf=mlir_type(recvbuf),
        datatype=mpi_datatype_name,
        op=mpi_op_name,
        location
    )

    recvbuf.mlir_data = IR.result(ret)
    return recvbuf
end

function get_mpi_op_name(op)
    if op == MPI.OP_NULL
        return "MPI_OP_NULL"
    elseif op == MPI.MAX
        return "MPI_MAX"
    elseif op == MPI.MIN
        return "MPI_MIN"
    elseif op == MPI.SUM
        return "MPI_SUM"
    elseif op == MPI.PROD
        return "MPI_PROD"
    elseif op == MPI.LAND
        return "MPI_LAND"
    elseif op == MPI.BAND
        return "MPI_BAND"
    elseif op == MPI.LOR
        return "MPI_LOR"
    elseif op == MPI.BOR
        return "MPI_BOR"
    elseif op == MPI.LXOR
        return "MPI_LXOR"
    elseif op == MPI.BXOR
        return "MPI_BXOR"
    elseif op == MPI.REPLACE
        return "MPI_REPLACE"
    elseif op == MPI.NO_OP
        return "MPI_NO_OP"
    else
        throw(ArgumentError("Unknown MPI operation `$op`"))
    end
end

function get_mpi_datatype_name(datatype)
    if datatype == MPI.DATATYPE_NULL
        return "MPI_DATATYPE_NULL"
    elseif datatype == MPI.BYTE
        return "MPI_BYTE"
        # elseif datatype == MPI.PACKED
        #     return "MPI_PACKED"
    elseif datatype == MPI.CHAR
        return "MPI_CHAR"
    elseif datatype == MPI.SHORT
        return "MPI_SHORT"
    elseif datatype == MPI.INT
        return "MPI_INT"
    elseif datatype == MPI.LONG
        return "MPI_LONG"
    elseif datatype == MPI.FLOAT
        return "MPI_FLOAT"
    elseif datatype == MPI.DOUBLE
        return "MPI_DOUBLE"
    elseif datatype == MPI.UNSIGNED_CHAR
        return "MPI_UNSIGNED_CHAR"
    elseif datatype == MPI.SIGNED_CHAR
        return "MPI_SIGNED_CHAR"
    elseif datatype == MPI.UNSIGNED_SHORT
        return "MPI_UNSIGNED_SHORT"
    elseif datatype == MPI.UNSIGNED_LONG
        return "MPI_UNSIGNED_LONG"
    elseif datatype == MPI.UNSIGNED
        return "MPI_UNSIGNED"
        # elseif datatype == MPI.FLOAT_INT
        #     return "MPI_FLOAT_INT"
        # elseif datatype == MPI.DOUBLE_INT
        #     return "MPI_DOUBLE_INT"
        # elseif datatype == MPI.LONG_DOUBLE_INT
        #     return "MPI_LONG_DOUBLE_INT"
        # elseif datatype == MPI.LONG_INT
        #     return "MPI_LONG_INT"
        # elseif datatype == MPI.SHORT_INT
        #     return "MPI_SHORT_INT"
        # elseif datatype == MPI.UB
        #     return "MPI_UB"
        # elseif datatype == MPI.LB
        #     return "MPI_LB"
    elseif datatype == MPI.WCHAR
        return "MPI_WCHAR"
    elseif datatype == MPI.LONG_LONG_INT
        return "MPI_LONG_LONG_INT"
    elseif datatype == MPI.UNSIGNED_LONG_LONG
        return "MPI_UNSIGNED_LONG_LONG"
    elseif datatype == MPI.INT8_T
        return "MPI_INT8_T"
    elseif datatype == MPI.UINT8_T
        return "MPI_UINT8_T"
    elseif datatype == MPI.INT16_T
        return "MPI_INT16_T"
    elseif datatype == MPI.UINT16_T
        return "MPI_UINT16_T"
    elseif datatype == MPI.INT32_T
        return "MPI_INT32_T"
    elseif datatype == MPI.UINT32_T
        return "MPI_UINT32_T"
    elseif datatype == MPI.INT64_T
        return "MPI_INT64_T"
    elseif datatype == MPI.UINT64_T
        return "MPI_UINT64_T"
    elseif datatype == MPI.AINT
        return "MPI_AINT"
    elseif datatype == MPI.OFFSET
        return "MPI_OFFSET"
    elseif datatype == MPI.C_BOOL
        return "MPI_C_BOOL"
    elseif datatype == MPI.C_FLOAT_COMPLEX
        return "MPI_C_FLOAT_COMPLEX"
    elseif datatype == MPI.C_DOUBLE_COMPLEX
        return "MPI_C_DOUBLE_COMPLEX"
    elseif datatype == MPI.COUNT
        return "MPI_COUNT"
    else
        throw(ArgumentError("Unknown MPI datatype `$datatype`"))
    end
end

end # module
