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
    dest::TracedRNumber,
    tag::TracedRNumber;
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
   buf::TracedRArray,
   src::TracedRNumber,
   tag::TracedRNumber;
   location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    count = Reactant.Ops.constant(Int32(length(buf)))
    ret = enzymexla.recv(buf.mlir_data, 
                         count.mlir_data, 
                         src.mlir_data, 
                         tag.mlir_data; 
                         outbuf=mlir_type(buf), 
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

function inject_mpi_op!(op)
    if op == MPI.OP_NULL
        IR.inject!("MPI_OP_NULL", "llvm.mlir.global constant @MPI_OP_NULL() : !llvm.ptr")
        return "MPI_OP_NULL"
    elseif op == MPI.MAX
        IR.inject!("MPI_MAX", "llvm.mlir.global constant @MPI_MAX() : !llvm.ptr")
        return "MPI_MAX"
    elseif op == MPI.MIN
        IR.inject!("MPI_MIN", "llvm.mlir.global constant @MPI_MIN() : !llvm.ptr")
        return "MPI_MIN"
    elseif op == MPI.SUM
        IR.inject!("MPI_SUM", "llvm.mlir.global constant @MPI_SUM() : !llvm.ptr")
        return "MPI_SUM"
    elseif op == MPI.PROD
        IR.inject!("MPI_PROD", "llvm.mlir.global constant @MPI_PROD() : !llvm.ptr")
        return "MPI_PROD"
    elseif op == MPI.LAND
        IR.inject!("MPI_LAND", "llvm.mlir.global constant @MPI_LAND() : !llvm.ptr")
        return "MPI_LAND"
    elseif op == MPI.BAND
        IR.inject!("MPI_BAND", "llvm.mlir.global constant @MPI_BAND() : !llvm.ptr")
        return "MPI_BAND"
    elseif op == MPI.LOR
        IR.inject!("MPI_LOR", "llvm.mlir.global constant @MPI_LOR() : !llvm.ptr")
        return "MPI_LOR"
    elseif op == MPI.BOR
        IR.inject!("MPI_BOR", "llvm.mlir.global constant @MPI_BOR() : !llvm.ptr")
        return "MPI_BOR"
    elseif op == MPI.LXOR
        IR.inject!("MPI_LXOR", "llvm.mlir.global constant @MPI_LXOR() : !llvm.ptr")
        return "MPI_LXOR"
    elseif op == MPI.BXOR
        IR.inject!("MPI_BXOR", "llvm.mlir.global constant @MPI_BXOR() : !llvm.ptr")
        return "MPI_BXOR"
    elseif op == MPI.REPLACE
        IR.inject!("MPI_REPLACE", "llvm.mlir.global constant @MPI_REPLACE() : !llvm.ptr")
        return "MPI_REPLACE"
    elseif op == MPI.NO_OP
        IR.inject!("MPI_NO_OP", "llvm.mlir.global constant @MPI_NO_OP() : !llvm.ptr")
        return "MPI_NO_OP"
    else
        throw(ArgumentError("Unknown MPI operation `$op`"))
    end
end

function inject_mpi_datatype!(datatype)
    if datatype == MPI.DATATYPE_NULL
        IR.inject!(
            "MPI_DATATYPE_NULL",
            "llvm.mlir.global constant @MPI_DATATYPE_NULL() : !llvm.ptr",
        )
        return "MPI_DATATYPE_NULL"
    elseif datatype == MPI.BYTE
        IR.inject!("MPI_BYTE", "llvm.mlir.global constant @MPI_BYTE() : !llvm.ptr")
        return "MPI_BYTE"
        # elseif datatype == MPI.PACKED
        #     IR.inject!("MPI_PACKED", "llvm.mlir.global constant @MPI_PACKED() : !llvm.ptr")
        #     return "MPI_PACKED"
    elseif datatype == MPI.CHAR
        IR.inject!("MPI_CHAR", "llvm.mlir.global constant @MPI_CHAR() : !llvm.ptr")
        return "MPI_CHAR"
    elseif datatype == MPI.SHORT
        IR.inject!("MPI_SHORT", "llvm.mlir.global constant @MPI_SHORT() : !llvm.ptr")
        return "MPI_SHORT"
    elseif datatype == MPI.INT
        IR.inject!("MPI_INT", "llvm.mlir.global constant @MPI_INT() : !llvm.ptr")
        return "MPI_INT"
    elseif datatype == MPI.LONG
        IR.inject!("MPI_LONG", "llvm.mlir.global constant @MPI_LONG() : !llvm.ptr")
        return "MPI_LONG"
    elseif datatype == MPI.FLOAT
        IR.inject!("MPI_FLOAT", "llvm.mlir.global constant @MPI_FLOAT() : !llvm.ptr")
        return "MPI_FLOAT"
    elseif datatype == MPI.DOUBLE
        IR.inject!("MPI_DOUBLE", "llvm.mlir.global constant @MPI_DOUBLE() : !llvm.ptr")
        return "MPI_DOUBLE"
    elseif datatype == MPI.UNSIGNED_CHAR
        IR.inject!(
            "MPI_UNSIGNED_CHAR",
            "llvm.mlir.global constant @MPI_UNSIGNED_CHAR() : !llvm.ptr",
        )
        return "MPI_UNSIGNED_CHAR"
    elseif datatype == MPI.SIGNED_CHAR
        IR.inject!(
            "MPI_SIGNED_CHAR", "llvm.mlir.global constant @MPI_SIGNED_CHAR() : !llvm.ptr"
        )
        return "MPI_SIGNED_CHAR"
    elseif datatype == MPI.UNSIGNED_SHORT
        IR.inject!(
            "MPI_UNSIGNED_SHORT",
            "llvm.mlir.global constant @MPI_UNSIGNED_SHORT() : !llvm.ptr",
        )
        return "MPI_UNSIGNED_SHORT"
    elseif datatype == MPI.UNSIGNED_LONG
        IR.inject!(
            "MPI_UNSIGNED_LONG",
            "llvm.mlir.global constant @MPI_UNSIGNED_LONG() : !llvm.ptr",
        )
        return "MPI_UNSIGNED_LONG"
    elseif datatype == MPI.UNSIGNED
        IR.inject!("MPI_UNSIGNED", "llvm.mlir.global constant @MPI_UNSIGNED() : !llvm.ptr")
        return "MPI_UNSIGNED"
        # elseif datatype == MPI.FLOAT_INT
        #     IR.inject!(
        #         "MPI_FLOAT_INT", "llvm.mlir.global constant @MPI_FLOAT_INT() : !llvm.ptr"
        #     )
        #     return "MPI_FLOAT_INT"
        # elseif datatype == MPI.DOUBLE_INT
        #     IR.inject!(
        #         "MPI_DOUBLE_INT", "llvm.mlir.global constant @MPI_DOUBLE_INT() : !llvm.ptr"
        #     )
        #     return "MPI_DOUBLE_INT"
        # elseif datatype == MPI.LONG_DOUBLE_INT
        #     IR.inject!(
        #         "MPI_LONG_DOUBLE_INT",
        #         "llvm.mlir.global constant @MPI_LONG_DOUBLE_INT() : !llvm.ptr",
        #     )
        #     return "MPI_LONG_DOUBLE_INT"
        # elseif datatype == MPI.LONG_INT
        #     IR.inject!("MPI_LONG_INT", "llvm.mlir.global constant @MPI_LONG_INT() : !llvm.ptr")
        #     return "MPI_LONG_INT"
        # elseif datatype == MPI.SHORT_INT
        #     IR.inject!(
        #         "MPI_SHORT_INT", "llvm.mlir.global constant @MPI_SHORT_INT() : !llvm.ptr"
        #     )
        #     return "MPI_SHORT_INT"
        # elseif datatype == MPI.UB
        #     IR.inject!("MPI_UB", "llvm.mlir.global constant @MPI_UB() : !llvm.ptr")
        #     return "MPI_UB"
        # elseif datatype == MPI.LB
        #     IR.inject!("MPI_LB", "llvm.mlir.global constant @MPI_LB() : !llvm.ptr")
        #     return "MPI_LB"
    elseif datatype == MPI.WCHAR
        IR.inject!("MPI_WCHAR", "llvm.mlir.global constant @MPI_WCHAR() : !llvm.ptr")
        return "MPI_WCHAR"
    elseif datatype == MPI.LONG_LONG_INT
        IR.inject!(
            "MPI_LONG_LONG_INT",
            "llvm.mlir.global constant @MPI_LONG_LONG_INT() : !llvm.ptr",
        )
        return "MPI_LONG_LONG_INT"
    elseif datatype == MPI.UNSIGNED_LONG_LONG
        IR.inject!(
            "MPI_UNSIGNED_LONG_LONG",
            "llvm.mlir.global constant @MPI_UNSIGNED_LONG_LONG() : !llvm.ptr",
        )
        return "MPI_UNSIGNED_LONG_LONG"
    elseif datatype == MPI.INT8_T
        IR.inject!("MPI_INT8_T", "llvm.mlir.global constant @MPI_INT8_T() : !llvm.ptr")
        return "MPI_INT8_T"
    elseif datatype == MPI.UINT8_T
        IR.inject!("MPI_UINT8_T", "llvm.mlir.global constant @MPI_UINT8_T() : !llvm.ptr")
        return "MPI_UINT8_T"
    elseif datatype == MPI.INT16_T
        IR.inject!("MPI_INT16_T", "llvm.mlir.global constant @MPI_INT16_T() : !llvm.ptr")
        return "MPI_INT16_T"
    elseif datatype == MPI.UINT16_T
        IR.inject!("MPI_UINT16_T", "llvm.mlir.global constant @MPI_UINT16_T() : !llvm.ptr")
        return "MPI_UINT16_T"
    elseif datatype == MPI.INT32_T
        IR.inject!("MPI_INT32_T", "llvm.mlir.global constant @MPI_INT32_T() : !llvm.ptr")
        return "MPI_INT32_T"
    elseif datatype == MPI.UINT32_T
        IR.inject!("MPI_UINT32_T", "llvm.mlir.global constant @MPI_UINT32_T() : !llvm.ptr")
        return "MPI_UINT32_T"
    elseif datatype == MPI.INT64_T
        IR.inject!("MPI_INT64_T", "llvm.mlir.global constant @MPI_INT64_T() : !llvm.ptr")
        return "MPI_INT64_T"
    elseif datatype == MPI.UINT64_T
        IR.inject!("MPI_UINT64_T", "llvm.mlir.global constant @MPI_UINT64_T() : !llvm.ptr")
        return "MPI_UINT64_T"
    elseif datatype == MPI.AINT
        IR.inject!("MPI_AINT", "llvm.mlir.global constant @MPI_AINT() : !llvm.ptr")
        return "MPI_AINT"
    elseif datatype == MPI.OFFSET
        IR.inject!("MPI_OFFSET", "llvm.mlir.global constant @MPI_OFFSET() : !llvm.ptr")
        return "MPI_OFFSET"
    elseif datatype == MPI.C_BOOL
        IR.inject!("MPI_C_BOOL", "llvm.mlir.global constant @MPI_C_BOOL() : !llvm.ptr")
        return "MPI_C_BOOL"
    elseif datatype == MPI.C_FLOAT_COMPLEX
        IR.inject!(
            "MPI_C_FLOAT_COMPLEX",
            "llvm.mlir.global constant @MPI_C_FLOAT_COMPLEX() : !llvm.ptr",
        )
        return "MPI_C_FLOAT_COMPLEX"
    elseif datatype == MPI.C_DOUBLE_COMPLEX
        IR.inject!(
            "MPI_C_DOUBLE_COMPLEX",
            "llvm.mlir.global constant @MPI_C_DOUBLE_COMPLEX() : !llvm.ptr",
        )
        return "MPI_C_DOUBLE_COMPLEX"
    elseif datatype == MPI.COUNT
        IR.inject!("MPI_COUNT", "llvm.mlir.global constant @MPI_COUNT() : !llvm.ptr")
        return "MPI_COUNT"
    else
        throw(ArgumentError("Unknown MPI datatype `$datatype`"))
    end
end

end # module
