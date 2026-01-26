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

@noinline function comm_rank(;
    location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__)
)
    rank = mlir_type(TracedRArray{Int32,0}, ())
    res = IR.result(enzymexla.mpi_comm_rank(; rank, location))
    return TracedRNumber{Int32}((), res)
end

@noinline function comm_size(;
    location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__)
)
    size = mlir_type(TracedRArray{Int32,0}, ())
    res = IR.result(enzymexla.mpi_comm_size(; size, location))
    return TracedRNumber{Int32}((), res)
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
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))

    enzymexla.mpi_send(
        buf.mlir_data,
        count.mlir_data,
        dest.mlir_data,
        tag.mlir_data;
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        location,
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
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))
    request = mlir_type(TracedRArray{Int64,0}, ())

    res = IR.result(
        enzymexla.mpi_isend(
            buf.mlir_data,
            count.mlir_data,
            dest.mlir_data,
            tag.mlir_data;
            request,
            datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(
                IR.current_context(), mpi_datatype
            ),
            location,
        ),
    )

    return TracedRNumber{Int64}((), res)
end

@noinline function recv!(
    buf::TracedRArray,
    src::TracedRNumber,
    tag::TracedRNumber;
    location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))

    ret = enzymexla.mpi_recv(
        buf.mlir_data,
        count.mlir_data,
        src.mlir_data,
        tag.mlir_data;
        outbuf=mlir_type(buf),
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        location,
    )

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
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))
    request = mlir_type(TracedRArray{Int64,0}, ())

    ret = enzymexla.mpi_irecv(
        buf.mlir_data,
        count.mlir_data,
        src.mlir_data,
        tag.mlir_data;
        outbuf=mlir_type(buf),
        request,
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        location,
    )

    buf.mlir_data = IR.result(ret, 1)
    request = TracedRNumber{Int64}((), IR.result(ret, 2))
    return request
end

@noinline function wait(
    req::TracedRequest; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    enzymexla.mpi_wait(req.mlir_data; location)
    return nothing
end

@noinline function allreduce!(
    op,
    sendbuf::TracedRArray,
    recvbuf::TracedRArray;
    location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__),
)
    mpi_op = get_mpi_op_enum(op)

    T = Reactant.unwrapped_eltype(sendbuf)
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(sendbuf)))

    ret = enzymexla.mpi_allreduce(
        sendbuf.mlir_data,
        recvbuf.mlir_data,
        count.mlir_data;
        outbuf=mlir_type(recvbuf),
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        op=MLIR.API.enzymexlaMPIOpAttrGet(IR.current_context(), mpi_op),
        location,
    )

    recvbuf.mlir_data = IR.result(ret)
    return recvbuf
end

@enum MPIOpEnum begin
    MPI_OP_NULL_ENUM = 0
    MPI_BAND_ENUM = 1
    MPI_BOR_ENUM = 2
    MPI_BXOR_ENUM = 3
    MPI_LAND_ENUM = 4
    MPI_LOR_ENUM = 5
    MPI_LXOR_ENUM = 6
    MPI_MAX_ENUM = 7
    MPI_MIN_ENUM = 8
    MPI_PROD_ENUM = 9
    MPI_REPLACE_ENUM = 10
    MPI_SUM_ENUM = 11
    MPI_NO_OP_ENUM = 12
end

const MPI_OP_MAP = Dict(
    MPI.OP_NULL.val => MPI_OP_NULL_ENUM,
    MPI.BAND.val => MPI_BAND_ENUM,
    MPI.BOR.val => MPI_BOR_ENUM,
    MPI.BXOR.val => MPI_BXOR_ENUM,
    MPI.LAND.val => MPI_LAND_ENUM,
    MPI.LOR.val => MPI_LOR_ENUM,
    MPI.LXOR.val => MPI_LXOR_ENUM,
    MPI.MAX.val => MPI_MAX_ENUM,
    MPI.MIN.val => MPI_MIN_ENUM,
    MPI.PROD.val => MPI_PROD_ENUM,
    MPI.REPLACE.val => MPI_REPLACE_ENUM,
    MPI.SUM.val => MPI_SUM_ENUM,
    MPI.NO_OP.val => MPI_NO_OP_ENUM,
)

function get_mpi_op_enum(op)
    return get(MPI_OP_MAP, op.val) do
        throw(ArgumentError("Unknown MPI op `$op`"))
    end
end

@enum MPIDataTypeEnum begin
    MPI_DATATYPE_NULL_ENUM = 0
    MPI_INT8_T_ENUM = 1
    MPI_UINT8_T_ENUM = 2
    MPI_INT16_T_ENUM = 3
    MPI_UINT16_T_ENUM = 4
    MPI_INT32_T_ENUM = 5
    MPI_UINT32_T_ENUM = 6
    MPI_INT64_T_ENUM = 7
    MPI_UINT64_T_ENUM = 8
    MPI_BYTE_ENUM = 9
    MPI_SHORT_ENUM = 10
    MPI_UNSIGNED_SHORT_ENUM = 11
    MPI_INT_ENUM = 12
    MPI_UNSIGNED_ENUM = 13
    MPI_LONG_ENUM = 14
    MPI_UNSIGNED_LONG_ENUM = 15
    MPI_LONG_LONG_INT_ENUM = 16
    MPI_UNSIGNED_LONG_LONG_ENUM = 17
    MPI_CHAR_ENUM = 18
    MPI_SIGNED_CHAR_ENUM = 19
    MPI_UNSIGNED_CHAR_ENUM = 20
    MPI_WCHAR_ENUM = 21
    MPI_FLOAT_ENUM = 22
    MPI_DOUBLE_ENUM = 23
    MPI_C_FLOAT_COMPLEX_ENUM = 24
    MPI_C_DOUBLE_COMPLEX_ENUM = 25
    MPI_C_BOOL_ENUM = 26
end

const MPI_DATATYPE_MAP = Dict(
    MPI.DATATYPE_NULL.val => MPI_DATATYPE_NULL_ENUM,
    MPI.INT8_T.val => MPI_INT8_T_ENUM,
    MPI.UINT8_T.val => MPI_UINT8_T_ENUM,
    MPI.INT16_T.val => MPI_INT16_T_ENUM,
    MPI.UINT16_T.val => MPI_UINT16_T_ENUM,
    MPI.INT32_T.val => MPI_INT32_T_ENUM,
    MPI.UINT32_T.val => MPI_UINT32_T_ENUM,
    MPI.INT64_T.val => MPI_INT64_T_ENUM,
    MPI.UINT64_T.val => MPI_UINT64_T_ENUM,
    MPI.BYTE.val => MPI_BYTE_ENUM,
    MPI.SHORT.val => MPI_SHORT_ENUM,
    MPI.UNSIGNED_SHORT.val => MPI_UNSIGNED_SHORT_ENUM,
    MPI.INT.val => MPI_INT_ENUM,
    MPI.UNSIGNED.val => MPI_UNSIGNED_ENUM,
    MPI.LONG.val => MPI_LONG_ENUM,
    MPI.UNSIGNED_LONG.val => MPI_UNSIGNED_LONG_ENUM,
    MPI.LONG_LONG_INT.val => MPI_LONG_LONG_INT_ENUM,
    MPI.UNSIGNED_LONG_LONG.val => MPI_UNSIGNED_LONG_LONG_ENUM,
    MPI.CHAR.val => MPI_CHAR_ENUM,
    MPI.SIGNED_CHAR.val => MPI_SIGNED_CHAR_ENUM,
    MPI.UNSIGNED_CHAR.val => MPI_UNSIGNED_CHAR_ENUM,
    MPI.WCHAR.val => MPI_WCHAR_ENUM,
    MPI.FLOAT.val => MPI_FLOAT_ENUM,
    MPI.DOUBLE.val => MPI_DOUBLE_ENUM,
    MPI.C_FLOAT_COMPLEX.val => MPI_C_FLOAT_COMPLEX_ENUM,
    MPI.C_DOUBLE_COMPLEX.val => MPI_C_DOUBLE_COMPLEX_ENUM,
    MPI.C_BOOL.val => MPI_C_BOOL_ENUM,
)

function get_mpi_datatype_enum(datatype)
    return get(MPI_DATATYPE_MAP, datatype.val) do
        throw(ArgumentError("Unknown MPI datatype `$datatype`"))
    end
end

end # module
