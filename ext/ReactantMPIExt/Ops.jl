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
    rank = Reactant.Ops.constant(Int32(-1))
    ret = enzymexla.mpi_comm_rank(; rank=mlir_type(rank), location)
    rank.mlir_data = IR.result(ret)
    return rank
end

@noinline function comm_size(;
    location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__)
)
    size = Reactant.Ops.constant(Int32(-1))
    ret = enzymexla.mpi_comm_size(; size=mlir_type(size), location)
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
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))

    enzymexla.mpi_send(
        buf.mlir_data,
        count.mlir_data,
        dest.mlir_data,
        tag.mlir_data;
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(MLIR.IR.context(), mpi_datatype),
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
    request = Reactant.Ops.constant(Int64(-1))

    ret = enzymexla.mpi_isend(
        buf.mlir_data,
        count.mlir_data,
        dest.mlir_data,
        tag.mlir_data;
        request=mlir_type(request),
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(MLIR.IR.context(), mpi_datatype),
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
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))

    ret = enzymexla.mpi_recv(
        buf.mlir_data,
        count.mlir_data,
        src.mlir_data,
        tag.mlir_data;
        outbuf=mlir_type(buf),
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(MLIR.IR.context(), mpi_datatype),
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
    request = Reactant.Ops.constant(Int64(-1))

    ret = enzymexla.mpi_irecv(
        buf.mlir_data,
        count.mlir_data,
        src.mlir_data,
        tag.mlir_data;
        outbuf=mlir_type(buf),
        request=mlir_type(request),
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(MLIR.IR.context(), mpi_datatype),
        location,
    )

    buf.mlir_data = IR.result(ret, 1)
    request.mlir_data = IR.result(ret, 2)
    return request # we return a TracedRNumber, converted to TracedRequest in Overrides.jl
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
    mpi_op_name = get_mpi_op_name(op)

    T = Reactant.unwrapped_eltype(sendbuf)
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(sendbuf)))

    ret = enzymexla.mpi_allreduce(
        sendbuf.mlir_data,
        recvbuf.mlir_data,
        count.mlir_data;
        outbuf=mlir_type(recvbuf),
        datatype=MLIR.API.enzymexlaMPIDatatypeAttrGet(MLIR.IR.context(), mpi_datatype),
        op=mpi_op_name,
        location,
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

@enum MPIDataTypeEnum begin
    MPI_DATATYPE_NULL_ENUM = 0
    MPI_BYTE_ENUM = 1
    MPI_CHAR_ENUM = 2
    MPI_SHORT_ENUM = 3
    MPI_INT_ENUM = 4
    MPI_LONG_ENUM = 5
    MPI_FLOAT_ENUM = 6
    MPI_DOUBLE_ENUM = 7
    MPI_UNSIGNED_CHAR_ENUM = 8
    MPI_SIGNED_CHAR_ENUM = 9
    MPI_UNSIGNED_SHORT_ENUM = 10
    MPI_UNSIGNED_LONG_ENUM = 11
    MPI_UNSIGNED_ENUM = 12
    MPI_WCHAR_ENUM = 13
    MPI_LONG_LONG_INT_ENUM = 14
    MPI_UNSIGNED_LONG_LONG_ENUM = 15
    MPI_INT8_T_ENUM = 16
    MPI_UINT8_T_ENUM = 17
    MPI_INT16_T_ENUM = 18
    MPI_UINT16_T_ENUM = 19
    MPI_INT32_T_ENUM = 20
    MPI_UINT32_T_ENUM = 21
    MPI_INT64_T_ENUM = 22
    MPI_UINT64_T_ENUM = 23
    MPI_AINT_ENUM = 24
    MPI_OFFSET_ENUM = 25
    MPI_C_BOOL_ENUM = 26
    MPI_C_FLOAT_COMPLEX_ENUM = 27
    MPI_C_DOUBLE_COMPLEX_ENUM = 28
    MPI_COUNT_ENUM = 29
    MPI_PACKED_ENUM = 30
    MPI_FLOAT_INT_ENUM = 31
    MPI_DOUBLE_INT_ENUM = 32
    MPI_LONG_DOUBLE_INT_ENUM = 33
    MPI_LONG_INT_ENUM = 34
    MPI_SHORT_INT_ENUM = 35
    MPI_UB_ENUM = 36
    MPI_LB_ENUM = 37
end

const MPI_DATATYPE_MAP = Dict(
    MPI.DATATYPE_NULL => MPI_DATATYPE_NULL_ENUM,
    MPI.BYTE => MPI_BYTE_ENUM,
    # MPI.PACKED => MPI_PACKED_ENUM,
    MPI.CHAR => MPI_CHAR_ENUM,
    MPI.SHORT => MPI_SHORT_ENUM,
    MPI.INT => MPI_INT_ENUM,
    MPI.LONG => MPI_LONG_ENUM,
    MPI.FLOAT => MPI_FLOAT_ENUM,
    MPI.DOUBLE => MPI_DOUBLE_ENUM,
    MPI.UNSIGNED_CHAR => MPI_UNSIGNED_CHAR_ENUM,
    MPI.SIGNED_CHAR => MPI_SIGNED_CHAR_ENUM,
    MPI.UNSIGNED_SHORT => MPI_UNSIGNED_SHORT_ENUM,
    MPI.UNSIGNED_LONG => MPI_UNSIGNED_LONG_ENUM,
    MPI.UNSIGNED => MPI_UNSIGNED_ENUM,
    # MPI.FLOAT_INT => MPI_FLOAT_INT_ENUM,
    # MPI.DOUBLE_INT => MPI_DOUBLE_INT_ENUM,
    # MPI.LONG_DOUBLE_INT => MPI_LONG_DOUBLE_INT_ENUM,
    # MPI.LONG_INT => MPI_LONG_INT_ENUM,
    # MPI.SHORT_INT => MPI_SHORT_INT_ENUM,
    # MPI.UB => MPI_UB_ENUM,
    # MPI.LB => MPI_LB_ENUM,
    MPI.WCHAR => MPI_WCHAR_ENUM,
    MPI.LONG_LONG_INT => MPI_LONG_LONG_INT_ENUM,
    MPI.UNSIGNED_LONG_LONG => MPI_UNSIGNED_LONG_LONG_ENUM,
    MPI.INT8_T => MPI_INT8_T_ENUM,
    MPI.UINT8_T => MPI_UINT8_T_ENUM,
    MPI.INT16_T => MPI_INT16_T_ENUM,
    MPI.UINT16_T => MPI_UINT16_T_ENUM,
    MPI.INT32_T => MPI_INT32_T_ENUM,
    MPI.UINT32_T => MPI_UINT32_T_ENUM,
    MPI.INT64_T => MPI_INT64_T_ENUM,
    MPI.UINT64_T => MPI_UINT64_T_ENUM,
    # MPI.AINT => MPI_AINT_ENUM,
    # MPI.OFFSET => MPI_OFFSET_ENUM,
    MPI.C_BOOL => MPI_C_BOOL_ENUM,
    MPI.C_FLOAT_COMPLEX => MPI_C_FLOAT_COMPLEX_ENUM,
    MPI.C_DOUBLE_COMPLEX => MPI_C_DOUBLE_COMPLEX_ENUM,
    # MPI.COUNT => MPI_COUNT_ENUM
)

function get_mpi_datatype_enum(datatype)
    return get(MPI_DATATYPE_MAP, datatype) do
        throw(ArgumentError("Unknown MPI datatype `$datatype`"))
    end
end

end # module
