module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.Ops: mlir_stacktrace, mlir_type
using MPI: MPI

# TODO(#2242)
# function init(; location=mlir_stacktrace("mpi.init", @__FILE__, @__LINE__))
#     return mpi.init(; location)
# end

# TODO(#2242)
# function finalize(; location=mlir_stacktrace("mpi.finalize", @__FILE__, @__LINE__))
#     return mpi.finalize(; location)
# end

@noinline function comm_rank(;
    location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__)
)
    rank = mlir_type(TracedRArray{Int32,0}, ())
    op = create_operation(
        "enzymexla.mpi.comm_rank",
        location;
        attributes=[MLIR.IR.NamedAttribute("rank", rank)],
    )
    res = IR.result(op)
    return TracedRNumber{Int32}((), res)
end

@noinline function comm_size(;
    location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__)
)
    size = mlir_type(TracedRArray{Int32,0}, ())
    op = create_operation(
        "enzymexla.mpi.comm_size",
        location;
        attributes=[MLIR.IR.NamedAttribute("size", size)],
    )
    res = IR.result(op)
    return TracedRNumber{Int32}((), res)
end

@noinline function barrier(; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    create_operation("enzymexla.mpi.barrier", location)
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

    operands = Reactant.TracedUtils.get_mlir_data.([buf, count, dest, tag])
    attributes = [
        MLIR.IR.NamedAttribute(
            "datatype",
            MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        ),
    ]
    create_operation("enzymexla.mpi.send", location; operands, attributes)

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

    attributes = [
        MLIR.IR.NamedAttribute(
            "datatype",
            MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        ),
    ]
    operands = Reactant.TracedUtils.get_mlir_data.([buf, count, dest, tag])
    results = [mlir_type(TracedRArray{Int32,0}, ())]
    op = create_operation("enzymexla.mpi.isend", location; operands, attributes, results)
    return TracedRNumber{Int32}((), IR.result(op))
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

    attributes = [
        MLIR.IR.NamedAttribute(
            "datatype",
            MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        ),
    ]
    operands = Reactant.TracedUtils.get_mlir_data.([buf, count, src, tag])
    op = create_operation(
        "enzymexla.mpi.recv", location; operands, attributes, results=[mlir_type(buf)]
    )

    buf.mlir_data = IR.result(op)
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

    attributes = [
        MLIR.IR.NamedAttribute(
            "datatype",
            MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        ),
    ]
    operands = Reactant.TracedUtils.get_mlir_data.([buf, count, src, tag])
    results = [mlir_type(buf), mlir_type(TracedRArray{Int32,0}, ())]
    op = create_operation("enzymexla.mpi.irecv", location; operands, attributes, results)

    buf.mlir_data = IR.result(op, 1)
    request = TracedRNumber{Int32}((), IR.result(op, 2))
    return request
end

@noinline function wait(
    req::TracedRNumber; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    operands = [Reactant.TracedUtils.get_mlir_data(req)]
    create_operation("enzymexla.mpi.wait", location; operands)
    return nothing
end

@noinline function waitall(
    req::TracedRArray; location=mlir_stacktrace("mpi.waitall", @__FILE__, @__LINE__)
)
    count = Reactant.Ops.constant(Int32(length(req)))
    operands = Reactant.TracedUtils.get_mlir_data.([req, count])
    create_operation("enzymexla.mpi.waitall", location; operands)
    return nothing
end

@noinline function allreduce!(
    op,
    sendbuf::TracedRArray,
    recvbuf::TracedRArray;
    location=mlir_stacktrace("mpi.allreduce", @__FILE__, @__LINE__),
)
    mpi_op = get_mpi_op_enum(op)

    T = Reactant.unwrapped_eltype(sendbuf)
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(sendbuf)))

    attributes = [
        MLIR.IR.NamedAttribute(
            "datatype",
            MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        ),
        MLIR.IR.NamedAttribute(
            "op", MLIR.API.enzymexlaMPIOpAttrGet(IR.current_context(), mpi_op)
        ),
    ]
    operands = Reactant.TracedUtils.get_mlir_data.([sendbuf, recvbuf, count])
    op = create_operation(
        "enzymexla.mpi.allreduce",
        location;
        operands,
        attributes,
        results=[mlir_type(recvbuf)],
    )

    recvbuf.mlir_data = IR.result(op)
    return recvbuf
end

@noinline function bcast!(
    buf::TracedRArray,
    root::TracedRNumber;
    location=mlir_stacktrace("mpi.bcast", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = get_mpi_datatype_enum(MPI.Datatype(T))

    count = Reactant.Ops.constant(Int32(length(buf)))

    attributes = [
        MLIR.IR.NamedAttribute(
            "datatype",
            MLIR.API.enzymexlaMPIDatatypeAttrGet(IR.current_context(), mpi_datatype),
        ),
    ]
    operands = Reactant.TracedUtils.get_mlir_data.([buf, count, root])
    op = create_operation(
        "enzymexla.mpi.bcast", location; operands, attributes, results=[mlir_type(buf)]
    )

    buf.mlir_data = IR.result(op)
    return buf
end

const MPI_OP_MAP = Dict(
    MPI.OP_NULL.val => MLIR.API.ENZYMEXLA_MPI_OP_NULL,
    MPI.BAND.val => MLIR.API.ENZYMEXLA_MPI_BAND,
    MPI.BOR.val => MLIR.API.ENZYMEXLA_MPI_BOR,
    MPI.BXOR.val => MLIR.API.ENZYMEXLA_MPI_BXOR,
    MPI.LAND.val => MLIR.API.ENZYMEXLA_MPI_LAND,
    MPI.LOR.val => MLIR.API.ENZYMEXLA_MPI_LOR,
    MPI.LXOR.val => MLIR.API.ENZYMEXLA_MPI_LXOR,
    MPI.MAX.val => MLIR.API.ENZYMEXLA_MPI_MAX,
    MPI.MIN.val => MLIR.API.ENZYMEXLA_MPI_MIN,
    MPI.PROD.val => MLIR.API.ENZYMEXLA_MPI_PROD,
    MPI.REPLACE.val => MLIR.API.ENZYMEXLA_MPI_REPLACE,
    MPI.SUM.val => MLIR.API.ENZYMEXLA_MPI_SUM,
    MPI.NO_OP.val => MLIR.API.ENZYMEXLA_MPI_NO_OP,
)

function get_mpi_op_enum(op)
    return get(MPI_OP_MAP, op.val) do
        throw(ArgumentError("Unknown MPI op `$op`"))
    end
end

const MPI_DATATYPE_MAP = Dict(
    MPI.DATATYPE_NULL.val => MLIR.API.ENZYMEXLA_MPI_DATATYPE_NULL,
    MPI.INT8_T.val => MLIR.API.ENZYMEXLA_MPI_INT8_T,
    MPI.UINT8_T.val => MLIR.API.ENZYMEXLA_MPI_UINT8_T,
    MPI.INT16_T.val => MLIR.API.ENZYMEXLA_MPI_INT16_T,
    MPI.UINT16_T.val => MLIR.API.ENZYMEXLA_MPI_UINT16_T,
    MPI.INT32_T.val => MLIR.API.ENZYMEXLA_MPI_INT32_T,
    MPI.UINT32_T.val => MLIR.API.ENZYMEXLA_MPI_UINT32_T,
    MPI.INT64_T.val => MLIR.API.ENZYMEXLA_MPI_INT64_T,
    MPI.UINT64_T.val => MLIR.API.ENZYMEXLA_MPI_UINT64_T,
    MPI.BYTE.val => MLIR.API.ENZYMEXLA_MPI_BYTE,
    MPI.SHORT.val => MLIR.API.ENZYMEXLA_MPI_SHORT,
    MPI.UNSIGNED_SHORT.val => MLIR.API.ENZYMEXLA_MPI_UNSIGNED_SHORT,
    MPI.INT.val => MLIR.API.ENZYMEXLA_MPI_INT,
    MPI.UNSIGNED.val => MLIR.API.ENZYMEXLA_MPI_UNSIGNED,
    MPI.LONG.val => MLIR.API.ENZYMEXLA_MPI_LONG,
    MPI.UNSIGNED_LONG.val => MLIR.API.ENZYMEXLA_MPI_UNSIGNED_LONG,
    MPI.LONG_LONG_INT.val => MLIR.API.ENZYMEXLA_MPI_LONG_LONG_INT,
    MPI.UNSIGNED_LONG_LONG.val => MLIR.API.ENZYMEXLA_MPI_UNSIGNED_LONG_LONG,
    MPI.CHAR.val => MLIR.API.ENZYMEXLA_MPI_CHAR,
    MPI.SIGNED_CHAR.val => MLIR.API.ENZYMEXLA_MPI_SIGNED_CHAR,
    MPI.UNSIGNED_CHAR.val => MLIR.API.ENZYMEXLA_MPI_UNSIGNED_CHAR,
    MPI.WCHAR.val => MLIR.API.ENZYMEXLA_MPI_WCHAR,
    MPI.FLOAT.val => MLIR.API.ENZYMEXLA_MPI_FLOAT,
    MPI.DOUBLE.val => MLIR.API.ENZYMEXLA_MPI_DOUBLE,
    MPI.C_FLOAT_COMPLEX.val => MLIR.API.ENZYMEXLA_MPI_C_FLOAT_COMPLEX,
    MPI.C_DOUBLE_COMPLEX.val => MLIR.API.ENZYMEXLA_MPI_C_DOUBLE_COMPLEX,
    MPI.C_BOOL.val => MLIR.API.ENZYMEXLA_MPI_C_BOOL,
)

function get_mpi_datatype_enum(datatype)
    return get(MPI_DATATYPE_MAP, datatype.val) do
        throw(ArgumentError("Unknown MPI datatype `$datatype`"))
    end
end

end # module
