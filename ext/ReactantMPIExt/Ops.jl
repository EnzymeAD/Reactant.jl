module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.MLIR.IR: @mlir_str
using Reactant.MLIR.Dialects: mpi, func, llvm, enzymexla
using Reactant.Ops: mlir_stacktrace, mlir_type
using ..ReactantMPIExt: TracedRequest
using MPI: MPI

# TODO we might need to have a `TracedComm` for communicators created during the compiled function

# function init(; location=mlir_stacktrace("mpi.init", @__FILE__, @__LINE__))
#     return mpi.init(; location)
# end

# function finalize(; location=mlir_stacktrace("mpi.finalize", @__FILE__, @__LINE__))
#     return mpi.finalize(; location)
# end

function comm_rank(; location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Comm_rank"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!("MPI_Comm_rank", "llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32")

    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%rank_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %errcode = llvm.call @MPI_Comm_rank(%comm, %rank_ptr) : (!llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    rank_placeholder = Reactant.Ops.constant(fill(Cint(-1)))
    output_operand_aliases = IR.Attribute([
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 0, C_NULL, 0, 0, C_NULL
            ),
        ),
    ])

    res = IR.result(
        enzymexla.jit_call(
            IR.Value[rank_placeholder.mlir_data];
            fn=sym_attr,
            result_0=[IR.TensorType(Int[], IR.Type(Cint))],
            location,
            output_operand_aliases,
        ),
    )
    return TracedRNumber{Cint}((), res)
end

function comm_size(; location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Comm_size"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!("MPI_Comm_size", "llvm.func @MPI_Comm_size(!llvm.ptr, !llvm.ptr) -> i32")

    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%size_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %errcode = llvm.call @MPI_Comm_size(%comm, %size_ptr) : (!llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    size_placeholder = Reactant.Ops.constant(fill(Cint(-1)))
    output_operand_aliases = IR.Attribute([
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 0, C_NULL, 0, 0, C_NULL
            ),
        ),
    ])

    res = IR.result(
        enzymexla.jit_call(
            IR.Value[size_placeholder.mlir_data];
            fn=sym_attr,
            result_0=[IR.TensorType(Int[], IR.Type(Cint))],
            output_operand_aliases,
            location,
        ),
    )
    return TracedRNumber{Cint}((), res)
end

function barrier(; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Barrier"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!("MPI_Barrier", "llvm.func @MPI_Barrier(!llvm.ptr) -> i32")

    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name() -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %status = llvm.call @MPI_Barrier(%comm) : (!llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    output_operand_aliases = IR.Attribute(IR.Attribute[])
    enzymexla.jit_call(
        IR.Value[]; fn=sym_attr, result_0=IR.Type[], output_operand_aliases, location
    )

    return nothing
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

function send(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.send", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = inject_mpi_datatype!(mpi_datatype)

    sym_name = "enzymexla_wrapper_MPI_Send_$(mpi_datatype_name)"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!(
        "MPI_Send",
        "llvm.func @MPI_Send(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32",
    )

    # int MPI_Send(const void* buf, int count, MPI_Datatype datatype, 
    #              int dest, int tag, MPI_Comm comm)
    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%buf : !llvm.ptr, %count_ptr : !llvm.ptr, %dest_ptr : !llvm.ptr, %tag_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %datatype = llvm.mlir.addressof @$(mpi_datatype_name) : !llvm.ptr
            %count = llvm.load %count_ptr : !llvm.ptr -> i32
            %dest = llvm.load %dest_ptr : !llvm.ptr -> i32
            %tag = llvm.load %tag_ptr : !llvm.ptr -> i32
            llvm.call @MPI_Send(%buf, %count, %datatype, %dest, %tag, %comm) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    count = Reactant.Ops.constant(Int32(length(buf)))

    enzymexla.jit_call(
        IR.Value[buf.mlir_data, count.mlir_data, dest.mlir_data, tag.mlir_data];
        fn=sym_attr,
        result_0=IR.Type[],
        output_operand_aliases=IR.Attribute(IR.Attribute[]),
        location,
    )

    return nothing
end

# TODO need c-function for creating MLIR `mpi.request` type?
function isend(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.isend", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = inject_mpi_datatype!(mpi_datatype)

    sym_name = "enzymexla_wrapper_MPI_Isend_$(mpi_datatype_name)"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!(
        "MPI_Isend",
        "llvm.func @MPI_Isend(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32",
    )

    # int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, 
    #               int dest, int tag, MPI_Comm comm, MPI_Request* request)
    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%buf : !llvm.ptr, %count_ptr : !llvm.ptr, %dest_ptr : !llvm.ptr, %tag_ptr : !llvm.ptr, %req_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %datatype = llvm.mlir.addressof @$(mpi_datatype_name) : !llvm.ptr
            %count = llvm.load %count_ptr : !llvm.ptr -> i32
            %dest = llvm.load %dest_ptr : !llvm.ptr -> i32
            %tag = llvm.load %tag_ptr : !llvm.ptr -> i32
            %res = llvm.call @MPI_Isend(%buf, %count, %datatype, %dest, %tag, %comm, %req_ptr) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    count = Reactant.Ops.constant(Int32(length(buf)))
    request = Reactant.Ops.constant(Int64(-1))

    output_operand_aliases = IR.Attribute([
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 0, C_NULL, 4, 0, C_NULL
            ),
        ),
    ])

    ret = enzymexla.jit_call(
        IR.Value[
            buf.mlir_data, count.mlir_data, dest.mlir_data, tag.mlir_data, request.mlir_data
        ];
        fn=sym_attr,
        result_0=IR.Type[mlir_type(request)],
        output_operand_aliases=output_operand_aliases,
        location,
    )

    request.mlir_data = IR.result(ret)
    return request # we return a TracedRNumber, converted to TracedRequest in Overrides.jl
end

function recv!(
    recvbuf::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(recvbuf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = inject_mpi_datatype!(mpi_datatype)

    sym_name = "enzymexla_wrapper_MPI_Recv_$(mpi_datatype_name)"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!(
        "MPI_STATUS_IGNORE", "llvm.mlir.global constant @MPI_STATUS_IGNORE() : !llvm.ptr"
    )
    IR.inject!(
        "MPI_Recv",
        "llvm.func @MPI_Recv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32",
    )

    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%buf : !llvm.ptr, %count_ptr : !llvm.ptr, %source_ptr : !llvm.ptr, %tag_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %datatype = llvm.mlir.addressof @$mpi_datatype_name : !llvm.ptr
            %status = llvm.mlir.addressof @MPI_STATUS_IGNORE : !llvm.ptr
            %count = llvm.load %count_ptr : !llvm.ptr -> i32
            %source = llvm.load %source_ptr : !llvm.ptr -> i32
            %tag = llvm.load %tag_ptr : !llvm.ptr -> i32
            llvm.call @MPI_Recv(%buf, %count, %datatype, %source, %tag, %comm, %status) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    count = Reactant.Ops.constant(Int32(length(recvbuf)))

    output_operand_aliases = IR.Attribute([
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 0, C_NULL, 0, 0, C_NULL
            ),
        ),
    ])

    ret = enzymexla.jit_call(
        IR.Value[recvbuf.mlir_data, count.mlir_data, src.mlir_data, tag.mlir_data];
        fn=sym_attr,
        result_0=[mlir_type(recvbuf)],
        output_operand_aliases,
        location,
    )

    recvbuf.mlir_data = IR.result(ret)

    return recvbuf
end

# TODO need c-function for creating MLIR `mpi.request` type?
function irecv!(
    buf::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.irecv", @__FILE__, @__LINE__),
)
    T = Reactant.unwrapped_eltype(buf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = inject_mpi_datatype!(mpi_datatype)

    sym_name = "enzymexla_wrapper_MPI_Irecv_$(mpi_datatype_name)"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!(
        "MPI_Irecv",
        "llvm.func @MPI_Irecv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32",
    )

    # int MPI_Irecv(void* buf, int count, MPI_Datatype datatype,
    #               int source, int tag, MPI_Comm comm, MPI_Request* request)
    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%buf : !llvm.ptr, %count_ptr : !llvm.ptr, %src_ptr : !llvm.ptr, %tag_ptr : !llvm.ptr, %req_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %datatype = llvm.mlir.addressof @$(mpi_datatype_name) : !llvm.ptr
            %count = llvm.load %count_ptr : !llvm.ptr -> i32
            %src = llvm.load %src_ptr : !llvm.ptr -> i32
            %tag = llvm.load %tag_ptr : !llvm.ptr -> i32
            %res = llvm.call @MPI_Irecv(%buf, %count, %datatype, %src, %tag, %comm, %req_ptr) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    count = Reactant.Ops.constant(Int32(length(buf)))
    request = Reactant.Ops.constant(Int64(-1))

    output_operand_aliases = IR.Attribute([
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 1, Ref{Int64}(0), 0, 0, C_NULL
            ),
        ),
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 1, Ref{Int64}(1), 4, 0, C_NULL
            ),
        ),
    ])

    ret = enzymexla.jit_call(
        IR.Value[
            buf.mlir_data, count.mlir_data, src.mlir_data, tag.mlir_data, request.mlir_data
        ];
        fn=sym_attr,
        result_0=[mlir_type(buf), mlir_type(request)],
        output_operand_aliases=output_operand_aliases,
        location,
    )

    buf.mlir_data = IR.result(ret, 1)
    request.mlir_data = IR.result(ret, 2)
    return request # we return a TracedRNumber, converted to TracedRequest in Overrides.jl
end

function wait(
    req::TracedRequest; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    sym_name = "enzymexla_wrapper_MPI_Wait"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    # likely isend/irecv will have injected MPI_COMM_WORLD already
    IR.tryinject!(
        "MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr"
    )

    IR.inject!("MPI_Wait", "llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32")

    # NOTE: Size of status is implem dependent, we try to set it to the max
    # int MPI_Wait(MPI_Request* request, MPI_Status* status)
    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%req : !llvm.ptr) -> () {
            %c1_i32 = arith.constant 1 : i32
            %status = llvm.alloca %c1_i32 x !llvm.array<6 x i32>  : (i32) -> !llvm.ptr
            llvm.call @MPI_Wait(%req, %status) : (!llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    enzymexla.jit_call(
        IR.Value[req.mlir_data];
        fn=sym_attr,
        result_0=IR.Type[],
        location,
        output_operand_aliases=IR.Attribute(IR.Attribute[]),
    )

    return nothing
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

function allreduce!(
    op, sendbuf, recvbuf; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    @assert Reactant.unwrapped_eltype(sendbuf) == Reactant.unwrapped_eltype(recvbuf)
    @assert length(sendbuf) == length(recvbuf)

    op_name = inject_mpi_op!(op)
    T = Reactant.unwrapped_eltype(sendbuf)
    mpi_datatype = MPI.Datatype(T)
    mpi_datatype_name = inject_mpi_datatype!(mpi_datatype)

    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!(
        "MPI_Allreduce",
        "llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32",
    )

    sym_name = "enzymexla_wrapper_MPI_Allreduce_$(op_name)_$(mpi_datatype_name)"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    # TODO is okay to use `i32`? how can we use word-size value or map C's `int` to MLIR? can we use `index`?
    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%sendbuf : !llvm.ptr, %recvbuf : !llvm.ptr, %count_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %op = llvm.mlir.addressof @$op_name : !llvm.ptr
            %datatype = llvm.mlir.addressof @$mpi_datatype_name : !llvm.ptr
            %count = llvm.load %count_ptr : !llvm.ptr -> i32
            %errcode = llvm.call @MPI_Allreduce(%sendbuf, %recvbuf, %count, %datatype, %op, %comm) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    count = Reactant.Ops.constant(fill(length(sendbuf)))

    output_operand_aliases = IR.Attribute([
        IR.Attribute(
            MLIR.API.stablehloOutputOperandAliasGet(
                MLIR.IR.context(), 0, C_NULL, 1, 0, C_NULL
            ),
        ),
    ])

    res = IR.result(
        enzymexla.jit_call(
            IR.Value[sendbuf.mlir_data, recvbuf.mlir_data, count.mlir_data];
            fn=sym_attr,
            result_0=IR.Type[Reactant.Ops.mlir_type(typeof(recvbuf), size(recvbuf))],
            location,
            output_operand_aliases,
        ),
    )

    recvbuf.mlir_data = res

    return recvbuf
end

end # module
