module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.MLIR.IR: @mlir_str
using Reactant.MLIR.Dialects: mpi, func, llvm, enzymexla
using Reactant.Ops: mlir_stacktrace
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

    # dirty hack: since MPI constants are i32, we pass the info as the pointer and then bitcast
    # DONT LOAD FROM THEM!
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

    # dirty hack: since MPI constants are i32, we pass the info as the pointer and then bitcast
    # DONT LOAD FROM THEM!
    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!("MPI_Comm_size", "llvm.func @MPI_Comm_size(!llvm.ptr, !llvm.ptr) -> i32")

    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%size_ptr : !llvm.ptr) -> () {
            %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %errcode = llvm.call @MPI_Comm_rank(%comm, %size_ptr) : (!llvm.ptr, !llvm.ptr) -> (i32)
            func.return
        }
    """)
    #! format: on

    comm = Reactant.Ops.constant(Base.unsafe_convert(Cint, comm))
    value_out = Reactant.Ops.constant(fill(Cint(-1)))
    inputs = IR.Value[comm.mlir_data, value_out.mlir_data]

    tensor_int_type = IR.TensorType(Int[], IR.Type(Cint))
    signature = IR.Type[tensor_int_type, tensor_int_type]

    # TODO output_operand_aliases
    res = IR.result(
        enzymexla.jit_call(inputs; fn=sym_attr, result_0=signature, location), 2
    )
    return TracedRNumber{Cint}((), res)
end

function barrier(; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Barrier"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    # dirty hack: since MPI constants are i32, we pass the info as the pointer and then bitcast
    # DONT LOAD FROM THEM!
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

    tensor_int_type = IR.TensorType(Int[], IR.Type(Cint))
    signature = IR.Type[tensor_int_type]

    comm = Reactant.Ops.constant(Base.unsafe_convert(Cint, comm))
    inputs = [comm.mlir_data]
    enzymexla.jit_call(inputs; fn=sym_attr, result_0=signature, location)

    return nothing
end

# TODO emit wrapper if not found
function send(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.send", @__FILE__, @__LINE__),
)
    # return mpi.send(buf.mlir_data, tag.mlir_data, dest.mlir_data; location)

    # TODO emit constant for size and datatype, and pass as args
    inputs = IR.Value[buf.mlir_data, tag.mlir_data, dest.mlir_data]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Send")
    rettype = IR.Type[]

    return enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location)
end

# TODO need c-function for creating MLIR `mpi.request` type?
function isend(
    buf::TracedRArray,
    tag::TracedRNumber,
    dest::TracedRNumber;
    location=mlir_stacktrace("mpi.isend", @__FILE__, @__LINE__),
)
    # return TracedRequest(
    #     IR.result(mpi.isend(buf.mlir_data, tag.mlir_data, dest.mlir_data; location))
    # )

    # TODO emit constant for size and datatype, and pass as args
    inputs = IR.Value[buf.mlir_data, tag.mlir_data, dest.mlir_data]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Isend")
    rettype = IR.Type[] # TODO return MPI_Request -> use i32 or opaque?

    return TracedRequest(
        IR.result(enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location))
    )
end

function recv!(
    ref::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.recv", @__FILE__, @__LINE__),
)
    # return mpi.recv(ref.mlir_data, tag.mlir_data, src.mlir_data; location)

    # TODO emit constant for size and datatype, and pass as args
    inputs = IR.Value[ref.mlir_data, tag.mlir_data, src.mlir_data]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Recv")
    rettype = IR.Type[]

    IR.result(enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location))
    return ref
end

# TODO need c-function for creating MLIR `mpi.request` type?
function irecv!(
    ref::TracedRArray,
    tag::TracedRNumber,
    src::TracedRNumber;
    location=mlir_stacktrace("mpi.irecv", @__FILE__, @__LINE__),
)
    # return TracedRequest(
    #     MLIR.IR.result(mpi.irecv(ref.mlir_data, tag.mlir_data, src.mlir_data; location))
    # )
    inputs = IR.Value[ref.mlir_data, tag.mlir_data, src.mlir_data]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Irecv")
    rettype = IR.Type[]

    IR.result(enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location))
    return ref
end

function wait(
    req::TracedRequest; location=mlir_stacktrace("mpi.wait", @__FILE__, @__LINE__)
)
    sym_name = "enzymexla_wrapper_MPI_Wait"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    # dirty hack: since MPI constants are i32, we pass the info as the pointer and then bitcast
    # DONT LOAD FROM THEM!
    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : !llvm.ptr")
    IR.inject!("MPI_Wait", "llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32")

    #! format: off
    IR.inject!(sym_name, """
    func.func @$sym_name(%req : !llvm.ptr) -> () {
        %comm = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
        %errcode = llvm.call @MPI_Wait(%req, %comm) : (!llvm.ptr, !llvm.ptr) -> (i32)
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

end # module
