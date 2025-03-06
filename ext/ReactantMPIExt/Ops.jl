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
    IR.inject!("MPI_COMM_WORLD", "llvm.mlir.global constant @MPI_COMM_WORLD() : i32")
    IR.inject!("MPI_Comm_rank", "llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32")

    #! format: off
    IR.inject!(sym_name, """
        func.func @$sym_name(%rank_ptr : !llvm.ptr) -> () {
            %comm_ptr = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
            %comm = llvm.ptrtoint %comm_ptr : !llvm.ptr to i32
            %status = llvm.call @MPI_Comm_rank(%comm, %rank_ptr) : (i32, !llvm.ptr) -> (i32)
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

function comm_size(comm; location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Comm_size"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    #! format: off
    try_inject_to_top_block!(sym_name, """
        module {
            llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
            func.func @$sym_name(%comm_ptr : !llvm.ptr, %size_ptr : !llvm.ptr) -> () {
                %comm = llvm.load %comm_ptr : !llvm.ptr -> i32
                %status = llvm.call @MPI_Comm_size(%comm, %rank_ptr) : (i32, !llvm.ptr) -> (i32)
                func.return
            }
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

function barrier(comm; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Barrier"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    tensor_int_type = IR.TensorType(Int[], IR.Type(Cint))
    signature = IR.Type[tensor_int_type]

    #! format: off
    try_inject_to_top_block!(sym_name, """
        module {
            llvm.func @MPI_Barrier(i32) -> i32
            func.func @$sym_name(%comm_ptr : !llvm.ptr) -> () {
                %comm = llvm.load %comm_ptr : !llvm.ptr -> i32
                %status = llvm.call @MPI_Barrier(%comm) : (i32) -> (i32)
                func.return
            }
        }
    """)
    #! format: on

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
    # return mpi.wait(req.mlir_data; location)
    inputs = IR.Value[req.mlir_data]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Wait")
    rettype = IR.Type[]

    return IR.result(enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location))
end

end # module
