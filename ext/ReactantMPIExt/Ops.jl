module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.MLIR.IR: @mlir_str
using Reactant.MLIR.Dialects: mpi, func, llvm, enzymexla
using Reactant.Ops: mlir_stacktrace
using ..ReactantMPIExt: TracedRequest
using MPI: MPI

# TODO add communicators

# function init(; location=mlir_stacktrace("mpi.init", @__FILE__, @__LINE__))
#     return mpi.init(; location)
# end

# function finalize(; location=mlir_stacktrace("mpi.finalize", @__FILE__, @__LINE__))
#     return mpi.finalize(; location)
# end

function comm_rank(comm; location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Comm_rank"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    current_module = IR.mmodule()
    fn = IR.lookup(IR.SymbolTable(IR.Operation(current_module)), sym_name)

    if isnothing(fn)
        top_level_block = MLIR.IR.body(current_module)
        #! format: off
        code = parse(IR.Module, """
            module {
                llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
                func.func @$sym_name(%comm_ptr : !llvm.ptr, %rank_ptr : !llvm.ptr) -> () {
                    %comm = llvm.load %comm_ptr : !llvm.ptr -> i32
                    %status = llvm.call @MPI_Comm_rank(%comm, %rank_ptr) : (i32, !llvm.ptr) -> (i32)
                    func.return
                }
            }
        """) |> IR.body
        #! format: on

        # using `collect` because if we remove the op, then the `OperationIterator` state is broken and skips ops
        for op in collect(IR.OperationIterator(code))
            IR.rmfromparent!(op)
            push!(top_level_block, op)
        end
    end

    # NOTE we assume here that `MPI_Comm` is of word-size
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

function comm_size(comm; location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Comm_size"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    current_module = IR.mmodule()
    fn = IR.lookup(IR.SymbolTable(IR.Operation(current_module)), sym_name)

    if isnothing(fn)
        top_level_block = MLIR.IR.body(current_module)
        #! format: off
        code = parse(IR.Module, """
            module {
                llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
                func.func @$sym_name(%comm_ptr : !llvm.ptr, %size_ptr : !llvm.ptr) -> () {
                    %comm = llvm.load %comm_ptr : !llvm.ptr -> i32
                    %status = llvm.call @MPI_Comm_size(%comm, %rank_ptr) : (i32, !llvm.ptr) -> (i32)
                    func.return
                }
            }
        """) |> IR.body
        #! format: on

        # using `collect` because if we remove the op, then the `OperationIterator` state is broken and skips ops
        for op in collect(IR.OperationIterator(code))
            IR.rmfromparent!(op)
            push!(top_level_block, op)
        end
    end

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

# TODO emit wrapper if not found
# TODO should we emit `stablehlo.optimization_barrier` here too?
function barrier(; location=mlir_stacktrace("mpi.barrier", @__FILE__, @__LINE__))
    inputs = IR.Value[]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Barrier")
    rettype = IR.Type[]

    # TODO should we return `TracedRNumber{Nothing}`?
    return IR.result(enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location))
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
