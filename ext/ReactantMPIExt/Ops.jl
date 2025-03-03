module Ops
using Reactant: Reactant, TracedRArray, TracedRNumber
using Reactant: MLIR
using Reactant.MLIR: IR
using Reactant.MLIR.IR: @mlir_str
using Reactant.MLIR.Dialects: mpi, func, llvm, enzymexla
using Reactant.Ops: mlir_stacktrace
using ..ReactantMPIExt: TracedRequest

# TODO add communicators

# function init(; location=mlir_stacktrace("mpi.init", @__FILE__, @__LINE__))
#     return mpi.init(; location)
# end

# function finalize(; location=mlir_stacktrace("mpi.finalize", @__FILE__, @__LINE__))
#     return mpi.finalize(; location)
# end

# TODO emit wrapper if not found
function comm_rank(; location=mlir_stacktrace("mpi.comm_rank", @__FILE__, @__LINE__))
    sym_name = "enzymexla_wrapper_MPI_Comm_rank"
    sym_attr = IR.FlatSymbolRefAttribute(sym_name)

    # rettype = [IR.TensorType(Int[], IR.Type(Cint))]

    current_module = IR.mmodule()
    fn = IR.lookup(IR.SymbolTable(IR.Operation(current_module)), sym_name)

    if isnothing(fn)
        # arg_type = IR.Type[IR.TensorType(Int[], IR.Type(Cint))]
        arg_type = IR.Type[MLIR.IR.Type(
            MLIR.API.mlirLLVMPointerTypeGet(IR.context(), Cuint(0))
        )]
        function_type = IR.FunctionType(arg_type, IR.Type[])

        @show arg_type function_type

        wrapper = IR.block!(IR.body(current_module)) do
            func.func_(; sym_name, function_type, body=IR.Region())
        end
        wrapper_body = IR.Block(arg_type, [IR.Location()])
        push!(IR.region(wrapper, 1), wrapper_body)

        # @show wrapper

        # fill the wrapper body
        IR.block!(wrapper_body) do
            # llvm.call(
            #     IR.Value[],
            #     IR.Value[];
            #     callee=IR.FlatSymbolRefAttribute("MPI_Comm_rank"),
            #     op_bundle_sizes=MLIR.IR.Attribute(Cint[]),
            # )
            # [IR.Type(Cint)],
            # [IR.Type(Cint)],
            # [IR.Type(Cint)],
            # value = Reactant.Ops.constant(fill(Int32(1)))
            # c = IR.result(llvm.mlir_constant(; res=IR.Type(Cint), value=1))
            # llvm.store(c, ...)
            func.return_(IR.Value[])
        end
    end

    # world = Reactant.Ops.constant(fill(0))
    value_out = Reactant.Ops.constant(fill(0))
    # inputs = IR.Value[world.mlir_data]
    inputs = IR.Value[value_out.mlir_data]

    res = IR.result(enzymexla.jit_call(inputs; fn=sym_attr, result_0=IR.Type[], location))
    return TracedRNumber{Cint}((), res)
end

# TODO emit wrapper if not found
function comm_size(; location=mlir_stacktrace("mpi.comm_size", @__FILE__, @__LINE__))
    inputs = IR.Value[]
    sym = IR.FlatSymbolRefAttribute("enzymexla_wrapper_MPI_Comm_size")
    rettype = [IR.TensorType(Int[], IR.Type(Cint))]

    res = IR.result(enzymexla.jit_call(inputs; fn=sym, result_0=rettype, location))
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
