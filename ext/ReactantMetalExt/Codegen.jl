# element type and MLIR dimensions of the tensor type `type`, which must be representable in MPSGraph
function tensor_type(type)
    IR.istensor(type) ||
        throw(unimplemented("values of type `$type` are not supported by `MetalClient`"))
    T = IR.julia_type(eltype(type))
    # e.g. Metal has no `Float64`
    hasmethod(convert, Tuple{Type{MPSDataType},Type{T}}) ||
        throw(unimplemented("tensors of `$T` are not supported by `MetalClient`"))
    # MPSGraph aborts on tensors without elements
    dims = collect(Int64, size(type))
    all(>(0), dims) ||
        throw(unimplemented("dynamic and empty shapes are not supported by `MetalClient`"))
    return (; eltype=T, size=dims)
end

# each op becomes nodes of the graph through its handler, `handle(Val(opname), graph, operands, attributes, results)`.
# ops with regions also get `regions`, a tuple with the block of each region.
# `codegen!` translates the ops of `block` into nodes of `graph` and returns the tensors of the operands of its
# terminator. `vars` maps MLIR values to their tensors
function codegen!(graph, block, vars)
    for op in block
        opname = IR.name(op)
        operands = [vars[IR.operand(op, i).ref.ptr] for i in 1:IR.noperands(op)]
        opname in ("func.return", "stablehlo.return") && return operands
        codegen_op!(graph, op, opname, operands, vars)
    end
    throw(invalid_argument("block has no terminator"))
end

function codegen_op!(graph, op, opname, operands, vars)
    opval = Val(Symbol(opname))
    regions = Tuple(IR.first_block(IR.region(op, r)) for r in 1:IR.nregions(op))

    signature = if isempty(regions)
        Tuple{typeof(opval),MPSGraph,Tuple,NamedTuple,Tuple}
    else
        Tuple{typeof(opval),MPSGraph,Tuple,NamedTuple,Tuple,Tuple}
    end
    hasmethod(handle, signature) ||
        throw(unimplemented("`$opname` is not supported by `MetalClient`"))

    attributes = (;
        (
            Symbol(String(API.mlirIdentifierStr(na.ref.name))) =>
                decode_attribute(IR.Attribute(na.ref.attribute)) for
            na in (IR.getattr(op, i) for i in 1:IR.nattrs(op))
        )...
    )
    nres = IR.nresults(op)
    results = ntuple(i -> tensor_type(IR.type(IR.result(op, i))), nres)

    tensors = if isempty(regions)
        handle(opval, graph, Tuple(operands), attributes, results)
    else
        handle(opval, graph, Tuple(operands), attributes, results, regions)
    end
    if nres == 1
        vars[IR.result(op, 1).ref.ptr] = tensors
    else
        for i in 1:nres
            vars[IR.result(op, i).ref.ptr] = tensors[i]
        end
    end
    return nothing
end
