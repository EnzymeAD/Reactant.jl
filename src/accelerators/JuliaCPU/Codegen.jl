# convert an MLIR attribute into a Julia value that doesn't depend on its MLIR context.
# dense elements become arrays with reversed dimensions (like operands), dense arrays keep MLIR order (e.g. dimension
# indices are 0-based MLIR dimensions), enums become a `Symbol` with their value (e.g.
# `#stablehlo<comparison_direction LT>` becomes `:LT`), and attributes of any other kind become their textual form.
function decode_attribute(attr)
    # reads a list of dimensions through the size and element getters of the C API
    getdims(getsize, getelem) = Int64[getelem(attr, i) for i in 0:(getsize(attr) - 1)]

    if IR.isdenseelements(attr)
        T = IR.julia_type(eltype(IR.type(attr)))
        dims = reverse(size(IR.type(attr)))
        if IR.issplat(attr)
            splat = IR.Attribute(API.mlirDenseElementsAttrGetSplatValue(attr))
            x = if T === Bool
                Bool(splat)
            elseif T <: Complex
                # complex splats are an array attribute `[re, im]` (indexing is 0-based)
                R = real(T)
                re, im = splat[0], splat[1]
                R <: Integer ? T(Int64(re), Int64(im)) : T(Float64(re), Float64(im))
            elseif T <: Integer
                splat = T <: Unsigned ? UInt64(splat) : Int64(splat)
                T(splat)
            else
                T(Float64(splat))
            end
            return fill(x, dims)
        end

        # i1 elements are stored as one byte per element (0x00/0x01), so `Bool` can be read like any other type
        return copy(
            unsafe_wrap(Array, Ptr{T}(API.mlirDenseElementsAttrGetRawData(attr)), dims)
        )
    elseif IR.isdensearray(attr, Int64)
        return Int64[API.mlirDenseI64ArrayGetElement(attr, i) for i in 0:(length(attr) - 1)]
    elseif API.mlirAttributeIsADenseBoolArray(attr)
        return Bool[
            API.mlirDenseBoolArrayGetElement(attr, i) for
            i in 0:(API.mlirDenseArrayGetNumElements(attr) - 1)
        ]
    elseif IR.isbool(attr)
        return Bool(attr)
    elseif IR.isinteger(attr)
        return IR.isunsigned(IR.type(attr)) ? UInt64(attr) : Int64(attr)
    elseif IR.isfloat(attr)
        return Float64(attr)
    elseif IR.isstring(attr)
        return String(attr)
    elseif API.stablehloAttributeIsADotDimensionNumbers(attr)
        return (;
            lhs_batching_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize,
                API.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem,
            ),
            rhs_batching_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize,
                API.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem,
            ),
            lhs_contracting_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize,
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem,
            ),
            rhs_contracting_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize,
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem,
            ),
        )
    elseif API.stablehloAttributeIsAGatherDimensionNumbers(attr)
        return (;
            offset_dims=getdims(
                API.stablehloGatherDimensionNumbersGetOffsetDimsSize,
                API.stablehloGatherDimensionNumbersGetOffsetDimsElem,
            ),
            collapsed_slice_dims=getdims(
                API.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize,
                API.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem,
            ),
            operand_batching_dims=getdims(
                API.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize,
                API.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem,
            ),
            start_indices_batching_dims=getdims(
                API.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize,
                API.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem,
            ),
            start_index_map=getdims(
                API.stablehloGatherDimensionNumbersGetStartIndexMapSize,
                API.stablehloGatherDimensionNumbersGetStartIndexMapElem,
            ),
            index_vector_dim=API.stablehloGatherDimensionNumbersGetIndexVectorDim(attr),
        )
    elseif API.stablehloAttributeIsAScatterDimensionNumbers(attr)
        return (;
            update_window_dims=getdims(
                API.stablehloScatterDimensionNumbersGetUpdateWindowDimsSize,
                API.stablehloScatterDimensionNumbersGetUpdateWindowDimsElem,
            ),
            inserted_window_dims=getdims(
                API.stablehloScatterDimensionNumbersGetInsertedWindowDimsSize,
                API.stablehloScatterDimensionNumbersGetInsertedWindowDimsElem,
            ),
            input_batching_dims=getdims(
                API.stablehloScatterDimensionNumbersGetInputBatchingDimsSize,
                API.stablehloScatterDimensionNumbersGetInputBatchingDimsElem,
            ),
            scatter_indices_batching_dims=getdims(
                API.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize,
                API.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem,
            ),
            scatter_dims_to_operand_dims=getdims(
                API.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize,
                API.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem,
            ),
            # the C API exports the scatter getter of `index_vector_dim` under this name
            index_vector_dim=API.stablehloDimensionNumbersGetIndexVectorDim(attr),
        )
    elseif API.stablehloAttributeIsAConvDimensionNumbers(attr)
        return (;
            input_batch_dimension=API.stablehloConvDimensionNumbersGetInputBatchDimension(
                attr
            ),
            input_feature_dimension=API.stablehloConvDimensionNumbersGetInputFeatureDimension(
                attr
            ),
            input_spatial_dimensions=getdims(
                API.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize,
                API.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem,
            ),
            kernel_input_feature_dimension=API.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
                attr
            ),
            kernel_output_feature_dimension=API.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
                attr
            ),
            kernel_spatial_dimensions=getdims(
                API.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize,
                API.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem,
            ),
            output_batch_dimension=API.stablehloConvDimensionNumbersGetOutputBatchDimension(
                attr
            ),
            output_feature_dimension=API.stablehloConvDimensionNumbersGetOutputFeatureDimension(
                attr
            ),
            output_spatial_dimensions=getdims(
                API.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize,
                API.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem,
            ),
        )
    elseif IR.isarray(attr)
        return Any[
            decode_attribute(IR.Attribute(API.mlirArrayAttrGetElement(attr, i))) for
            i in 0:(API.mlirArrayAttrGetNumElements(attr) - 1)
        ]
    end

    str = string(attr)
    m = match(r"^#\w+<\w+ (\w+)>$", str)
    return isnothing(m) ? str : Symbol(m[1])
end

# each op becomes a call to its handler, `%N = handle(Val(opname), operands, attributes, results)`
# like in MLIR, an op with several results binds `%N#0`, `%N#1`, ... instead.
# ops with regions also get `regions`, a tuple with an anonymous function per region, which takes the block
# arguments and returns a tuple with the operands of the region terminator.
# `codegen!` translates the ops of `block` into `exprs` and returns the operands of its terminator. `vars` maps MLIR
# values to the names of their Julia variables, and `nvals` counts the ops with results to name them
function codegen!(exprs, block, vars, nvals)
    for op in block
        opname = IR.name(op)
        operands = [vars[IR.operand(op, i).ref.ptr] for i in 1:IR.noperands(op)]
        opname in ("func.return", "stablehlo.return") && return operands
        codegen_op!(exprs, op, opname, operands, vars, nvals)
    end
    throw(invalid_argument("block has no terminator"))
end

function codegen_op!(exprs, op, opname, operands, vars, nvals)
    opval = Val(Symbol(opname))

    # `func.call` and `stablehlo.composite` have no regions, but the function they call (named by their `callee` and
    # `decomposition` attributes) is passed to their handler as if it were one
    symattr = if opname == "func.call"
        "callee"
    elseif opname == "stablehlo.composite"
        "decomposition"
    else
        nothing
    end
    blocks = if !isnothing(symattr)
        mod = IR.parent_op(op)
        while IR.name(mod) != "builtin.module"
            mod = IR.parent_op(mod)
        end
        callee = IR.flatsymbol(IR.getattr(op, symattr))
        func = IR.lookup(IR.SymbolTable(mod), callee)
        isnothing(func) && throw(invalid_argument("function `@$callee` not found"))
        [IR.first_block(IR.region(func, 1))]
    else
        [IR.first_block(IR.region(op, r)) for r in 1:IR.nregions(op)]
    end
    nregions = length(blocks)

    signature = if nregions == 0
        Tuple{typeof(opval),Tuple,NamedTuple,Tuple}
    else
        Tuple{typeof(opval),Tuple,NamedTuple,Tuple,Tuple}
    end
    hasmethod(handle, signature) ||
        throw(unimplemented("`$opname` is not supported by `HostClient`"))

    # attributes are decoded here, since the MLIR context doesn't outlive `compile`
    attributes = (;
        (
            Symbol(String(API.mlirIdentifierStr(na.ref.name))) =>
                decode_attribute(IR.Attribute(na.ref.attribute)) for
            na in (IR.getattr(op, i) for i in 1:IR.nattrs(op))
        )...
    )
    nres = IR.nresults(op)
    results = ntuple(nres) do i
        type = IR.type(IR.result(op, i))
        API.stablehloTypeIsAToken(type) && return Token
        IR.istensor(type) || throw(
            unimplemented(
                "`$opname` returns a value that is neither a tensor nor a token, which is not supported",
            ),
        )
        T = IR.julia_type(eltype(type))
        dims = reverse(size(type))

        # `eltype` is typed as `Type{T}` instead of `DataType`, so that handlers that build their result from it
        # (e.g. `Array{result.eltype}`) are type-stable
        NamedTuple{(:eltype, :size),Tuple{Type{T},typeof(dims)}}((T, dims))
    end

    call = :(handle($opval, ($(operands...),), $attributes, $results))
    if nregions > 0
        id = nvals[]
        nvals[] += 1
        regions = map(enumerate(blocks)) do (r, rblock)
            args = map(1:IR.nargs(rblock)) do i
                vars[IR.argument(rblock, i).ref.ptr] = Symbol(
                    "%", id, "^", r - 1, "#arg", i - 1
                )
            end
            rexprs = Expr(:block)
            rets = codegen!(rexprs, rblock, vars, nvals)
            Expr(
                :->,
                Expr(:tuple, args...),
                Expr(:block, rexprs.args..., :(return ($(rets...),))),
            )
        end
        push!(call.args, Expr(:tuple, regions...))
    end

    if nres == 0
        push!(exprs.args, call)
    elseif nres == 1
        var = vars[IR.result(op, 1).ref.ptr] = Symbol("%", nvals[])
        push!(exprs.args, :($var = $call))
        nvals[] += 1
    else
        resvars = map(1:nres) do i
            vars[IR.result(op, i).ref.ptr] = Symbol("%", nvals[], "#", i - 1)
        end
        push!(exprs.args, :(($(resvars...),) = $call::NTuple{$nres,Any}))
        nvals[] += 1
    end
    return nothing
end
