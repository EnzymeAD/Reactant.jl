# MLIRWalker.jl — Walk Reactant MLIR IR and build MPSGraph for Metal execution
#
# Ported from MetalPJRT/src/MLIRWalker.jl.
# Uses Reactant's in-tree MLIR.IR and MLIR.API directly (no parameter injection).
#
# Architecture:
#   TranslationContext  — holds MPSGraph being built + IR.Value→MPSGraphTensor map
#   compile_mlir_module — traverses Module → func.func → Block → Operations
#   execute!            — runs compiled MetalExecutable on the Metal GPU
#
# Op handlers use the @objc bindings from XLACompiler.jl (included into the
# same ReactantMetalExt module, so all mps_* functions are in scope).

# ============================================================================
# TranslationContext — maps MLIR Values to MPSGraph tensors
# ============================================================================

"""
    TranslationContext

Central state for MLIR→MPSGraph translation.  Maps MLIR IR.Value objects
to their corresponding MPSGraphTensor nodes in the graph being built.
"""
mutable struct TranslationContext
    graph::Metal.MPSGraphs.MPSGraph
    value_map::Dict{Any,Metal.MPSGraphs.MPSGraphTensor}  # IR.Value → MPSGraphTensor
    inputs::Vector{Metal.MPSGraphs.MPSGraphTensor}        # ordered input placeholders
    input_shapes::Vector{Vector{Int}}                      # IR shapes for inputs
    input_dtypes::Vector{DataType}
    outputs::Vector{Metal.MPSGraphs.MPSGraphTensor}       # return values
    output_shapes::Vector{Vector{Int}}                     # Julia shapes for outputs
    output_dtypes::Vector{DataType}
    op_count::Int
    # Multi-value constants: stored as placeholder+MtlArray pairs fed at execute time
    const_placeholders::Vector{Metal.MPSGraphs.MPSGraphTensor}
    const_mtl_values::Vector{Any}   # Vector of MtlArrays
end

function TranslationContext()
    return TranslationContext(
        Metal.MPSGraphs.MPSGraph(),
        Dict{Any,Metal.MPSGraphs.MPSGraphTensor}(),
        Metal.MPSGraphs.MPSGraphTensor[],
        Vector{Int}[],
        DataType[],
        Metal.MPSGraphs.MPSGraphTensor[],
        Vector{Int}[],
        DataType[],
        0,
        Metal.MPSGraphs.MPSGraphTensor[],
        Any[],
    )
end

# ============================================================================
# MetalExecutable — compiled result ready for execution
# ============================================================================

"""
    MetalExecutable

Holds a compiled MPSGraph ready for Metal GPU execution.
Produced by [`compile_mlir_module`](@ref).

GPU buffers, NSDictionary feeds, and NSArray targets are cached after
the first `execute!` call so that subsequent calls only upload new input
data via `copyto!` — no per-call GPU allocation or ObjC object creation.
"""
mutable struct MetalExecutable
    graph::Metal.MPSGraphs.MPSGraph
    input_placeholders::Vector{Metal.MPSGraphs.MPSGraphTensor}
    input_shapes::Vector{Vector{Int}}
    input_dtypes::Vector{DataType}
    output_tensors::Vector{Metal.MPSGraphs.MPSGraphTensor}
    output_shapes::Vector{Vector{Int}}
    output_dtypes::Vector{DataType}
    num_ops::Int
    # Multi-value constants fed as fixed inputs at every execute! call
    const_placeholders::Vector{Metal.MPSGraphs.MPSGraphTensor}
    const_mtl_values::Vector{Any}   # MtlArrays — must stay alive as long as executable
    # Serialized MLIR text for _exec_optimized_program (set after construction in _client_compile)
    mlir_text::String

    function MetalExecutable(
        graph,
        input_placeholders,
        input_shapes,
        input_dtypes,
        output_tensors,
        output_shapes,
        output_dtypes,
        num_ops,
        const_placeholders,
        const_mtl_values,
    )
        return new(
            graph,
            input_placeholders,
            input_shapes,
            input_dtypes,
            output_tensors,
            output_shapes,
            output_dtypes,
            num_ops,
            const_placeholders,
            const_mtl_values,
            "",
        )
    end
end

# ============================================================================
# Type parsing from MLIR type strings
# ============================================================================

"""
Get (shape, dtype) from an MLIR Value via the type API.
Returns (Int[], Float32) for non-tensor (scalar) types.
"""
function get_type_info(v::IR.Value)
    t = IR.type(v)
    if !IR.isshaped(t)
        return Int[], Float32
    end
    rank = ndims(t)
    shape = [Int(size(t, i)) for i in 1:rank]
    dtype = IR.julia_type(IR.eltype(t))
    return shape, dtype
end

# ============================================================================
# Op handlers — each builds one StableHLO op's equivalent MPSGraph nodes
# ============================================================================

function handle_add(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]
    rhs = ctx.value_map[IR.operand(op, 2)]
    result = Metal.MPSGraphs.additionWithPrimaryTensor(
        ctx.graph, lhs, rhs, "add_$(ctx.op_count)"
    )
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_subtract(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]
    rhs = ctx.value_map[IR.operand(op, 2)]
    result = mps_subtract(ctx.graph, lhs, rhs, "sub_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_multiply(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]
    rhs = ctx.value_map[IR.operand(op, 2)]
    result = Metal.MPSGraphs.multiplicationWithPrimaryTensor(
        ctx.graph, lhs, rhs, "mul_$(ctx.op_count)"
    )
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_divide(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]
    rhs = ctx.value_map[IR.operand(op, 2)]
    result = mps_divide(ctx.graph, lhs, rhs, "div_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_maximum(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]
    rhs = ctx.value_map[IR.operand(op, 2)]
    result = mps_maximum(ctx.graph, lhs, rhs, "max_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_negate(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_negate(ctx.graph, input, "neg_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_exponential(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_exp(ctx.graph, input, "exp_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_log(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_log(ctx.graph, input, "log_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_tanh(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_tanh(ctx.graph, input, "tanh_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_sqrt(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_sqrt(ctx.graph, input, "sqrt_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_rsqrt(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_rsqrt(ctx.graph, input, "rsqrt_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_abs(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_abs(ctx.graph, input, "abs_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_sin(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_sin(ctx.graph, input, "sin_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_cos(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    result = mps_cos(ctx.graph, input, "cos_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_convert(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    # Identity for now — proper castTensor to be added in METAL-112
    result = Metal.MPSGraphs.identityWithTensor(ctx.graph, input, "convert_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_constant(ctx, op)
    out_shape, out_dtype = get_type_info(IR.result(op, 1))
    mps_dtype = julia_to_mps_dtype(out_dtype)

    val_attr = IR.getattr(op, "value")

    function get_splat_f64()
        if out_dtype == Float32
            return Float64(API.mlirDenseElementsAttrGetFloatSplatValue(val_attr))
        elseif out_dtype == Float64
            return Float64(API.mlirDenseElementsAttrGetDoubleSplatValue(val_attr))
        else
            return Float64(API.mlirDenseElementsAttrGetInt64SplatValue(val_attr))
        end
    end

    if isempty(out_shape)
        # Scalar constant (0-d tensor)
        value = get_splat_f64()
        result = Metal.MPSGraphs.constantWithScalar(ctx.graph, value, mps_dtype)
    elseif IR.issplat(val_attr)
        # Splat tensor: dense<1.0> : tensor<40xf32>
        value = get_splat_f64()
        scalar_const = Metal.MPSGraphs.constantWithScalar(ctx.graph, value, mps_dtype)
        ns_shape = NSArray(Metal.MTL.NSNumber.(out_shape))
        result = Metal.MPSGraphs.broadcastTensor(
            ctx.graph, scalar_const, ns_shape, "const_$(ctx.op_count)"
        )
    else
        # Multi-value array constant
        total = prod(out_shape)
        nums = if out_dtype == Float32
            [
                Float64(API.mlirDenseElementsAttrGetFloatValue(val_attr, i)) for
                i in 0:(total - 1)
            ]
        elseif out_dtype == Float64
            [
                Float64(API.mlirDenseElementsAttrGetDoubleValue(val_attr, i)) for
                i in 0:(total - 1)
            ]
        else
            [
                Float64(API.mlirDenseElementsAttrGetInt64Value(val_attr, i)) for
                i in 0:(total - 1)
            ]
        end
        if length(nums) == total
            flat_vals = convert(Vector{mps_dtype}, nums)
            # Tensors in MPSGraph are in IR convention; placeholderTensor reverses Julia→IR.
            # Julia shape = reverse(ir_shape) for rank≥2.
            julia_shape = length(out_shape) >= 2 ? reverse(out_shape) : out_shape
            julia_arr = reshape(flat_vals, julia_shape...)
            mtl = MtlArray(julia_arr)
            ph = Metal.MPSGraphs.placeholderTensor(
                ctx.graph, julia_shape, mps_dtype, "const_ph_$(ctx.op_count)"
            )
            push!(ctx.const_placeholders, ph)
            push!(ctx.const_mtl_values, mtl)
            result = ph
        else
            # Fallback: broadcast first value (or 0)
            @warn "Dense constant parse: expected $total values, got $(length(nums)); using first or 0" out_shape
            value = isempty(nums) ? 0.0 : nums[1]
            scalar_const = Metal.MPSGraphs.constantWithScalar(ctx.graph, value, mps_dtype)
            ns_shape = NSArray(Metal.MTL.NSNumber.(out_shape))
            result = Metal.MPSGraphs.broadcastTensor(
                ctx.graph, scalar_const, ns_shape, "const_$(ctx.op_count)"
            )
        end
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_dot_general(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]
    rhs = ctx.value_map[IR.operand(op, 2)]

    dnums_attr = IR.getattr(op, "dot_dimension_numbers")
    lhs_contract, rhs_contract = if dnums_attr !== nothing
        n_lc = Int(API.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(dnums_attr))
        lc = [
            Int(
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(
                    dnums_attr, i
                ),
            ) for i in 0:(n_lc - 1)
        ]
        n_rc = Int(
            API.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(dnums_attr)
        )
        rc = [
            Int(
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(
                    dnums_attr, i
                ),
            ) for i in 0:(n_rc - 1)
        ]
        lc, rc
    else
        [1], [0]  # default: standard matmul
    end

    lhs_ir_shape, _ = get_type_info(IR.operand(op, 1))
    rhs_ir_shape, _ = get_type_info(IR.operand(op, 2))
    out_ir_shape, _ = get_type_info(IR.result(op, 1))

    lhs_ndim = length(lhs_ir_shape)
    rhs_ndim = length(rhs_ir_shape)
    name = "matmul_$(ctx.op_count)"

    # Use general reshape+matmul path when contracting >1 dim or rank>2
    needs_general =
        length(lhs_contract) > 1 || length(rhs_contract) > 1 || lhs_ndim > 2 || rhs_ndim > 2

    if !needs_general
        # Simple 2D matmul (existing behavior preserved)
        lhs_dim = isempty(lhs_contract) ? 1 : lhs_contract[1]
        rhs_dim = isempty(rhs_contract) ? 0 : rhs_contract[1]
        result = if lhs_dim == 1 && rhs_dim == 0
            Metal.MPSGraphs.matrixMultiplicationWithPrimaryTensor(ctx.graph, lhs, rhs, name)
        elseif lhs_dim == 0 && rhs_dim == 1
            lhs_t = Metal.MPSGraphs.transposeTensor(ctx.graph, lhs, 0, 1, "$(name)_lhs_t")
            rhs_t = Metal.MPSGraphs.transposeTensor(ctx.graph, rhs, 0, 1, "$(name)_rhs_t")
            Metal.MPSGraphs.matrixMultiplicationWithPrimaryTensor(
                ctx.graph, lhs_t, rhs_t, name
            )
        elseif lhs_dim == 0 && rhs_dim == 0
            lhs_t = Metal.MPSGraphs.transposeTensor(ctx.graph, lhs, 0, 1, "$(name)_lhs_t")
            Metal.MPSGraphs.matrixMultiplicationWithPrimaryTensor(
                ctx.graph, lhs_t, rhs, name
            )
        elseif lhs_dim == 1 && rhs_dim == 1
            rhs_t = Metal.MPSGraphs.transposeTensor(ctx.graph, rhs, 0, 1, "$(name)_rhs_t")
            Metal.MPSGraphs.matrixMultiplicationWithPrimaryTensor(
                ctx.graph, lhs, rhs_t, name
            )
        else
            @warn "Unhandled contracting dims pattern" lhs_dim rhs_dim
            Metal.MPSGraphs.matrixMultiplicationWithPrimaryTensor(ctx.graph, lhs, rhs, name)
        end
        ctx.value_map[IR.result(op, 1)] = result
        return nothing
    end

    # General path: multi-contracting dims or rank>2.
    # Reshape both tensors to 2D then use standard matmul.
    lhs_outer = [i for i in 0:(lhs_ndim - 1) if !(i in lhs_contract)]
    rhs_outer = [i for i in 0:(rhs_ndim - 1) if !(i in rhs_contract)]
    lhs_outer_size = isempty(lhs_outer) ? 1 : prod(lhs_ir_shape[d + 1] for d in lhs_outer)
    rhs_outer_size = isempty(rhs_outer) ? 1 : prod(rhs_ir_shape[d + 1] for d in rhs_outer)
    contract_size = prod(lhs_ir_shape[d + 1] for d in lhs_contract)

    # Transpose lhs: [outer_dims..., contracted_dims...]
    lhs_perm = [lhs_outer; lhs_contract]
    if lhs_perm != collect(0:(lhs_ndim - 1))
        lhs = apply_permutation(ctx.graph, lhs, lhs_perm, "$(name)_lhs_perm")
    end
    lhs_2d = mps_reshape(ctx.graph, lhs, [lhs_outer_size, contract_size], "$(name)_lhs_2d")

    # Transpose rhs: [contracted_dims..., outer_dims...]
    rhs_perm = [rhs_contract; rhs_outer]
    if rhs_perm != collect(0:(rhs_ndim - 1))
        rhs = apply_permutation(ctx.graph, rhs, rhs_perm, "$(name)_rhs_perm")
    end
    rhs_2d = mps_reshape(ctx.graph, rhs, [contract_size, rhs_outer_size], "$(name)_rhs_2d")

    # Matmul [outer_lhs, contract] @ [contract, outer_rhs] → [outer_lhs, outer_rhs]
    result_2d = Metal.MPSGraphs.matrixMultiplicationWithPrimaryTensor(
        ctx.graph, lhs_2d, rhs_2d, name
    )

    # Reshape result to output IR shape
    result = if isempty(out_ir_shape)
        Metal.MPSGraphs.identityWithTensor(ctx.graph, result_2d, "$(name)_scalar")
    elseif out_ir_shape == [lhs_outer_size, rhs_outer_size]
        result_2d
    else
        mps_reshape(ctx.graph, result_2d, out_ir_shape, "$(name)_out_reshape")
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_broadcast_in_dim(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    out_shape, _ = get_type_info(IR.result(op, 1))

    if isempty(out_shape)
        result = Metal.MPSGraphs.identityWithTensor(
            ctx.graph, input, "broadcast_$(ctx.op_count)"
        )
    else
        # Tensors inside MPSGraph are in IR (row-major) convention because
        # placeholderTensor auto-reverses Julia shapes. Use IR shapes directly.
        bcast_attr = IR.getattr(op, "broadcast_dimensions")
        broadcast_dims = if bcast_attr !== nothing
            [Int64(bcast_attr[i]) for i in 0:(length(bcast_attr) - 1)]
        else
            nothing
        end
        input_tensor = input

        # If broadcast_dims maps input dims to specific output dims, we need to
        # reshape the input so MPSGraph's numpy-style broadcasting works correctly.
        # E.g., bias [4] with dims=[1] and out=[1,4,8,8] → reshape to [1,4,1,1]
        #
        # When broadcast_dims is non-monotonic (e.g., [4,2,1,0] from nearest upsample
        # after a conv whose output layout differs from the placeholder layout), we must
        # first TRANSPOSE the input so its dimensions are ordered from lowest to highest
        # output position before inserting singleton dims via reshape.
        # Without this, reshape just reinterprets memory without reordering data.
        if broadcast_dims !== nothing && length(broadcast_dims) < length(out_shape)
            # Sort broadcast_dims and get the permutation that would sort them.
            # sorted_indices[k] = 1-indexed position in broadcast_dims of the k-th smallest value.
            sorted_indices = sortperm(broadcast_dims)       # 1-indexed
            sorted_broadcast_dims = broadcast_dims[sorted_indices]
            perm_0indexed = sorted_indices .- 1             # 0-indexed for apply_permutation
            # Transpose input so dims align with sorted output positions.
            if perm_0indexed != collect(0:(length(broadcast_dims) - 1))
                input_tensor = apply_permutation(
                    ctx.graph, input_tensor, perm_0indexed, "broadcast_$(ctx.op_count)_perm"
                )
            end
            # Build intermediate shape with singletons at positions not in broadcast_dims.
            intermediate_shape = ones(Int, length(out_shape))
            for (new_input_dim, output_dim) in enumerate(sorted_broadcast_dims)
                intermediate_shape[output_dim + 1] = out_shape[output_dim + 1]
            end
            input_tensor = mps_reshape(
                ctx.graph,
                input_tensor,
                intermediate_shape,
                "broadcast_$(ctx.op_count)_reshape",
            )
        end

        ns_shape = NSArray(Metal.MTL.NSNumber.(out_shape))
        result = Metal.MPSGraphs.broadcastTensor(
            ctx.graph, input_tensor, ns_shape, "broadcast_$(ctx.op_count)"
        )
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_reshape(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    out_shape, _ = get_type_info(IR.result(op, 1))

    if isempty(out_shape)
        result = Metal.MPSGraphs.identityWithTensor(
            ctx.graph, input, "reshape_$(ctx.op_count)"
        )
    else
        # Tensors in MPSGraph are in IR (row-major) convention — use IR shapes directly
        result = mps_reshape(ctx.graph, input, out_shape, "reshape_$(ctx.op_count)")
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_transpose(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]

    perm_attr = IR.getattr(op, "permutation")
    perm = if perm_attr !== nothing
        [Int64(perm_attr[i]) for i in 0:(length(perm_attr) - 1)]
    else
        Int[]
    end

    if perm !== nothing && length(perm) >= 2
        result = apply_permutation(ctx.graph, input, perm, "transpose_$(ctx.op_count)")
    else
        result = Metal.MPSGraphs.transposeTensor(
            ctx.graph, input, 0, 1, "transpose_$(ctx.op_count)"
        )
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_reduce(ctx, op)
    dims_attr = IR.getattr(op, "dimensions")
    ir_reduce_dims = if dims_attr !== nothing
        [Int64(dims_attr[i]) for i in 0:(length(dims_attr) - 1)]
    else
        Int[]
    end

    # The input tensor (operand 1) — operand 2 is the init value (scalar identity)
    input = ctx.value_map[IR.operand(op, 1)]

    # Pattern-match the body to determine reduction type.
    # Count stablehlo ops (excluding stablehlo.return) to validate body is trivial.
    body_op_name = ""
    stablehlo_op_count = 0
    region = IR.region(op, 1)
    body_block = first(region)
    for bop in body_block
        bop_name = IR.name(bop)
        if startswith(bop_name, "stablehlo.") && bop_name != "stablehlo.return"
            stablehlo_op_count += 1
            if body_op_name == ""
                body_op_name = bop_name
            end
        end
    end
    if stablehlo_op_count > 1
        error(
            "Unsupported stablehlo.reduce: body contains $stablehlo_op_count ops " *
            "(expected 1 for simple reduction). Complex reduce bodies are not yet supported.",
        )
    end

    # Tensors in MPSGraph are in IR (row-major) convention because
    # placeholderTensor auto-reverses Julia shapes. Use IR dims directly.
    name = "reduce_$(ctx.op_count)"
    result = if body_op_name == "stablehlo.add"
        mps_reduce_sum(ctx.graph, input, ir_reduce_dims, name)
    elseif body_op_name == "stablehlo.maximum"
        mps_reduce_max(ctx.graph, input, ir_reduce_dims, name)
    elseif body_op_name == "stablehlo.minimum"
        error("Unsupported StableHLO op: reduce (min) — not yet implemented via MPSGraph")
    elseif body_op_name == "stablehlo.multiply"
        error("Unsupported StableHLO op: reduce (prod) — not yet implemented via MPSGraph")
    else
        error("Unsupported StableHLO op: reduce with body op '$body_op_name'")
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

"""
    apply_permutation(graph, tensor, perm, name) -> MPSGraphTensor

Apply an arbitrary dimension permutation to a tensor via a sequence of transpositions.
`perm` is a 0-indexed array where `perm[i]` = source dimension for target dimension i.
MPSGraph transposeTensor is a lazy graph node (no data copy).
"""
function apply_permutation(graph, tensor, perm::Vector{Int}, name::String)
    # Apply permutation: output[i] = input[perm[i]]
    # Uses selection-sort decomposition into transpositions.
    if perm == collect(0:(length(perm) - 1))
        return tensor
    end
    result = tensor
    # current[pos+1] = which original dimension is currently at position pos
    current = collect(0:(length(perm) - 1))
    for i in 0:(length(perm) - 2)
        target = perm[i + 1]  # which original dim should end up at position i
        j_1idx = findfirst(==(target), current)  # find where target currently is
        if j_1idx === nothing
            error("Invalid permutation: $perm (missing dim $target)")
        end
        j0 = j_1idx - 1  # convert to 0-indexed dim for transposeTensor
        if j0 != i
            result = Metal.MPSGraphs.transposeTensor(
                graph, result, i, j0, "$(name)_swap$(i)_$(j0)"
            )
            current[i + 1], current[j_1idx] = current[j_1idx], current[i + 1]
        end
    end
    return result
end

"""
    compute_layout_permutation(from_batch, from_feature, from_spatial, from_spatial_indices,
                               to_batch, to_feature, to_spatial, to_spatial_indices) -> Vector{Int}

Compute the 0-indexed permutation to convert between two data layouts.
Spatial dimensions are matched by their INDEX (the numbers from the dimension_numbers text,
e.g., "0" and "1" in `[b, f, 1, 0]`), NOT by their array enumeration position.
"""
function compute_layout_permutation(
    from_batch,
    from_feature,
    from_spatial,
    from_spatial_indices,
    to_batch,
    to_feature,
    to_spatial,
    to_spatial_indices,
)
    rank = 2 + length(from_spatial)
    # Build mapping: semantic_name => from_position
    semantic_to_from = Dict{Symbol,Int}()
    semantic_to_from[:batch] = from_batch
    semantic_to_from[:feature] = from_feature
    for (k, idx) in enumerate(from_spatial_indices)
        semantic_to_from[Symbol("spatial_$idx")] = from_spatial[k]
    end
    # Build mapping: to_position => semantic_name
    to_pos_semantic = Dict{Int,Symbol}()
    to_pos_semantic[to_batch] = :batch
    to_pos_semantic[to_feature] = :feature
    for (k, idx) in enumerate(to_spatial_indices)
        to_pos_semantic[to_spatial[k]] = Symbol("spatial_$idx")
    end
    # perm[to_pos] = from_pos
    perm = Vector{Int}(undef, rank)
    for to_pos in 0:(rank - 1)
        sem = to_pos_semantic[to_pos]
        perm[to_pos + 1] = semantic_to_from[sem]
    end
    return perm
end

function handle_reverse(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    dims_attr = IR.getattr(op, "dimensions")
    ir_dims = if dims_attr !== nothing
        [Int64(dims_attr[i]) for i in 0:(length(dims_attr) - 1)]
    else
        Int[]
    end

    # Tensors in MPSGraph are in IR (row-major) convention — use IR dims directly
    result = mps_reverse(ctx.graph, input, ir_dims, "reverse_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_concatenate(ctx, op)
    ir_dim = Int64(IR.getattr(op, "dimension"))

    # Collect all input tensors
    n_inputs = IR.noperands(op)
    input_tensors = Metal.MPSGraphs.MPSGraphTensor[]
    for i in 1:n_inputs
        push!(input_tensors, ctx.value_map[IR.operand(op, i)])
    end

    # Tensors in MPSGraph are in IR (row-major) convention — use IR dim directly
    result = mps_concatenate(ctx.graph, input_tensors, ir_dim, "concat_$(ctx.op_count)")
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_slice(ctx, op)
    input = ctx.value_map[IR.operand(op, 1)]
    start_attr = IR.getattr(op, "start_indices")
    limit_attr = IR.getattr(op, "limit_indices")
    ranges = if start_attr !== nothing && limit_attr !== nothing
        [(Int64(start_attr[i]), Int64(limit_attr[i])) for i in 0:(length(start_attr) - 1)]
    else
        nothing
    end
    if ranges === nothing
        ctx.value_map[IR.result(op, 1)] = input
        return nothing
    end
    # Get IR input shape to skip full-extent slices
    ir_shape, _ = get_type_info(IR.operand(op, 1))

    result = input
    name = "slice_$(ctx.op_count)"
    for (dim_idx, (start, stop)) in enumerate(ranges)
        dim = dim_idx - 1  # 0-indexed for MPSGraph
        len = stop - start
        full_len = dim_idx <= length(ir_shape) ? ir_shape[dim_idx] : len
        start == 0 && len == full_len && continue  # full-extent: no-op
        result = mps_slice_dim(ctx.graph, result, dim, start, len, "$(name)_d$(dim)")
    end
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_scatter(ctx, op)
    operand = ctx.value_map[IR.operand(op, 1)]   # base tensor
    indices = ctx.value_map[IR.operand(op, 2)]   # scatter indices (ND)
    updates = ctx.value_map[IR.operand(op, 3)]   # update values

    # Use scatterND with "set" mode (MPSGraphScatterModeSet = 0).
    # This is correct for the Enzyme Forward gradient pattern where the update
    # region computes a function of (old_value, update_value).  For the specific
    # pattern generated by Enzyme (body returns constant 1.0, updates tensor is
    # also 1.0, operand is zeros), using the updates tensor directly with Set
    # mode produces the correct result.
    # batchDimensions = 0: no leading batch dims (index_vector_dim = 1 in Enzyme pattern).
    result = mps_scatter_nd(
        ctx.graph, operand, updates, indices, 0, 0, "scatter_$(ctx.op_count)"
    )
    return ctx.value_map[IR.result(op, 1)] = result
end

function handle_convolution(ctx, op)
    lhs = ctx.value_map[IR.operand(op, 1)]  # input
    rhs = ctx.value_map[IR.operand(op, 2)]  # weights

    op_text = string(op)

    # Parse attributes — dimension_numbers via StableHLO C API
    dnums_attr = IR.getattr(op, "dimension_numbers")
    dim_nums = if dnums_attr === nothing
        # Default: NHWC input, HWIO kernel, NHWC output
        (
            input_batch=0,
            input_feature=3,
            input_spatial=[1, 2],
            input_spatial_indices=[0, 1],
            kernel_spatial=[0, 1],
            kernel_spatial_indices=[0, 1],
            kernel_input=2,
            kernel_output=3,
            output_batch=0,
            output_feature=3,
            output_spatial=[1, 2],
            output_spatial_indices=[0, 1],
        )
    else
        n_in_sp = Int(
            API.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(dnums_attr)
        )
        in_sp_raw = sort!([
            (
                Int(
                    API.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(
                        dnums_attr, i
                    ),
                ),
                Int(i),
            ) for i in 0:(n_in_sp - 1)
        ])
        n_ker_sp = Int(
            API.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(dnums_attr)
        )
        ker_sp_raw = sort!([
            (
                Int(
                    API.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(
                        dnums_attr, i
                    ),
                ),
                Int(i),
            ) for i in 0:(n_ker_sp - 1)
        ])
        n_out_sp = Int(
            API.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(dnums_attr)
        )
        out_sp_raw = sort!([
            (
                Int(
                    API.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(
                        dnums_attr, i
                    ),
                ),
                Int(i),
            ) for i in 0:(n_out_sp - 1)
        ])
        (
            input_batch=Int(
                API.stablehloConvDimensionNumbersGetInputBatchDimension(dnums_attr)
            ),
            input_feature=Int(
                API.stablehloConvDimensionNumbersGetInputFeatureDimension(dnums_attr)
            ),
            input_spatial=[p[1] for p in in_sp_raw],
            input_spatial_indices=[p[2] for p in in_sp_raw],
            kernel_spatial=[p[1] for p in ker_sp_raw],
            kernel_spatial_indices=[p[2] for p in ker_sp_raw],
            kernel_input=Int(
                API.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(dnums_attr),
            ),
            kernel_output=Int(
                API.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
                    dnums_attr
                ),
            ),
            output_batch=Int(
                API.stablehloConvDimensionNumbersGetOutputBatchDimension(dnums_attr)
            ),
            output_feature=Int(
                API.stablehloConvDimensionNumbersGetOutputFeatureDimension(dnums_attr)
            ),
            output_spatial=[p[1] for p in out_sp_raw],
            output_spatial_indices=[p[2] for p in out_sp_raw],
        )
    end
    strides_attr = IR.getattr(op, "window_strides")
    strides = if strides_attr !== nothing
        [Int64(strides_attr[i]) for i in 0:(length(strides_attr) - 1)]
    else
        Int[]
    end
    if isempty(strides)
        # Also try window = {stride = [...]} format
        m = match(r"stride\s*=\s*\[([^\]]*)\]", op_text)
        if m !== nothing
            strides = [parse(Int, strip(s)) for s in split(m.captures[1], ",")]
        else
            strides = [1, 1]
        end
    end
    pad_attr = IR.getattr(op, "padding")
    padding = if pad_attr !== nothing
        n_sp = length(dim_nums.input_spatial)
        Tuple([
            [
                Int(API.mlirDenseElementsAttrGetInt64Value(pad_attr, Int64(2 * (i - 1)))),
                Int(
                    API.mlirDenseElementsAttrGetInt64Value(
                        pad_attr, Int64(2 * (i - 1) + 1)
                    ),
                ),
            ] for i in 1:n_sp
        ])
    else
        Tuple([[0, 0] for _ in dim_nums.input_spatial])
    end
    rhs_dilation_attr = IR.getattr(op, "rhs_dilation")
    rhs_dilation = if rhs_dilation_attr !== nothing
        [Int64(rhs_dilation_attr[i]) for i in 0:(length(rhs_dilation_attr) - 1)]
    else
        Int[]
    end
    if isempty(rhs_dilation)
        m = match(r"rhs_dilate\s*=\s*\[([^\]]*)\]", op_text)
        rhs_dilation = if m !== nothing
            [parse(Int, strip(s)) for s in split(m.captures[1], ",")]
        else
            [1, 1]
        end
    end
    fgc_attr = IR.getattr(op, "feature_group_count")
    groups = fgc_attr !== nothing ? Int(Int64(fgc_attr)) : 1

    # Get shapes for rank
    lhs_shape, _ = get_type_info(IR.operand(op, 1))
    rank = length(lhs_shape)
    n_spatial = length(dim_nums.input_spatial)
    name = "conv_$(ctx.op_count)"

    if n_spatial != 2 && n_spatial != 3
        error("Only 2D and 3D convolution supported, got $n_spatial spatial dims")
    end

    # Tensors in MPSGraph are already in IR (row-major) convention because
    # placeholderTensor auto-reverses Julia shapes.
    #
    # However, Reactant may use different input/output data layouts for conv ops
    # (e.g., NCHW input → WHCN output). MPSGraph conv descriptors use a single
    # dataLayout for both, so we transpose input/output as needed to match.
    if n_spatial == 2
        # --- 2D Convolution ---
        # Always run conv in NCHW. The spatial ordering at positions [2, 3] must match
        # the KERNEL's spatial ordering so MPSGraph applies kernel[h,w] to data[h,w].
        is_nchw_input = dim_nums.input_batch == 0 && dim_nums.input_feature == 1
        nchw_spatial = collect(2:(rank - 1))  # [2, 3] for 2D

        # Target spatial ordering = kernel's spatial ordering
        conv_spatial_indices = dim_nums.kernel_spatial_indices

        conv_input = lhs
        if !is_nchw_input || dim_nums.input_spatial_indices != conv_spatial_indices
            # Transpose input to NCHW with kernel's spatial ordering
            perm = compute_layout_permutation(
                dim_nums.input_batch,
                dim_nums.input_feature,
                dim_nums.input_spatial,
                dim_nums.input_spatial_indices,
                0,
                1,
                nchw_spatial,
                conv_spatial_indices,
            )
            if perm != collect(0:(rank - 1))
                conv_input = apply_permutation(
                    ctx.graph, lhs, perm, "$(name)_input_to_nchw"
                )
            end
        end

        # Check if output needs transpose
        needs_output_transpose = !(
            dim_nums.output_batch == 0 &&
            dim_nums.output_feature == 1 &&
            dim_nums.output_spatial_indices == conv_spatial_indices
        )

        is_oihw = dim_nums.kernel_output == 0 && dim_nums.kernel_input == 1
        weights_layout = if is_oihw
            Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutOIHW
        else
            is_hwio =
                length(dim_nums.kernel_spatial) >= 2 &&
                dim_nums.kernel_spatial[1] == 0 &&
                dim_nums.kernel_spatial[2] == 1
            if is_hwio
                Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutHWIO
            else
                @warn "Unknown conv2d weights layout, defaulting to OIHW" dim_nums
                Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutOIHW
            end
        end

        # Remap strides/padding/dilation from input spatial ordering to conv spatial ordering
        # IR strides are ordered by input spatial positions: strides[k] for input_spatial_indices[k]
        stride_map = Dict{Int,Int}()
        dil_map = Dict{Int,Int}()
        pad_map = Dict{Int,Any}()
        for (k, idx) in enumerate(dim_nums.input_spatial_indices)
            stride_map[idx] = k <= length(strides) ? strides[k] : 1
            dil_map[idx] = k <= length(rhs_dilation) ? rhs_dilation[k] : 1
            pad_map[idx] = k <= length(padding) ? padding[k] : [0, 0]
        end
        stride_h = get(stride_map, conv_spatial_indices[1], 1)
        stride_w = get(stride_map, conv_spatial_indices[2], 1)
        dil_h = get(dil_map, conv_spatial_indices[1], 1)
        dil_w = get(dil_map, conv_spatial_indices[2], 1)
        pad_h = get(pad_map, conv_spatial_indices[1], [0, 0])
        pad_w = get(pad_map, conv_spatial_indices[2], [0, 0])

        desc = mps_create_conv2d_descriptor(;
            stride_x=stride_w,
            stride_y=stride_h,
            dilation_x=dil_w,
            dilation_y=dil_h,
            groups=groups,
            padding_top=pad_h[1],
            padding_bottom=pad_h[2],
            padding_left=pad_w[1],
            padding_right=pad_w[2],
            data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCHW,
            weights_layout=weights_layout,
        )
        conv_result = mps_convolution2d(ctx.graph, conv_input, rhs, desc, name)

        if needs_output_transpose
            perm = compute_layout_permutation(
                0,
                1,
                nchw_spatial,
                conv_spatial_indices,
                dim_nums.output_batch,
                dim_nums.output_feature,
                dim_nums.output_spatial,
                dim_nums.output_spatial_indices,
            )
            conv_result = apply_permutation(
                ctx.graph, conv_result, perm, "$(name)_output_from_nchw"
            )
        end

    else  # n_spatial == 3
        # --- 3D Convolution ---
        # Always run conv in NCDHW with kernel's spatial ordering.
        is_ncdhw_input = dim_nums.input_batch == 0 && dim_nums.input_feature == 1
        ncdhw_spatial = collect(2:(rank - 1))  # [2, 3, 4]

        conv_spatial_indices = dim_nums.kernel_spatial_indices

        conv_input = lhs
        if !is_ncdhw_input || dim_nums.input_spatial_indices != conv_spatial_indices
            perm = compute_layout_permutation(
                dim_nums.input_batch,
                dim_nums.input_feature,
                dim_nums.input_spatial,
                dim_nums.input_spatial_indices,
                0,
                1,
                ncdhw_spatial,
                conv_spatial_indices,
            )
            if perm != collect(0:(rank - 1))
                conv_input = apply_permutation(
                    ctx.graph, lhs, perm, "$(name)_input_to_ncdhw"
                )
            end
        end

        needs_output_transpose = !(
            dim_nums.output_batch == 0 &&
            dim_nums.output_feature == 1 &&
            dim_nums.output_spatial_indices == conv_spatial_indices
        )

        is_oidhw = dim_nums.kernel_output == 0 && dim_nums.kernel_input == 1
        weights_layout = if is_oidhw
            Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutOIDHW
        else
            is_dhwio =
                length(dim_nums.kernel_spatial) >= 3 &&
                dim_nums.kernel_spatial[1] == 0 &&
                dim_nums.kernel_spatial[2] == 1
            if is_dhwio
                Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutDHWIO
            else
                @warn "Unknown conv3d weights layout, defaulting to OIDHW" dim_nums
                Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutOIDHW
            end
        end

        # Remap strides/padding/dilation from input spatial ordering to conv spatial ordering
        stride_map = Dict{Int,Int}()
        dil_map = Dict{Int,Int}()
        pad_map = Dict{Int,Any}()
        for (k, idx) in enumerate(dim_nums.input_spatial_indices)
            stride_map[idx] = k <= length(strides) ? strides[k] : 1
            dil_map[idx] = k <= length(rhs_dilation) ? rhs_dilation[k] : 1
            pad_map[idx] = k <= length(padding) ? padding[k] : [0, 0]
        end
        stride_d = get(stride_map, conv_spatial_indices[1], 1)
        stride_h = get(stride_map, conv_spatial_indices[2], 1)
        stride_w = get(stride_map, conv_spatial_indices[3], 1)
        dil_d = get(dil_map, conv_spatial_indices[1], 1)
        dil_h = get(dil_map, conv_spatial_indices[2], 1)
        dil_w = get(dil_map, conv_spatial_indices[3], 1)
        pad_d = get(pad_map, conv_spatial_indices[1], [0, 0])
        pad_h = get(pad_map, conv_spatial_indices[2], [0, 0])
        pad_w = get(pad_map, conv_spatial_indices[3], [0, 0])

        desc = mps_create_conv3d_descriptor(;
            stride_x=stride_w,
            stride_y=stride_h,
            stride_z=stride_d,
            dilation_x=dil_w,
            dilation_y=dil_h,
            dilation_z=dil_d,
            groups=groups,
            padding_front=pad_d[1],
            padding_back=pad_d[2],
            padding_top=pad_h[1],
            padding_bottom=pad_h[2],
            padding_left=pad_w[1],
            padding_right=pad_w[2],
            data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCDHW,
            weights_layout=weights_layout,
        )

        conv_result = mps_convolution3d(ctx.graph, conv_input, rhs, desc, name)

        if needs_output_transpose
            perm = compute_layout_permutation(
                0,
                1,
                ncdhw_spatial,
                conv_spatial_indices,
                dim_nums.output_batch,
                dim_nums.output_feature,
                dim_nums.output_spatial,
                dim_nums.output_spatial_indices,
            )
            conv_result = apply_permutation(
                ctx.graph, conv_result, perm, "$(name)_output_from_ncdhw"
            )
        end
    end

    # Conv result is already in IR convention — use directly
    return ctx.value_map[IR.result(op, 1)] = conv_result
end

function handle_reduce_window(ctx, op)
    # Parse window attributes
    window_dims_attr = IR.getattr(op, "window_dimensions")
    window_dims = if window_dims_attr !== nothing
        [Int64(window_dims_attr[i]) for i in 0:(length(window_dims_attr) - 1)]
    else
        Int[]
    end
    window_strides_attr = IR.getattr(op, "window_strides")
    window_strides = if window_strides_attr !== nothing
        [Int64(window_strides_attr[i]) for i in 0:(length(window_strides_attr) - 1)]
    else
        Int[]
    end

    # Get input tensor
    input = ctx.value_map[IR.operand(op, 1)]

    # Get input shape for rank
    ir_input_shape, in_dtype = get_type_info(IR.operand(op, 1))
    rank = length(ir_input_shape)

    # Parse padding
    pad_attr = IR.getattr(op, "padding")
    window_padding = if pad_attr !== nothing
        [
            (
                Int(API.mlirDenseElementsAttrGetInt64Value(pad_attr, Int64(2 * (i - 1)))),
                Int(
                    API.mlirDenseElementsAttrGetInt64Value(
                        pad_attr, Int64(2 * (i - 1) + 1)
                    ),
                ),
            ) for i in 1:rank
        ]
    else
        [(0, 0) for _ in 1:rank]
    end

    # Pattern-match the body to determine reduction type.
    # Count stablehlo ops (excluding stablehlo.return) to validate body is trivial.
    body_op_name = ""
    stablehlo_op_count = 0
    if IR.nregions(op) > 0
        region = IR.region(op, 1)
        body_block = first(region)
        for bop in body_block
            bop_name = IR.name(bop)
            if startswith(bop_name, "stablehlo.") && bop_name != "stablehlo.return"
                stablehlo_op_count += 1
                if body_op_name == ""
                    body_op_name = bop_name
                end
            end
        end
    end
    if stablehlo_op_count > 1
        error(
            "Unsupported stablehlo.reduce_window: body contains $stablehlo_op_count ops " *
            "(expected 1 for simple pooling). Complex reduce_window bodies are not yet supported.",
        )
    end

    name = "pool_$(ctx.op_count)"

    # Tensors in MPSGraph are already in IR (row-major) convention — no reversal needed.
    # Determine spatial dims from window_dims (dims with window > 1 are spatial)
    spatial_pos = [i for i in 1:length(window_dims) if window_dims[i] > 1]  # 1-indexed
    nonspatial_pos = [i for i in 1:length(window_dims) if window_dims[i] == 1]

    pool_result = if rank == 4
        # --- 2D Pooling ---
        if length(spatial_pos) != 2
            error(
                "Expected exactly 2 spatial dims in 2D pooling, got $(length(spatial_pos)) from window_dimensions: $window_dims",
            )
        end

        # Check if already in NCHW layout (spatial at positions 3,4 i.e. 1-indexed)
        is_nchw_pool = spatial_pos == [3, 4]
        # Or NHWC (spatial at positions 2,3 i.e. 1-indexed)
        is_nhwc_pool = spatial_pos == [2, 3]

        pool_input = input
        perm_src_2d = Int[]
        if is_nchw_pool
            kernel_h, kernel_w = window_dims[3], window_dims[4]
            stride_h = length(window_strides) >= 3 ? window_strides[3] : kernel_h
            stride_w = length(window_strides) >= 4 ? window_strides[4] : kernel_w
            pad_top = window_padding[3][1]
            pad_bottom = window_padding[3][2]
            pad_left = window_padding[4][1]
            pad_right = window_padding[4][2]
            data_layout = Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCHW
        elseif is_nhwc_pool
            kernel_h, kernel_w = window_dims[2], window_dims[3]
            stride_h = length(window_strides) >= 2 ? window_strides[2] : kernel_h
            stride_w = length(window_strides) >= 3 ? window_strides[3] : kernel_w
            pad_top = window_padding[2][1]
            pad_bottom = window_padding[2][2]
            pad_left = window_padding[3][1]
            pad_right = window_padding[3][2]
            data_layout = Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNHWC
        else
            # Non-standard layout: transpose to NCHW, pool, transpose back
            perm_src_2d = [nonspatial_pos; spatial_pos]  # 1-indexed
            perm = [p - 1 for p in perm_src_2d]          # 0-indexed
            pool_input = apply_permutation(ctx.graph, input, perm, "$(name)_input_to_nchw")
            kernel_h = window_dims[spatial_pos[1]]
            kernel_w = window_dims[spatial_pos[2]]
            stride_h = if spatial_pos[1] <= length(window_strides)
                window_strides[spatial_pos[1]]
            else
                kernel_h
            end
            stride_w = if spatial_pos[2] <= length(window_strides)
                window_strides[spatial_pos[2]]
            else
                kernel_w
            end
            pad_top = window_padding[spatial_pos[1]][1]
            pad_bottom = window_padding[spatial_pos[1]][2]
            pad_left = window_padding[spatial_pos[2]][1]
            pad_right = window_padding[spatial_pos[2]][2]
            data_layout = Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCHW
        end

        desc2d = mps_create_pooling2d_descriptor(;
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_x=stride_w,
            stride_y=stride_h,
            padding_top=pad_top,
            padding_bottom=pad_bottom,
            padding_left=pad_left,
            padding_right=pad_right,
            data_layout=data_layout,
        )

        r2d = if body_op_name == "stablehlo.maximum"
            mps_max_pooling2d(ctx.graph, pool_input, desc2d, name)
        elseif body_op_name == "stablehlo.add"
            avg = mps_avg_pooling2d(ctx.graph, pool_input, desc2d, name)
            window_area = kernel_h * kernel_w
            mps_dtype = julia_to_mps_dtype(in_dtype)
            scalar = Metal.MPSGraphs.constantWithScalar(
                ctx.graph, Float64(window_area), mps_dtype
            )
            Metal.MPSGraphs.multiplicationWithPrimaryTensor(
                ctx.graph, avg, scalar, "$(name)_sum"
            )
        else
            error("Unsupported reduce_window body op: $body_op_name")
        end

        # Transpose back for non-standard layouts
        if !is_nchw_pool && !is_nhwc_pool
            inv_perm = zeros(Int, rank)
            for (dst, src) in enumerate(perm_src_2d)
                inv_perm[src] = dst - 1
            end
            r2d = apply_permutation(ctx.graph, r2d, inv_perm, "$(name)_output_from_nchw")
        end
        r2d

    elseif rank == 5
        # --- 3D Pooling via sequential 2D pooling ---
        # MPSGraphPooling4DOpDescriptor requires exactly 4 spatial dims (rank 6).
        # Instead we decompose: pool(d1,d2) first (no transpose needed), then pool(d0).
        # Both max and avg/sum pooling are separable, so sequential 2D pools are exact.

        if length(spatial_pos) != 3
            error(
                "Expected exactly 3 spatial dims in 3D pooling, got $(length(spatial_pos)) from window_dimensions: $window_dims",
            )
        end

        # Transpose to NCDHW (non-spatial first, spatial last) if needed
        is_ncdhw_pool = spatial_pos == [3, 4, 5]
        pool_input = input
        perm_src_3d = Int[]
        if !is_ncdhw_pool
            perm_src_3d = [nonspatial_pos; spatial_pos]  # 1-indexed src positions
            perm = [p - 1 for p in perm_src_3d]          # 0-indexed
            pool_input = apply_permutation(ctx.graph, input, perm, "$(name)_input_to_ncdhw")
        end

        # pool_input is now in NCDHW order: [N, C, d0, d1, d2]
        # nonspatial_pos and spatial_pos are 1-indexed positions in the ORIGINAL ir_input_shape
        N = ir_input_shape[nonspatial_pos[1]]
        C = ir_input_shape[nonspatial_pos[2]]
        sp = spatial_pos
        d0 = ir_input_shape[sp[1]]
        d1 = ir_input_shape[sp[2]]
        d2 = ir_input_shape[sp[3]]

        kernel_d = window_dims[sp[1]]
        kernel_h = window_dims[sp[2]]
        kernel_w = window_dims[sp[3]]
        stride_d = sp[1] <= length(window_strides) ? window_strides[sp[1]] : kernel_d
        stride_h = sp[2] <= length(window_strides) ? window_strides[sp[2]] : kernel_h
        stride_w = sp[3] <= length(window_strides) ? window_strides[sp[3]] : kernel_w
        pad_front = window_padding[sp[1]][1]
        pad_back = window_padding[sp[1]][2]
        pad_top = window_padding[sp[2]][1]
        pad_bottom = window_padding[sp[2]][2]
        pad_left = window_padding[sp[3]][1]
        pad_right = window_padding[sp[3]][2]

        # Compute output spatial sizes
        d0p = (d0 + pad_front + pad_back - kernel_d) ÷ stride_d + 1
        d1p = (d1 + pad_top + pad_bottom - kernel_h) ÷ stride_h + 1
        d2p = (d2 + pad_left + pad_right - kernel_w) ÷ stride_w + 1

        # --- Step 1: Pool d1, d2 (treat (N,C,d0) as batch — d2 is innermost, valid reshape) ---
        # Reshape [N,C,d0,d1,d2] → [N*C*d0, 1, d1, d2]
        r3d = mps_reshape(ctx.graph, pool_input, [N * C * d0, 1, d1, d2], "$(name)_r1")
        # 2D pool over d1 (H) and d2 (W)
        desc_hw = mps_create_pooling2d_descriptor(;
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_x=stride_w,
            stride_y=stride_h,
            padding_top=pad_top,
            padding_bottom=pad_bottom,
            padding_left=pad_left,
            padding_right=pad_right,
            data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCHW,
        )
        r3d = if body_op_name == "stablehlo.maximum"
            mps_max_pooling2d(ctx.graph, r3d, desc_hw, "$(name)_p1")
        else  # stablehlo.add → avg (we'll scale at the end)
            mps_avg_pooling2d(ctx.graph, r3d, desc_hw, "$(name)_p1")
        end
        # Result: [N*C*d0, 1, d1p, d2p]
        # Reshape → [N, C, d0, d1p, d2p]
        r3d = mps_reshape(ctx.graph, r3d, [N, C, d0, d1p, d2p], "$(name)_r2")

        # --- Step 2: Pool d0 (move it inward: transpose → reshape → pool → reshape → transpose) ---
        # Transpose [N,C,d0,d1p,d2p] → [N,C,d1p,d2p,d0] so d0 is innermost
        r3d = apply_permutation(ctx.graph, r3d, [0, 1, 3, 4, 2], "$(name)_t2a")
        # Reshape [N,C,d1p,d2p,d0] → [N*C*d1p*d2p, 1, d0, 1]
        r3d = mps_reshape(ctx.graph, r3d, [N * C * d1p * d2p, 1, d0, 1], "$(name)_r3")
        # 2D pool over d0 (H) with kernel=1 in W
        desc_d = mps_create_pooling2d_descriptor(;
            kernel_h=kernel_d,
            kernel_w=1,
            stride_x=1,
            stride_y=stride_d,
            padding_top=pad_front,
            padding_bottom=pad_back,
            padding_left=0,
            padding_right=0,
            data_layout=Metal.MPSGraphs.MPSGraphTensorNamedDataLayoutNCHW,
        )
        r3d = if body_op_name == "stablehlo.maximum"
            mps_max_pooling2d(ctx.graph, r3d, desc_d, "$(name)_p2")
        else
            mps_avg_pooling2d(ctx.graph, r3d, desc_d, "$(name)_p2")
        end
        # Result: [N*C*d1p*d2p, 1, d0p, 1]
        # Reshape → [N, C, d1p, d2p, d0p]
        r3d = mps_reshape(ctx.graph, r3d, [N, C, d1p, d2p, d0p], "$(name)_r4")
        # Transpose [N,C,d1p,d2p,d0p] → [N,C,d0p,d1p,d2p]
        r3d = apply_permutation(ctx.graph, r3d, [0, 1, 4, 2, 3], "$(name)_t2b")

        # For sum pooling: avg1*avg2 = avg over 3D window; multiply by volume to get sum
        if body_op_name == "stablehlo.add"
            window_vol = kernel_d * kernel_h * kernel_w
            mps_dtype = julia_to_mps_dtype(in_dtype)
            scalar = Metal.MPSGraphs.constantWithScalar(
                ctx.graph, Float64(window_vol), mps_dtype
            )
            r3d = Metal.MPSGraphs.multiplicationWithPrimaryTensor(
                ctx.graph, r3d, scalar, "$(name)_sum"
            )
        elseif body_op_name != "stablehlo.maximum"
            error("Unsupported reduce_window body op: $body_op_name")
        end

        # Transpose back to original layout if we rearranged at start
        if !is_ncdhw_pool
            inv_perm = zeros(Int, rank)
            for (dst, src) in enumerate(perm_src_3d)
                inv_perm[src] = dst - 1
            end
            r3d = apply_permutation(ctx.graph, r3d, inv_perm, "$(name)_output_from_ncdhw")
        end
        r3d

    else
        error(
            "reduce_window only supported for rank 4 (2D pooling) and rank 5 (3D pooling), got rank $rank",
        )
    end

    return ctx.value_map[IR.result(op, 1)] = pool_result
end

# ============================================================================
# compile_mlir_module — the main entry point
# ============================================================================

"""
    compile_mlir_module(mod::MLIR.IR.Module) -> MetalExecutable

Walk a Reactant MLIR.IR.Module and compile it to a MetalExecutable.

Uses Reactant.MLIR.IR and Reactant.MLIR.API directly (imported in the
parent ReactantMetalExt module).

Returns a MetalExecutable ready for execution via [`execute!`](@ref).
"""
function compile_mlir_module(mod)
    ctx = TranslationContext()

    # Navigate: Module → func.func → Block
    body_block = IR.body(mod)
    func_op = first(body_block)
    func_region = IR.region(func_op, 1)
    func_block = first(func_region)

    # Create input placeholders from block arguments
    n_args = IR.nargs(func_block)
    for i in 1:n_args
        arg = IR.argument(func_block, i)
        ir_shape, dtype = get_type_info(arg)
        mps_dtype = julia_to_mps_dtype(dtype)

        # Placeholder uses Julia shape (reversed from IR for 2D+)
        julia_shape = length(ir_shape) >= 2 ? reverse(ir_shape) : ir_shape

        placeholder = Metal.MPSGraphs.placeholderTensor(
            ctx.graph, julia_shape, mps_dtype, "arg_$i"
        )
        ctx.value_map[arg] = placeholder
        push!(ctx.inputs, placeholder)
        push!(ctx.input_shapes, ir_shape)
        push!(ctx.input_dtypes, dtype)
    end

    # Walk operations
    for op in func_block
        op_name = IR.name(op)
        ctx.op_count += 1

        if op_name == "func.return"
            for j in 1:IR.noperands(op)
                ret_val = IR.operand(op, j)
                if haskey(ctx.value_map, ret_val)
                    push!(ctx.outputs, ctx.value_map[ret_val])
                    ir_shape, dtype = get_type_info(ret_val)
                    julia_shape = length(ir_shape) >= 2 ? reverse(ir_shape) : ir_shape
                    push!(ctx.output_shapes, julia_shape)
                    push!(ctx.output_dtypes, dtype)
                else
                    @warn "Return value not found in value_map"
                end
            end
        elseif op_name == "stablehlo.add"
            handle_add(ctx, op)
        elseif op_name == "stablehlo.subtract"
            handle_subtract(ctx, op)
        elseif op_name == "stablehlo.multiply"
            handle_multiply(ctx, op)
        elseif op_name == "stablehlo.divide"
            handle_divide(ctx, op)
        elseif op_name == "stablehlo.maximum"
            handle_maximum(ctx, op)
        elseif op_name == "stablehlo.negate"
            handle_negate(ctx, op)
        elseif op_name == "stablehlo.exponential" || op_name == "stablehlo.exp"
            handle_exponential(ctx, op)
        elseif op_name == "stablehlo.log"
            handle_log(ctx, op)
        elseif op_name == "stablehlo.tanh"
            handle_tanh(ctx, op)
        elseif op_name == "stablehlo.sqrt"
            handle_sqrt(ctx, op)
        elseif op_name == "stablehlo.rsqrt"
            handle_rsqrt(ctx, op)
        elseif op_name == "stablehlo.abs"
            handle_abs(ctx, op)
        elseif op_name == "stablehlo.sine" || op_name == "stablehlo.sin"
            handle_sin(ctx, op)
        elseif op_name == "stablehlo.cosine" || op_name == "stablehlo.cos"
            handle_cos(ctx, op)
        elseif op_name == "stablehlo.convert"
            handle_convert(ctx, op)
        elseif op_name == "stablehlo.constant"
            handle_constant(ctx, op)
        elseif op_name == "stablehlo.dot_general" || op_name == "stablehlo.dot"
            handle_dot_general(ctx, op)
        elseif op_name == "stablehlo.broadcast_in_dim"
            handle_broadcast_in_dim(ctx, op)
        elseif op_name == "stablehlo.reshape"
            handle_reshape(ctx, op)
        elseif op_name == "stablehlo.transpose"
            handle_transpose(ctx, op)
        elseif op_name == "stablehlo.reverse"
            handle_reverse(ctx, op)
        elseif op_name == "stablehlo.concatenate"
            handle_concatenate(ctx, op)
        elseif op_name == "stablehlo.convolution"
            handle_convolution(ctx, op)
        elseif op_name == "stablehlo.slice"
            handle_slice(ctx, op)
        elseif op_name == "stablehlo.scatter"
            handle_scatter(ctx, op)
        elseif op_name == "stablehlo.reduce"
            handle_reduce(ctx, op)
        elseif op_name == "stablehlo.reduce_window"
            handle_reduce_window(ctx, op)
        else
            error("Unsupported StableHLO op: $op_name")
        end
    end

    return MetalExecutable(
        ctx.graph,
        ctx.inputs,
        ctx.input_shapes,
        ctx.input_dtypes,
        ctx.outputs,
        ctx.output_shapes,
        ctx.output_dtypes,
        ctx.op_count,
        ctx.const_placeholders,
        ctx.const_mtl_values,
    )
end

# ============================================================================
# execute! — run a MetalExecutable on the GPU
# ============================================================================

"""
    execute!(exec::MetalExecutable, input_bufs::Vector) -> Vector{MTLBuffer}

Execute a compiled MetalExecutable on the Metal GPU using raw MTLBuffers.

- `exec`:       MetalExecutable from compile_mlir_module
- `input_bufs`: Vector of raw MTLBuffer objects

Returns a Vector of raw MTLBuffer objects (SharedStorage, one per output).
The caller (PJRTPlugin._loaded_exec_execute) wraps them in MTLTensor handles.

Input MTLBuffers are fed directly to MPSGraph via MPSGraphTensorData — no
MtlArray intermediary, no CPU→GPU upload. Output data is extracted from
MPSGraph results via exportDataWithCommandBuffer to fresh raw MTLBuffers.
"""
function execute!(exec::MetalExecutable, input_bufs::Vector)
    # Build feeds dictionary: raw MTLBuffer → MPSGraphTensorData
    feeds = Dict{Metal.MPSGraphs.MPSGraphTensor,Metal.MPSGraphs.MPSGraphTensorData}()
    for (k, mtl_buf) in enumerate(input_bufs)
        ir_shape = exec.input_shapes[k]
        mps_dtype = julia_to_mps_dtype(exec.input_dtypes[k])
        # ir_shape is MLIR order = Apple's row-major convention (no reversal needed)
        mps_shape = convert(Metal.MPS.MPSShape, ir_shape)
        feeds[exec.input_placeholders[k]] = Metal.MPSGraphs.MPSGraphTensorData(
            mtl_buf, mps_shape, mps_dtype
        )
    end

    # Add fixed constant inputs (multi-value dense constants compiled as placeholders)
    for (ph, mtl) in zip(exec.const_placeholders, exec.const_mtl_values)
        feeds[ph] = Metal.MPSGraphs.MPSGraphTensorData(mtl)
    end

    feeds_ns = NSDictionary(feeds)
    targets_ns = NSArray(exec.output_tensors)

    dev = Metal.device()
    queue = Metal.global_queue(dev)
    results = Metal.MPSGraphs.run(exec.graph, queue, feeds_ns, targets_ns)
    Metal.synchronize()

    # Extract output data from MPSGraph results into raw MTLBuffers
    output_bufs = Metal.MTL.MTLBuffer[]
    for (i, output_tensor) in enumerate(exec.output_tensors)
        result_ptr = results[output_tensor]
        if result_ptr != nil
            result_data = reinterpret(Metal.MPSGraphs.MPSGraphTensorData, result_ptr)
            ndarray = Metal.MPS.MPSNDArray(result_data)

            out_shape = exec.output_shapes[i]  # Julia order
            out_dtype = exec.output_dtypes[i]
            mps_dtype = julia_to_mps_dtype(out_dtype)
            nbytes = max(1, prod(out_shape) * sizeof(mps_dtype))

            # Allocate raw MTLBuffer for output data
            out_buf = Metal.alloc(dev, nbytes; storage=Metal.SharedStorage)

            # Export MPSNDArray data directly to raw MTLBuffer (no MtlArray needed)
            cmdbuf = Metal.MTL.MTLCommandBuffer(queue) do cmdbuf
                Metal.MPS.exportDataWithCommandBuffer(
                    ndarray, cmdbuf, out_buf, mps_dtype, UInt(0)
                )
            end
            Metal.MTL.wait_completed(cmdbuf)

            push!(output_bufs, out_buf)
        end
    end

    return output_bufs
end
