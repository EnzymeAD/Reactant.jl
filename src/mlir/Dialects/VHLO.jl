module vhlo
using ...IR
import ...IR:
    NamedAttribute,
    Value,
    Location,
    Block,
    Region,
    Attribute,
    create_operation,
    context,
    IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

function abs_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.abs_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function add_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.add_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function after_all_v1(inputs::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.after_all_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function all_gather_v1(
    operand::Value;
    result::IR.Type,
    all_gather_dim,
    replica_groups,
    channel_id,
    use_global_device_ids,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("all_gather_dim", all_gather_dim),
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
        namedattribute("use_global_device_ids", use_global_device_ids),
    ]

    return create_operation(
        "vhlo.all_gather_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function all_gather_v2(
    operands::Vector{Value};
    results::Vector{IR.Type},
    all_gather_dim,
    replica_groups,
    channel_id,
    use_global_device_ids,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("all_gather_dim", all_gather_dim),
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
        namedattribute("use_global_device_ids", use_global_device_ids),
    ]

    return create_operation(
        "vhlo.all_gather_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function all_reduce_v1(
    operand::Value;
    result::IR.Type,
    replica_groups,
    channel_id,
    use_global_device_ids,
    computation::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[computation,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
        namedattribute("use_global_device_ids", use_global_device_ids),
    ]

    return create_operation(
        "vhlo.all_reduce_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function all_reduce_v2(
    operands::Vector{Value};
    results::Vector{IR.Type},
    replica_groups,
    channel_id,
    use_global_device_ids,
    computation::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[computation,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
        namedattribute("use_global_device_ids", use_global_device_ids),
    ]

    return create_operation(
        "vhlo.all_reduce_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function all_to_all_v1(
    operand::Value;
    result::IR.Type,
    split_dimension,
    concat_dimension,
    split_count,
    replica_groups,
    channel_id,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("split_dimension", split_dimension),
        namedattribute("concat_dimension", concat_dimension),
        namedattribute("split_count", split_count),
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
    ]

    return create_operation(
        "vhlo.all_to_all_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function all_to_all_v2(
    operands::Vector{Value};
    results::Vector{IR.Type},
    split_dimension,
    concat_dimension,
    split_count,
    replica_groups,
    channel_id,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("split_dimension", split_dimension),
        namedattribute("concat_dimension", concat_dimension),
        namedattribute("split_count", split_count),
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
    ]

    return create_operation(
        "vhlo.all_to_all_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function and_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.and_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function atan2_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.atan2_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function batch_norm_grad_v1(
    operand::Value,
    scale::Value,
    mean::Value,
    variance::Value,
    grad_output::Value;
    grad_operand::IR.Type,
    grad_scale::IR.Type,
    grad_offset::IR.Type,
    epsilon,
    feature_index,
    location=Location(),
)
    op_ty_results = IR.Type[grad_operand, grad_scale, grad_offset]
    operands = Value[operand, scale, mean, variance, grad_output]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("epsilon", epsilon), namedattribute("feature_index", feature_index)
    ]

    return create_operation(
        "vhlo.batch_norm_grad_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function batch_norm_inference_v1(
    operand::Value,
    scale::Value,
    offset::Value,
    mean::Value,
    variance::Value;
    result::IR.Type,
    epsilon,
    feature_index,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, scale, offset, mean, variance]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("epsilon", epsilon), namedattribute("feature_index", feature_index)
    ]

    return create_operation(
        "vhlo.batch_norm_inference_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function batch_norm_training_v1(
    operand::Value,
    scale::Value,
    offset::Value;
    output::IR.Type,
    batch_mean::IR.Type,
    batch_var::IR.Type,
    epsilon,
    feature_index,
    location=Location(),
)
    op_ty_results = IR.Type[output, batch_mean, batch_var]
    operands = Value[operand, scale, offset]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("epsilon", epsilon), namedattribute("feature_index", feature_index)
    ]

    return create_operation(
        "vhlo.batch_norm_training_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function bitcast_convert_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.bitcast_convert_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function broadcast_in_dim_v1(
    operand::Value; result::IR.Type, broadcast_dimensions, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute(
        "broadcast_dimensions", broadcast_dimensions
    ),]

    return create_operation(
        "vhlo.broadcast_in_dim_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function broadcast_v1(operand::Value; result::IR.Type, broadcast_sizes, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("broadcast_sizes", broadcast_sizes),]

    return create_operation(
        "vhlo.broadcast_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function call_v1(
    operands::Vector{Value}; results::Vector{IR.Type}, callee, location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("callee", callee),]

    return create_operation(
        "vhlo.call_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function case_v1(
    index::Value; results::Vector{IR.Type}, branches::Vector{Region}, location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[index,]
    owned_regions = Region[branches...,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.case_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cbrt_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.cbrt_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cbrt_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.cbrt_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ceil_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.ceil_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cholesky_v1(a::Value; result::IR.Type, lower, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[a,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("lower", lower),]

    return create_operation(
        "vhlo.cholesky_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function clamp_v1(
    min::Value, operand::Value, max::Value; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[min, operand, max]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.clamp_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function count_leading_zeros_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.count_leading_zeros_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function collective_broadcast_v1(
    operand::Value; result::IR.Type, replica_groups, channel_id, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
    ]

    return create_operation(
        "vhlo.collective_broadcast_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function collective_permute_v1(
    operand::Value; result::IR.Type, source_target_pairs, channel_id, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("source_target_pairs", source_target_pairs),
        namedattribute("channel_id", channel_id),
    ]

    return create_operation(
        "vhlo.collective_permute_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function compare_v1(
    lhs::Value,
    rhs::Value;
    result::IR.Type,
    comparison_direction,
    compare_type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("comparison_direction", comparison_direction),
        namedattribute("compare_type", compare_type),
    ]

    return create_operation(
        "vhlo.compare_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function complex_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.complex_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function composite_v1(
    inputs::Vector{Value};
    results::Vector{IR.Type},
    name,
    composite_attributes,
    decomposition,
    version,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("name", name),
        namedattribute("composite_attributes", composite_attributes),
        namedattribute("decomposition", decomposition),
        namedattribute("version", version),
    ]

    return create_operation(
        "vhlo.composite_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function concatenate_v1(
    inputs::Vector{Value}; result::IR.Type, dimension, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return create_operation(
        "vhlo.concatenate_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function constant_v1(; output::IR.Type, value, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value),]

    return create_operation(
        "vhlo.constant_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function convert_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.convert_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function convolution_v1(
    lhs::Value,
    rhs::Value;
    result::IR.Type,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    window_reversal,
    input_batch_dimension,
    input_feature_dimension,
    input_spatial_dimensions,
    kernel_input_feature_dimension,
    kernel_output_feature_dimension,
    kernel_spatial_dimensions,
    output_batch_dimension,
    output_feature_dimension,
    output_spatial_dimensions,
    feature_group_count,
    batch_group_count,
    precision_config,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("window_strides", window_strides),
        namedattribute("padding", padding),
        namedattribute("lhs_dilation", lhs_dilation),
        namedattribute("rhs_dilation", rhs_dilation),
        namedattribute("window_reversal", window_reversal),
        namedattribute("input_batch_dimension", input_batch_dimension),
        namedattribute("input_feature_dimension", input_feature_dimension),
        namedattribute("input_spatial_dimensions", input_spatial_dimensions),
        namedattribute("kernel_input_feature_dimension", kernel_input_feature_dimension),
        namedattribute("kernel_output_feature_dimension", kernel_output_feature_dimension),
        namedattribute("kernel_spatial_dimensions", kernel_spatial_dimensions),
        namedattribute("output_batch_dimension", output_batch_dimension),
        namedattribute("output_feature_dimension", output_feature_dimension),
        namedattribute("output_spatial_dimensions", output_spatial_dimensions),
        namedattribute("feature_group_count", feature_group_count),
        namedattribute("batch_group_count", batch_group_count),
        namedattribute("precision_config", precision_config),
    ]

    return create_operation(
        "vhlo.convolution_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cosine_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.cosine_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cosine_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.cosine_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function create_token_v1(; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.create_token_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cross_replica_sum_v1(
    operand::Value; result::IR.Type, replica_groups, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("replica_groups", replica_groups),]

    return create_operation(
        "vhlo.cross-replica-sum_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function custom_call_v1(
    inputs::Vector{Value};
    results::Vector{IR.Type},
    call_target_name,
    has_side_effect,
    backend_config,
    api_version,
    called_computations,
    operand_layouts,
    result_layouts,
    output_operand_aliases,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("call_target_name", call_target_name),
        namedattribute("has_side_effect", has_side_effect),
        namedattribute("backend_config", backend_config),
        namedattribute("api_version", api_version),
        namedattribute("called_computations", called_computations),
        namedattribute("operand_layouts", operand_layouts),
        namedattribute("result_layouts", result_layouts),
        namedattribute("output_operand_aliases", output_operand_aliases),
    ]

    return create_operation(
        "vhlo.custom_call_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function divide_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.divide_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dot_general_v1(
    lhs::Value,
    rhs::Value;
    result::IR.Type,
    lhs_batching_dimensions,
    rhs_batching_dimensions,
    lhs_contracting_dimensions,
    rhs_contracting_dimensions,
    precision_config,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs_batching_dimensions", lhs_batching_dimensions),
        namedattribute("rhs_batching_dimensions", rhs_batching_dimensions),
        namedattribute("lhs_contracting_dimensions", lhs_contracting_dimensions),
        namedattribute("rhs_contracting_dimensions", rhs_contracting_dimensions),
        namedattribute("precision_config", precision_config),
    ]

    return create_operation(
        "vhlo.dot_general_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dot_general_v2(
    lhs::Value,
    rhs::Value;
    result::IR.Type,
    lhs_batching_dimensions,
    rhs_batching_dimensions,
    lhs_contracting_dimensions,
    rhs_contracting_dimensions,
    precision_config,
    lhs_precision_type,
    rhs_precision_type,
    accumulation_type,
    lhs_component_count,
    rhs_component_count,
    num_primitive_operations,
    allow_imprecise_accumulation,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("lhs_batching_dimensions", lhs_batching_dimensions),
        namedattribute("rhs_batching_dimensions", rhs_batching_dimensions),
        namedattribute("lhs_contracting_dimensions", lhs_contracting_dimensions),
        namedattribute("rhs_contracting_dimensions", rhs_contracting_dimensions),
        namedattribute("precision_config", precision_config),
        namedattribute("lhs_precision_type", lhs_precision_type),
        namedattribute("rhs_precision_type", rhs_precision_type),
        namedattribute("accumulation_type", accumulation_type),
        namedattribute("lhs_component_count", lhs_component_count),
        namedattribute("rhs_component_count", rhs_component_count),
        namedattribute("num_primitive_operations", num_primitive_operations),
        namedattribute("allow_imprecise_accumulation", allow_imprecise_accumulation),
    ]

    return create_operation(
        "vhlo.dot_general_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dot_v1(
    lhs::Value, rhs::Value; result::IR.Type, precision_config, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("precision_config", precision_config),]

    return create_operation(
        "vhlo.dot_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_broadcast_in_dim_v1(
    operand::Value,
    output_dimensions::Value;
    result::IR.Type,
    broadcast_dimensions,
    known_expanding_dimensions,
    known_nonexpanding_dimensions,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, output_dimensions]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("broadcast_dimensions", broadcast_dimensions),
        namedattribute("known_expanding_dimensions", known_expanding_dimensions),
        namedattribute("known_nonexpanding_dimensions", known_nonexpanding_dimensions),
    ]

    return create_operation(
        "vhlo.dynamic_broadcast_in_dim_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_conv_v1(
    lhs::Value,
    rhs::Value,
    d_padding::Value;
    result::IR.Type,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    window_reversal,
    input_batch_dimension,
    input_feature_dimension,
    input_spatial_dimensions,
    kernel_input_feature_dimension,
    kernel_output_feature_dimension,
    kernel_spatial_dimensions,
    output_batch_dimension,
    output_feature_dimension,
    output_spatial_dimensions,
    feature_group_count,
    batch_group_count,
    precision_config,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs, d_padding]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("window_strides", window_strides),
        namedattribute("padding", padding),
        namedattribute("lhs_dilation", lhs_dilation),
        namedattribute("rhs_dilation", rhs_dilation),
        namedattribute("window_reversal", window_reversal),
        namedattribute("input_batch_dimension", input_batch_dimension),
        namedattribute("input_feature_dimension", input_feature_dimension),
        namedattribute("input_spatial_dimensions", input_spatial_dimensions),
        namedattribute("kernel_input_feature_dimension", kernel_input_feature_dimension),
        namedattribute("kernel_output_feature_dimension", kernel_output_feature_dimension),
        namedattribute("kernel_spatial_dimensions", kernel_spatial_dimensions),
        namedattribute("output_batch_dimension", output_batch_dimension),
        namedattribute("output_feature_dimension", output_feature_dimension),
        namedattribute("output_spatial_dimensions", output_spatial_dimensions),
        namedattribute("feature_group_count", feature_group_count),
        namedattribute("batch_group_count", batch_group_count),
        namedattribute("precision_config", precision_config),
    ]

    return create_operation(
        "vhlo.dynamic_conv_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_conv_v2(
    lhs::Value,
    rhs::Value,
    padding::Value;
    result::IR.Type,
    window_strides,
    lhs_dilation,
    rhs_dilation,
    window_reversal,
    input_batch_dimension,
    input_feature_dimension,
    input_spatial_dimensions,
    kernel_input_feature_dimension,
    kernel_output_feature_dimension,
    kernel_spatial_dimensions,
    output_batch_dimension,
    output_feature_dimension,
    output_spatial_dimensions,
    feature_group_count,
    batch_group_count,
    precision_config,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs, padding]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("window_strides", window_strides),
        namedattribute("lhs_dilation", lhs_dilation),
        namedattribute("rhs_dilation", rhs_dilation),
        namedattribute("window_reversal", window_reversal),
        namedattribute("input_batch_dimension", input_batch_dimension),
        namedattribute("input_feature_dimension", input_feature_dimension),
        namedattribute("input_spatial_dimensions", input_spatial_dimensions),
        namedattribute("kernel_input_feature_dimension", kernel_input_feature_dimension),
        namedattribute("kernel_output_feature_dimension", kernel_output_feature_dimension),
        namedattribute("kernel_spatial_dimensions", kernel_spatial_dimensions),
        namedattribute("output_batch_dimension", output_batch_dimension),
        namedattribute("output_feature_dimension", output_feature_dimension),
        namedattribute("output_spatial_dimensions", output_spatial_dimensions),
        namedattribute("feature_group_count", feature_group_count),
        namedattribute("batch_group_count", batch_group_count),
        namedattribute("precision_config", precision_config),
    ]

    return create_operation(
        "vhlo.dynamic_conv_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_gather_v1(
    operand::Value,
    start_indices::Value,
    slice_sizes::Value;
    result::IR.Type,
    offset_dims,
    collapsed_slice_dims,
    start_index_map,
    index_vector_dim,
    indices_are_sorted,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices, slice_sizes]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("offset_dims", offset_dims),
        namedattribute("collapsed_slice_dims", collapsed_slice_dims),
        namedattribute("start_index_map", start_index_map),
        namedattribute("index_vector_dim", index_vector_dim),
        namedattribute("indices_are_sorted", indices_are_sorted),
    ]

    return create_operation(
        "vhlo.dynamic_gather_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_gather_v2(
    operand::Value,
    start_indices::Value,
    slice_sizes::Value;
    result::IR.Type,
    offset_dims,
    collapsed_slice_dims,
    operand_batching_dims,
    start_indices_batching_dims,
    start_index_map,
    index_vector_dim,
    indices_are_sorted,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices, slice_sizes]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("offset_dims", offset_dims),
        namedattribute("collapsed_slice_dims", collapsed_slice_dims),
        namedattribute("operand_batching_dims", operand_batching_dims),
        namedattribute("start_indices_batching_dims", start_indices_batching_dims),
        namedattribute("start_index_map", start_index_map),
        namedattribute("index_vector_dim", index_vector_dim),
        namedattribute("indices_are_sorted", indices_are_sorted),
    ]

    return create_operation(
        "vhlo.dynamic_gather_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_iota_v1(
    output_shape::Value; result::IR.Type, iota_dimension, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[output_shape,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("iota_dimension", iota_dimension),]

    return create_operation(
        "vhlo.dynamic_iota_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_pad_v1(
    operand::Value,
    padding_value::Value,
    edge_padding_low::Value,
    edge_padding_high::Value,
    interior_padding::Value;
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[
        operand, padding_value, edge_padding_low, edge_padding_high, interior_padding
    ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.dynamic_pad_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_reshape_v1(
    operand::Value, output_shape::Value; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, output_shape]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.dynamic_reshape_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_slice_v1(
    operand::Value,
    start_indices::Vector{Value};
    result::IR.Type,
    slice_sizes,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("slice_sizes", slice_sizes),]

    return create_operation(
        "vhlo.dynamic_slice_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_update_slice_v1(
    operand::Value,
    update::Value,
    start_indices::Vector{Value};
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, update, start_indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.dynamic_update_slice_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function einsum_v1(
    lhs::Value, rhs::Value; result::IR.Type, einsum_config, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("einsum_config", einsum_config),]

    return create_operation(
        "vhlo.einsum_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function exponential_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.exponential_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function exponential_v2(
    operand::Value; result::IR.Type, result_accuracy, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.exponential_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function exponential_minus_one_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.exponential_minus_one_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function exponential_minus_one_v2(
    operand::Value; result::IR.Type, result_accuracy, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.exponential_minus_one_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fft_v1(operand::Value; result::IR.Type, fft_type, fft_length, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("fft_type", fft_type), namedattribute("fft_length", fft_length)
    ]

    return create_operation(
        "vhlo.fft_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function floor_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.floor_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function func_v1(;
    sym_name,
    function_type,
    sym_visibility,
    arg_attrs,
    res_attrs,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sym_name", sym_name),
        namedattribute("function_type", function_type),
        namedattribute("sym_visibility", sym_visibility),
        namedattribute("arg_attrs", arg_attrs),
        namedattribute("res_attrs", res_attrs),
    ]

    return create_operation(
        "vhlo.func_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gather_v1(
    operand::Value,
    start_indices::Value;
    result::IR.Type,
    offset_dims,
    collapsed_slice_dims,
    start_index_map,
    index_vector_dim,
    slice_sizes,
    indices_are_sorted,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("offset_dims", offset_dims),
        namedattribute("collapsed_slice_dims", collapsed_slice_dims),
        namedattribute("start_index_map", start_index_map),
        namedattribute("index_vector_dim", index_vector_dim),
        namedattribute("slice_sizes", slice_sizes),
        namedattribute("indices_are_sorted", indices_are_sorted),
    ]

    return create_operation(
        "vhlo.gather_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gather_v2(
    operand::Value,
    start_indices::Value;
    result::IR.Type,
    offset_dims,
    collapsed_slice_dims,
    operand_batching_dims,
    start_indices_batching_dims,
    start_index_map,
    index_vector_dim,
    slice_sizes,
    indices_are_sorted,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("offset_dims", offset_dims),
        namedattribute("collapsed_slice_dims", collapsed_slice_dims),
        namedattribute("operand_batching_dims", operand_batching_dims),
        namedattribute("start_indices_batching_dims", start_indices_batching_dims),
        namedattribute("start_index_map", start_index_map),
        namedattribute("index_vector_dim", index_vector_dim),
        namedattribute("slice_sizes", slice_sizes),
        namedattribute("indices_are_sorted", indices_are_sorted),
    ]

    return create_operation(
        "vhlo.gather_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function get_dimension_size_v1(
    operand::Value; result::IR.Type, dimension, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return create_operation(
        "vhlo.get_dimension_size_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function get_tuple_element_v1(operand::Value; result::IR.Type, index, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("index", index),]

    return create_operation(
        "vhlo.get_tuple_element_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function if_v1(
    pred::Value;
    results::Vector{IR.Type},
    true_branch::Region,
    false_branch::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[pred,]
    owned_regions = Region[true_branch, false_branch]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.if_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function imag_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.imag_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function infeed_v1(
    token::Value; results::Vector{IR.Type}, infeed_config, layout, location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[token,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("infeed_config", infeed_config), namedattribute("layout", layout)
    ]

    return create_operation(
        "vhlo.infeed_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function iota_v1(; output::IR.Type, iota_dimension, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("iota_dimension", iota_dimension),]

    return create_operation(
        "vhlo.iota_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function is_finite_v1(x::Value; y::IR.Type, location=Location())
    op_ty_results = IR.Type[y,]
    operands = Value[x,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.is_finite_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function log_plus_one_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.log_plus_one_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function log_plus_one_v2(
    operand::Value; result::IR.Type, result_accuracy, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.log_plus_one_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function log_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.log_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function log_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.log_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function logistic_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.logistic_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function logistic_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.logistic_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function map_v1(
    inputs::Vector{Value};
    result::IR.Type,
    dimensions,
    computation::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[inputs...,]
    owned_regions = Region[computation,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions),]

    return create_operation(
        "vhlo.map_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function maximum_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.maximum_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function minimum_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.minimum_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function multiply_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.multiply_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function negate_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.negate_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function not_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.not_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function optimization_barrier_v1(
    operand::Vector{Value}; result::Vector{IR.Type}, location=Location()
)
    op_ty_results = IR.Type[result...,]
    operands = Value[operand...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.optimization_barrier_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function or_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.or_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function outfeed_v1(
    inputs::Vector{Value},
    token::Value;
    result::IR.Type,
    outfeed_config,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[inputs..., token]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("outfeed_config", outfeed_config),]

    return create_operation(
        "vhlo.outfeed_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pad_v1(
    operand::Value,
    padding_value::Value;
    result::IR.Type,
    edge_padding_low,
    edge_padding_high,
    interior_padding,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, padding_value]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("edge_padding_low", edge_padding_low),
        namedattribute("edge_padding_high", edge_padding_high),
        namedattribute("interior_padding", interior_padding),
    ]

    return create_operation(
        "vhlo.pad_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function partition_id_v1(; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.partition_id_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function popcnt_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.popcnt_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function power_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.power_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function real_dynamic_slice_v1(
    operand::Value,
    start_indices::Value,
    limit_indices::Value,
    strides::Value;
    result::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, start_indices, limit_indices, strides]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.real_dynamic_slice_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function real_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.real_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function recv_v1(
    token::Value;
    results::Vector{IR.Type},
    channel_id,
    channel_type,
    is_host_transfer,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[token,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("channel_id", channel_id),
        namedattribute("channel_type", channel_type),
        namedattribute("is_host_transfer", is_host_transfer),
    ]

    return create_operation(
        "vhlo.recv_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function recv_v2(
    token::Value;
    results::Vector{IR.Type},
    channel_id,
    channel_type,
    is_host_transfer,
    source_target_pairs,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[token,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("channel_id", channel_id),
        namedattribute("channel_type", channel_type),
        namedattribute("is_host_transfer", is_host_transfer),
        namedattribute("source_target_pairs", source_target_pairs),
    ]

    return create_operation(
        "vhlo.recv_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reduce_v1(
    inputs::Vector{Value},
    init_values::Vector{Value};
    results::Vector{IR.Type},
    dimensions,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs..., init_values...]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions),]

    return create_operation(
        "vhlo.reduce_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reduce_precision_v1(
    operand::Value; output::IR.Type, exponent_bits, mantissa_bits, location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("exponent_bits", exponent_bits),
        namedattribute("mantissa_bits", mantissa_bits),
    ]

    return create_operation(
        "vhlo.reduce_precision_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reduce_scatter_v1(
    operand::Value;
    result::IR.Type,
    scatter_dimension,
    replica_groups,
    channel_id,
    use_global_device_ids,
    computation::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[computation,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("scatter_dimension", scatter_dimension),
        namedattribute("replica_groups", replica_groups),
        namedattribute("channel_id", channel_id),
        namedattribute("use_global_device_ids", use_global_device_ids),
    ]

    return create_operation(
        "vhlo.reduce_scatter_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reduce_window_v1(
    inputs::Vector{Value},
    init_values::Vector{Value};
    results::Vector{IR.Type},
    window_dimensions,
    window_strides,
    base_dilations,
    window_dilations,
    padding,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs..., init_values...]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("window_dimensions", window_dimensions),
        namedattribute("window_strides", window_strides),
        namedattribute("base_dilations", base_dilations),
        namedattribute("window_dilations", window_dilations),
        namedattribute("padding", padding),
    ]

    return create_operation(
        "vhlo.reduce_window_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function remainder_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.remainder_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function replica_id_v1(; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.replica_id_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reshape_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.reshape_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function return_v1(results::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[results...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.return_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reverse_v1(operand::Value; result::IR.Type, dimensions, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions),]

    return create_operation(
        "vhlo.reverse_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function rng_bit_generator_v1(
    initial_state::Value;
    output_state::IR.Type,
    output::IR.Type,
    rng_algorithm,
    location=Location(),
)
    op_ty_results = IR.Type[output_state, output]
    operands = Value[initial_state,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rng_algorithm", rng_algorithm),]

    return create_operation(
        "vhlo.rng_bit_generator_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function rng_v1(
    a::Value, b::Value, shape::Value; result::IR.Type, rng_distribution, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[a, b, shape]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rng_distribution", rng_distribution),]

    return create_operation(
        "vhlo.rng_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function round_nearest_even_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.round_nearest_even_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function round_nearest_afz_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.round_nearest_afz_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function rsqrt_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.rsqrt_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function rsqrt_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.rsqrt_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function scatter_v1(
    inputs::Vector{Value},
    scatter_indices::Value,
    updates::Vector{Value};
    results::Vector{IR.Type},
    update_window_dims,
    inserted_window_dims,
    scatter_dims_to_operand_dims,
    index_vector_dim,
    indices_are_sorted,
    unique_indices,
    update_computation::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs..., scatter_indices, updates...]
    owned_regions = Region[update_computation,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("update_window_dims", update_window_dims),
        namedattribute("inserted_window_dims", inserted_window_dims),
        namedattribute("scatter_dims_to_operand_dims", scatter_dims_to_operand_dims),
        namedattribute("index_vector_dim", index_vector_dim),
        namedattribute("indices_are_sorted", indices_are_sorted),
        namedattribute("unique_indices", unique_indices),
    ]

    return create_operation(
        "vhlo.scatter_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function scatter_v2(
    inputs::Vector{Value},
    scatter_indices::Value,
    updates::Vector{Value};
    results::Vector{IR.Type},
    update_window_dims,
    inserted_window_dims,
    input_batching_dims,
    scatter_indices_batching_dims,
    scatter_dims_to_operand_dims,
    index_vector_dim,
    indices_are_sorted,
    unique_indices,
    update_computation::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs..., scatter_indices, updates...]
    owned_regions = Region[update_computation,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("update_window_dims", update_window_dims),
        namedattribute("inserted_window_dims", inserted_window_dims),
        namedattribute("input_batching_dims", input_batching_dims),
        namedattribute("scatter_indices_batching_dims", scatter_indices_batching_dims),
        namedattribute("scatter_dims_to_operand_dims", scatter_dims_to_operand_dims),
        namedattribute("index_vector_dim", index_vector_dim),
        namedattribute("indices_are_sorted", indices_are_sorted),
        namedattribute("unique_indices", unique_indices),
    ]

    return create_operation(
        "vhlo.scatter_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function select_and_scatter_v1(
    operand::Value,
    source::Value,
    init_value::Value;
    result::IR.Type,
    window_dimensions,
    window_strides,
    padding,
    select::Region,
    scatter::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, source, init_value]
    owned_regions = Region[select, scatter]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("window_dimensions", window_dimensions),
        namedattribute("window_strides", window_strides),
        namedattribute("padding", padding),
    ]

    return create_operation(
        "vhlo.select_and_scatter_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function select_v1(
    pred::Value, on_true::Value, on_false::Value; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[pred, on_true, on_false]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.select_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function send_v1(
    inputs::Vector{Value},
    token::Value;
    result::IR.Type,
    channel_id,
    channel_type,
    is_host_transfer,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[inputs..., token]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("channel_id", channel_id),
        namedattribute("channel_type", channel_type),
        namedattribute("is_host_transfer", is_host_transfer),
    ]

    return create_operation(
        "vhlo.send_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function send_v2(
    inputs::Vector{Value},
    token::Value;
    result::IR.Type,
    channel_id,
    channel_type,
    is_host_transfer,
    source_target_pairs,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[inputs..., token]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("channel_id", channel_id),
        namedattribute("channel_type", channel_type),
        namedattribute("is_host_transfer", is_host_transfer),
        namedattribute("source_target_pairs", source_target_pairs),
    ]

    return create_operation(
        "vhlo.send_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function set_dimension_size_v1(
    operand::Value, size::Value; result::IR.Type, dimension, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, size]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return create_operation(
        "vhlo.set_dimension_size_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function shift_left_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.shift_left_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function shift_right_arithmetic_v1(
    lhs::Value, rhs::Value; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.shift_right_arithmetic_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function shift_right_logical_v1(
    lhs::Value, rhs::Value; result::IR.Type, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.shift_right_logical_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sign_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.sign_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sine_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.sine_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sine_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.sine_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function slice_v1(
    operand::Value;
    result::IR.Type,
    start_indices,
    limit_indices,
    strides,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("start_indices", start_indices),
        namedattribute("limit_indices", limit_indices),
        namedattribute("strides", strides),
    ]

    return create_operation(
        "vhlo.slice_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sort_v1(
    inputs::Vector{Value};
    results::Vector{IR.Type},
    dimension,
    is_stable,
    comparator::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[inputs...,]
    owned_regions = Region[comparator,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("dimension", dimension), namedattribute("is_stable", is_stable)
    ]

    return create_operation(
        "vhlo.sort_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sqrt_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.sqrt_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sqrt_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.sqrt_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function subtract_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.subtract_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tan_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.tan_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tan_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.tan_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tanh_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.tanh_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tanh_v2(operand::Value; result::IR.Type, result_accuracy, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_accuracy", result_accuracy),]

    return create_operation(
        "vhlo.tanh_v2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function torch_index_select_v1(
    operand::Value, index::Value; result::IR.Type, dim, batch_dims, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand, index]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("dim", dim), namedattribute("batch_dims", batch_dims)
    ]

    return create_operation(
        "vhlo.torch_index_select_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function transpose_v1(operand::Value; result::IR.Type, permutation, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation", permutation),]

    return create_operation(
        "vhlo.transpose_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function triangular_solve_v1(
    a::Value,
    b::Value;
    result::IR.Type,
    left_side,
    lower,
    unit_diagonal,
    transpose_a,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[a, b]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("left_side", left_side),
        namedattribute("lower", lower),
        namedattribute("unit_diagonal", unit_diagonal),
        namedattribute("transpose_a", transpose_a),
    ]

    return create_operation(
        "vhlo.triangular_solve_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tuple_v1(val::Vector{Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[val...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.tuple_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function unary_einsum_v1(
    operand::Value; result::IR.Type, einsum_config, location=Location()
)
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("einsum_config", einsum_config),]

    return create_operation(
        "vhlo.unary_einsum_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function uniform_dequantize_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.uniform_dequantize_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function uniform_quantize_v1(operand::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.uniform_quantize_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function while_v1(
    operand::Vector{Value};
    results::Vector{IR.Type},
    cond::Region,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[operand...,]
    owned_regions = Region[cond, body]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.while_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function xor_v1(lhs::Value, rhs::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "vhlo.xor_v1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # vhlo
