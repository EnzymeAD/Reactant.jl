module tpu
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
import ..Dialects: namedattribute, operandsegmentsizes, c
import ...API
using EnumX

"""
`ReductionKind`
Reduction kind
"""
@enumx ReductionKind SUM MAX MIN
const ReductionKindStorage = ["sum", "max", "min"]

function IR.Attribute(e::ReductionKind.T)
    return parse(Attribute, "#tpu<reduction_kind <$(ReductionKindStorage[Int(e)+1])>>")
end

"""
`RoundingMode`
Rounding mode
"""
@enumx RoundingMode kTowardsZero kToNearestEven
const RoundingModeStorage = ["towards_zero", "to_nearest_even"]

function IR.Attribute(e::RoundingMode.T)
    return parse(Attribute, "#tpu<rounding_mode <$(RoundingModeStorage[Int(e)+1])>>")
end

"""
`ContractPrecision`
Contraction precision
"""
@enumx ContractPrecision kBF16 kFP32
const ContractPrecisionStorage = ["bf16", "fp32"]

function IR.Attribute(e::ContractPrecision.T)
    return parse(
        Attribute, "#tpu<contract_precision <$(ContractPrecisionStorage[Int(e)+1])>>"
    )
end

"""
`PackFormat`
Pack format
"""
@enumx PackFormat kCompressed kInterleaved
const PackFormatStorage = ["compressed", "interleaved"]

function IR.Attribute(e::PackFormat.T)
    return parse(Attribute, "#tpu<pack_format <$(PackFormatStorage[Int(e)+1])>>")
end

"""
`CoreType`
Core type
"""
@enumx CoreType kTc kScScalarSubcore kScVectorSubcore
const CoreTypeStorage = ["tc", "sc_scalar_subcore", "sc_vector_subcore"]

function IR.Attribute(e::CoreType.T)
    return parse(Attribute, "#tpu<core_type <$(CoreTypeStorage[Int(e)+1])>>")
end

function all_reduce(
    input::Value;
    output::Union{Nothing,IR.Type}=nothing,
    dim::Int64,
    kind::ReductionKind.T,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dim", dim), namedattribute("kind", kind)]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "tpu.all_reduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function sem_alloc(; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.sem_alloc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function assume_layout(input::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.assume_layout",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function assume_multiple(
    value::Value;
    result::Union{Nothing,IR.Type}=nothing,
    multiple::Int32,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("multiple", multiple),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tpu.assume_multiple",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function bitcast(input::Value; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.bitcast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function bitcast_vreg(input::Value; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.bitcast_vreg",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`broadcast_in_sublanes`

For each sublane `i`, broadcasts the value in lane `lane + i` along the entire
sublane. If `lane + i` is not in [0, lane_count), then the value in sublane `i`
is not defined (can be anything).
"""
function broadcast_in_sublanes(
    source::Value; output::IR.Type, lane::Int32, location::Location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("lane", lane),]

    return create_operation(
        "tpu.broadcast_in_sublanes",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function concatenate(
    sources::Vector{Value}; output::IR.Type, dimension::Int32, location::Location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[sources...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return create_operation(
        "tpu.concatenate",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function create_mask(
    low::Vector{Value}, high::Vector{Value}; output::IR.Type, location::Location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[low..., high...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.create_mask",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

"""
`create_subelement_mask`

The \"half-sublanes\", \"quarter-sublanes\", etc. (unit is determined by
the type of `output`) of the mask are masked in the range specified by
`from` and `to`.

- If `from <= to`, the range `[from, to)` is set and the rest is unset.
- If `to <= from`, the range `[to, from)` is unset and the rest is set.

All lanes are set identically.

# Example

```mlir
%msk = tpu.create_subelement_mask 3, 9 : vector<8x128x2xi1>
```

This creates a mask `%msk` where, for all `lane`s, `%msk[*][lane][*]` is:

```
[[0, 0], [0, 1], [1, 1], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]
```

It is currently only supported:
- In TPU v4, for `num_subelems` of 1 and 2.
- In TPU v5, for `num_subelems` of 1, 2, and 4.
"""
function create_subelement_mask(;
    output::IR.Type, from::Int32, to::Int32, location::Location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("from", from), namedattribute("to", to)]

    return create_operation(
        "tpu.create_subelement_mask",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function delay(nanos::Value; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[nanos,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.delay",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function device_id(; result::Union{Nothing,IR.Type}=nothing, location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tpu.device_id",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function dynamic_gather(
    source::Value,
    indices::Value;
    output::IR.Type,
    dimension::Int32,
    location::Location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[source, indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return create_operation(
        "tpu.dynamic_gather",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function dynamic_rotate(
    value::Value,
    amount::Value;
    result::IR.Type,
    dimension::Int32,
    stride::Union{Int32,Nothing}=nothing,
    stride_dimension::Union{Int32,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[value, amount]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]
    !isnothing(stride) && push!(attributes, namedattribute("stride", stride))
    !isnothing(stride_dimension) &&
        push!(attributes, namedattribute("stride_dimension", stride_dimension))

    return create_operation(
        "tpu.dynamic_rotate",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function enqueue_dma(
    source::Value,
    source_semaphore::Union{Nothing,Value}=nothing;
    target::Value,
    target_semaphore::Value,
    device_id::Union{Nothing,Value}=nothing,
    core_id::Union{Nothing,Value}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, target, target_semaphore]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(source_semaphore) && push!(operands, source_semaphore)
    !isnothing(device_id) && push!(operands, device_id)
    !isnothing(core_id) && push!(operands, core_id)
    push!(
        attributes,
        operandsegmentsizes([
            1,
            (source_semaphore == nothing) ? 0 : 11,
            1,
            if (device_id == nothing)
                0
            elseif 1(core_id == nothing)
                0
            else
                1
            end,
        ]),
    )

    return create_operation(
        "tpu.enqueue_dma",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function erase_memref_layout(operand::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.erase_memref_layout",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fptosi(
    input::Value;
    output::IR.Type,
    rounding_mode::RoundingMode.T,
    location::Location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rounding_mode", rounding_mode),]

    return create_operation(
        "tpu.fptosi",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function gather(
    source::Value;
    output::IR.Type,
    indices::IR.DenseAttribute{Int32},
    dimension::Int32,
    location::Location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("indices", indices), namedattribute("dimension", dimension)
    ]

    return create_operation(
        "tpu.gather",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sem_barrier(; semaphore::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[semaphore,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.sem_barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function internal_scratch(; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.internal_scratch",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function iteration_bound(;
    result::Union{Nothing,IR.Type}=nothing, dim::Int32, location::Location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dim", dim),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tpu.iteration_bound",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function iota(;
    output::IR.Type, dimension::Union{Int32,Nothing}=nothing, location::Location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dimension) && push!(attributes, namedattribute("dimension", dimension))

    return create_operation(
        "tpu.iota",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function load(
    base::Value,
    indices::Vector{Value};
    result::IR.Type,
    sublane_mask::IR.DenseAttribute{Bool},
    sublane_stride::Union{Int32,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sublane_mask", sublane_mask),]
    !isnothing(sublane_stride) &&
        push!(attributes, namedattribute("sublane_stride", sublane_stride))

    return create_operation(
        "tpu.load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function log_buffer(
    input::Value;
    shape::IR.DenseAttribute{Int64},
    tag::String,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("shape", shape), namedattribute("tag", tag)]

    return create_operation(
        "tpu.log_buffer",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function log(
    inputs::Vector{Value};
    tag::String,
    formatted::Union{Bool,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[inputs...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("tag", tag),]
    !isnothing(formatted) && push!(attributes, namedattribute("formatted", formatted))

    return create_operation(
        "tpu.log",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mask_cast(input::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.mask_cast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function matmul(
    lhs::Value,
    rhs::Value,
    acc::Value;
    result::IR.Type,
    transpose_lhs::Union{Bool,Nothing}=nothing,
    transpose_rhs::Union{Bool,Nothing}=nothing,
    precision::Union{ContractPrecision.T,Nothing}=nothing,
    dimension_numbers=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[lhs, rhs, acc]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(transpose_lhs) &&
        push!(attributes, namedattribute("transpose_lhs", transpose_lhs))
    !isnothing(transpose_rhs) &&
        push!(attributes, namedattribute("transpose_rhs", transpose_rhs))
    !isnothing(precision) && push!(attributes, namedattribute("precision", precision))
    !isnothing(dimension_numbers) &&
        push!(attributes, namedattribute("dimension_numbers", dimension_numbers))

    return create_operation(
        "tpu.matmul",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function memref_bitcast(input::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.memref_bitcast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function memref_reshape(input::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.memref_reshape",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function memref_slice(
    mem_ref::Value,
    base_idx::Vector{Value},
    dynamic_sizes::Vector{Value};
    result::IR.Type,
    location::Location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[mem_ref, base_idx..., dynamic_sizes...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(base_idx), length(dynamic_sizes)]))

    return create_operation(
        "tpu.memref_slice",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function memref_squeeze(input::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.memref_squeeze",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function prng_random_bits(; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.prng_random_bits",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function prng_set_seed_32(seeds::Vector{Value}; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[seeds...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.prng_set_seed_32",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pack_vmsk(low::Value, high::Value; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[low, high]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.pack_vmsk",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pack_subelements(
    sources::Vector{Value};
    output::IR.Type,
    positions::IR.DenseAttribute{Int32},
    pack_format::PackFormat.T,
    location::Location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[sources...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("positions", positions), namedattribute("pack_format", pack_format)
    ]

    return create_operation(
        "tpu.pack_subelements",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function region(;
    results::Base.AbstractVecOrTuple{IR.Type}, region::Region, location::Location=Location()
)
    op_ty_results = IR.Type[results...,]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.region",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function reinterpret_cast(input::Value; result::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.reinterpret_cast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function relayout(
    input::Value; output::Union{Nothing,IR.Type}=nothing, location::Location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "tpu.relayout",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function repeat(
    source::Value;
    output::IR.Type,
    dimension::Int32,
    times::Int32,
    location::Location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("dimension", dimension), namedattribute("times", times)
    ]

    return create_operation(
        "tpu.repeat",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function roll_vectors(input::Vector{Value}; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.roll_vectors",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function rotate(
    value::Value;
    result::Union{Nothing,IR.Type}=nothing,
    amount::Int32,
    dimension::Int32,
    stride::Union{Int32,Nothing}=nothing,
    stride_dimension::Union{Int32,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("amount", amount), namedattribute("dimension", dimension)
    ]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(stride) && push!(attributes, namedattribute("stride", stride))
    !isnothing(stride_dimension) &&
        push!(attributes, namedattribute("stride_dimension", stride_dimension))

    return create_operation(
        "tpu.rotate",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function sem_read(
    semaphore::Value; result::Union{Nothing,IR.Type}=nothing, location::Location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[semaphore,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tpu.sem_read",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(isempty(op_ty_results) ? nothing : op_ty_results),
        result_inference=isempty(op_ty_results),
    )
end

function sem_signal(
    semaphore::Value,
    amount::Value,
    device_id::Union{Nothing,Value}=nothing;
    core_id::Union{Nothing,Value}=nothing,
    core_type::Union{CoreType.T,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[semaphore, amount]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device_id) && push!(operands, device_id)
    !isnothing(core_id) && push!(operands, core_id)
    push!(attributes, operandsegmentsizes([
        1,
        1,
        if (device_id == nothing)
            0
        elseif 1(core_id == nothing)
            0
        else
            1
        end,
    ]))
    !isnothing(core_type) && push!(attributes, namedattribute("core_type", core_type))

    return create_operation(
        "tpu.sem_signal",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sem_wait(semaphore::Value, amount::Value; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[semaphore, amount]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.sem_wait",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function shuffled_load(
    base::Value,
    indices::Vector{Value};
    result::IR.Type,
    sublane_mask::IR.DenseAttribute{Bool},
    sublane_offsets::IR.DenseAttribute{Int32},
    location::Location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sublane_mask", sublane_mask),
        namedattribute("sublane_offsets", sublane_offsets),
    ]

    return create_operation(
        "tpu.shuffled_load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function shuffled_store(
    valueToStore::Value,
    base::Value,
    indices::Vector{Value};
    sublane_mask::IR.DenseAttribute{Bool},
    sublane_offsets::IR.DenseAttribute{Int32},
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[valueToStore, base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("sublane_mask", sublane_mask),
        namedattribute("sublane_offsets", sublane_offsets),
    ]

    return create_operation(
        "tpu.shuffled_store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function store(
    valueToStore::Value,
    base::Value,
    indices::Vector{Value},
    mask::Union{Nothing,Value}=nothing;
    sublane_mask::IR.DenseAttribute{Bool},
    sublane_stride::Union{Int32,Nothing}=nothing,
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[valueToStore, base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sublane_mask", sublane_mask),]
    !isnothing(mask) && push!(operands, mask)
    push!(
        attributes, operandsegmentsizes([1, 1, length(indices), (mask == nothing) ? 0 : 1])
    )
    !isnothing(sublane_stride) &&
        push!(attributes, namedattribute("sublane_stride", sublane_stride))

    return create_operation(
        "tpu.store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function strided_load(
    base::Value,
    indices::Vector{Value};
    result::IR.Type,
    strides::IR.DenseAttribute{Int32},
    location::Location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("strides", strides),]

    return create_operation(
        "tpu.strided_load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function strided_store(
    valueToStore::Value,
    base::Value,
    indices::Vector{Value};
    strides::IR.DenseAttribute{Int32},
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[valueToStore, base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("strides", strides),]

    return create_operation(
        "tpu.strided_store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function trace(;
    results::Base.AbstractVecOrTuple{IR.Type},
    message::String,
    level::Int32,
    region::Region,
    location::Location=Location(),
)
    op_ty_results = IR.Type[results...,]
    operands = Value[]
    owned_regions = Region[region,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("message", message), namedattribute("level", level)
    ]

    return create_operation(
        "tpu.trace",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function trace_start(; message::String, level::Int32, location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("message", message), namedattribute("level", level)
    ]

    return create_operation(
        "tpu.trace_start",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function trace_stop(; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.trace_stop",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function unpack_subelements(
    source::Value;
    output::IR.Type,
    index::Int32,
    pack_format::PackFormat.T,
    location::Location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("index", index), namedattribute("pack_format", pack_format)
    ]

    return create_operation(
        "tpu.unpack_subelements",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function unroll_vectors(
    input::Value; output::Base.AbstractVecOrTuple{IR.Type}, location::Location=Location()
)
    op_ty_results = IR.Type[output...,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.unroll_vectors",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function vector_store(
    valueToStore::Value,
    base::Value,
    indices::Vector{Value},
    mask::Union{Nothing,Value}=nothing;
    strides::IR.DenseAttribute{Int32},
    location::Location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[valueToStore, base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("strides", strides),]
    !isnothing(mask) && push!(operands, mask)
    push!(
        attributes, operandsegmentsizes([1, 1, length(indices), (mask == nothing) ? 0 : 1])
    )

    return create_operation(
        "tpu.vector_store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function wait_dma(semaphore::Value, ref::Value; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[semaphore, ref]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.wait_dma",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function weird(input::Value; output::IR.Type, location::Location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.weird",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function yield(results::Vector{Value}; location::Location=Location())
    op_ty_results = IR.Type[]
    operands = Value[results...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.yield",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # tpu
