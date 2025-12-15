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
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

function all_reduce(input::Value; output::IR.Type, dim, kind, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dim", dim), namedattribute("kind", kind)]

    return create_operation(
        "tpu.all_reduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sem_alloc(; result::IR.Type, location=Location())
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

function assume_layout(input::Value; result::IR.Type, location=Location())
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

"""
`assume_multiple`

This operation is a hint to the compiler that the input `value` is guaranteed
to be a multiple of `multiple`. This can be used to satisfy divisibility checks
in some compiler passes.

The result is the same as the input `value`.

# Example

```mlir
%val = tpu.assume_multiple %arg0, 16 : index
```
"""
function assume_multiple(
    value::Value; result=nothing::Union{Nothing,IR.Type}, multiple, location=Location()
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
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`barrier`

Performs barrier synchronization across all SC vector subcores at the
specified barrier id.
"""
function barrier(barrier_id::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[barrier_id,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function bitcast(input::Value; output::IR.Type, location=Location())
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

function bitcast_vreg(input::Value; output::IR.Type, location=Location())
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

For each sublane `i`, broadcasts the value in lane `lane + i` along the
entire sublane. For packed type, imagine the data is compressed unpacked
along sublane dimension, and the sublane count is multiplied by the packing
factor.
For example, for i16 with sublane count 8, `i` above is in [0, 8 * 2).
If `lane + i` is not in [0, lane_count), then the value in sublane `i` is
not defined (can be anything).
"""
function broadcast_in_sublanes(source::Value; output::IR.Type, lane, location=Location())
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
    sources::Vector{Value};
    output=nothing::Union{Nothing,IR.Type},
    dimension,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[sources...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "tpu.concatenate",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function create_mask(
    low::Vector{Value}, high::Vector{Value}; output::IR.Type, location=Location()
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
function create_subelement_mask(; output::IR.Type, from, to, location=Location())
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

function delay(nanos::Value; location=Location())
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

function device_id(; result=nothing::Union{Nothing,IR.Type}, location=Location())
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
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`dynamic_gather`

Gathers elements from `source` using `indices`.

The specified `dimensions` of `source` are collapsed together and indexed by
`indices`.

Given a shape `N0 x N1 x ...`,  the `output[i0, i1, ...]` is given by
`collapsed_source[j0, j1, ..., indices[i0, i1, ...] mod M]` where
- `collapsed_source` is the result of collapsing `dimensions` of `source`
  into a new trailing dimension of size `M`.
- `jk` is the subsequence of `in` for `n` not in `dimensions`.

When a single dimension is specified, this is similar to
`np.take_along_axis`.
"""
function dynamic_gather(
    source::Value,
    indices::Value;
    output=nothing::Union{Nothing,IR.Type},
    dimensions,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, indices]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions),]
    !isnothing(output) && push!(op_ty_results, output)

    return create_operation(
        "tpu.dynamic_gather",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function dynamic_rotate(
    value::Value,
    amount::Value;
    result::IR.Type,
    dimension,
    stride=nothing,
    stride_dimension=nothing,
    location=Location(),
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
    source_semaphore=nothing::Union{Nothing,Value};
    target::Value,
    target_semaphore::Value,
    device_id=nothing::Union{Nothing,Value},
    core_id=nothing::Union{Nothing,Value},
    priority=nothing,
    strict_ordering=nothing,
    location=Location(),
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
            (source_semaphore == nothing) ? 0 : 1,
            1,
            1,
            (device_id == nothing) ? 0 : 1,
            (core_id == nothing) ? 0 : 1,
        ]),
    )
    !isnothing(priority) && push!(attributes, namedattribute("priority", priority))
    !isnothing(strict_ordering) &&
        push!(attributes, namedattribute("strict_ordering", strict_ordering))

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

function enqueue_indirect_dma(
    source::Value,
    target::Value,
    offsets::Value,
    semaphore::Value,
    offset_filter=nothing::Union{Nothing,Value};
    add=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, target, offsets, semaphore]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(offset_filter) && push!(operands, offset_filter)
    !isnothing(add) && push!(attributes, namedattribute("add", add))

    return create_operation(
        "tpu.enqueue_indirect_dma",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function erase_memref_layout(
    operand::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tpu.erase_memref_layout",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function extf(in::Value; out::IR.Type, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.extf",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fptosi(input::Value; output::IR.Type, rounding_mode, location=Location())
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

function gather(source::Value; output::IR.Type, indices, dimension, location=Location())
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

function sem_barrier(; semaphore::IR.Type, location=Location())
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

function internal_scratch(; result::IR.Type, location=Location())
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

function iteration_bound(; result=nothing::Union{Nothing,IR.Type}, dim, location=Location())
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
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

"""
`iota`

Creates a vector that with values that start at 0 and increase along a
dimension resulting from collapsing the given `dimensions` together in
row-major order.

# Example
```
tpu.iota {dimensions = array<i32: 2, 0>} : vector<4x3x2xi16>
```
This produces a vector with the following values:
```
[[[0, 4], [0, 4], [0, 4]]
 [[1, 5], [1, 5], [1, 5]]
 [[2, 6], [2, 6], [2, 6]]
 [[3, 7], [3, 7], [3, 7]]]
```
"""
function iota(; output::IR.Type, dimensions, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimensions", dimensions),]

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

"""
`load`

Similar to `vector::LoadOp` but with `sublane_mask` and `sublane_stride`.
When `indices` are negative, it means loading from negative offset
of `base` address.
"""
function load(
    base::Value,
    indices::Vector{Value};
    result::IR.Type,
    sublane_mask,
    sublane_stride=nothing,
    location=Location(),
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

function log_buffer(input::Value; shape, tag, location=Location())
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

function log(inputs::Vector{Value}; tag, formatted=nothing, location=Location())
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

"""
`mask_cast`

Cast a mask register into a different packing.

If casting to a type with smaller packing, then values being packed together
must be identical. For example, for 8x128x4xi1 -> 8x128x2xi1,
input[i, j, 0] == input[i, j, 1] and input[i, j, 2] == input[i, j, 3] must
hold for all i, j. Otherwise, the result is undefined.
"""
function mask_cast(input::Value; result::IR.Type, location=Location())
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
    transpose_lhs=nothing,
    transpose_rhs=nothing,
    precision=nothing,
    dimension_numbers=nothing,
    location=Location(),
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

function memref_bitcast(input::Value; result::IR.Type, location=Location())
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

function memref_reshape(input::Value; result::IR.Type, location=Location())
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
    location=Location(),
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

function memref_squeeze(input::Value; result::IR.Type, location=Location())
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

function prng_random_bits(; output::IR.Type, location=Location())
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

function prng_set_seed_32(seeds::Vector{Value}; location=Location())
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

"""
`pack_elementwise`

Packs multiple `sources` elementwise into a single vector of a narrower `target_type`.

The number of `sources` must equal the packing factor, which is the ratio of
the element bitwidth of the `sources` to the element bitwidth of the
`target_type`. Elements from the `sources` are interleaved and packed into
each word of the `output`, ordered from lowest to highest bits,
corresponding to their order in the `sources`.
"""
function pack_elementwise(
    sources::Vector{Value}; output::IR.Type, target_type, location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[sources...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("target_type", target_type),]

    return create_operation(
        "tpu.pack_elementwise",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function pack_vmsk(low::Value, high::Value; output::IR.Type, location=Location())
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
    sources::Vector{Value}; output::IR.Type, positions, pack_format, location=Location()
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

function reciprocal(
    input::Value;
    output=nothing::Union{Nothing,IR.Type},
    approx=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(op_ty_results, output)
    !isnothing(approx) && push!(attributes, namedattribute("approx", approx))

    return create_operation(
        "tpu.reciprocal",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function reduce_index(input::Value; output::IR.Type, axis, kind, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), namedattribute("kind", kind)]

    return create_operation(
        "tpu.reduce_index",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function region(; results::Vector{IR.Type}, region::Region, location=Location())
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

function reinterpret_cast(input::Value; result::IR.Type, location=Location())
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

function relayout(input::Value; output=nothing::Union{Nothing,IR.Type}, location=Location())
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
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function repeat(source::Value; output::IR.Type, dimension, times, location=Location())
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

function reshape(source::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.reshape",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function roll_vectors(input::Vector{Value}; output::IR.Type, location=Location())
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

"""
`rotate`

Rotates the given vector by the given amount in the given dimension, i.e.,
for a 2D vector of shape (m, n), rotating dim 0 by `amount` will shift a row
at index `i` to index `(i + amount) % m`
"""
function rotate(
    value::Value;
    result=nothing::Union{Nothing,IR.Type},
    amount,
    dimension,
    stride=nothing,
    stride_dimension=nothing,
    location=Location(),
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
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function sitofp(in::Value; output::IR.Type, rounding_mode, location=Location())
    op_ty_results = IR.Type[output,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rounding_mode", rounding_mode),]

    return create_operation(
        "tpu.sitofp",
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
`scan_count`

ScanCountOp calculates the running duplicate occurrence count of the elements
in the input vector, %values. The output vector, %counts, contains the running
duplicate occurrence count for the corresponding element in
the input vector, where the count is performed in ascending order of element
indices. For example, if the elements of %values at indices 0, 5, and 7 had
duplicate values, then the elements of %counts at indices 0, 5, and 7 would
be 1, 2, and 3, respectively.

A mask vector, %in_mask, specifies which of the elements in the input vector
are eligible for counting. An element in %values that has its mask set to 0
will always have a count of 1 in %counts, regardless of the position in the
vector, or whether there were duplicates or not.
"""
function scan_count(
    in_mask::Value,
    values::Value;
    out_mask=nothing::Union{Nothing,IR.Type},
    counts=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[in_mask, values]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(out_mask) && push!(op_ty_results, out_mask)
    !isnothing(counts) && push!(op_ty_results, counts)

    return create_operation(
        "tpu.scan_count",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function scan(
    input::Value,
    mask=nothing::Union{Nothing,Value};
    output::IR.Type,
    kind,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[input,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind),]
    !isnothing(mask) && push!(operands, mask)

    return create_operation(
        "tpu.scan",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function sem_read(
    semaphore::Value; result=nothing::Union{Nothing,IR.Type}, location=Location()
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
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function sem_signal(
    semaphore::Value,
    amount::Value,
    device_id=nothing::Union{Nothing,Value};
    core_id=nothing::Union{Nothing,Value},
    core_type=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[semaphore, amount]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device_id) && push!(operands, device_id)
    !isnothing(core_id) && push!(operands, core_id)
    push!(
        attributes,
        operandsegmentsizes([
            1, 1, (device_id == nothing) ? 0 : 1, (core_id == nothing) ? 0 : 1
        ]),
    )
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

function sem_wait(semaphore::Value, amount::Value; location=Location())
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
    sublane_mask,
    sublane_offsets,
    location=Location(),
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
    sublane_mask,
    sublane_offsets,
    location=Location(),
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

"""
`sort`

tpu.sort performs a stable sort of key/value pairs in ascending or
descending order based on keys. Masked-out keys and values are placed at the
end of the output vectors. An output mask indicates which outputs
correspond to the valid inputs.
"""
function sort(
    keys::Value,
    values::Value,
    mask=nothing::Union{Nothing,Value};
    output_mask::IR.Type,
    sorted_keys::IR.Type,
    sorted_values::IR.Type,
    descending=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[output_mask, sorted_keys, sorted_values]
    operands = Value[keys, values]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(mask) && push!(operands, mask)
    !isnothing(descending) && push!(attributes, namedattribute("descending", descending))

    return create_operation(
        "tpu.sort",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function stochastic_convert_elementwise(
    input::Value, random::Value; output::IR.Type, dst_type, location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[input, random]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dst_type", dst_type),]

    return create_operation(
        "tpu.stochastic_convert_elementwise",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function stochastic_convert(
    input::Value, random::Value; output::IR.Type, location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[input, random]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.stochastic_convert",
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
    mask=nothing::Union{Nothing,Value};
    sublane_mask,
    sublane_stride=nothing,
    location=Location(),
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
    base::Value, indices::Vector{Value}; result::IR.Type, strides, location=Location()
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
    valueToStore::Value, base::Value, indices::Vector{Value}; strides, location=Location()
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

function sublane_shuffle(
    lhs::Value,
    rhs::Value;
    result=nothing::Union{Nothing,IR.Type},
    pattern,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pattern", pattern),]
    !isnothing(result) && push!(op_ty_results, result)

    return create_operation(
        "tpu.sublane_shuffle",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function trace(;
    results::Vector{IR.Type}, message, level, region::Region, location=Location()
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

function trace_start(; message, level, location=Location())
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

function trace_stop(; location=Location())
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

function transpose(vector::Value; result::IR.Type, permutation, location=Location())
    op_ty_results = IR.Type[result,]
    operands = Value[vector,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation", permutation),]

    return create_operation(
        "tpu.transpose",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function truncf(in::Value; out::IR.Type, rounding_mode, location=Location())
    op_ty_results = IR.Type[out,]
    operands = Value[in,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rounding_mode", rounding_mode),]

    return create_operation(
        "tpu.truncf",
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
`unpack_elementwise`

Unpacks a single vector from `source`, which contains multiple `source_type`
vectors packed elementwise.

The `index` selects which packed value to extract from each word of `source`.
An `index` of 0 corresponds to the lowest bits. The extracted values are
cast to the output element type.
"""
function unpack_elementwise(
    source::Value; output::IR.Type, source_type, index, location=Location()
)
    op_ty_results = IR.Type[output,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("source_type", source_type), namedattribute("index", index)
    ]

    return create_operation(
        "tpu.unpack_elementwise",
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
    index,
    pack_format,
    sign_extended=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[output,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("index", index), namedattribute("pack_format", pack_format)
    ]
    !isnothing(sign_extended) &&
        push!(attributes, namedattribute("sign_extended", sign_extended))

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

function unroll_vectors(input::Value; output::Vector{IR.Type}, location=Location())
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

function vector_load_idx(
    base::Value,
    indices::Vector{Value},
    mask=nothing::Union{Nothing,Value};
    value::IR.Type,
    location=Location(),
)
    op_ty_results = IR.Type[value,]
    operands = Value[base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(mask) && push!(operands, mask)
    push!(attributes, operandsegmentsizes([1, length(indices), (mask == nothing) ? 0 : 1]))

    return create_operation(
        "tpu.vector_load_idx",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function vector_load(
    base::Value,
    indices::Vector{Value},
    mask=nothing::Union{Nothing,Value};
    result::IR.Type,
    strides,
    location=Location(),
)
    op_ty_results = IR.Type[result,]
    operands = Value[base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("strides", strides),]
    !isnothing(mask) && push!(operands, mask)
    push!(attributes, operandsegmentsizes([1, length(indices), (mask == nothing) ? 0 : 1]))

    return create_operation(
        "tpu.vector_load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function vector_store_idx(
    valueToStore::Value,
    base::Value,
    indices::Vector{Value},
    mask=nothing::Union{Nothing,Value};
    add=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[valueToStore, base, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(mask) && push!(operands, mask)
    push!(
        attributes, operandsegmentsizes([1, 1, length(indices), (mask == nothing) ? 0 : 1])
    )
    !isnothing(add) && push!(attributes, namedattribute("add", add))

    return create_operation(
        "tpu.vector_store_idx",
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
    mask=nothing::Union{Nothing,Value};
    strides,
    add=nothing,
    location=Location(),
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
    !isnothing(add) && push!(attributes, namedattribute("add", add))

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

function wait_dma2(
    semaphore::Value,
    src::Value,
    dst::Value,
    device_id=nothing::Union{Nothing,Value};
    core_id=nothing::Union{Nothing,Value},
    strict_ordering=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[semaphore, src, dst]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device_id) && push!(operands, device_id)
    !isnothing(core_id) && push!(operands, core_id)
    push!(
        attributes,
        operandsegmentsizes([
            1, 1, 1, (device_id == nothing) ? 0 : 1, (core_id == nothing) ? 0 : 1
        ]),
    )
    !isnothing(strict_ordering) &&
        push!(attributes, namedattribute("strict_ordering", strict_ordering))

    return create_operation(
        "tpu.wait_dma2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function wait_dma(semaphore::Value, ref::Value; location=Location())
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

function wait_indirect_dma(semaphore::Value, src::Value, dst::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[semaphore, src, dst]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "tpu.wait_indirect_dma",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function weird(input::Value; output::IR.Type, location=Location())
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

function yield(results::Vector{Value}; location=Location())
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
