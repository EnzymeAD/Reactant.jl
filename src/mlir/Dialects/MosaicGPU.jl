module mosaic_gpu
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

function arrive_expect_tx(barrier::Value; expect_tx, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[barrier,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("expect_tx", expect_tx),]

    return create_operation(
        "mosaic_gpu.arrive_expect_tx",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function arrive(barrier::Value; orders_tensor_core, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[barrier,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("orders_tensor_core", orders_tensor_core),]

    return create_operation(
        "mosaic_gpu.arrive",
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
`async_load`

Schedules an async copy of the contents of the `source` MemRef in GMEM to
the `destination` MemRef in SMEM. The `destination` MemRef in SMEM must be
contiguous.

Upon completion of the copy, the `complete-tx(complete-count)` operation
will always be executed on the provided `barrier`.

The `indices` and `slice_lengths` inputs define what slice of the GMEM
`source` corresponds to the SMEM `destination`. Both `indices` and
`slice_lengths` must have a length equal to the rank of the `source`. The
values in `indices` are the starting indices of each dimension and the
values in `slice_lengths` are the lengths. Providing -1 in `slice_lengths`
indicates that the slice length is 1 and that the corresponding dimension
should be collapsed and does not appear in the `destination` MemRef.

The data is written in row-major order to the contiguous SMEM `destination`.
The `source` data does not need to be contiguous, except for the last
(and minor-most) dimension.

The `collective` attribute can be provided to use TMA multicast to more
efficiently load the GMEM data in cases where multiple thread blocks are
grouped together in a cluster and need to load the same data. Each block in
a cluster will first load a slice from GMEM to SMEM and then the slices will
be multicast to all other blocks in the cluster. In this way TMA multicast
guarantees L2 cache hits. The `collective` attribute is the list of
cluster dimensions along which to partition the input data loads.

The `predicate` allows scheduling the transfer conditionally. The async copy
   is always scheduled by at most a single lane in the warpgroup.
"""
function async_load(
    source::Value,
    destination::Value,
    barrier::Value,
    indices::Vector{Value},
    predicate=nothing::Union{Nothing,Value};
    slice_lengths,
    collective,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, destination, barrier, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("slice_lengths", slice_lengths),
        namedattribute("collective", collective),
    ]
    !isnothing(predicate) && push!(operands, predicate)
    push!(
        attributes,
        operandsegmentsizes([1, 1, 1, length(indices), Int(!isnothing(predicate))]),
    )

    return create_operation(
        "mosaic_gpu.async_load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function async_load_tmem(
    source::Value; result_0=nothing::Union{Nothing,IR.Type}, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "mosaic_gpu.async_load_tmem",
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
`async_prefetch`

Schedules an async prefetch of the contents of the `source` MemRef in GMEM
to the L2 cache, making subsequent loads of the same data from GMEM faster.

The `indices` and `slice_lengths` inputs define what slice of the GMEM
`source` is going to be prefetched. Both `indices` and `slice_lengths` must
have a length equal to the rank of the `source`. The values in `indices` are
the starting indices of each dimension and the values in `slice_lengths` are
the lengths. Providing -1 in `slice_lengths` indicates that the slice length
is 1.

The `collective` attribute can be provided to partition the prefetch over
multiple blocks in a cluster.

The `predicate` allows scheduling the prefetch conditionally.
"""
function async_prefetch(
    source::Value,
    indices::Vector{Value},
    predicate=nothing::Union{Nothing,Value};
    slice_lengths,
    collective,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("slice_lengths", slice_lengths),
        namedattribute("collective", collective),
    ]
    !isnothing(predicate) && push!(operands, predicate)
    push!(attributes, operandsegmentsizes([1, length(indices), Int(!isnothing(predicate))]))

    return create_operation(
        "mosaic_gpu.async_prefetch",
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
`async_store`

Schedules an async store of the contents of the `source` MemRef in SMEM to
the `destination` MemRef in GMEM. The `source` MemRef in SMEM must be
contiguous.

The `indices` and `slice_lengths` inputs define what slice of the GMEM
`destination` corresponds to the SMEM `source`. Both `indices` and
`slice_lengths` must have a length equal to the rank of the `destination`.
The values in `indices` are the starting indices of each dimension and the
values in `slice_lengths` are the lengths. Providing -1 in `slice_lengths`
indicates that this dimension is collapsed in the `source` and needs to be
expanded to a slice of size 1 in the `destination`.

The data is written in row-major order to the GMEM `destination`. The
`source` data in SMEM needs to be contiguous, but the `destination` GMEM
does not.

The `predicate` allows scheduling the transfer conditionally. The async copy
is always scheduled by at most a single lane in the warpgroup.

The `reduction_op` attribute can be provided to perform a reduction when
storing to GMEM. For example, using `add` will add the SMEM values to
existing values in GMEM.
"""
function async_store(
    source::Value,
    destination::Value,
    indices::Vector{Value},
    predicate=nothing::Union{Nothing,Value};
    slice_lengths,
    commit_group=nothing,
    reduction_op=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source, destination, indices...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("slice_lengths", slice_lengths),]
    !isnothing(predicate) && push!(operands, predicate)
    push!(
        attributes, operandsegmentsizes([1, 1, length(indices), Int(!isnothing(predicate))])
    )
    !isnothing(commit_group) &&
        push!(attributes, namedattribute("commit_group", commit_group))
    !isnothing(reduction_op) &&
        push!(attributes, namedattribute("reduction_op", reduction_op))

    return create_operation(
        "mosaic_gpu.async_store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function async_store_tmem(source::Value, destination::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[source, destination]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mosaic_gpu.async_store_tmem",
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
`broadcast_in_dim`

`broadcast_dimensions` must have the same size as the rank of the input
vector and for each input dimension, specifies which output dimension it
corresponds to.
"""
function broadcast_in_dim(
    operand::Value; result_0::IR.Type, broadcast_dimensions, location=Location()
)
    op_ty_results = IR.Type[result_0,]
    operands = Value[operand,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute(
        "broadcast_dimensions", broadcast_dimensions
    ),]

    return create_operation(
        "mosaic_gpu.broadcast_in_dim",
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
`broadcasted_iota`

Creates an array that has the specified shape and holds values starting at
zero and incrementing by one along the specified dimension.
"""
function broadcasted_iota(; result_0::IR.Type, dimension, location=Location())
    op_ty_results = IR.Type[result_0,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension),]

    return create_operation(
        "mosaic_gpu.broadcasted_iota",
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
`custom_primitive`

Allows defining a custom Mosaic GPU primitive.

Custom primitives should carry input and output layouts for each of their
vector operands and outputs, and input transforms for each of their memref
operands that live in SMEM.

Custom primitives can only return vectors.
"""
function custom_primitive(
    operands::Vector{Value};
    result_0::Vector{IR.Type},
    in_layouts,
    in_transforms,
    out_layouts,
    body::Region,
    location=Location(),
)
    op_ty_results = IR.Type[result_0...,]
    operands = Value[operands...,]
    owned_regions = Region[body,]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("in_layouts", in_layouts),
        namedattribute("in_transforms", in_transforms),
        namedattribute("out_layouts", out_layouts),
    ]

    return create_operation(
        "mosaic_gpu.custom_primitive",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function debug_print(value::Value; format, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("format", format),]

    return create_operation(
        "mosaic_gpu.debug_print",
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
`initialize_barrier`

Initializes `num_barriers` barriers each meant to synchronize exactly
`arrival_count` threads.

`base_pointer` must be a pointer to a shared memory location.
"""
function initialize_barrier(
    base_pointer::Value; arrival_count, num_barriers, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[base_pointer,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("arrival_count", arrival_count),
        namedattribute("num_barriers", num_barriers),
    ]

    return create_operation(
        "mosaic_gpu.initialize_barrier",
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
`layout_cast`
Casts a vector value to a new strided or tiled layout.
"""
function layout_cast(
    x::Value; result_0=nothing::Union{Nothing,IR.Type}, new_layout, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[x,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("new_layout", new_layout),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "mosaic_gpu.layout_cast",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function optimization_barrier(
    operands::Vector{Value};
    result_0=nothing::Union{Nothing,Vector{IR.Type}},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0...)

    return create_operation(
        "mosaic_gpu.optimization_barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

function print_layout(value::Value; format, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("format", format),]

    return create_operation(
        "mosaic_gpu.print_layout",
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
`return_`

The `return` op is a terminator that indicates the end of execution
within a `CustomPrimitiveOp`\'s region. It can optionally return some values,
which become the results of the parent `CustomPrimitiveOp`.

The declared results of the parent `CustomPrimitiveOp` must match the
operand types of this op.
"""
function return_(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mosaic_gpu.return",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function slice_smem(offset::Value; result_0::IR.Type, location=Location())
    op_ty_results = IR.Type[result_0,]
    operands = Value[offset,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mosaic_gpu.slice_smem",
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
`slice_tmem`

The principal use case for this op is to do a single TMEM allocation and
slice it into multiple smaller TMEM references. `source` is the large TMEM
allocation and `offset` is the number of columns to start slicing from.
"""
function slice_tmem(source::Value; result_0::IR.Type, offset, location=Location())
    op_ty_results = IR.Type[result_0,]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("offset", offset),]

    return create_operation(
        "mosaic_gpu.slice_tmem",
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
`tcgen05_mma`

Schedules `tcgen05.mma` instructions that perform the following matrix
multiply and accumulate:

  accumulator += a * b

This operation supports larger inputs than the PTX-level MMA instruction
and will schedule as many PTX-level MMA instructions as needed to
accomplish the calculation.

The inputs should have the following shapes:
  - a: [groups_m * m, groups_k * s]
  - b: [groups_k * s, groups_n * s]
  - accumulator: [groups_m * m, groups_n * s]
where `s == swizzle / element_bytewidth` and `m` is specified according to
https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape.

The `accumulator`, `a` and `b` matrices need to be provided as 2-dimensional
memrefs. The `accumulator` is always in TMEM and `b` is always in SMEM.
`a` can be in TMEM or SMEM. `a` and `b` must have the same element
type and when `a` is in TMEM only F16 or BF16 are supported.

`a_scale` and `b_scale` are optional scaling matrices that reside in TMEM.
When set the operation is defined as:

  accumulator += (a * a_scale) * (b * b_scale)

`accumulate` is a boolean that indicates whether to perform the accumulate
step.
"""
function tcgen05_mma(
    accumulator::Value,
    a::Value,
    b::Value,
    accumulate::Value,
    a_scale=nothing::Union{Nothing,Value};
    b_scale=nothing::Union{Nothing,Value},
    collective=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[accumulator, a, b, accumulate]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(a_scale) && push!(operands, a_scale)
    !isnothing(b_scale) && push!(operands, b_scale)
    push!(
        attributes,
        operandsegmentsizes([
            1, 1, 1, 1, Int(!isnothing(a_scale)), Int(!isnothing(b_scale))
        ]),
    )
    !isnothing(collective) && push!(attributes, namedattribute("collective", collective))

    return create_operation(
        "mosaic_gpu.tcgen05_mma",
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
`tmem_alloc`

This op allocates a chunk of TMEM and stores the pointer to the memory
in the provided SMEM memref.

The `smem_ptr` is a pointer in SMEM where a pointer to the allocated
TMEM will be stored. The op returns a memref to the allocated TMEM. The
result must have a shape with dimensions [rows, logical_columns]. If
`packing` is 1, then the number of logical (unpacked) columns is equal to
the number of allocated columns in TMEM. Otherwise, these constraints
must hold:

    packing = 32 / bitwidth(element type of result)
    unpacked_columns = allocated_columns * packing

The number of allocated columns in TMEM can be any power of two in the
range [32, 512]. If the calculated number of allocated columns is less than
32 or not a power of two, then it will be rounded up to the nearest power of
two larger or equal to 32.

If `collective` is `true` 2 CTAs will perform the allocation collectively,
otherwise, only one CTA will perform the allocation.
"""
function tmem_alloc(
    smem_ptr::Value;
    result_0::IR.Type,
    collective=nothing,
    packing=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[result_0,]
    operands = Value[smem_ptr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(collective) && push!(attributes, namedattribute("collective", collective))
    !isnothing(packing) && push!(attributes, namedattribute("packing", packing))

    return create_operation(
        "mosaic_gpu.tmem_alloc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tmem_dealloc(tmem_ref::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tmem_ref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mosaic_gpu.tmem_dealloc",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function tmem_layout_cast(
    ref::Value; result_0=nothing::Union{Nothing,IR.Type}, new_layout, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[ref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("new_layout", new_layout),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "mosaic_gpu.tmem_layout_cast",
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
`tmem_relinquish_alloc_permit`

The instruction specifies that the CTA of the executing thread is
relinquishing the right to allocate Tensor Memory. So, it is illegal for a
CTA to perform `tmem_alloc` after any of its constituent threads execute
`tmem_relinquish_alloc_permit`.

If `collective` is `true`, applies to collective TMEM allocations.
"""
function tmem_relinquish_alloc_permit(; collective=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(collective) && push!(attributes, namedattribute("collective", collective))

    return create_operation(
        "mosaic_gpu.tmem_relinquish_alloc_permit",
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
`vector_load`

Similar to `vector.load` (vector dialect) but supports loading from
non-contiguous memory.

If `optimized` is true, raises an error if we cannot generate an optimised
transfer. If unset, fall back to a non-optimized transfer if unable to
generate an optimized transfer.
"""
function vector_load(
    source::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    optimized=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[source,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)
    !isnothing(optimized) && push!(attributes, namedattribute("optimized", optimized))

    return create_operation(
        "mosaic_gpu.vector_load",
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
`vector_store`

Similar to `vector.store` (vector dialect) but supports storing to
non-contiguous memory.

If `optimized` is true, raises an error if we cannot generate an optimised
transfer. If unset, fall back to a non-optimized transfer if unable to
generate an optimized transfer.
"""
function vector_store(
    valueToStore::Value, destination::Value; optimized=nothing, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[valueToStore, destination]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(optimized) && push!(attributes, namedattribute("optimized", optimized))

    return create_operation(
        "mosaic_gpu.vector_store",
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
`wgmma`

Schedules WGMMA operations that perform the following matrix multiply and
accumulate:

  accumulator = a * b + accumulator

This operation supports larger inputs than the PTX-level WGMMA operation
and will schedule as many PTX-level WGMMA operations as needed to
accomplish the calculation. The `b` matrix, and optionally `a`, need to be
provided as a 2-dimensional memref.

The inputs should have the following shapes:
  - a: [groups_m * 64, groups_k * s]
  - b: [groups_k * s, groups_n * s]
  - accumulator: [groups_m * 64, groups_n * s]
where `s == swizzle / element_bytewidth`.

The output has an identical shape and type as the input accumulator.

The `accumulator` is always in registers and `b` is always in shared memory.
`a` and `b` must have the same element type and when `a` is in
registers only F16 or BF16 are supported.

The `accumulator` must be a vector with a FragmentedLayout. The WGMMA
operation will be executed in the async proxy and any inputs in
registers need to be synchronized with a memory fence.

Usually `a` is read from shared memory if it is used directly in the WGMMA
operation. If `a` needs to be transformed before it is used in the WGMMA
operation, it may be more convenient to read it directly form registers.
This avoids the need to store the data and wait for a fence.
"""
function wgmma(
    accumulator::Value,
    a::Value,
    b::Value;
    result_0=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[accumulator, a, b]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "mosaic_gpu.wgmma",
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
`wait`

All threads in the warpgroup will block, waiting on the provided barrier
until:
  - all pending threads have arrived on the barrier
  - all expected byte transfers have been completed
  - the barrier\'s parity matches the provided parity
"""
function wait(barrier::Value, parity::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[barrier, parity]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "mosaic_gpu.wait",
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
`with_transforms`

This op enforces the provided transforms on the parameter memref.
"""
function with_transforms(
    ref::Value; result_0=nothing::Union{Nothing,IR.Type}, transforms, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[ref,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("transforms", transforms),]
    !isnothing(result_0) && push!(op_ty_results, result_0)

    return create_operation(
        "mosaic_gpu.with_transforms",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false),
    )
end

end # mosaic_gpu
