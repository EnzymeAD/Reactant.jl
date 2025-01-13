module nvvm
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

function barrier0(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.barrier0",
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
`barrier_arrive`

Thread that executes this op announces their arrival at the barrier with 
given id and continue their execution.

The default barrier id is 0 that is similar to `nvvm.barrier` Op. When 
`barrierId` is not present, the default barrier id is used. 

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
"""
function barrier_arrive(
    barrierId=nothing::Union{Nothing,Value}; numberOfThreads::Value, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[numberOfThreads,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(barrierId) && push!(operands, barrierId)

    return create_operation(
        "nvvm.barrier.arrive",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function barrier(
    barrierId=nothing::Union{Nothing,Value};
    numberOfThreads=nothing::Union{Nothing,Value},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(barrierId) && push!(operands, barrierId)
    !isnothing(numberOfThreads) && push!(operands, numberOfThreads)
    push!(attributes, operandsegmentsizes([
        if (barrierId == nothing)
            0
        elseif 1(numberOfThreads == nothing)
            0
        else
            1
        end,
    ]))

    return create_operation(
        "nvvm.barrier",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_ntid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.ntid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_ntid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.ntid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_ntid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.ntid.z",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_ctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.ctaid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_ctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.ctaid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_ctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.ctaid.z",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_ctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.ctaid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_ctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.ctaid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_ctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.ctaid.z",
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
`breakpoint`

Breakpoint suspends execution of the program for debugging.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt)
"""
function breakpoint(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.breakpoint",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_clock64(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.clock64",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_clock(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.clock",
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
`cluster_arrive`

The `cluster.arrive` can be used by the threads within the cluster for synchronization and
communication. The `cluster.arrive` instruction marks the warps\' arrival at the barrier
without causing the executing thread to wait for other participating threads.

The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
"""
function cluster_arrive(; aligned=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned) && push!(attributes, namedattribute("aligned", aligned))

    return create_operation(
        "nvvm.cluster.arrive",
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
`cluster_arrive_relaxed`

The `cluster.arrive` can be used by the threads within the cluster for synchronization and
communication. The `cluster.arrive` instruction marks the warps\' arrival at the barrier
without causing the executing thread to wait for other participating threads.

The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.
The .relaxed qualifier on `cluster.arrive` specifies that there are no memory
ordering and visibility guarantees provided for the memory accesses performed prior to
`cluster.arrive`.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
"""
function cluster_arrive_relaxed(; aligned=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned) && push!(attributes, namedattribute("aligned", aligned))

    return create_operation(
        "nvvm.cluster.arrive.relaxed",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_nctarank(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.nctarank",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_nctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.nctaid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_nctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.nctaid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_nctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.nctaid.z",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nclusterid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nclusterid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nclusterid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nclusterid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nclusterid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nclusterid.z",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_cluster_ctarank(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.cluster.ctarank",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_clusterid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.clusterid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_clusterid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.clusterid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_clusterid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.clusterid.z",
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
`cluster_wait`

The `cluster.wait` causes the executing thread to wait for all non-exited threads
of the cluster to perform `cluster.arrive`. The `aligned` attribute, when provided,
generates the .aligned version of the PTX instruction.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
"""
function cluster_wait(; aligned=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned) && push!(attributes, namedattribute("aligned", aligned))

    return create_operation(
        "nvvm.cluster.wait",
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
`cp_async_bulk_commit_group`

This Op commits all prior initiated but uncommitted cp.async.bulk
instructions into a cp.async.bulk-group.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)
"""
function cp_async_bulk_commit_group(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.cp.async.bulk.commit.group",
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
`cp_async_bulk_tensor_shared_cluster_global`

Initiates an asynchronous copy operation on the tensor data from global 
memory to shared memory. 

The Op operates has two load modes:
1) Tiled Mode: It\'s the default mode. The source multi-dimensional tensor 
layout is preserved at the destination. 

2) Im2col Mode: This mode is used when `im2colOffsets` operands are present.
the elements in the Bounding Box of the source tensor are rearranged into
columns at the destination. In this mode, the tensor has to be at least 
3-dimensional. 

The `multicastMask` operand is optional. When it is present, the Op copies
data from global memory to shared memory of multiple CTAs in the cluster.
Operand `multicastMask` specifies the destination CTAs in the cluster such 
that each bit position in the 16-bit `multicastMask` operand corresponds to
the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.     

The `l2CacheHint` operand is optional, and it is used to specify cache 
eviction policy that may be used during the memory access.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
"""
function cp_async_bulk_tensor_shared_cluster_global(
    dstMem::Value,
    tmaDescriptor::Value,
    coordinates::Vector{Value},
    mbar::Value,
    im2colOffsets::Vector{Value},
    multicastMask=nothing::Union{Nothing,Value};
    l2CacheHint=nothing::Union{Nothing,Value},
    predicate=nothing::Union{Nothing,Value},
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[dstMem, tmaDescriptor, coordinates..., mbar, im2colOffsets...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(multicastMask) && push!(operands, multicastMask)
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    !isnothing(predicate) && push!(operands, predicate)
    push!(
        attributes,
        operandsegmentsizes([
            1,
            1,
            length(coordinates),
            1,
            length(im2colOffsets),
            if (multicastMask == nothing)
                0
            elseif 1(l2CacheHint == nothing)
                0
            elseif 1(predicate == nothing)
                0
            else
                1
            end,
        ]),
    )

    return create_operation(
        "nvvm.cp.async.bulk.tensor.shared.cluster.global",
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
`cp_async_bulk_tensor_prefetch`

Initiates an asynchronous prefetch operation on the tensor data from global
memory to L2 cache.

The Op has two modes:
1) Tiled Mode: It\'s the default mode. The source multi-dimensional tensor
layout is preserved at the destination.

2) Im2col Mode: This mode is used when `im2colOffsets` operands are present.
the elements in the Bounding Box of the source tensor are rearranged into
columns at the destination. In this mode, the tensor has to be at least
3-dimensional.

The `l2CacheHint` operand is optional, and it is used to specify cache
eviction policy that may be used during the memory access.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
"""
function cp_async_bulk_tensor_prefetch(
    tmaDescriptor::Value,
    coordinates::Vector{Value},
    im2colOffsets::Vector{Value},
    l2CacheHint=nothing::Union{Nothing,Value};
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, coordinates..., im2colOffsets...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    push!(
        attributes,
        operandsegmentsizes([
            1, length(coordinates), length(im2colOffsets), (l2CacheHint == nothing) ? 0 : 1
        ]),
    )

    return create_operation(
        "nvvm.cp.async.bulk.tensor.prefetch",
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
`cp_async_bulk_tensor_reduce`

Initiates an asynchronous reduction operation of tensor data in
global memory with tensor data in shared memory.

The `mode` attribute indicates whether the copy mode is tile or im2col.
The `redOp` attribute specifies the reduction operations applied.
The supported reduction operations are:
{add, min, max, inc, dec, and, or, xor}

The `l2CacheHint` operand is optional, and it is used to specify cache
eviction policy that may be used during the memory access.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)
"""
function cp_async_bulk_tensor_reduce(
    tmaDescriptor::Value,
    srcMem::Value,
    coordinates::Vector{Value},
    l2CacheHint=nothing::Union{Nothing,Value};
    redKind,
    mode=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, srcMem, coordinates...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("redKind", redKind),]
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    push!(
        attributes,
        operandsegmentsizes([1, 1, length(coordinates), (l2CacheHint == nothing) ? 0 : 1]),
    )
    !isnothing(mode) && push!(attributes, namedattribute("mode", mode))

    return create_operation(
        "nvvm.cp.async.bulk.tensor.reduce",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cp_async_bulk_tensor_global_shared_cta(
    tmaDescriptor::Value,
    srcMem::Value,
    coordinates::Vector{Value},
    predicate=nothing::Union{Nothing,Value};
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, srcMem, coordinates...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    push!(
        attributes,
        operandsegmentsizes([1, 1, length(coordinates), (predicate == nothing) ? 0 : 1]),
    )

    return create_operation(
        "nvvm.cp.async.bulk.tensor.global.shared.cta",
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
`cp_async_bulk_wait_group`

Op waits for completion of the most recent bulk async-groups.

The `\$group` operand tells waiting has to be done until for \$group or fewer
of the most recent bulk async-groups. If `\$group` is 0, the op wait until 
all the most recent bulk async-groups have completed.

The `\$read` indicates that the waiting has to be done until all the bulk 
async operations in the specified bulk async-group have completed reading 
from their source locations.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
"""
function cp_async_bulk_wait_group(; group, read=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("group", group),]
    !isnothing(read) && push!(attributes, namedattribute("read", read))

    return create_operation(
        "nvvm.cp.async.bulk.wait_group",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cp_async_commit_group(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.cp.async.commit.group",
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
`cp_async_mbarrier_arrive`

The `cp.async.mbarrier.arrive` Op makes the mbarrier object track
all prior cp.async operations initiated by the executing thread.
The `addr` operand specifies the address of the mbarrier object
in generic address space. The `noinc` attr impacts how the
mbarrier\'s state is updated.
[For more information, refer PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
"""
function cp_async_mbarrier_arrive(addr::Value; noinc=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(noinc) && push!(attributes, namedattribute("noinc", noinc))

    return create_operation(
        "nvvm.cp.async.mbarrier.arrive",
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
`cp_async_mbarrier_arrive_shared`

The `cp.async.mbarrier.arrive.shared` Op makes the mbarrier object
track all prior cp.async operations initiated by the executing thread.
The `addr` operand specifies the address of the mbarrier object in
shared memory. The `noinc` attr impacts how the mbarrier\'s state
is updated. [For more information, refer PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
"""
function cp_async_mbarrier_arrive_shared(addr::Value; noinc=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(noinc) && push!(attributes, namedattribute("noinc", noinc))

    return create_operation(
        "nvvm.cp.async.mbarrier.arrive.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cp_async_shared_global(
    dst::Value,
    src::Value,
    cpSize=nothing::Union{Nothing,Value};
    size,
    modifier,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[dst, src]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("size", size), namedattribute("modifier", modifier)
    ]
    !isnothing(cpSize) && push!(operands, cpSize)

    return create_operation(
        "nvvm.cp.async.shared.global",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function cp_async_wait_group(; n, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("n", n),]

    return create_operation(
        "nvvm.cp.async.wait.group",
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
`elect_sync`

The `elect.sync` instruction elects one predicated active leader
thread from among a set of threads specified in membermask.
The membermask is set to `0xFFFFFFFF` for the current version
of this Op. The predicate result is set to `True` for the
leader thread, and `False` for all other threads.

[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)
"""
function elect_sync(; pred::IR.Type, location=Location())
    op_ty_results = IR.Type[pred,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.elect.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg0(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg0",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg1(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg1",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg2(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg2",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg3(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg3",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg4(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg4",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg5(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg5",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg6(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg6",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg7(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg7",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg8(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg8",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg9(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg9",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg10(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg10",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg11(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg11",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg12(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg12",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg13(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg13",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg14(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg14",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg15(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg15",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg16(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg16",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg17(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg17",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg18(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg18",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg19(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg19",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg20(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg20",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg21(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg21",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg22(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg22",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg23(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg23",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg24(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg24",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg25(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg25",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg26(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg26",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg27(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg27",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg28(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg28",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg29(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg29",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg30(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg30",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_envreg31(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.envreg31",
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
`exit`

Ends execution of a thread.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit)
"""
function exit(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.exit",
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
`fence_mbarrier_init`

Fence operation that applies on the prior nvvm.mbarrier.init
[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_mbarrier_init(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.fence.mbarrier.init",
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
`fence_proxy_acquire`

`fence.proxy.acquire` is a uni-directional fence used to establish ordering
between a prior memory access performed via the generic proxy and a
subsequent memory access performed via the tensormap proxy

The address operand `addr` and the operand `size` together specify the
memory range `[addr, addr+size)` on which the ordering guarantees on the
memory accesses across the proxies is to be provided. The only supported
value for the `size` operand is 128 and must be an immediate. Generic Addressing
is used unconditionally, and the address specified by the operand `addr` must
fall within the `.global` state space. Otherwise, the behavior is undefined
[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_proxy_acquire(
    addr::Value, size::Value; scope, fromProxy=nothing, toProxy=nothing, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[addr, size]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("scope", scope),]
    !isnothing(fromProxy) && push!(attributes, namedattribute("fromProxy", fromProxy))
    !isnothing(toProxy) && push!(attributes, namedattribute("toProxy", toProxy))

    return create_operation(
        "nvvm.fence.proxy.acquire",
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
`fence_proxy`

Fence operation with proxy to establish an ordering between memory accesses
that may happen through different proxies.
[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_proxy(; kind, space=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind),]
    !isnothing(space) && push!(attributes, namedattribute("space", space))

    return create_operation(
        "nvvm.fence.proxy",
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
`fence_proxy_release`

`fence.proxy.release` is a uni-directional fence used to establish ordering
between a prior memory access performed via the generic proxy and a
subsequent memory access performed via the tensormap proxy. `fence.proxy.release`
operation can form a release sequence that synchronizes with an acquire
sequence that contains the fence.proxy.acquire proxy fence operation
[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_proxy_release(;
    scope, fromProxy=nothing, toProxy=nothing, location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("scope", scope),]
    !isnothing(fromProxy) && push!(attributes, namedattribute("fromProxy", fromProxy))
    !isnothing(toProxy) && push!(attributes, namedattribute("toProxy", toProxy))

    return create_operation(
        "nvvm.fence.proxy.release",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function fence_sc_cluster(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.fence.sc.cluster",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_globaltimer(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.globaltimer",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nctaid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nctaid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nctaid.z",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_gridid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.gridid",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_laneid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.laneid",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_lanemask_eq(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.lanemask.eq",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_lanemask_ge(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.lanemask.ge",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_lanemask_gt(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.lanemask.gt",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_lanemask_le(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.lanemask.le",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_lanemask_lt(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.read.ptx.sreg.lanemask.lt",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function ldmatrix(ptr::Value; res::IR.Type, num, layout, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[ptr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("num", num), namedattribute("layout", layout)
    ]

    return create_operation(
        "nvvm.ldmatrix",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_arrive_expect_tx(
    addr::Value,
    txcount::Value,
    predicate=nothing::Union{Nothing,Value};
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[addr, txcount]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)

    return create_operation(
        "nvvm.mbarrier.arrive.expect_tx",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_arrive_expect_tx_shared(
    addr::Value,
    txcount::Value,
    predicate=nothing::Union{Nothing,Value};
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[addr, txcount]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)

    return create_operation(
        "nvvm.mbarrier.arrive.expect_tx.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_arrive_nocomplete(
    addr::Value, count::Value; res::IR.Type, location=Location()
)
    op_ty_results = IR.Type[res,]
    operands = Value[addr, count]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.arrive.nocomplete",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_arrive_nocomplete_shared(
    addr::Value, count::Value; res::IR.Type, location=Location()
)
    op_ty_results = IR.Type[res,]
    operands = Value[addr, count]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.arrive.nocomplete.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_arrive(addr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.arrive",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_arrive_shared(addr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.arrive.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_init(
    addr::Value, count::Value, predicate=nothing::Union{Nothing,Value}; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[addr, count]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)

    return create_operation(
        "nvvm.mbarrier.init",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_init_shared(
    addr::Value, count::Value, predicate=nothing::Union{Nothing,Value}; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[addr, count]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)

    return create_operation(
        "nvvm.mbarrier.init.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_inval(addr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.inval",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_inval_shared(addr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.inval.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_test_wait(addr::Value, state::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[addr, state]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.test.wait",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_test_wait_shared(
    addr::Value, state::Value; res::IR.Type, location=Location()
)
    op_ty_results = IR.Type[res,]
    operands = Value[addr, state]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.test.wait.shared",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_try_wait_parity(
    addr::Value, phase::Value, ticks::Value; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[addr, phase, ticks]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.try_wait.parity",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function mbarrier_try_wait_parity_shared(
    addr::Value, phase::Value, ticks::Value; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[addr, phase, ticks]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.mbarrier.try_wait.parity.shared",
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
`mma_sync`

The `nvvm.mma.sync` operation collectively performs the operation
`D = matmul(A, B) + C` using all threads in a warp.

All the threads in the warp must execute the same `mma.sync` operation.

For each possible multiplicand PTX data type, there are one or more possible
instruction shapes given as \"mMnNkK\". The below table describes the posssibilities
as well as the types required for the operands. Note that the data type for
C (the accumulator) and D (the result) can vary independently when there are
multiple possibilities in the \"C/D Type\" column.

When an optional attribute cannot be immediately inferred from the types of
the operands and the result during parsing or validation, an error will be
raised.

`b1Op` is only relevant when the binary (b1) type is given to
`multiplicandDataType`. It specifies how the multiply-and-acumulate is
performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

`intOverflowBehavior` is only relevant when the `multiplicandType` attribute
is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
in the accumulator. When the attribute is `satfinite`, the accumulator values
are clamped in the int32 range on overflow. This is the default behavior.
Alternatively, accumulator behavior `wrapped` can also be specified, in
which case overflow wraps from one end of the range to the other.

`layoutA` and `layoutB` are required and should generally be set to
`#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
combinations are possible for certain layouts according to the table below.

```
| A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
|----------|-----------|---------|---------|----------|----------|-------------------|
| f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
| f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
|          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| bf16     | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
|          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
| u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
```


# Example
```mlir

%128 = nvvm.mma.sync A[%120, %121, %122, %123]
                     B[%124, %125]
                     C[%126, %127]
                     {layoutA = #nvvm.mma_layout<row>,
                      layoutB = #nvvm.mma_layout<col>,
                      shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
       -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
```
"""
function mma_sync(
    operandA::Vector{Value},
    operandB::Vector{Value},
    operandC::Vector{Value};
    res::IR.Type,
    shape,
    b1Op=nothing,
    intOverflowBehavior=nothing,
    layoutA,
    layoutB,
    multiplicandAPtxType=nothing,
    multiplicandBPtxType=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[operandA..., operandB..., operandC...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("shape", shape),
        namedattribute("layoutA", layoutA),
        namedattribute("layoutB", layoutB),
    ]
    push!(
        attributes,
        operandsegmentsizes([length(operandA), length(operandB), length(operandC)]),
    )
    !isnothing(b1Op) && push!(attributes, namedattribute("b1Op", b1Op))
    !isnothing(intOverflowBehavior) &&
        push!(attributes, namedattribute("intOverflowBehavior", intOverflowBehavior))
    !isnothing(multiplicandAPtxType) &&
        push!(attributes, namedattribute("multiplicandAPtxType", multiplicandAPtxType))
    !isnothing(multiplicandBPtxType) &&
        push!(attributes, namedattribute("multiplicandBPtxType", multiplicandBPtxType))

    return create_operation(
        "nvvm.mma.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function prefetch_tensormap(
    tmaDescriptor::Value, predicate=nothing::Union{Nothing,Value}; location=Location()
)
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)

    return create_operation(
        "nvvm.prefetch.tensormap",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function rcp_approx_ftz_f(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[arg,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.rcp.approx.ftz.f",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function redux_sync(
    val::Value, mask_and_clamp::Value; res::IR.Type, kind, location=Location()
)
    op_ty_results = IR.Type[res,]
    operands = Value[val, mask_and_clamp]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind),]

    return create_operation(
        "nvvm.redux.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function setmaxregister(; regCount, action, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("regCount", regCount), namedattribute("action", action)
    ]

    return create_operation(
        "nvvm.setmaxregister",
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
`shfl_sync`

The `shfl.sync` Op implements data shuffle within threads of a warp.
The `thread_mask` denotes the threads participating in the Op where
the bit position corresponds to a particular thread’s laneid.
The `offset` specifies a source lane or source lane offset
(depending on `kind`). The `val` is the input value to be copied from
the source. The `mask_and_clamp` contains two packed values specifying
a mask for logically splitting warps into sub-segments and an upper bound
for clamping the source lane index.
[For more information, refer PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync)
"""
function shfl_sync(
    thread_mask::Value,
    val::Value,
    offset::Value,
    mask_and_clamp::Value;
    res::IR.Type,
    kind,
    return_value_and_is_valid=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[thread_mask, val, offset, mask_and_clamp]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind),]
    !isnothing(return_value_and_is_valid) && push!(
        attributes,
        namedattribute("return_value_and_is_valid", return_value_and_is_valid),
    )

    return create_operation(
        "nvvm.shfl.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nsmid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nsmid",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_smid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.smid",
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
`stmatrix`

Collectively store one or more matrices across all threads in a warp to the
location indicated by the address operand \$ptr in shared memory.
[For more information, see PTX ISA]
(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)
"""
function stmatrix(ptr::Value, sources::Vector{Value}; layout, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, sources...]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("layout", layout),]

    return create_operation(
        "nvvm.stmatrix",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function bar_warp_sync(mask::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[mask,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.bar.warp.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_tid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.tid.x",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_tid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.tid.y",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_tid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.tid.z",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function vote_ballot_sync(mask::Value, pred::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[mask, pred]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.vote.ballot.sync",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function wmma_load(
    ptr::Value,
    stride::Value;
    res::IR.Type,
    m,
    n,
    k,
    layout,
    eltype,
    frag,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[ptr, stride]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("m", m),
        namedattribute("n", n),
        namedattribute("k", k),
        namedattribute("layout", layout),
        namedattribute("eltype", eltype),
        namedattribute("frag", frag),
    ]

    return create_operation(
        "nvvm.wmma.load",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function wmma_mma(
    args::Vector{Value};
    res::IR.Type,
    m,
    n,
    k,
    layoutA,
    layoutB,
    eltypeA,
    eltypeB,
    location=Location(),
)
    op_ty_results = IR.Type[res,]
    operands = Value[args...,]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("m", m),
        namedattribute("n", n),
        namedattribute("k", k),
        namedattribute("layoutA", layoutA),
        namedattribute("layoutB", layoutB),
        namedattribute("eltypeA", eltypeA),
        namedattribute("eltypeB", eltypeB),
    ]

    return create_operation(
        "nvvm.wmma.mma",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function wmma_store(
    ptr::Value,
    args::Vector{Value},
    stride::Value;
    m,
    n,
    k,
    layout,
    eltype,
    location=Location(),
)
    op_ty_results = IR.Type[]
    operands = Value[ptr, args..., stride]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("m", m),
        namedattribute("n", n),
        namedattribute("k", k),
        namedattribute("layout", layout),
        namedattribute("eltype", eltype),
    ]

    return create_operation(
        "nvvm.wmma.store",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_nwarpid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.nwarpid",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_warpid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.warpid",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

function read_ptx_sreg_warpsize(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res,]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, namedattribute("range", range))

    return create_operation(
        "nvvm.read.ptx.sreg.warpsize",
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
`wgmma_fence_aligned`

Enforce an ordering of register accesses between warpgroup level matrix 
multiplication and other operations. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
"""
function wgmma_fence_aligned(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.wgmma.fence.aligned",
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
`wgmma_commit_group_sync_aligned`

Commits all prior uncommitted warpgroup level matrix multiplication operations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)
"""
function wgmma_commit_group_sync_aligned(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]

    return create_operation(
        "nvvm.wgmma.commit.group.sync.aligned",
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
`wgmma_mma_async`

The warpgroup (128 threads) level matrix multiply and accumulate operation 
has either of the following forms, where matrix D is called accumulator:
  D = A * B + D
  D = A * B, where the input from accumulator D is disabled.

Supported shapes:  
```
|--------------|--------------|------------|--------------|---------------|
|              |              |            |              |f16+=e4m3*e4m3 |
|              |              |            |              |f16+=e5m2*e5m2 |
|f32+=tf32*tf32|f16+=f16 *f16 | s32+=s8*s8 |s32 += b1 * b1|f16+=e5m2*e4m3 |
|              |f32+=f16 *f16 | s32+=u8*u8 |              |f16+=e4m3*e5m2 |
|              |f32+=bf16*bf16| s32+=u8*u8 |              |f16+=e4m3*e5m2 |
|              |f32+=bf16*bf16| s32+=s8*u8 |              |f32+=e4m3*e4m3 |
|              |              | s32+=u8*s8 |              |f32+=e5m2*e5m2 |
|              |              |            |              |f32+=e4m3*e5m2 |
|              |              |            |              |f32+=e4m3*e5m2 |
|--------------|--------------|------------|--------------|---------------|
|   .m64n8k8   |  .m64n8k16   | .m64n8k32  | .m64n8k256   | .m64n8k32     |
|   .m64n16k8  |  .m64n16k16  | .m64n16k32 | .m64n16k256  | .m64n16k32    |
|   .m64n24k8  |  .m64n24k16  | .m64n24k32 | .m64n24k256  | .m64n24k32    |
|   .m64n32k8  |  .m64n32k16  | .m64n32k32 | .m64n32k256  | .m64n32k32    |
|   .m64n40k8  |  .m64n40k16  | .m64n48k32 | .m64n48k256  | .m64n40k32    |
|   .m64n48k8  |  .m64n48k16  | .m64n64k32 | .m64n64k256  | .m64n48k32    |
|   .m64n56k8  |  .m64n56k16  | .m64n80k32 | .m64n80k256  | .m64n56k32    |
|   .m64n64k8  |  .m64n64k16  | .m64n96k32 | .m64n96k256  | .m64n64k32    |
|   .m64n72k8  |  .m64n72k16  | .m64n112k32| .m64n112k256 | .m64n72k32    |
|   .m64n80k8  |  .m64n80k16  | .m64n128k32| .m64n128k256 | .m64n80k32    |
|   .m64n88k8  |  .m64n88k16  | .m64n144k32| .m64n144k256 | .m64n88k32    |
|   .m64n96k8  |  .m64n96k16  | .m64n160k32| .m64n160k256 | .m64n96k32    |
|   .m64n104k8 |  .m64n104k16 | .m64n176k32| .m64n176k256 | .m64n104k32   |
|   .m64n112k8 |  .m64n112k16 | .m64n192k32| .m64n192k256 | .m64n112k32   |
|   .m64n120k8 |  .m64n120k16 | .m64n208k32| .m64n208k256 | .m64n120k32   |
|   .m64n128k8 |  .m64n128k16 | .m64n224k32| .m64n224k256 | .m64n128k32   |
|   .m64n136k8 |  .m64n136k16 | .m64n240k32| .m64n240k256 | .m64n136k32   |
|   .m64n144k8 |  .m64n144k16 | .m64n256k32| .m64n256k256 | .m64n144k32   |
|   .m64n152k8 |  .m64n152k16 |            |              | .m64n152k32   |
|   .m64n160k8 |  .m64n160k16 |            |              | .m64n160k32   |
|   .m64n168k8 |  .m64n168k16 |            |              | .m64n168k32   |
|   .m64n176k8 |  .m64n176k16 |            |              | .m64n176k32   |
|   .m64n184k8 |  .m64n184k16 |            |              | .m64n184k32   |
|   .m64n192k8 |  .m64n192k16 |            |              | .m64n192k32   |
|   .m64n200k8 |  .m64n200k16 |            |              | .m64n200k32   |
|   .m64n208k8 |  .m64n208k16 |            |              | .m64n208k32   |
|   .m64n216k8 |  .m64n216k16 |            |              | .m64n216k32   |
|   .m64n224k8 |  .m64n224k16 |            |              | .m64n224k32   |
|   .m64n232k8 |  .m64n232k16 |            |              | .m64n232k32   |
|   .m64n240k8 |  .m64n240k16 |            |              | .m64n240k32   |
|   .m64n248k8 |  .m64n248k16 |            |              | .m64n248k32   |
|   .m64n256k8 |  .m64n256k16 |            |              | .m64n256k32   |
|--------------|--------------|------------|--------------|---------------|
```


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)
"""
function wgmma_mma_async(
    inouts::Value,
    descriptorA::Value,
    descriptorB::Value;
    results::IR.Type,
    shape,
    typeA,
    typeB,
    typeD,
    scaleD,
    scaleA,
    scaleB,
    layoutA,
    layoutB,
    satfinite=nothing,
    location=Location(),
)
    op_ty_results = IR.Type[results,]
    operands = Value[inouts, descriptorA, descriptorB]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[
        namedattribute("shape", shape),
        namedattribute("typeA", typeA),
        namedattribute("typeB", typeB),
        namedattribute("typeD", typeD),
        namedattribute("scaleD", scaleD),
        namedattribute("scaleA", scaleA),
        namedattribute("scaleB", scaleB),
        namedattribute("layoutA", layoutA),
        namedattribute("layoutB", layoutB),
    ]
    !isnothing(satfinite) && push!(attributes, namedattribute("satfinite", satfinite))

    return create_operation(
        "nvvm.wgmma.mma_async",
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
`wgmma_wait_group_sync_aligned`

Signal the completion of a preceding warpgroup operation.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
"""
function wgmma_wait_group_sync_aligned(; group, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("group", group),]

    return create_operation(
        "nvvm.wgmma.wait.group.sync.aligned",
        location;
        operands,
        owned_regions,
        successors,
        attributes,
        results=op_ty_results,
        result_inference=false,
    )
end

end # nvvm
