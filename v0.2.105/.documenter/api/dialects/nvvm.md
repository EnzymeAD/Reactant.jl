


# NVVM Dialect {#NVVM-Dialect}

Refer to the [official documentation](https://mlir.llvm.org/docs/Dialects/NVVMDialect/) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.barrier_arrive' href='#Reactant.MLIR.Dialects.nvvm.barrier_arrive'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.barrier_arrive</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`barrier_arrive`

Thread that executes this op announces their arrival at the barrier with  given id and continue their execution.

The default barrier id is 0 that is similar to `nvvm.barrier` Op. When  `barrierId` is not present, the default barrier id is used. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L35-L45" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.breakpoint-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.breakpoint-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.breakpoint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`breakpoint`

Breakpoint suspends execution of the program for debugging. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L282-L287" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cluster_arrive-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.cluster_arrive-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cluster_arrive</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cluster_arrive`

The `cluster.arrive` can be used by the threads within the cluster for synchronization and communication. The `cluster.arrive` instruction marks the warps&#39; arrival at the barrier without causing the executing thread to wait for other participating threads.

The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L376-L386" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cluster_arrive_relaxed-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.cluster_arrive_relaxed-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cluster_arrive_relaxed</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cluster_arrive_relaxed`

The `cluster.arrive` can be used by the threads within the cluster for synchronization and communication. The `cluster.arrive` instruction marks the warps&#39; arrival at the barrier without causing the executing thread to wait for other participating threads.

The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction. The .relaxed qualifier on `cluster.arrive` specifies that there are no memory ordering and visibility guarantees provided for the memory accesses performed prior to `cluster.arrive`.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L407-L420" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cluster_wait-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.cluster_wait-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cluster_wait</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cluster_wait`

The `cluster.wait` causes the executing thread to wait for all non-exited threads of the cluster to perform `cluster.arrive`. The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L661-L669" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_commit_group-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_commit_group-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_commit_group</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cp_async_bulk_commit_group`

This Op commits all prior initiated but uncommitted cp.async.bulk instructions into a cp.async.bulk-group.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L690-L697" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_global_shared_cta' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_global_shared_cta'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_global_shared_cta</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`cp_async_bulk_global_shared_cta`

Initiates an asynchronous copy operation from Shared CTA memory to global memory.

The `l2CacheHint` operand is optional, and it is used to specify cache eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L776-L786" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_global' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_global'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_global</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`cp_async_bulk_shared_cluster_global`

Initiates an asynchronous copy operation from global memory to cluster&#39;s shared memory.

The `multicastMask` operand is optional. When it is present, the Op copies data from global memory to shared memory of multiple CTAs in the cluster. Operand `multicastMask` specifies the destination CTAs in the cluster such that each bit position in the 16-bit `multicastMask` operand corresponds to the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.

The `l2CacheHint` operand is optional, and it is used to specify cache eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L717-L733" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_shared_cta-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_shared_cta-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_shared_cta</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cp_async_bulk_shared_cluster_shared_cta`

Initiates an asynchronous copy operation from Shared CTA memory to Shared cluster memory.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L813-L820" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_prefetch' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_prefetch'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_prefetch</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`cp_async_bulk_tensor_prefetch`

Initiates an asynchronous prefetch operation on the tensor data from global memory to L2 cache.

The Op has two modes:
1. Tiled Mode: It&#39;s the default mode. The source multi-dimensional tensor
  

layout is preserved at the destination.
1. Im2col Mode: This mode is used when `im2colOffsets` operands are present.
  

the elements in the Bounding Box of the source tensor are rearranged into columns at the destination. In this mode, the tensor has to be at least 3-dimensional.

The `l2CacheHint` operand is optional, and it is used to specify cache eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L919-L938" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_reduce' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_reduce'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_reduce</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`cp_async_bulk_tensor_reduce`

Initiates an asynchronous reduction operation of tensor data in global memory with tensor data in shared memory.

The `mode` attribute indicates whether the copy mode is tile or im2col. The `redOp` attribute specifies the reduction operations applied. The supported reduction operations are: {add, min, max, inc, dec, and, or, xor}

The `l2CacheHint` operand is optional, and it is used to specify cache eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L971-L986" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_shared_cluster_global' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_shared_cluster_global'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_shared_cluster_global</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`cp_async_bulk_tensor_shared_cluster_global`

Initiates an asynchronous copy operation on the tensor data from global  memory to shared memory. 

The Op operates has two load modes:
1. Tiled Mode: It&#39;s the default mode. The source multi-dimensional tensor 
  

layout is preserved at the destination. 
1. Im2col Mode: This mode is used when `im2colOffsets` operands are present.
  

the elements in the Bounding Box of the source tensor are rearranged into columns at the destination. In this mode, the tensor has to be at least  3-dimensional. 

The `multicastMask` operand is optional. When it is present, the Op copies data from global memory to shared memory of multiple CTAs in the cluster. Operand `multicastMask` specifies the destination CTAs in the cluster such  that each bit position in the 16-bit `multicastMask` operand corresponds to the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.     

The `l2CacheHint` operand is optional, and it is used to specify cache  eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L842-L867" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_bulk_wait_group-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_wait_group-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_bulk_wait_group</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cp_async_bulk_wait_group`

Op waits for completion of the most recent bulk async-groups.

The `$group` operand tells waiting has to be done until for $group or fewer of the most recent bulk async-groups. If `$group` is 0, the op wait until  all the most recent bulk async-groups have completed.

The `$read` indicates that the waiting has to be done until all the bulk  async operations in the specified bulk async-group have completed reading  from their source locations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1050-L1064" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cp_async_mbarrier_arrive`

The `cp.async.mbarrier.arrive` Op makes the mbarrier object track all prior cp.async operations initiated by the executing thread. The `addr` operand specifies the address of the mbarrier object in generic address space. The `noinc` attr impacts how the mbarrier&#39;s state is updated.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1104-L1114" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive_shared-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive_shared-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive_shared</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cp_async_mbarrier_arrive_shared`

The `cp.async.mbarrier.arrive.shared` Op makes the mbarrier object track all prior cp.async operations initiated by the executing thread. The `addr` operand specifies the address of the mbarrier object in shared memory. The `noinc` attr impacts how the mbarrier&#39;s state is updated. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1135-L1145" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cvt_float_to_tf32-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.cvt_float_to_tf32-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cvt_float_to_tf32</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cvt_float_to_tf32`

This Op converts the given f32 input to tf32. The result `res` is represented as an i32 type. The `relu` attribute, when set, lowers to the &#39;.relu&#39; variant of the cvt instruction. The `rnd` and `sat` attributes specify the the rounding and saturation modes respectively.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1214-L1224" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.cvt_to_f6x2-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.cvt_to_f6x2-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.cvt_to_f6x2</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`cvt_to_f6x2`

This Op converts each of the given float inputs to the specified fp6 type. The result `dst` is represented either as an i16 type or as a vector of two i8 types. If `dst` is returned as an i16 type, the converted values are packed such  that the value converted from `a` is stored in the upper 8 bits of `dst`  with 2 MSB bits padded with zeros and the value converted from `b` is  stored in the lower 8 bits of `dst` with 2 MSB bits padded with zeros. If `dst` is returned as a vector type, each converted value is stored as an  i8 element in the vector. The `relu` attribute, when set, lowers to the &#39;.relu&#39; variant of the cvt instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1249-L1265" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.elect_sync-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.elect_sync-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.elect_sync</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`elect_sync`

The `elect.sync` instruction elects one predicated active leader thread from among a set of threads specified in membermask. The membermask is set to `0xFFFFFFFF` for the current version of this Op. The predicate result is set to `True` for the leader thread, and `False` for all other threads.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1288-L1298" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.exit-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.exit-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.exit</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`exit`

Ends execution of a thread. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1926-L1931" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.fence_mbarrier_init-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.fence_mbarrier_init-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.fence_mbarrier_init</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fence_mbarrier_init`

Fence operation that applies on the prior nvvm.mbarrier.init

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1951-L1957" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.fence_proxy-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.fence_proxy-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.fence_proxy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fence_proxy`

Fence operation with proxy to establish an ordering between memory accesses that may happen through different proxies.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2016-L2023" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.fence_proxy_acquire-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.fence_proxy_acquire-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.fence_proxy_acquire</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fence_proxy_acquire`

`fence.proxy.acquire` is a uni-directional fence used to establish ordering between a prior memory access performed via the generic proxy and a subsequent memory access performed via the tensormap proxy

The address operand `addr` and the operand `size` together specify the memory range `[addr, addr+size)` on which the ordering guarantees on the memory accesses across the proxies is to be provided. The only supported value for the `size` operand is 128 and must be an immediate. Generic Addressing is used unconditionally, and the address specified by the operand `addr` must fall within the `.global` state space. Otherwise, the behavior is undefined

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L1977-L1992" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.fence_proxy_release-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.fence_proxy_release-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.fence_proxy_release</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fence_proxy_release`

`fence.proxy.release` is a uni-directional fence used to establish ordering between a prior memory access performed via the generic proxy and a subsequent memory access performed via the tensormap proxy. `fence.proxy.release` operation can form a release sequence that synchronizes with an acquire sequence that contains the fence.proxy.acquire proxy fence operation

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2044-L2054" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.griddepcontrol_launch_dependents-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.griddepcontrol_launch_dependents-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.griddepcontrol_launch_dependents</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`griddepcontrol_launch_dependents`

Signals that specific dependents the runtime system designated to react to  this instruction can be scheduled as soon as all other CTAs in the grid  issue the same instruction or have completed.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2196-L2205" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.griddepcontrol_wait-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.griddepcontrol_wait-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.griddepcontrol_wait</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`griddepcontrol_wait`

Causes the executing thread to wait until all prerequisite grids in flight  have completed and all the memory operations from the prerequisite grids  are performed and made visible to the current grid.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2225-L2234" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.match_sync-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.match_sync-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.match_sync</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`match_sync`

The `match.sync` op performs broadcast and compare of operand `val` across  all non-exited threads in `thread_mask` and returns a mask depending on the  kind and an optional predicate.

The matching operation kinds are:
- `any`: Returns a mask corresponding to the non-exited threads in the 
  

`thread_mask` that have the same value of operand `val`.
- `all`: Returns a mask and a predicate. If all non-exited threads in the 
  

`thread_mask` have the same value of operand `val`, the predicate is set to  true and the mask corresponds to the non-exited threads in the  `thread_mask`. Otherwise, the predicate is set to false and the mask is 0.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2703-L2719" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.mma_sync-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.nvvm.mma_sync-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.mma_sync</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mma_sync`

The `nvvm.mma.sync` operation collectively performs the operation `D = matmul(A, B) + C` using all threads in a warp.

All the threads in the warp must execute the same `mma.sync` operation.

For each possible multiplicand PTX data type, there are one or more possible instruction shapes given as &quot;mMnNkK&quot;. The below table describes the posssibilities as well as the types required for the operands. Note that the data type for C (the accumulator) and D (the result) can vary independently when there are multiple possibilities in the &quot;C/D Type&quot; column.

When an optional attribute cannot be immediately inferred from the types of the operands and the result during parsing or validation, an error will be raised.

`b1Op` is only relevant when the binary (b1) type is given to `multiplicandDataType`. It specifies how the multiply-and-acumulate is performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

`intOverflowBehavior` is only relevant when the `multiplicandType` attribute is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled in the accumulator. When the attribute is `satfinite`, the accumulator values are clamped in the int32 range on overflow. This is the default behavior. Alternatively, accumulator behavior `wrapped` can also be specified, in which case overflow wraps from one end of the range to the other.

`layoutA` and `layoutB` are required and should generally be set to `#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other combinations are possible for certain layouts according to the table below.

```
| A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
|----------|-----------|---------|---------|----------|----------|-------------------|
| f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
| f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
|          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| bf16     | .m16n8k8  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
|          | .m16n8k16 | row     | col     | 4x i32   | 2x i32   | 4x f32            |
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


**Example**

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



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2739-L2806" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.redux_sync-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.redux_sync-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.redux_sync</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`redux_sync`

`redux.sync` performs a reduction operation `kind` of the 32 bit source  register across all non-exited threads in the membermask.

The `abs` and `nan` attributes can be used in the case of f32 input type,  where the `abs` attribute causes the absolute value of the input to be used  in the reduction operation, and the `nan` attribute causes the reduction  operation to return NaN if any of the inputs to participating threads are  NaN.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2895-L2908" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.shfl_sync-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.shfl_sync-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.shfl_sync</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`shfl_sync`

The `shfl.sync` Op implements data shuffle within threads of a warp. The `thread_mask` denotes the threads participating in the Op where the bit position corresponds to a particular thread’s laneid. The `offset` specifies a source lane or source lane offset (depending on `kind`). The `val` is the input value to be copied from the source. The `mask_and_clamp` contains two packed values specifying a mask for logically splitting warps into sub-segments and an upper bound for clamping the source lane index.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L2959-L2972" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.st_bulk-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.st_bulk-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.st_bulk</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`st_bulk`

Initializes a region of shared memory at the address given by `addr`. The `size` operand specifies the number of bytes to initialize and must be  a multiple of 8. The `initVal` operand specifies the value to initialize the memory to. The  only supported value is 0.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L307-L317" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.stmatrix-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.nvvm.stmatrix-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.stmatrix</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`stmatrix`

Collectively store one or more matrices across all threads in a warp to the location indicated by the address operand ptr in shared memory.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3045-L3052" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_alloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_alloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_alloc</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_alloc`

The `tcgen05.alloc` Op allocates tensor core memory for the amount specified by `nCols` and writes the destination address to the `addr` argument. The `nCols` operand specifies the number of columns to be allocated and it must be a power-of-two. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3091-L3099" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_commit' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_commit'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_commit</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`tcgen05_commit`

The `tcgen05.commit` makes the mbarrier object, specified by the operand `addr`, track the completion of all the prior async-tcgen05 operations initiated by the executing thread. The multicast variants allow signaling on the mbarrier objects of multiple CTAs within the cluster. Operand `multicastMask`, when present, specifies the destination CTAs in the cluster such that each bit position in the 16-bit `multicastMask` operand corresponds to the `nvvm.read.ptx.sreg.ctaid` of the destination CTA. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3120-L3132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_cp-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_cp-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_cp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_cp`

Instruction tcgen05.cp initiates an asynchronous copy operation from shared memory to the location specified by the address operand `taddr` in the Tensor Memory. The 64-bit register operand `smem_desc` specifies the matrix descriptor representing the source matrix in the shared memory that needs to be copied.

**Example**

```mlir
  nvvm.tcgen05.cp %taddr, %smem_desc {
    group = #nvvm.tcgen05_group<cta_2>,
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_01_23>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }
```


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-cp)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3159-L3178" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_dealloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_dealloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_dealloc</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_dealloc`

The `tcgen05.dealloc` Op de-allocates the tensor core memory specified by `tmemAddr`, which must be from a previous tensor memory allocation. The `nCols` operand specifies the number of columns to be de-allocated, and it must be a power-of-two. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3209-L3217" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_fence-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_fence-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_fence</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_fence`

The `tcgen05.fence<before>` orders all prior async tcgen05 operations with respect to the subsequent tcgen05 and execution ordering operations. The `tcgen05.fence<after>` orders all subsequent async tcgen05 operations with respect to the prior tcgen05 and execution ordering operations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-fence)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3238-L3247" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_ld' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_ld'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_ld</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`tcgen05_ld`

Instruction `tcgen05.ld` asynchronously loads data from the Tensor Memory at the location specified by the 32-bit address operand `tmemAddr` into the destination register `res`, collectively across all threads of the warps.

The `shape` and the `num` attribute together determines the total dimension of the data which is loaded from the Tensor Memory. The `shape` attribute indicates the base dimension of data to be accessed as described in the Data Movement Shape. The `num` attribute indicates the repeat factor on the base dimension resulting in the total dimension of the data that is accessed.

The shape `16x32bx2` performs two accesses into Tensor Memory of the shape `16x32b`. The base address of the first access is specified by `tmemAddr` and the base address of the second access is specified by `tmemAddr + offset`, where `offset` is an immediate argument.

The unit attribute `pack` can be used to pack two 16-bit elements from adjacent columns into a single 32-bit element during the load.

The following table describes the size of the vector for various combinations of `num` and `shape` attributes |=====================================================================| | num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  | |=====================================================================| | x1             |          1                 |    2       |    4     | | x2             |          2                 |    4       |    8     | | x4             |          4                 |    8       |    16    | | x8             |          8                 |    16      |    32    | | x16            |          16                |    32      |    64    | | x32            |          32                |    64      |    128   | | x64            |          64                |    128     |    NA    | | x128           |          128               |    NA      |    NA    | |=====================================================================|

**Example**

```mlir
  nvvm.tcgen05.ld %tmemAddr, %offset pack {
    shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
  } : <2xi32>
```


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3267-L3312" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_relinquish_alloc_permit-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_relinquish_alloc_permit-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_relinquish_alloc_permit</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_relinquish_alloc_permit`

The `tcgen05.relinquish_alloc_permit` Op specifies that the CTA of the executing thread is relinquishing the right to allocate Tensor Memory. So, it is illegal for a CTA to perform `tcgen05.alloc` after any of its constituent threads execute `tcgen05.relinquish_alloc_permit`. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3341-L3349" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_shift-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_shift-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_shift</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_shift`

The `tcgen05.shift` is an asynchronous instruction which initiates the shifting of 32-byte elements downwards across all the rows, except the last, by one row. The operand `taddr` specifies the base address of the matrix in Tensor Memory whose rows must be down shifted.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3370-L3379" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_st' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_st'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_st</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`tcgen05_st`

Instruction `tcgen05.st` asynchronously stores data from the source register `r` into the Tensor Memory at the location specified by the 32-bit address operand `tmemAddr`, collectively across all threads of the warps.

The `shape` and the `num` attribute together determines the total dimension of the data which is stored to the Tensor Memory. The `shape` indicates the base dimension of data to be accessed. The `num` attribute indicates the repeat factor on the base dimension resulting in the total dimension of the data that is accessed.

The shape `16x32bx2` performs two accesses into Tensor Memory of the shape `16x32b`. The base address of the first access is specified by `tmemAddr` and the base address of the second access is specified by `tmemAddr + offset`, where `offset` is an immediate argument.

The unit attribute `unpack` can be used to unpack a 32-bit element in the register into two 16-bit elements and store them in adjacent columns.

The following table describes the size of the vector for various combinations of `num` and `shape` attributes |=====================================================================| | num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  | |=====================================================================| | x1             |          1                 |    2       |    4     | | x2             |          2                 |    4       |    8     | | x4             |          4                 |    8       |    16    | | x8             |          8                 |    16      |    32    | | x16            |          16                |    32      |    64    | | x32            |          32                |    64      |    128   | | x64            |          64                |    128     |    NA    | | x128           |          128               |    NA      |    NA    | |=====================================================================|

**Example**

```mlir
  nvvm.tcgen05.st %tmemAddr, %val, %offset unpack {
    shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
  } : <2xi32>
```


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3400-L3444" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.tcgen05_wait-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.tcgen05_wait-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.tcgen05_wait</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`tcgen05_wait`

The `tcgen05.wait<load>` causes the executing thread to block until all prior `tcgen05.ld` operations issued by the executing thread have completed. Similarly, the `tcgen05.wait<store>` causes the executing thread to block until all prior `tcgen05.st` operations issued by the executing thread have completed. [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3473-L3482" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.vote_sync-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.vote_sync-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.vote_sync</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`vote_sync`

The `vote.sync` op will cause executing thread to wait until all non-exited threads corresponding to membermask have executed `vote.sync` with the same qualifiers and same membermask value before resuming execution.

The vote operation kinds are:
- `any`: True if source predicate is True for some thread in membermask.
  
- `all`: True if source predicate is True for all non-exited threads in membermask. 
  
- `uni`: True if source predicate has the same value in all non-exited threads in membermask.
  
- `ballot`: In the ballot form, the destination result is a 32 bit integer. In this form, the predicate from each thread in membermask are copied into the corresponding bit position of the result, where the bit position corresponds to the thread’s lane id.
  

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3562-L3581" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.wgmma_commit_group_sync_aligned-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.wgmma_commit_group_sync_aligned-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.wgmma_commit_group_sync_aligned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`wgmma_commit_group_sync_aligned`

Commits all prior uncommitted warpgroup level matrix multiplication operations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3798-L3804" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.wgmma_fence_aligned-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.wgmma_fence_aligned-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.wgmma_fence_aligned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`wgmma_fence_aligned`

Enforce an ordering of register accesses between warpgroup level matrix  multiplication and other operations. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3771-L3778" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.wgmma_mma_async-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.nvvm.wgmma_mma_async-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.wgmma_mma_async</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`wgmma_mma_async`

The warpgroup (128 threads) level matrix multiply and accumulate operation  has either of the following forms, where matrix D is called accumulator:   D = A * B + D   D = A * B, where the input from accumulator D is disabled.

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


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3824-L3882" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.nvvm.wgmma_wait_group_sync_aligned-Tuple{}' href='#Reactant.MLIR.Dialects.nvvm.wgmma_wait_group_sync_aligned-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.nvvm.wgmma_wait_group_sync_aligned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`wgmma_wait_group_sync_aligned`

Signal the completion of a preceding warpgroup operation.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Nvvm.jl#L3929-L3935" target="_blank" rel="noreferrer">source</a></Badge>

</details>

