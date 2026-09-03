


# MPI Dialect {#MPI-Dialect}

Refer to the [official documentation](https://mlir.llvm.org/docs/Dialects/MPI/) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.allreduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.allreduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.allreduce</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`allreduce`

MPI_Allreduce performs a reduction operation on the values in the sendbuf array and stores the result in the recvbuf array. The operation is  performed across all processes in the communicator.

The `op` attribute specifies the reduction operation to be performed. Currently only the `MPI_Op` predefined in the standard (e.g. `MPI_SUM`) are supported.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L16-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.barrier-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.barrier-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.barrier</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`barrier`

MPI_Barrier blocks execution until all processes in the communicator have reached this routine.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L57-L65" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.comm_rank-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.comm_rank-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.comm_rank</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`comm_rank`

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L86-L91" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.comm_size-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.comm_size-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.comm_size</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`comm_size`

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L114-L119" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.comm_split-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.comm_split-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.comm_split</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`comm_split`

This operation splits the communicator into multiple sub-communicators. The color value determines the group of processes that will be part of the new communicator. The key value determines the rank of the calling process in the new communicator.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L142-L152" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.comm_world-Tuple{}' href='#Reactant.MLIR.Dialects.mpi.comm_world-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.comm_world</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`comm_world`

This operation returns the predefined MPI_COMM_WORLD communicator.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L180-L184" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.error_class-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.error_class-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.error_class</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`error_class`

`MPI_Error_class` maps return values from MPI calls to a set of well-known MPI error classes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L204-L209" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.finalize-Tuple{}' href='#Reactant.MLIR.Dialects.mpi.finalize-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.finalize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`finalize`

This function cleans up the MPI state. Afterwards, no MPI methods may  be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized). Notably, MPI_Init cannot be called again in the same program.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L229-L238" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.init-Tuple{}' href='#Reactant.MLIR.Dialects.mpi.init-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.init</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`init`

This operation must preceed most MPI calls (except for very few exceptions, please consult with the MPI specification on these).

Passing &amp;argc, &amp;argv is not supported currently.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L337-L347" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.irecv-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.irecv-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.irecv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`irecv`

MPI_Irecv begins a non-blocking receive of `size` elements of type `dtype`  from rank `source`. The `tag` value and communicator enables the library to determine the matching of multiple sends and receives between the same  ranks.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L259-L269" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.isend-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.isend-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.isend</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`isend`

MPI_Isend begins a non-blocking send of `size` elements of type `dtype` to rank `dest`. The `tag` value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L298-L308" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.recv-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.recv-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.recv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`recv`

MPI_Recv performs a blocking receive of `size` elements of type `dtype`  from rank `source`. The `tag` value and communicator enables the library to determine the matching of multiple sends and receives between the same  ranks.

The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object  is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L368-L381" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.retval_check-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.retval_check-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.retval_check</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`retval_check`

This operation compares MPI status codes to known error class constants such as `MPI_SUCCESS`, or `MPI_ERR_COMM`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L409-L414" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.send-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.send-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.send</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`send`

MPI_Send performs a blocking send of `size` elements of type `dtype` to rank `dest`. The `tag` value and communicator enables the library to determine  the matching of multiple sends and receives between the same ranks.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L434-L443" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.mpi.wait-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.mpi.wait-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.mpi.wait</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`wait`

MPI_Wait blocks execution until the request has completed.

The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object  is not yet ported to MLIR.

This operation can optionally return an `!mpi.retval` value that can be used to check for errors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/MPI.jl#L471-L481" target="_blank" rel="noreferrer">source</a></Badge>

</details>

