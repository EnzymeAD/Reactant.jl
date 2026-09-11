


# Enzyme Dialect {#Enzyme-Dialect}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.enzyme.addTo-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.enzyme.addTo-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.enzyme.addTo</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`addTo`

TODO


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Enzyme.jl#L16-L20" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.enzyme.broadcast-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.enzyme.broadcast-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.enzyme.broadcast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast`

Broadcast the operand by adding extra dimensions with sizes provided by the `shape` attribute to the front. For scalar operands, ranked tensor is created.

NOTE: Only works for scalar and _ranked_ tensor operands for now.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Enzyme.jl#L95-L102" target="_blank" rel="noreferrer">source</a></Badge>

</details>

