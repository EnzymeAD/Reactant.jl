


# Higher level API {#Higher-level-API}
<details class='jldocstring custom-block' >
<summary><a id='Core.Bool-Tuple{Reactant.MLIR.IR.Attribute}' href='#Core.Bool-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Core.Bool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Bool(attr)
```


Returns the value stored in the given bool attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L210-L214" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Core.Float64-Tuple{Reactant.MLIR.IR.Attribute}' href='#Core.Float64-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Core.Float64</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Float64(attr)
```


Returns the value stored in the given floating point attribute, interpreting the value as double.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L126-L130" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Core.Int64-Tuple{Reactant.MLIR.IR.Attribute}' href='#Core.Int64-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Core.Int64</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Int64(attr)
```


Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L174-L178" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Core.String-Tuple{Reactant.MLIR.IR.Attribute}' href='#Core.String-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Core.String</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
String(attr)
```


Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L287-L291" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Core.String-Tuple{Reactant.MLIR.IR.Identifier}' href='#Core.String-Tuple{Reactant.MLIR.IR.Identifier}'><span class="jlbinding">Core.String</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
String(ident)
```


Gets the string value of the identifier.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Identifier.jl#L29-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Core.UInt64-Tuple{Reactant.MLIR.IR.Attribute}' href='#Core.UInt64-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Core.UInt64</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
UInt64(attr)
```


Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L186-L190" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.AffineMap-Tuple{Any, Any, Vector{Reactant.MLIR.IR.AffineExpr}}' href='#Reactant.MLIR.IR.AffineMap-Tuple{Any, Any, Vector{Reactant.MLIR.IR.AffineExpr}}'><span class="jlbinding">Reactant.MLIR.IR.AffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
AffineMap(ndims, nsymbols, affineExprs; context=context())
```


Creates an affine map with results defined by the given list of affine expressions. The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L50-L55" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.AffineMap-Tuple{Any, Any}' href='#Reactant.MLIR.IR.AffineMap-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.AffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
AffineMap(ndims, nsymbols; context=context())
```


Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L41-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.AffineMap-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.AffineMap-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.AffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
AffineMap(attr)
```


Returns the affine map wrapped in the given affine map attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L64-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.AffineMap-Tuple{}' href='#Reactant.MLIR.IR.AffineMap-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.AffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
AffineMap(; context=context())
```


Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L10-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{AbstractString}' href='#Reactant.MLIR.IR.Attribute-Tuple{AbstractString}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(str; context=context())
```


Creates a string attribute in the given context containing the given string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L270-L274" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{Bool}' href='#Reactant.MLIR.IR.Attribute-Tuple{Bool}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(value; context=context())
```


Creates a bool attribute in the given context with the given value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L203-L207" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{Dict}' href='#Reactant.MLIR.IR.Attribute-Tuple{Dict}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(elements; context=context())
```


Creates a dictionary attribute containing the given list of elements in the provided context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L93-L97" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.Attribute-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(affineMap)
```


Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L57-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{Reactant.MLIR.IR.Type, AbstractString}' href='#Reactant.MLIR.IR.Attribute-Tuple{Reactant.MLIR.IR.Type, AbstractString}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(type, str)
```


Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L278-L282" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.Attribute-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(type)
```


Creates a type attribute wrapping the given type in the same context as the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L376-L380" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{T} where T<:AbstractFloat' href='#Reactant.MLIR.IR.Attribute-Tuple{T} where T<:AbstractFloat'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(float; context=context(), location=Location(), check=false)
```


Creates a floating point attribute in the given context with the given double value and double-precision FP semantics. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L110-L115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{T} where T<:Complex' href='#Reactant.MLIR.IR.Attribute-Tuple{T} where T<:Complex'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(complex; context=context(), location=Location(), check=false)
```


Creates a complex attribute in the given context with the given complex value and double-precision FP semantics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L136-L140" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{Vector{Reactant.MLIR.IR.Attribute}}' href='#Reactant.MLIR.IR.Attribute-Tuple{Vector{Reactant.MLIR.IR.Attribute}}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(elements; context=context())
```


Creates an array element containing the given list of elements in the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L78-L82" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Tuple{}' href='#Reactant.MLIR.IR.Attribute-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute()
```


Returns an empty attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L5-L9" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Attribute-Union{Tuple{T}, Tuple{T, Any}} where T<:Integer' href='#Reactant.MLIR.IR.Attribute-Union{Tuple{T}, Tuple{T, Any}} where T<:Integer'><span class="jlbinding">Reactant.MLIR.IR.Attribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Attribute(int)
```


Creates an integer attribute of the given type with the given integer value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L166-L170" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Block-Tuple{Vector{Reactant.MLIR.IR.Type}, Vector{Reactant.MLIR.IR.Location}}' href='#Reactant.MLIR.IR.Block-Tuple{Vector{Reactant.MLIR.IR.Type}, Vector{Reactant.MLIR.IR.Location}}'><span class="jlbinding">Reactant.MLIR.IR.Block</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Block(args, locs)
```


Creates a new empty block with the given argument types and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L17-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.BlockIterator' href='#Reactant.MLIR.IR.BlockIterator'><span class="jlbinding">Reactant.MLIR.IR.BlockIterator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BlockIterator(region::Region)
```


Iterates over all blocks in the given region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Iterators.jl#L1-L5" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Context-Tuple{}' href='#Reactant.MLIR.IR.Context-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.Context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Context()
```


Creates an MLIR context and transfers its ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Context.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ExecutionEngine' href='#Reactant.MLIR.IR.ExecutionEngine'><span class="jlbinding">Reactant.MLIR.IR.ExecutionEngine</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ExecutionEngine(op, optLevel, sharedlibs = [])
```


Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is expected to be &quot;translatable&quot; to LLVM IR (only contains operations in dialects that implement the `LLVMTranslationDialectInterface`). The module ownership stays with the client and can be destroyed as soon as the call returns. `optLevel` is the optimization level to be used for transformation and code generation. LLVM passes at `optLevel` are run before code generation. The number and array of paths corresponding to shared libraries that will be loaded are specified via `numPaths` and `sharedLibPaths` respectively. TODO: figure out other options.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/ExecutionEngine.jl#L10-L20" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Identifier-Tuple{String}' href='#Reactant.MLIR.IR.Identifier-Tuple{String}'><span class="jlbinding">Reactant.MLIR.IR.Identifier</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Identifier(context, str)
```


Gets an identifier with the given string value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Identifier.jl#L5-L9" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.IntegerSet-NTuple{4, Any}' href='#Reactant.MLIR.IR.IntegerSet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.IR.IntegerSet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
IntegerSet(ndims, nsymbols, constraints, eqflags; context=context())
```


Gets or creates a new integer set in the given context. The set is defined by a list of affine constraints, with the given number of input dimensions and symbols, which are treated as either equalities (eqflags is 1) or inequalities (eqflags is 0). Both `constraints` and `eqflags` need to be arrays of the same length.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L18-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.IntegerSet-Tuple{Any, Any}' href='#Reactant.MLIR.IR.IntegerSet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.IntegerSet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Integerset(ndims, nsymbols; context=context())
```


Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.LogicalResult' href='#Reactant.MLIR.IR.LogicalResult'><span class="jlbinding">Reactant.MLIR.IR.LogicalResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LogicalResult
```


A logical result value, essentially a boolean with named states. LLVM convention for using boolean values to designate success or failure of an operation is a moving target, so MLIR opted for an explicit class. Instances of [`LogicalResult`](/api/mlirc#Reactant.MLIR.IR.LogicalResult) must only be inspected using the associated functions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/LogicalResult.jl#L1-L7" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Module' href='#Reactant.MLIR.IR.Module'><span class="jlbinding">Reactant.MLIR.IR.Module</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Module(location=Location())
```


Creates a new, empty module and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Module.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.NamedAttribute-Tuple{Any, Any}' href='#Reactant.MLIR.IR.NamedAttribute-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.NamedAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
NamedAttribute(name, attr)
```


Associates an attribute with the name. Takes ownership of neither.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L850-L854" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.OpPassManager-Tuple{Reactant.MLIR.IR.OpPassManager, Any}' href='#Reactant.MLIR.IR.OpPassManager-Tuple{Reactant.MLIR.IR.OpPassManager, Any}'><span class="jlbinding">Reactant.MLIR.IR.OpPassManager</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
OpPassManager(opPassManager, operationName)
```


Nest an `OpPassManager` under the provided `OpPassManager`, the nested passmanager will only run on operations matching the provided name. The returned `OpPassManager` will be destroyed when the parent is destroyed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L192-L196" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.OpPassManager-Tuple{Reactant.MLIR.IR.PassManager, Any}' href='#Reactant.MLIR.IR.OpPassManager-Tuple{Reactant.MLIR.IR.PassManager, Any}'><span class="jlbinding">Reactant.MLIR.IR.OpPassManager</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
OpPassManager(passManager, operationName)
```


Nest an `OpPassManager` under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned `OpPassManager` will be destroyed when the parent is destroyed. To further nest more `OpPassManager` under the newly returned one, see `mlirOpPassManagerNest` below.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L183-L188" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.OpPassManager-Tuple{Reactant.MLIR.IR.PassManager}' href='#Reactant.MLIR.IR.OpPassManager-Tuple{Reactant.MLIR.IR.PassManager}'><span class="jlbinding">Reactant.MLIR.IR.OpPassManager</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
OpPassManager(passManager)
```


Cast a top-level `PassManager` to a generic `OpPassManager`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L175-L179" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Operation-Tuple{Reactant.MLIR.IR.Module}' href='#Reactant.MLIR.IR.Operation-Tuple{Reactant.MLIR.IR.Module}'><span class="jlbinding">Reactant.MLIR.IR.Operation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Operation(module)
```


Views the module as a generic operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Module.jl#L50-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.OperationIterator' href='#Reactant.MLIR.IR.OperationIterator'><span class="jlbinding">Reactant.MLIR.IR.OperationIterator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
OperationIterator(block::Block)
```


Iterates over all operations for the given block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Iterators.jl#L66-L70" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.PassManager-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.PassManager-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.PassManager</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
PassManager(anchorOp; context=context())
```


Create a new top-level PassManager anchored on `anchorOp`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L28-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.PassManager-Tuple{}' href='#Reactant.MLIR.IR.PassManager-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.PassManager</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
PassManager(; context=context())
```


Create a new top-level PassManager.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Region-Tuple{}' href='#Reactant.MLIR.IR.Region-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.Region</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Region()
```


Creates a new empty region and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L15-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.RegionIterator' href='#Reactant.MLIR.IR.RegionIterator'><span class="jlbinding">Reactant.MLIR.IR.RegionIterator</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RegionIterator(::Operation)
```


Iterates over all sub-regions for the given operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Iterators.jl#L34-L38" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.SymbolTable-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.SymbolTable-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.SymbolTable</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableCreate(operation)
```


Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/SymbolTable.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.Type-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(attr)
```


Returns the type stored in the given type attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L383-L387" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Integer}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Integer}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(T::Core.Type{<:Integer}; context=context()
```


Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L80-L84" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E4M3B11FNUZ}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E4M3B11FNUZ}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Reactant.F8E4M3B11FNUZ}; context=context())
```


Creates a f8e4m3b11fnuz type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L212-L216" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E4M3FNUZ}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E4M3FNUZ}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Reactant.F8E4M3FNUZ}; context=context())
```


Creates a f8e4m3fnuz type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L230-L234" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E4M3FN}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E4M3FN}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Reactant.F8E4M3FN}; context=context())
```


Creates a f8e4m3fn type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L203-L207" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E5M2FNUZ}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E5M2FNUZ}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Reactant.F8E5M2FNUZ}; context=context())
```


Creates a f8e5m2fnuz type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L221-L225" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E5M2}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.F8E5M2}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Reactant.F8E5M2}; context=context())
```


Creates a f8e5m2 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L194-L198" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.TF32}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Reactant.TF32}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Reactant.TF32}; context=context())
```


Creates a tf32 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L239-L243" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Signed}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Signed}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(T::Core.Type{<:Signed}; context=context()
```


Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L88-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{<:Unsigned}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{<:Unsigned}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(T::Core.Type{<:Unsigned}; context=context()
```


Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L96-L100" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{Bool}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{Bool}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(T::Core.Type{Bool}; context=context()
```


Creates a 1-bit signless integer type in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L71-L75" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{Float16}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{Float16}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Float16}; context=context())
```


Creates an f16 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L164-L168" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{Float32}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{Float32}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(Core.Type{Float32}; context=context())
```


Creates an f32 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L180-L184" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{Float64}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{Float64}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(Core.Type{Float64}; context=context())
```


Creates a f64 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L187-L191" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Type{Nothing}}' href='#Reactant.MLIR.IR.Type-Tuple{Type{Nothing}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(::Core.Type{Nothing}; context=context())
```


Creates a None type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L42-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Tuple{Vector{Reactant.MLIR.IR.Type}}' href='#Reactant.MLIR.IR.Type-Tuple{Vector{Reactant.MLIR.IR.Type}}'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(elements; context=context())
Type(::Core.Type{<:Tuple{T...}}; context=context())
```


Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L642-L647" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Type-Union{Tuple{Type{Complex{T}}}, Tuple{T}} where T' href='#Reactant.MLIR.IR.Type-Union{Tuple{Type{Complex{T}}}, Tuple{T}} where T'><span class="jlbinding">Reactant.MLIR.IR.Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Type(Complex{T}) where {T}
```


Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L319-L323" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}' href='#Base.:-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.:*</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
*(lhs, rhs)
```


Creates an affine mul expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L157-L161" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:+-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}' href='#Base.:+-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.:+</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
+(lhs, rhs)
```


Creates an affine add expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L143-L147" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}' href='#Base.:==-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(a, b)
```


Returns `true` if the two affine expressions are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L12-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.AffineMap, Reactant.MLIR.IR.AffineMap}' href='#Base.:==-Tuple{Reactant.MLIR.IR.AffineMap, Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(a, b)
```


Checks if two affine maps are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L20-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.Attribute, Reactant.MLIR.IR.Attribute}' href='#Base.:==-Tuple{Reactant.MLIR.IR.Attribute, Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(a1, a2)
```


Checks if two attributes are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L22-L26" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Block}' href='#Base.:==-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Block}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(block, other)
```


Checks whether two blocks handles point to the same block. This does not perform deep comparison.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L27-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.Identifier, Reactant.MLIR.IR.Identifier}' href='#Base.:==-Tuple{Reactant.MLIR.IR.Identifier, Reactant.MLIR.IR.Identifier}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(ident, other)
```


Checks whether two identifiers are the same.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Identifier.jl#L15-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.IntegerSet, Reactant.MLIR.IR.IntegerSet}' href='#Base.:==-Tuple{Reactant.MLIR.IR.IntegerSet, Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(s1, s2)
```


Checks if two integer set objects are equal. This is a &quot;shallow&quot; comparison of two objects. Only the sets with some small number of constraints are uniqued and compare equal here. Set objects that represent the same integer set with different constraints may be considered non-equal by this check. Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L50-L57" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Region}' href='#Base.:==-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Region}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(region, other)
```


Checks whether two region handles point to the same region. This does not perform deep comparison.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L25-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.Type, Reactant.MLIR.IR.Type}' href='#Base.:==-Tuple{Reactant.MLIR.IR.Type, Reactant.MLIR.IR.Type}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(t1, t2)
```


Checks if two types are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L20-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.TypeID, Reactant.MLIR.IR.TypeID}' href='#Base.:==-Tuple{Reactant.MLIR.IR.TypeID, Reactant.MLIR.IR.TypeID}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(typeID1, typeID2)
```


Checks if two type ids are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/TypeID.jl#L23-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.:==-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Base.:==-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Base.:==</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
==(value1, value2)
```


Returns 1 if two values are equal, 0 otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L14-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.cld-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}' href='#Base.cld-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.cld</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
cld(lhs, rhs)
```


Creates an affine ceildiv expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L203-L207" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.copy-Tuple{Reactant.MLIR.IR.Operation}' href='#Base.copy-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Base.copy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
copy(op)
```


Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.div-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}' href='#Base.div-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.div</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
div(lhs, rhs)
÷(lhs, rhs)
fld(lhs, rhs)
```


Creates an affine floordiv expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L185-L191" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.fill-Tuple{Reactant.MLIR.IR.Attribute, Reactant.MLIR.IR.Type}' href='#Base.fill-Tuple{Reactant.MLIR.IR.Attribute, Reactant.MLIR.IR.Type}'><span class="jlbinding">Base.fill</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
fill(attr, shapedType)
```


Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L435-L439" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.gcd-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Base.gcd-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.gcd</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
gcd(affineExpr)
```


Returns the greatest known integral divisor of this affine expression. The result is always positive.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L40-L44" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.hash-Tuple{Reactant.MLIR.IR.TypeID}' href='#Base.hash-Tuple{Reactant.MLIR.IR.TypeID}'><span class="jlbinding">Base.hash</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
hash(typeID)
```


Returns the hash value of the type id.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/TypeID.jl#L14-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.insert!-Tuple{Reactant.MLIR.IR.Block, Any, Reactant.MLIR.IR.Operation}' href='#Base.insert!-Tuple{Reactant.MLIR.IR.Block, Any, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Base.insert!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
insert!(block, index, operation)
```


Takes an operation owned by the caller and inserts it as `index` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L134-L139" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.insert!-Tuple{Reactant.MLIR.IR.Region, Any, Reactant.MLIR.IR.Block}' href='#Base.insert!-Tuple{Reactant.MLIR.IR.Region, Any, Reactant.MLIR.IR.Block}'><span class="jlbinding">Base.insert!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
insert!(region, index, block)
```


Takes a block owned by the caller and inserts it at `index` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L42-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.isempty-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Base.isempty-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Base.isempty</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isempty(affineMap)
```


Checks whether the given affine map is an empty affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L114-L118" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.isempty-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Base.isempty-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Base.isempty</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isempty(set)
```


Checks whether the given set is a canonical empty set, e.g., the set returned by [`Reactant.MLIR.API.mlirIntegerSetEmptyGet`](/api/mlirc#Reactant.MLIR.API.mlirIntegerSetEmptyGet-Tuple{Any,%20Any,%20Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L67-L71" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.isperm-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Base.isperm-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Base.isperm</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isperm(affineMap)
```


Checks whether the given affine map represents a symbol-less permutation map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L180-L184" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.mod-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}' href='#Base.mod-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Base.mod</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mod(lhs, rhs)
```


Creates an affine mod expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L171-L175" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.ndims-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Base.ndims-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Base.ndims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ndims(affineMap)
```


Returns the number of dimensions of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L138-L142" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.ndims-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Base.ndims-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Base.ndims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ndims(set)
```


Returns the number of dimensions in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L74-L78" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.ndims-Tuple{Reactant.MLIR.IR.Type}' href='#Base.ndims-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Base.ndims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ndims(type)
```


Returns the rank of the given ranked shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L348-L352" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.parse-Tuple{Reactant.MLIR.IR.OpPassManager, String}' href='#Base.parse-Tuple{Reactant.MLIR.IR.OpPassManager, String}'><span class="jlbinding">Base.parse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parse(opPassManager, pipeline)
```


Parse a textual MLIR pass pipeline and add it to the provided `OpPassManager`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L256-L260" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.parse-Tuple{Type{Reactant.MLIR.IR.Attribute}, Any}' href='#Base.parse-Tuple{Type{Reactant.MLIR.IR.Attribute}, Any}'><span class="jlbinding">Base.parse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parse(::Core.Type{Attribute}, str; context=context())
```


Parses an attribute. The attribute is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L14-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.parse-Tuple{Type{Reactant.MLIR.IR.Module}, Any}' href='#Base.parse-Tuple{Type{Reactant.MLIR.IR.Module}, Any}'><span class="jlbinding">Base.parse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parse(::Type{Module}, module; context=context())
```


Parses a module from the string and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Module.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.parse-Tuple{Type{Reactant.MLIR.IR.Operation}, Any}' href='#Base.parse-Tuple{Type{Reactant.MLIR.IR.Operation}, Any}'><span class="jlbinding">Base.parse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parse(::Type{Operation}, code; context=context())
```


Parses an operation from the string and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L245-L249" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.parse-Tuple{Type{Reactant.MLIR.IR.Type}, Any}' href='#Base.parse-Tuple{Type{Reactant.MLIR.IR.Type}, Any}'><span class="jlbinding">Base.parse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parse(type; context=context())
```


Parses a type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L12-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.push!-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Operation}' href='#Base.push!-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Base.push!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
push!(block, operation)
```


Takes an operation owned by the caller and appends it to the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L124-L128" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.push!-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Block}' href='#Base.push!-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Block}'><span class="jlbinding">Base.push!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
push!(region, block)
```


Takes a block owned by the caller and appends it to the given region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L32-L36" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.push!-Tuple{Reactant.MLIR.IR.SymbolTable, Reactant.MLIR.IR.Operation}' href='#Base.push!-Tuple{Reactant.MLIR.IR.SymbolTable, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Base.push!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
push!(symboltable, operation)
```


Inserts the given operation into the given symbol table. The operation must have the symbol trait. If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness. Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately. Returns the name of the symbol after insertion.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/SymbolTable.jl#L40-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.replace-Tuple{Reactant.MLIR.IR.AffineMap, Pair{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}, Any, Any}' href='#Base.replace-Tuple{Reactant.MLIR.IR.AffineMap, Pair{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineExpr}, Any, Any}'><span class="jlbinding">Base.replace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapReplace(affineMap, expression => replacement, numResultDims, numResultSyms)
```


Apply `AffineExpr::replace(map)` to each of the results and return a new new AffineMap with the new results and the specified number of dims and symbols.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L214-L218" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.replace-Tuple{Reactant.MLIR.IR.IntegerSet, Any, Any}' href='#Base.replace-Tuple{Reactant.MLIR.IR.IntegerSet, Any, Any}'><span class="jlbinding">Base.replace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)
```


Gets or creates a new integer set in which the values and dimensions of the given set are replaced with the given affine expressions. `dimReplacements` and `symbolReplacements` are expected to point to at least as many consecutive expressions as the given set has dimensions and symbols, respectively. The new set will have `numResultDims` and `numResultSymbols` dimensions and symbols, respectively.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L31-L37" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.reshape-Tuple{Reactant.MLIR.IR.Attribute, Vector{Int64}}' href='#Base.reshape-Tuple{Reactant.MLIR.IR.Attribute, Vector{Int64}}'><span class="jlbinding">Base.reshape</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Base.reshape(attr, shapedType)
```


Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L635-L639" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.size-Tuple{Reactant.MLIR.IR.Type, Int64}' href='#Base.size-Tuple{Reactant.MLIR.IR.Type, Int64}'><span class="jlbinding">Base.size</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
size(type, i)
```


Returns the `i`-th dimension of the given ranked shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L372-L376" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Base.write-Tuple{String, Reactant.MLIR.IR.ExecutionEngine}' href='#Base.write-Tuple{String, Reactant.MLIR.IR.ExecutionEngine}'><span class="jlbinding">Base.write</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
write(fileName, jit)
```


Dump as an object in `fileName`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/ExecutionEngine.jl#L54-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.AffineDimensionExpr-Tuple{Any}' href='#Reactant.MLIR.IR.AffineDimensionExpr-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.AffineDimensionExpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
AffineDimensionExpr(position; context=context)
```


Creates an affine dimension expression with &#39;position&#39; in the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L69-L73" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.BFloat16Type-Tuple{}' href='#Reactant.MLIR.IR.BFloat16Type-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.BFloat16Type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



BFloat16Type(; context=context())

Creates a bf16 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L157-L161" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ConstantAffineMap-Tuple{Any}' href='#Reactant.MLIR.IR.ConstantAffineMap-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.ConstantAffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ConstantAffineMap(val; context=context())
```


Creates a single constant result affine map in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L59-L63" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ConstantExpr-Tuple{Any}' href='#Reactant.MLIR.IR.ConstantExpr-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.ConstantExpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ConstantExpr(constant::Int; context=context())
```


Creates an affine constant expression with &#39;constant&#39; in the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L118-L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.DenseElementsAttribute-Tuple{AbstractArray{Bool}}' href='#Reactant.MLIR.IR.DenseElementsAttribute-Tuple{AbstractArray{Bool}}'><span class="jlbinding">Reactant.MLIR.IR.DenseElementsAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
DenseElementsAttribute(array::AbstractArray)
```


Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L499-L503" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.DenseElementsAttribute-Tuple{AbstractArray{String}}' href='#Reactant.MLIR.IR.DenseElementsAttribute-Tuple{AbstractArray{String}}'><span class="jlbinding">Reactant.MLIR.IR.DenseElementsAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
DenseElementsAttribute(array::AbstractArray{String})
```


Creates a dense elements attribute with the given shaped type from string elements.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L620-L624" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.DenseElementsAttribute-Tuple{Reactant.MLIR.IR.Type, AbstractArray}' href='#Reactant.MLIR.IR.DenseElementsAttribute-Tuple{Reactant.MLIR.IR.Type, AbstractArray}'><span class="jlbinding">Reactant.MLIR.IR.DenseElementsAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
DenseElementsAttribute(shapedType, elements)
```


Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L423-L427" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.FlatSymbolRefAttribute-Tuple{String}' href='#Reactant.MLIR.IR.FlatSymbolRefAttribute-Tuple{String}'><span class="jlbinding">Reactant.MLIR.IR.FlatSymbolRefAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
FlatSymbolRefAttribute(ctx, symbol)
```


Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L351-L355" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Float8E4M3FN-Tuple{}' href='#Reactant.MLIR.IR.Float8E4M3FN-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.Float8E4M3FN</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Float8E4M3FN(; context=context())
```


Creates an f8E4M3FN type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L150-L154" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.Float8E5M2-Tuple{}' href='#Reactant.MLIR.IR.Float8E5M2-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.Float8E5M2</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Float8E5M2(; context=context())
```


Creates an f8E5M2 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L143-L147" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.FunctionType-Tuple{Any, Any}' href='#Reactant.MLIR.IR.FunctionType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.FunctionType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
FunctionType(inputs, results; context=context())
```


Creates a function type, mapping a list of input types to result types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L672-L676" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.IdentityAffineMap-Tuple{Any}' href='#Reactant.MLIR.IR.IdentityAffineMap-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.IdentityAffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
IdentityAffineMap(ndims; context=context())
```


Creates an affine map with &#39;ndims&#39; identity in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L67-L71" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.IndexType-Tuple{}' href='#Reactant.MLIR.IR.IndexType-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.IndexType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
IndexType(; context=context())
```


Creates an index type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L57-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.MemRefType-Tuple{Reactant.MLIR.IR.Type, Any, Any, Any}' href='#Reactant.MLIR.IR.MemRefType-Tuple{Reactant.MLIR.IR.Type, Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.MemRefType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
MemRefType(elementType, rank, shape, layout, memorySpace; location=Location(), check=false)
```


Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L529-L534" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.MemRefType-Tuple{Reactant.MLIR.IR.Type, Any, Any}' href='#Reactant.MLIR.IR.MemRefType-Tuple{Reactant.MLIR.IR.Type, Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.MemRefType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
MemRefType(elementType, rank, shape, memorySpace; location=Location(), check=false)
```


Creates a MemRef type with the given rank, shape, memory space and element type in the same context as the element type. The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L554-L560" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.MemRefType-Tuple{Reactant.MLIR.IR.Type, Any}' href='#Reactant.MLIR.IR.MemRefType-Tuple{Reactant.MLIR.IR.Type, Any}'><span class="jlbinding">Reactant.MLIR.IR.MemRefType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
MemRefType(elementType, memorySpace)
```


Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L575-L580" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.MinorIdentityAffineMap-Tuple{Any, Any}' href='#Reactant.MLIR.IR.MinorIdentityAffineMap-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.MinorIdentityAffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
MinorIdentityAffineMap(ndims, nresults; context=context())
```


Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L75-L80" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.OpaqueAttribute-Tuple{Any, Any, Any}' href='#Reactant.MLIR.IR.OpaqueAttribute-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.OpaqueAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
OpaqueAttribute(dialectNamespace, dataLength, data, type; context=context())
```


Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L234-L239" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.OpaqueType-Tuple{Any, Any}' href='#Reactant.MLIR.IR.OpaqueType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.OpaqueType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
OpaqueType(dialectNamespace, typeData; context=context())
```


Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L729-L733" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.PermutationAffineMap-Tuple{Any}' href='#Reactant.MLIR.IR.PermutationAffineMap-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.PermutationAffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
PermutationAffineMap(permutation; context=context())
```


Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid invalid permutation). The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L86-L93" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.SymbolExpr-Tuple{Any}' href='#Reactant.MLIR.IR.SymbolExpr-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.SymbolExpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
SymbolExpr(position; context=context())
```


Creates an affine symbol expression with &#39;position&#39; in the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L84-L88" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.SymbolRefAttribute-Tuple{String, Vector{Reactant.MLIR.IR.Attribute}}' href='#Reactant.MLIR.IR.SymbolRefAttribute-Tuple{String, Vector{Reactant.MLIR.IR.Attribute}}'><span class="jlbinding">Reactant.MLIR.IR.SymbolRefAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
SymbolRefAttribute(symbol, references; context=context())
```


Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L304-L309" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.TensorType' href='#Reactant.MLIR.IR.TensorType'><span class="jlbinding">Reactant.MLIR.IR.TensorType</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
TensorType(shape, elementType, encoding=Attribute(); location=Location(), check=false)
```


Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type. The type is owned by the context. Tensor types without any specific encoding field should assign [`Reactant.MLIR.API.mlirAttributeGetNull`](/api/mlirc#Reactant.MLIR.API.mlirAttributeGetNull-Tuple{}) to this parameter. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L439-L445" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.TensorType-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.TensorType-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.TensorType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
TensorType(elementType)
```


Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L463-L468" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.UnitAttribute-Tuple{}' href='#Reactant.MLIR.IR.UnitAttribute-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.UnitAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
UnitAttribute(; context=context())
```


Creates a unit attribute in the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L397-L401" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.VectorType-Tuple{Any, Any, Any}' href='#Reactant.MLIR.IR.VectorType-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.VectorType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
VectorType(rank, shape, elementType; location=Location(), check=false)
```


Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context. If `check=true`, emits appropriate diagnostics on illegal arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L413-L418" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.add_owned_pass!-Tuple{Reactant.MLIR.IR.OpPassManager, Any}' href='#Reactant.MLIR.IR.add_owned_pass!-Tuple{Reactant.MLIR.IR.OpPassManager, Any}'><span class="jlbinding">Reactant.MLIR.IR.add_owned_pass!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
add_owned_pass!(opPassManager, pass)
```


Add a pass and transfer ownership to the provided `OpPassManager`. If the pass is not a generic operation pass or matching the type of the provided `OpPassManager`, a new `OpPassManager` is implicitly nested under the provided `OpPassManager`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L246-L250" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.add_owned_pass!-Tuple{Reactant.MLIR.IR.PassManager, Any}' href='#Reactant.MLIR.IR.add_owned_pass!-Tuple{Reactant.MLIR.IR.PassManager, Any}'><span class="jlbinding">Reactant.MLIR.IR.add_owned_pass!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
add_owned_pass!(passManager, pass)
```


Add a pass and transfer ownership to the provided top-level `PassManager`. If the pass is not a generic operation pass or a `ModulePass`, a new `OpPassManager` is implicitly nested under the provided PassManager.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L236-L240" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.add_pipeline!-Tuple{Reactant.MLIR.IR.OpPassManager, Any}' href='#Reactant.MLIR.IR.add_pipeline!-Tuple{Reactant.MLIR.IR.OpPassManager, Any}'><span class="jlbinding">Reactant.MLIR.IR.add_pipeline!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
add_pipeline!(opPassManager, pipeline)
```


Parse a sequence of textual MLIR pass pipeline elements and add them to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L274-L278" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.affinemap-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.affinemap-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.affinemap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
affinemap(type)
```


Returns the affine map of the given MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L617-L621" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.argument-Tuple{Reactant.MLIR.IR.Block, Any}' href='#Reactant.MLIR.IR.argument-Tuple{Reactant.MLIR.IR.Block, Any}'><span class="jlbinding">Reactant.MLIR.IR.argument</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
argument(block, i)
```


Returns `i`-th argument of the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L70-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.attr!-Tuple{Reactant.MLIR.IR.Operation, Any, Any}' href='#Reactant.MLIR.IR.attr!-Tuple{Reactant.MLIR.IR.Operation, Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.attr!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
attr!(op, name, attr)
```


Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L202-L206" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.attr-Tuple{Reactant.MLIR.IR.Operation, AbstractString}' href='#Reactant.MLIR.IR.attr-Tuple{Reactant.MLIR.IR.Operation, AbstractString}'><span class="jlbinding">Reactant.MLIR.IR.attr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
attr(op, name)
```


Returns an attribute attached to the operation given its name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L189-L193" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.attr-Tuple{Reactant.MLIR.IR.Operation, Any}' href='#Reactant.MLIR.IR.attr-Tuple{Reactant.MLIR.IR.Operation, Any}'><span class="jlbinding">Reactant.MLIR.IR.attr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
attr(op, i)
```


Return `i`-th attribute of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L179-L183" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.bitwidth-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.bitwidth-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.bitwidth</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
bitwidth(type)
```


Returns the bitwidth of an integer type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L132-L136" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.block-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.block-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.block</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
block(op)
```


Gets the block that owns this operation, returning null if the operation is not owned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L56-L60" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.block_arg_num-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.block_arg_num-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.block_arg_num</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
block_arg_num(value)
```


Returns the position of the value in the argument list of its block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L69-L73" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.block_owner-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.block_owner-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.block_owner</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
block_owner(value)
```


Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L35-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.body-Tuple{Any}' href='#Reactant.MLIR.IR.body-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.body</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
body(module)
```


Gets the body of the module, i.e. the only block it contains.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Module.jl#L43-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.compose-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.compose-Tuple{Reactant.MLIR.IR.AffineExpr, Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.compose</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
compose(affineExpr, affineMap)
```


Composes the given map with the given expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L27-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.constraint-Tuple{Reactant.MLIR.IR.IntegerSet, Any}' href='#Reactant.MLIR.IR.constraint-Tuple{Reactant.MLIR.IR.IntegerSet, Any}'><span class="jlbinding">Reactant.MLIR.IR.constraint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetConstraint(set, i)
```


Returns `i`-th constraint of the set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L116-L120" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(affineExpr)
```


Gets the context that owns the affine expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L19-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(affineMap)
```


Gets the context that the given affine map was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L34-L38" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(attribute)
```


Gets the context that an attribute was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L29-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Identifier}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Identifier}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(ident)
```


Returns the context associated with this identifier


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Identifier.jl#L22-L26" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(set)
```


Gets the context in which the given integer set lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L60-L64" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Module}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Module}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(module)
```


Gets the context that a module was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Module.jl#L36-L40" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(op)
```


Gets the context this operation is associated with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L28-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.context-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.context</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
context(type)
```


Gets the context that a type was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L27-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.data-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.data-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.data</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
data(attr)
```


Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L253-L257" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.data-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.data-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.data</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueTypeGetData(type)
```


Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L751-L755" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.delete!-Tuple{Reactant.MLIR.IR.SymbolTable, Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.delete!-Tuple{Reactant.MLIR.IR.SymbolTable, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.delete!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
delete!(symboltable, operation)
```


Removes the given operation from the symbol table and erases it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/SymbolTable.jl#L50-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.dynsize-Tuple{}' href='#Reactant.MLIR.IR.dynsize-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.dynsize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
dynsize()
```


Returns the value indicating a dynamic size in a shaped type. Prefer [`isdynsize`](/api/mlirc#Reactant.MLIR.IR.isdynsize-Tuple{Any}) to direct comparisons with this value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L391-L395" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.dynstrideoroffset-Tuple{}' href='#Reactant.MLIR.IR.dynstrideoroffset-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.dynstrideoroffset</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeGetDynamicStrideOrOffset()
```


Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`isdynstrideoroffset`](/api/mlirc#Reactant.MLIR.IR.isdynstrideoroffset-Tuple{Any}) to direct comparisons with this value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L405-L409" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.enable_ir_printing!-Tuple{Any}' href='#Reactant.MLIR.IR.enable_ir_printing!-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.enable_ir_printing!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
enable_ir_printing!(passManager)
```


Enable mlir-print-ir-after-all.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L38-L42" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.enable_verifier!' href='#Reactant.MLIR.IR.enable_verifier!'><span class="jlbinding">Reactant.MLIR.IR.enable_verifier!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
enable_verifier!(passManager, enable)
```


Enable / disable verify-each.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L57-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.encoding-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.encoding-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.encoding</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
encoding(type)
```


Gets the &#39;encoding&#39; attribute from the ranked tensor type, returning a `nothing` if none.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L517-L521" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.erase_argument!-Tuple{Any, Any}' href='#Reactant.MLIR.IR.erase_argument!-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.erase_argument!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
erase_argument!(block, i)
```


Erase argument `i` of the block. Returns the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L88-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.failure-Tuple{}' href='#Reactant.MLIR.IR.failure-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.failure</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
failure()
```


Creates a logical result representing a failure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/LogicalResult.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.first_block-Tuple{Reactant.MLIR.IR.Region}' href='#Reactant.MLIR.IR.first_block-Tuple{Reactant.MLIR.IR.Region}'><span class="jlbinding">Reactant.MLIR.IR.first_block</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
first_block(region)
```


Gets the first block in the region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L73-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.first_op-Tuple{Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.first_op-Tuple{Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.first_op</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
first_op(block)
```


Returns the first operation in the block or `nothing` if empty.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L101-L105" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.first_use-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.first_use-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.first_use</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
first_use(value)
```


Returns an `OpOperand` representing the first use of the value, or a `nothing` if there are no uses.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/OpOperand.jl#L12-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.flatsymbol-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.flatsymbol-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.flatsymbol</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
flatsymbol(attr)
```


Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L359-L363" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.hasrank-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.hasrank-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.hasrank</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
hasrank(type)
```


Checks whether the given shaped type is ranked.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L341-L345" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.hasstaticshape-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.hasstaticshape-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.hasstaticshape</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
hasstaticshape(type)
```


Checks whether the given shaped type has a static shape.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L358-L362" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.input-Tuple{Reactant.MLIR.IR.Type, Any}' href='#Reactant.MLIR.IR.input-Tuple{Reactant.MLIR.IR.Type, Any}'><span class="jlbinding">Reactant.MLIR.IR.input</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
input(type, i)
```


Returns the `i`-th input type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L708-L712" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.insert_after!-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.insert_after!-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.insert_after!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
insert_after!(block, reference, operation)
```


Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L150-L154" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.insert_after!-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.insert_after!-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.insert_after!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
insert_after!(region, reference, block)
```


Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L57-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.insert_before!-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.insert_before!-Tuple{Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.insert_before!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
insert_before!(block, reference, operation)
```


Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L160-L164" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.insert_before!-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.insert_before!-Tuple{Reactant.MLIR.IR.Region, Reactant.MLIR.IR.Block, Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.insert_before!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
insert_before!(region, reference, block)
```


Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Region.jl#L65-L69" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.is_block_arg-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.is_block_arg-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.is_block_arg</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_block_arg(value)
```


Returns 1 if the value is a block argument, 0 otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.is_op_res-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.is_op_res-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.is_op_res</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_op_res(value)
```


Returns 1 if the value is an operation result, 0 otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L28-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.is_pure_affine-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.is_pure_affine-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.is_pure_affine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_pure_affine(affineExpr)
```


Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L33-L37" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.is_registered-Tuple{Any}' href='#Reactant.MLIR.IR.is_registered-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.is_registered</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_registered(name; context=context())
```


Returns whether the given fully-qualified operation (i.e. &#39;dialect.operation&#39;) is registered with the context. This will return true if the dialect is loaded and the operation is registered within the dialect.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L299-L304" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.is_symbolic_or_constant-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.is_symbolic_or_constant-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.is_symbolic_or_constant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
is_symbolic_or_constant(affineExpr)
```


Checks whether the given affine expression is made out of only symbols and constants.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L26-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isadd-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.isadd-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.isadd</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isadd(affineExpr)
```


Checks whether the given affine expression is an add expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L136-L140" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isaffinemap-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isaffinemap-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isaffinemap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isaffinemap(attr)
```


Checks whether the given attribute is an affine map attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L50-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isarray-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isarray-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isarray</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isarray(attr)
```


Checks whether the given attribute is an array attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L71-L75" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isbf16-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isbf16-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isbf16</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isbf16(type)
```


Checks whether the given type is a bf16 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L283-L287" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isbinary-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.isbinary-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.isbinary</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isbinary(affineExpr)
```


Checks whether the given affine expression is binary.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L211-L215" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isbool-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isbool-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isbool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isbool(attr)
```


Checks whether the given attribute is a bool attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L196-L200" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isceildiv-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.isceildiv-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.isceildiv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isceildiv(affineExpr)
```


Checks whether the given affine expression is an ceildiv expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L196-L200" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.iscomplex-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.iscomplex-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.iscomplex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
iscomplex(type)
```


Checks whether the given type is a Complex type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L326-L330" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isconstantexpr-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.isconstantexpr-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.isconstantexpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isconstantexpr(affineExpr)
```


Checks whether the given affine expression is a constant expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L111-L115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isconstrainteq-Tuple{Reactant.MLIR.IR.IntegerSet, Any}' href='#Reactant.MLIR.IR.isconstrainteq-Tuple{Reactant.MLIR.IR.IntegerSet, Any}'><span class="jlbinding">Reactant.MLIR.IR.isconstrainteq</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetIsConstraintEq(set, i)
```


Returns `true` of the `i`-th constraint of the set is an equality constraint, `false` otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L123-L127" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isdenseelements-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isdenseelements-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isdenseelements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isdenseelements(attr)
```


Checks whether the given attribute is a dense elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L414-L418" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isdict-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isdict-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isdict</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isdict(attr)
```


Checks whether the given attribute is a dictionary attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L86-L90" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isdimexpr-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.isdimexpr-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.isdimexpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isdimexpr(affineExpr)
```


Checks whether the given affine expression is a dimension expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L62-L66" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isdyndim-Tuple{Reactant.MLIR.IR.Type, Int64}' href='#Reactant.MLIR.IR.isdyndim-Tuple{Reactant.MLIR.IR.Type, Int64}'><span class="jlbinding">Reactant.MLIR.IR.isdyndim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isdyndim(type, i)
```


Checks wither the `i`-th dimension of the given shaped type is dynamic.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L365-L369" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isdynsize-Tuple{Any}' href='#Reactant.MLIR.IR.isdynsize-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.isdynsize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isdynsize(size)
```


Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L384-L388" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isdynstrideoroffset-Tuple{Any}' href='#Reactant.MLIR.IR.isdynstrideoroffset-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.isdynstrideoroffset</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeIsDynamicStrideOrOffset(val)
```


Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L398-L402" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.iselements-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.iselements-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.iselements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
iselements(attr)
```


Checks whether the given attribute is an elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L404-L408" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf16-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf16-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf16</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf16(type)
```


Checks whether the given type is an f16 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L290-L294" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf32-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf32-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf32</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf32(type)
```


Checks whether the given type is an f32 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L297-L301" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf64-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf64-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf64</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf64(type)
```


Checks whether the given type is an f64 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L304-L308" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf8e4m3b11fnuz-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf8e4m3b11fnuz-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf8e4m3b11fnuz</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf8e4m3b11fnuz(type)
```


Checks whether the given type is an f8E4M3B11FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L262-L266" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf8e4m3fn-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf8e4m3fn-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf8e4m3fn</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf8e4m3fn(type)
```


Checks whether the given type is an f8E4M3FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L255-L259" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf8e4m3fnuz-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf8e4m3fnuz-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf8e4m3fnuz</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf8e4m3fnuz(type)
```


Checks whether the given type is an f8E4M3FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L276-L280" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf8e5m2-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf8e5m2-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf8e5m2</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf8e5m2(type)
```


Checks whether the given type is an f8E5M2 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L248-L252" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isf8e5m2fnuz-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isf8e5m2fnuz-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isf8e5m2fnuz</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isf8e5m2fnuz(type)
```


Checks whether the given type is an f8E5M2FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L269-L273" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isfailure-Tuple{Reactant.MLIR.IR.LogicalResult}' href='#Reactant.MLIR.IR.isfailure-Tuple{Reactant.MLIR.IR.LogicalResult}'><span class="jlbinding">Reactant.MLIR.IR.isfailure</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isfailure(res)
```


Checks if the given logical result represents a failure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/LogicalResult.jl#L35-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isflatsymbolref-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isflatsymbolref-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isflatsymbolref</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isflatsymbolref(attr)
```


Checks whether the given attribute is a flat symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L344-L348" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isfloat-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isfloat-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isfloat</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isfloat(attr)
```


Checks whether the given attribute is a floating point attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L103-L107" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isfloordiv-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.isfloordiv-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.isfloordiv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isfloordiv(affineExpr)
```


Checks whether the given affine expression is an floordiv expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L178-L182" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isfunction-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isfunction-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isfunction</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isfunction(type)
```


Checks whether the given type is a function type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L665-L669" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isfunctionofdimexpr-Tuple{Reactant.MLIR.IR.AffineExpr, Any}' href='#Reactant.MLIR.IR.isfunctionofdimexpr-Tuple{Reactant.MLIR.IR.AffineExpr, Any}'><span class="jlbinding">Reactant.MLIR.IR.isfunctionofdimexpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isfunctionofdimexpr(affineExpr, position)
```


Checks whether the given affine expression involves AffineDimExpr &#39;position&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L54-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isidentity-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.isidentity-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.isidentity</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isidentity(affineMap)
```


Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L100-L104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isindex-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isindex-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isindex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isindex(type)
```


Checks whether the given type is an index type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L64-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isinteger-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isinteger-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isinteger</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isinteger(attr)
```


Checks whether the given attribute is an integer attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L159-L163" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isinteger-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isinteger-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isinteger</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isinteger(type)
```


Checks whether the given type is an integer type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L104-L108" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isintegerset-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isintegerset-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isintegerset</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isintegerset(attr)
```


Checks whether the given attribute is an integer set attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L220-L224" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ismemref-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.ismemref-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.ismemref</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ismemref(type)
```


Checks whether the given type is a MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L593-L597" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isminoridentity-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.isminoridentity-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.isminoridentity</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isminoridentity(affineMap)
```


Checks whether the given affine map is a minor identity affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L107-L111" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ismod-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.ismod-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.ismod</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ismod(affineExpr)
```


Checks whether the given affine expression is an mod expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L164-L168" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ismul-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.ismul-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.ismul</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ismul(affineExpr)
```


Checks whether the given affine expression is an mul expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L150-L154" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ismultipleof-Tuple{Reactant.MLIR.IR.AffineExpr, Any}' href='#Reactant.MLIR.IR.ismultipleof-Tuple{Reactant.MLIR.IR.AffineExpr, Any}'><span class="jlbinding">Reactant.MLIR.IR.ismultipleof</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ismultipleof(affineExpr, factor)
```


Checks whether the given affine expression is a multiple of &#39;factor&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L47-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isnone-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isnone-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isnone</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsANone(type)
```


Checks whether the given type is a None type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L49-L53" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isopaque-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isopaque-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isopaque</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isopaque(attr)
```


Checks whether the given attribute is an opaque attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L227-L231" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isopaque-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isopaque-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isopaque</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isopaque(type)
```


Checks whether the given type is an opaque type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L737-L741" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isprojperm-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.isprojperm-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.isprojperm</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isprojperm(affineMap)
```


Checks whether the given affine map represents a subset of a symbol-less permutation map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L173-L177" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isrankedtensor-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isrankedtensor-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isrankedtensor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isrankedtensor(type)
```


Checks whether the given type is a ranked tensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L503-L507" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isshaped-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isshaped-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isshaped</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isshaped(type)
```


Checks whether the given type is a Shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L334-L338" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issigned-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.issigned-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.issigned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issigned(type)
```


Checks whether the given integer type is signed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L111-L115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issignless-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.issignless-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.issignless</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issignless(type)
```


Checks whether the given integer type is signless.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L118-L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issingleconstant-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.issingleconstant-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.issingleconstant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issingleconstant(affineMap)
```


Checks whether the given affine map is a single result constant affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L121-L125" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issparseelements-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.issparseelements-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.issparseelements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issparseelements(attr)
```


Checks whether the given attribute is a sparse elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L660-L664" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issplat-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.issplat-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.issplat</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issplat(attr)
```


Checks whether the given dense elements attribute contains a single replicated value (splat).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L648-L652" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isstring-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isstring-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isstring</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isstring(attr)
```


Checks whether the given attribute is a string attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L263-L267" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issuccess-Tuple{Reactant.MLIR.IR.LogicalResult}' href='#Reactant.MLIR.IR.issuccess-Tuple{Reactant.MLIR.IR.LogicalResult}'><span class="jlbinding">Reactant.MLIR.IR.issuccess</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issuccess(res)
```


Checks if the given logical result represents a success.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/LogicalResult.jl#L28-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issymbolexpr-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.issymbolexpr-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.issymbolexpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issymbolexpr(affineExpr)
```


Checks whether the given affine expression is a symbol expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L77-L81" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.issymbolref-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.issymbolref-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.issymbolref</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
issymbolref(attr)
```


Checks whether the given attribute is a symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L297-L301" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.istensor-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.istensor-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.istensor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
istensor(type)
```


Checks whether the given type is a Tensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L496-L500" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.istf32-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.istf32-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.istf32</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
istf32(type)
```


Checks whether the given type is an tf32 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L311-L315" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.istuple-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.istuple-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.istuple</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
istuple(type)
```


Checks whether the given type is a tuple type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L657-L661" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.istype-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.istype-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.istype</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
istype(attr)
```


Checks whether the given attribute is a type attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L369-L373" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isunit-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.isunit-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.isunit</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isunit(attr)
```


Checks whether the given attribute is a unit attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L390-L394" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isunrankedmemref-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isunrankedmemref-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isunrankedmemref</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAUnrankedMemRef(type)
```


Checks whether the given type is an UnrankedMemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L600-L604" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isunrankedtensor-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isunrankedtensor-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isunrankedtensor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isunrankedtensor(type)
```


Checks whether the given type is an unranked tensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L510-L514" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isunsigned-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isunsigned-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isunsigned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isunsigned(type)
```


Checks whether the given integer type is unsigned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L125-L129" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.isvector-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.isvector-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.isvector</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
isvector(type)
```


Checks whether the given type is a Vector type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L431-L435" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.layout-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.layout-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.layout</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
layout(type)
```


Returns the layout of the given MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L607-L611" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.leafref-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.leafref-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.leafref</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
leafref(attr)
```


Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L324-L328" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.lhs-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.lhs-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.lhs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
lhs(affineExpr)
```


Returns the left hand side affine expression of the given affine binary operation expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L218-L222" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.location-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.location-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.location</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
location(op)
```


Gets the location of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L35-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.lookup-Tuple{Reactant.MLIR.IR.ExecutionEngine, String}' href='#Reactant.MLIR.IR.lookup-Tuple{Reactant.MLIR.IR.ExecutionEngine, String}'><span class="jlbinding">Reactant.MLIR.IR.lookup</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
lookup(jit, name)
```


Lookup a native function in the execution engine by name, returns nullptr if the name can&#39;t be looked-up.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/ExecutionEngine.jl#L38-L42" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.lookup-Tuple{Reactant.MLIR.IR.SymbolTable, AbstractString}' href='#Reactant.MLIR.IR.lookup-Tuple{Reactant.MLIR.IR.SymbolTable, AbstractString}'><span class="jlbinding">Reactant.MLIR.IR.lookup</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
lookup(symboltable, name)
```


Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol. If the symbol cannot be found, returns a null operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/SymbolTable.jl#L22-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.majorsubmap-Tuple{Reactant.MLIR.IR.AffineMap, Any}' href='#Reactant.MLIR.IR.majorsubmap-Tuple{Reactant.MLIR.IR.AffineMap, Any}'><span class="jlbinding">Reactant.MLIR.IR.majorsubmap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
majorsubmap(affineMap, nresults)
```


Returns the affine map consisting of the most major `nresults` results. Returns the null AffineMap if the `nresults` is equal to zero. Returns the `affineMap` if `nresults` is greater or equals to number of results of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L195-L201" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.memspace-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.memspace-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.memspace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetMemorySpace(type)
```


Returns the memory space of the given MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L627-L631" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.minorsubmap-Tuple{Reactant.MLIR.IR.AffineMap, Any}' href='#Reactant.MLIR.IR.minorsubmap-Tuple{Reactant.MLIR.IR.AffineMap, Any}'><span class="jlbinding">Reactant.MLIR.IR.minorsubmap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
minorsubmap(affineMap, nresults)
```


Returns the affine map consisting of the most minor `nresults` results. Returns the null AffineMap if the `nresults` is equal to zero. Returns the `affineMap` if `nresults` is greater or equals to number of results of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L205-L210" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.move_after!-Tuple{Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.move_after!-Tuple{Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.move_after!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
move_after!(op, other)
```


Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L276-L280" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.move_before!-Tuple{Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.move_before!-Tuple{Reactant.MLIR.IR.Operation, Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.move_before!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
move_before!(op, other)
```


Moves the given operation immediately before the other operation in its parent block. The given operation may be owner by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L286-L293" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.name-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.name-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.name</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
name(op)
```


Gets the name of the operation as an identifier.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L49-L53" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.namespace-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.namespace-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.namespace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueAttrGetDialectNamespace(attr)
```


Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L243-L247" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.namespace-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.namespace-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.namespace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueTypeGetDialectNamespace(type)
```


Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L744-L748" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nargs-Tuple{Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.nargs-Tuple{Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.nargs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nargs(block)
```


Returns the number of arguments of the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L63-L67" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nattrs-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.nattrs-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.nattrs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nattrs(op)
```


Returns the number of attributes attached to the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L172-L176" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nconstraints-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Reactant.MLIR.IR.nconstraints-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Reactant.MLIR.IR.nconstraints</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nconstraints(set)
```


Returns the number of constraints (equalities + inequalities) in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L95-L99" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nequalities-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Reactant.MLIR.IR.nequalities-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Reactant.MLIR.IR.nequalities</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nequalities(set)
```


Returns the number of equalities in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L102-L106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.next-Tuple{Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.next-Tuple{Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.next</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
next(block)
```


Returns the block immediately following the given block in its parent region or `nothing` if last.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L52-L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.next-Tuple{Reactant.MLIR.IR.OpOperand}' href='#Reactant.MLIR.IR.next-Tuple{Reactant.MLIR.IR.OpOperand}'><span class="jlbinding">Reactant.MLIR.IR.next</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
next(opOperand)
```


Returns an op operand representing the next use of the value, or `nothing` if there is no next use.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/OpOperand.jl#L37-L41" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ninequalities-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Reactant.MLIR.IR.ninequalities-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Reactant.MLIR.IR.ninequalities</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ninequalities(set)
```


Returns the number of inequalities in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L109-L113" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ninputs-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.ninputs-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.ninputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ninputs(affineMap)
```


Returns the number of inputs (dimensions + symbols) of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L166-L170" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ninputs-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Reactant.MLIR.IR.ninputs-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Reactant.MLIR.IR.ninputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ninputs(set)
```


Returns the number of inputs (dimensions + symbols) in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L88-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.ninputs-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.ninputs-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.ninputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ninputs(type)
```


Returns the number of input types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L688-L692" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nnestedrefs-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.nnestedrefs-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.nnestedrefs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nnestedrefs(attr)
```


Returns the number of references nested in the given symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L334-L338" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.noperands-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.noperands-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.noperands</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
noperands(op)
```


Returns the number of operands of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L120-L124" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nregions-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.nregions-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.nregions</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nregions(op)
```


Returns the number of regions attached to the given operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L85-L89" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nresults-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.nresults-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.nresults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nresults(affineMap)
```


Returns the number of results of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L152-L156" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nresults-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.nresults-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.nresults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nresults(op)
```


Returns the number of results of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L102-L106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nresults-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.nresults-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.nresults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nresults(type)
```


Returns the number of result types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L698-L702" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nsuccessors-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.nsuccessors-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.nsuccessors</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nsuccessors(op)
```


Returns the number of successor blocks of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L155-L159" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nsymbols-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.nsymbols-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.nsymbols</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nsymbols(affineMap)
```


Returns the number of symbols of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L145-L149" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.nsymbols-Tuple{Reactant.MLIR.IR.IntegerSet}' href='#Reactant.MLIR.IR.nsymbols-Tuple{Reactant.MLIR.IR.IntegerSet}'><span class="jlbinding">Reactant.MLIR.IR.nsymbols</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
nsymbols(set)
```


Returns the number of symbols in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IntegerSet.jl#L81-L85" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.op_owner-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.op_owner-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.op_owner</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
op_owner(value)
```


Returns an operation that produced this value as its result. Asserts if the value is not an op result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L45-L49" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.op_res_num-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.op_res_num-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.op_res_num</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
op_res_num(value)
```


Returns the position of the value in the list of results of the operation that produced it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L79-L83" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.operand' href='#Reactant.MLIR.IR.operand'><span class="jlbinding">Reactant.MLIR.IR.operand</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
operand(op, i)
```


Returns `i`-th operand of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L127-L131" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.operand!-Tuple{Reactant.MLIR.IR.Operation, Any, Any}' href='#Reactant.MLIR.IR.operand!-Tuple{Reactant.MLIR.IR.Operation, Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.operand!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
operand!(op, i, value)
```


Sets the `i`-th operand of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L144-L148" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.operandindex-Tuple{Reactant.MLIR.IR.OpOperand}' href='#Reactant.MLIR.IR.operandindex-Tuple{Reactant.MLIR.IR.OpOperand}'><span class="jlbinding">Reactant.MLIR.IR.operandindex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
operandindex(opOperand)
```


Returns the operand number of an op operand.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/OpOperand.jl#L30-L34" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.operands-Tuple{Any}' href='#Reactant.MLIR.IR.operands-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.operands</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
operands(op)
```


Return an array of all operands of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L137-L141" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.owner-Tuple{Reactant.MLIR.IR.OpOperand}' href='#Reactant.MLIR.IR.owner-Tuple{Reactant.MLIR.IR.OpOperand}'><span class="jlbinding">Reactant.MLIR.IR.owner</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
owner(opOperand)
```


Returns the owner operation of an op operand.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/OpOperand.jl#L23-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.parent_op-Tuple{Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.parent_op-Tuple{Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.parent_op</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parent_op(block)
```


Returns the closest surrounding operation that contains this block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L36-L40" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.parent_op-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.parent_op-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.parent_op</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parent_op(op)
```


Gets the operation that owns this operation, returning null if the operation is not owned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L63-L67" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.parent_region-Tuple{Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.parent_region-Tuple{Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.parent_region</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
parent_region(block)
```


Returns the region that contains this block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L43-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.pass_pipeline-Tuple{Reactant.MLIR.IR.OpPassManager}' href='#Reactant.MLIR.IR.pass_pipeline-Tuple{Reactant.MLIR.IR.OpPassManager}'><span class="jlbinding">Reactant.MLIR.IR.pass_pipeline</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
pass_pipeline(opPassManager) -> String
```


Returns the pass pipeline.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L202-L206" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.position-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.position-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.position</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
position(affineExpr)
```


Returns the position of the given affine dimension expression, affine symbol expression or ...


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L92-L96" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.print_pass_pipeline-Tuple{IO, Reactant.MLIR.IR.OpPassManager}' href='#Reactant.MLIR.IR.print_pass_pipeline-Tuple{IO, Reactant.MLIR.IR.OpPassManager}'><span class="jlbinding">Reactant.MLIR.IR.print_pass_pipeline</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
print_pass_pipeline(io::IO, opPassManager)
```


Prints the pass pipeline to the IO.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L209-L213" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.push_argument!-Tuple{Reactant.MLIR.IR.Block, Any}' href='#Reactant.MLIR.IR.push_argument!-Tuple{Reactant.MLIR.IR.Block, Any}'><span class="jlbinding">Reactant.MLIR.IR.push_argument!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
push_argument!(block, type; location=Location())
```


Appends an argument of the specified type to the block. Returns the newly added argument.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L80-L84" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.region-Tuple{Reactant.MLIR.IR.Operation, Any}' href='#Reactant.MLIR.IR.region-Tuple{Reactant.MLIR.IR.Operation, Any}'><span class="jlbinding">Reactant.MLIR.IR.region</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
region(op, i)
```


Returns `i`-th region attached to the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L92-L96" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.result' href='#Reactant.MLIR.IR.result'><span class="jlbinding">Reactant.MLIR.IR.result</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
result(op, i)
```


Returns `i`-th result of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L109-L113" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.result-2' href='#Reactant.MLIR.IR.result-2'><span class="jlbinding">Reactant.MLIR.IR.result</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
result(type, i)
```


Returns the `i`-th result type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L718-L722" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.result-Tuple{Reactant.MLIR.IR.AffineMap, Any}' href='#Reactant.MLIR.IR.result-Tuple{Reactant.MLIR.IR.AffineMap, Any}'><span class="jlbinding">Reactant.MLIR.IR.result</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
result(affineMap, pos)
```


Returns the result at the given position.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L159-L163" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.result-Tuple{Reactant.MLIR.IR.AffineMap}' href='#Reactant.MLIR.IR.result-Tuple{Reactant.MLIR.IR.AffineMap}'><span class="jlbinding">Reactant.MLIR.IR.result</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
result(affineMap)
```


Returns the constant result of the given affine map. The function asserts that the map has a single constant result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L128-L132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.rhs-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.rhs-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.rhs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
rhs(affineExpr)
```


Returns the right hand side affine expression of the given affine binary operation expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L225-L229" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.rmattr!-Tuple{Reactant.MLIR.IR.Operation, Any}' href='#Reactant.MLIR.IR.rmattr!-Tuple{Reactant.MLIR.IR.Operation, Any}'><span class="jlbinding">Reactant.MLIR.IR.rmattr!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
rmattr!(op, name)
```


Removes an attribute by name. Returns false if the attribute was not found and true if removed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L212-L216" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.rmfromparent!-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.rmfromparent!-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.rmfromparent!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
rmfromparent!(op)
```


Removes the given operation from its parent block. The operation is not destroyed. The ownership of the operation is transferred to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L71-L76" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.rootref-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.rootref-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.rootref</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
rootref(attr)
```


Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L314-L318" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.run!' href='#Reactant.MLIR.IR.run!'><span class="jlbinding">Reactant.MLIR.IR.run!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
run!(passManager, module)
```


Run the provided `passManager` on the given `module`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Pass.jl#L142-L146" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.submap-Tuple{Reactant.MLIR.IR.AffineMap, Vector{Int64}}' href='#Reactant.MLIR.IR.submap-Tuple{Reactant.MLIR.IR.AffineMap, Vector{Int64}}'><span class="jlbinding">Reactant.MLIR.IR.submap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
submap(affineMap, positions)
```


Returns the affine map consisting of the `positions` subset.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L187-L191" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.success-Tuple{}' href='#Reactant.MLIR.IR.success-Tuple{}'><span class="jlbinding">Reactant.MLIR.IR.success</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
success()
```


Creates a logical result representing a success.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/LogicalResult.jl#L14-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.successor-Tuple{Reactant.MLIR.IR.Operation, Any}' href='#Reactant.MLIR.IR.successor-Tuple{Reactant.MLIR.IR.Operation, Any}'><span class="jlbinding">Reactant.MLIR.IR.successor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
successor(op, i)
```


Returns `i`-th successor of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L162-L166" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.terminator-Tuple{Reactant.MLIR.IR.Block}' href='#Reactant.MLIR.IR.terminator-Tuple{Reactant.MLIR.IR.Block}'><span class="jlbinding">Reactant.MLIR.IR.terminator</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
terminator(block)
```


Returns the terminator operation in the block or `nothing` if no terminator.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Block.jl#L113-L117" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.type!-Tuple{Any, Any}' href='#Reactant.MLIR.IR.type!-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.IR.type!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
set_type!(value, type)
```


Sets the type of the block argument to the given type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L106-L110" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.type-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.type-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
type(attribute)
```


Gets the type of this attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L36-L40" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.type-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.IR.type-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.IR.type</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
type(value)
```


Returns the type of the value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Value.jl#L99-L103" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.typeid-Tuple{Reactant.MLIR.IR.Attribute}' href='#Reactant.MLIR.IR.typeid-Tuple{Reactant.MLIR.IR.Attribute}'><span class="jlbinding">Reactant.MLIR.IR.typeid</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
typeid(attribute)
```


Gets the type id of the attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Attribute.jl#L43-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.typeid-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.typeid-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.typeid</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
typeid(op)
```


Gets the type id of the operation. Returns null if the operation does not have a registered operation description.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L42-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.typeid-Tuple{Reactant.MLIR.IR.Type}' href='#Reactant.MLIR.IR.typeid-Tuple{Reactant.MLIR.IR.Type}'><span class="jlbinding">Reactant.MLIR.IR.typeid</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
typeid(type)
```


Gets the type ID of the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Type.jl#L34-L38" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.value-Tuple{Reactant.MLIR.IR.AffineExpr}' href='#Reactant.MLIR.IR.value-Tuple{Reactant.MLIR.IR.AffineExpr}'><span class="jlbinding">Reactant.MLIR.IR.value</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
value(affineExpr)
```


Returns the value of the given affine constant expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineExpr.jl#L126-L130" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.verify-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.verify-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.verify</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
verify(op)
```


Verify the operation and return true if it passes, false if it fails.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/Operation.jl#L269-L273" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.verifyall-Tuple{Reactant.MLIR.IR.Operation}' href='#Reactant.MLIR.IR.verifyall-Tuple{Reactant.MLIR.IR.Operation}'><span class="jlbinding">Reactant.MLIR.IR.verifyall</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
verifyall(operation; debug=false)
```


Prints the operations which could not be verified.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/IR.jl#L117-L121" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.IR.@affinemap-Tuple{Any}' href='#Reactant.MLIR.IR.@affinemap-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.IR.@affinemap</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@affinemap (d1, d2, d3, ...)[s1, s2, ...] -> (d0 + d1, ...)
```


Returns an affine map from the provided Julia expression. On the right hand side are allowed the following function calls:
- +, *, ÷, %, fld, cld
  

The rhs can only contains dimensions and symbols present on the left hand side or integer literals.

```julia
julia> using Reactant.MLIR: IR

julia> IR.context!(IR.Context()) do
           IR.@affinemap (d1, d2)[s0] -> (d1 + s0, d2 % 10)
       end
MLIR.IR.AffineMap(#= (d0, d1)[s0] -> (d0 + s0, d1 mod 10) =#)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/IR/AffineMap.jl#L249-L267" target="_blank" rel="noreferrer">source</a></Badge>

</details>


# MLIR C API {#MLIR-C-API}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMAttributeRef' href='#Reactant.MLIR.API.LLVMAttributeRef'><span class="jlbinding">Reactant.MLIR.API.LLVMAttributeRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Used to represent an attributes.

**See also**

llvm::Attribute


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9648-L9653" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMBasicBlockRef' href='#Reactant.MLIR.API.LLVMBasicBlockRef'><span class="jlbinding">Reactant.MLIR.API.LLVMBasicBlockRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents a basic block of instructions in LLVM IR.

This models llvm::BasicBlock.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9561-L9565" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMBinaryRef' href='#Reactant.MLIR.API.LLVMBinaryRef'><span class="jlbinding">Reactant.MLIR.API.LLVMBinaryRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::object::Binary


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9690-L9693" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMBool' href='#Reactant.MLIR.API.LLVMBool'><span class="jlbinding">Reactant.MLIR.API.LLVMBool</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



`LLVMCSupportTypes Types and Enumerations`

@{


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9506-L9510" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMBuilderRef' href='#Reactant.MLIR.API.LLVMBuilderRef'><span class="jlbinding">Reactant.MLIR.API.LLVMBuilderRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents an LLVM basic block builder.

This models llvm::IRBuilder.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9597-L9601" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMComdatRef' href='#Reactant.MLIR.API.LLVMComdatRef'><span class="jlbinding">Reactant.MLIR.API.LLVMComdatRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::Comdat


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9666-L9669" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMContextRef' href='#Reactant.MLIR.API.LLVMContextRef'><span class="jlbinding">Reactant.MLIR.API.LLVMContextRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



The top-level container for all LLVM global data. See the LLVMContext class.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9525-L9527" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMDIBuilderRef' href='#Reactant.MLIR.API.LLVMDIBuilderRef'><span class="jlbinding">Reactant.MLIR.API.LLVMDIBuilderRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents an LLVM debug info builder.

This models llvm::DIBuilder.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9606-L9610" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMDbgRecordRef' href='#Reactant.MLIR.API.LLVMDbgRecordRef'><span class="jlbinding">Reactant.MLIR.API.LLVMDbgRecordRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::DbgRecord


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9698-L9701" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMDiagnosticInfoRef' href='#Reactant.MLIR.API.LLVMDiagnosticInfoRef'><span class="jlbinding">Reactant.MLIR.API.LLVMDiagnosticInfoRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::DiagnosticInfo


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9658-L9661" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMJITEventListenerRef' href='#Reactant.MLIR.API.LLVMJITEventListenerRef'><span class="jlbinding">Reactant.MLIR.API.LLVMJITEventListenerRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::JITEventListener


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9682-L9685" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMMemoryBufferRef' href='#Reactant.MLIR.API.LLVMMemoryBufferRef'><span class="jlbinding">Reactant.MLIR.API.LLVMMemoryBufferRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Used to pass regions of memory through LLVM interfaces.

**See also**

llvm::MemoryBuffer


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9515-L9520" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMMetadataRef' href='#Reactant.MLIR.API.LLVMMetadataRef'><span class="jlbinding">Reactant.MLIR.API.LLVMMetadataRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents an LLVM Metadata.

This models llvm::Metadata.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9570-L9574" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMModuleFlagEntry' href='#Reactant.MLIR.API.LLVMModuleFlagEntry'><span class="jlbinding">Reactant.MLIR.API.LLVMModuleFlagEntry</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::Module::ModuleFlagEntry


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9674-L9677" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMModuleProviderRef' href='#Reactant.MLIR.API.LLVMModuleProviderRef'><span class="jlbinding">Reactant.MLIR.API.LLVMModuleProviderRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Interface used to provide a module to JIT or interpreter. This is now just a synonym for llvm::Module, but we have to keep using the different type to keep binary compatibility.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9615-L9617" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMModuleRef' href='#Reactant.MLIR.API.LLVMModuleRef'><span class="jlbinding">Reactant.MLIR.API.LLVMModuleRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



The top-level container for all other LLVM Intermediate Representation (IR) objects.

**See also**

llvm::Module


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9532-L9537" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMNamedMDNodeRef' href='#Reactant.MLIR.API.LLVMNamedMDNodeRef'><span class="jlbinding">Reactant.MLIR.API.LLVMNamedMDNodeRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents an LLVM Named Metadata Node.

This models llvm::NamedMDNode.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9579-L9583" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMOperandBundleRef' href='#Reactant.MLIR.API.LLVMOperandBundleRef'><span class="jlbinding">Reactant.MLIR.API.LLVMOperandBundleRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::OperandBundleDef


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9640-L9643" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMPassManagerRef' href='#Reactant.MLIR.API.LLVMPassManagerRef'><span class="jlbinding">Reactant.MLIR.API.LLVMPassManagerRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**See also**

llvm::PassManagerBase


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9622-L9625" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMTypeRef' href='#Reactant.MLIR.API.LLVMTypeRef'><span class="jlbinding">Reactant.MLIR.API.LLVMTypeRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Each value in the LLVM IR has a type, an [`LLVMTypeRef`](/api/mlirc#Reactant.MLIR.API.LLVMTypeRef).

**See also**

llvm::Type


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9542-L9547" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMUseRef' href='#Reactant.MLIR.API.LLVMUseRef'><span class="jlbinding">Reactant.MLIR.API.LLVMUseRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Used to get the users and usees of a Value.

**See also**

llvm::Use


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9630-L9635" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMValueMetadataEntry' href='#Reactant.MLIR.API.LLVMValueMetadataEntry'><span class="jlbinding">Reactant.MLIR.API.LLVMValueMetadataEntry</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents an entry in a Global Object&#39;s metadata attachments.

This models std::pair&lt;unsigned, MDNode *&gt;


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9588-L9592" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMValueRef' href='#Reactant.MLIR.API.LLVMValueRef'><span class="jlbinding">Reactant.MLIR.API.LLVMValueRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Represents an individual value in LLVM IR.

This models llvm::Value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9552-L9556" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirDiagnostic' href='#Reactant.MLIR.API.MlirDiagnostic'><span class="jlbinding">Reactant.MLIR.API.MlirDiagnostic</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirDiagnostic
```


An opaque reference to a diagnostic, always owned by the diagnostics engine (context). Must not be stored outside of the diagnostic handler.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6138-L6142" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirDiagnosticHandler' href='#Reactant.MLIR.API.MlirDiagnosticHandler'><span class="jlbinding">Reactant.MLIR.API.MlirDiagnosticHandler</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Diagnostic handler type. Accepts a reference to a diagnostic, which is only guaranteed to be live during the call. The handler is passed the `userData` that was provided when the handler was attached to a context. If the handler processed the diagnostic completely, it is expected to return success. Otherwise, it is expected to return failure to indicate that other handlers should attempt to process the diagnostic.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6165-L6167" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirDiagnosticHandlerID' href='#Reactant.MLIR.API.MlirDiagnosticHandlerID'><span class="jlbinding">Reactant.MLIR.API.MlirDiagnosticHandlerID</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Opaque identifier of a diagnostic handler, useful to detach a handler.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6159-L6161" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirDiagnosticSeverity' href='#Reactant.MLIR.API.MlirDiagnosticSeverity'><span class="jlbinding">Reactant.MLIR.API.MlirDiagnosticSeverity</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirDiagnosticSeverity
```


Severity of a diagnostic.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6147-L6151" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirExternalPassCallbacks' href='#Reactant.MLIR.API.MlirExternalPassCallbacks'><span class="jlbinding">Reactant.MLIR.API.MlirExternalPassCallbacks</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirExternalPassCallbacks
```


Structure of external [`MlirPass`](@ref) callbacks. All callbacks are required to be set unless otherwise specified.

| Field      | Note                                                                                                                                                                                             |
|:---------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| construct  | This callback is called from the pass is created. This is analogous to a C++ pass constructor.                                                                                                   |
| destruct   | This callback is called when the pass is destroyed This is analogous to a C++ pass destructor.                                                                                                   |
| initialize | This callback is optional. The callback is called before the pass is run, allowing a chance to initialize any complex state necessary for running the pass. See Pass::initialize(MLIRContext *). |
| clone      | This callback is called when the pass is cloned. See Pass::clonePass().                                                                                                                          |
| run        | This callback is called when the pass is run. See Pass::runOnOperation().                                                                                                                        |



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8937-L8949" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirLlvmThreadPool' href='#Reactant.MLIR.API.MlirLlvmThreadPool'><span class="jlbinding">Reactant.MLIR.API.MlirLlvmThreadPool</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirLlvmThreadPool
```


Re-export llvm::ThreadPool so as to avoid including the LLVM C API directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L48-L52" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirLogicalResult' href='#Reactant.MLIR.API.MlirLogicalResult'><span class="jlbinding">Reactant.MLIR.API.MlirLogicalResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirLogicalResult
```


A logical result value, essentially a boolean with named states. LLVM convention for using boolean values to designate success or failure of an operation is a moving target, so MLIR opted for an explicit class. Instances of [`MlirLogicalResult`](/api/mlirc#Reactant.MLIR.API.MlirLogicalResult) must only be inspected using the associated functions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L115-L119" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirNamedAttribute' href='#Reactant.MLIR.API.MlirNamedAttribute'><span class="jlbinding">Reactant.MLIR.API.MlirNamedAttribute</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirNamedAttribute
```


Named MLIR attribute.

A named attribute is essentially a (name, attribute) pair where the name is a string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L311-L317" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirOperationState' href='#Reactant.MLIR.API.MlirOperationState'><span class="jlbinding">Reactant.MLIR.API.MlirOperationState</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirOperationState
```


An auxiliary class for constructing operations.

This class contains all the information necessary to construct the operation. It owns the MlirRegions it has pointers to and does not own anything else. By default, the state can be constructed from a name and location, the latter being also used to access the context, and has no other components. These components can be added progressively until the operation is constructed. Users are not expected to rely on the internals of this class and should use mlirOperationState* functions instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1001-L1007" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirOperationWalkCallback' href='#Reactant.MLIR.API.MlirOperationWalkCallback'><span class="jlbinding">Reactant.MLIR.API.MlirOperationWalkCallback</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Operation walker type. The handler is passed an (opaque) reference to an operation and a pointer to a `userData`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1735-L1737" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirShapedTypeComponentsCallback' href='#Reactant.MLIR.API.MlirShapedTypeComponentsCallback'><span class="jlbinding">Reactant.MLIR.API.MlirShapedTypeComponentsCallback</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



These callbacks are used to return multiple shaped type components from functions while transferring ownership to the caller. The first argument is the has rank boolean followed by the the rank and a pointer to the shape (if applicable). The next argument is the element type, then the attribute. The last argument is an opaque pointer forwarded to the callback by the caller. This callback will be called potentially multiple times for each shaped type components.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8702-L8704" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirSparseTensorLevelType' href='#Reactant.MLIR.API.MlirSparseTensorLevelType'><span class="jlbinding">Reactant.MLIR.API.MlirSparseTensorLevelType</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Dimension level types (and properties) that define sparse tensors. See the documentation in SparseTensorAttrDefs.td for their meaning.

These correspond to SparseTensorEncodingAttr::LevelType in the C++ API. If updating, keep them in sync and update the static_assert in the impl file.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8186-L8190" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirStringCallback' href='#Reactant.MLIR.API.MlirStringCallback'><span class="jlbinding">Reactant.MLIR.API.MlirStringCallback</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



A callback for returning string references.

This function is called back by the functions that need to return a reference to the portion of the string with the following arguments: - an [`MlirStringRef`](/api/mlirc#Reactant.MLIR.API.MlirStringRef) representing the current portion of the string - a pointer to user data forwarded from the printing call.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L108-L112" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirStringRef' href='#Reactant.MLIR.API.MlirStringRef'><span class="jlbinding">Reactant.MLIR.API.MlirStringRef</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirStringRef
```


A pointer to a sized fragment of a string, not necessarily null-terminated. Does not own the underlying string. This is equivalent to llvm::StringRef.

| Field  | Note                         |
|:------ |:---------------------------- |
| data   | Pointer to the first symbol. |
| length | Length of the fragment.      |



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L65-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirTypesCallback' href='#Reactant.MLIR.API.MlirTypesCallback'><span class="jlbinding">Reactant.MLIR.API.MlirTypesCallback</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



These callbacks are used to return multiple types from functions while transferring ownership to the caller. The first argument is the number of consecutive elements pointed to by the second argument. The third argument is an opaque pointer forwarded to the callback by the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8654-L8656" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirWalkOrder' href='#Reactant.MLIR.API.MlirWalkOrder'><span class="jlbinding">Reactant.MLIR.API.MlirWalkOrder</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirWalkOrder
```


Traversal order for operation walk.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1724-L1728" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.MlirWalkResult' href='#Reactant.MLIR.API.MlirWalkResult'><span class="jlbinding">Reactant.MLIR.API.MlirWalkResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MlirWalkResult
```


Operation walk result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1713-L1717" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMAddSymbol-Tuple{Any, Any}' href='#Reactant.MLIR.API.LLVMAddSymbol-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.LLVMAddSymbol</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
LLVMAddSymbol(symbolName, symbolValue)
```


This functions permanently adds the symbol `symbolName` with the value `symbolValue`. These symbols are searched before any libraries.

**See also**

sys::DynamicLibrary::AddSymbol()


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9742-L9749" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMLoadLibraryPermanently-Tuple{Any}' href='#Reactant.MLIR.API.LLVMLoadLibraryPermanently-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.LLVMLoadLibraryPermanently</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
LLVMLoadLibraryPermanently(Filename)
```


This function permanently loads the dynamic library at the given path. It is safe to call this function multiple times for the same library.

**See also**

sys::DynamicLibrary::LoadLibraryPermanently()


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9704-L9711" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMParseCommandLineOptions-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.LLVMParseCommandLineOptions-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.LLVMParseCommandLineOptions</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
LLVMParseCommandLineOptions(argc, argv, Overview)
```


This function parses the given arguments using the LLVM command line parser. Note that the only stable thing about this function is its signature; you cannot rely on any particular set of command line arguments being interpreted the same way across LLVM versions.

**See also**

llvm::cl::ParseCommandLineOptions()


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9716-L9723" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.LLVMSearchForAddressOfSymbol-Tuple{Any}' href='#Reactant.MLIR.API.LLVMSearchForAddressOfSymbol-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.LLVMSearchForAddressOfSymbol</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
LLVMSearchForAddressOfSymbol(symbolName)
```


This function will search through all previously loaded dynamic libraries for the symbol `symbolName`. If it is found, the address of that symbol is returned. If not, null is returned.

**See also**

sys::DynamicLibrary::SearchForAddressOfSymbol()


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9730-L9737" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineAddExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineAddExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineAddExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineAddExprGet(lhs, rhs)
```


Creates an affine add expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2845-L2849" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineBinaryOpExprGetLHS-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineBinaryOpExprGetLHS-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineBinaryOpExprGetLHS</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineBinaryOpExprGetLHS(affineExpr)
```


Returns the left hand side affine expression of the given affine binary operation expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2945-L2949" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineBinaryOpExprGetRHS-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineBinaryOpExprGetRHS-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineBinaryOpExprGetRHS</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineBinaryOpExprGetRHS(affineExpr)
```


Returns the right hand side affine expression of the given affine binary operation expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2954-L2958" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineCeilDivExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineCeilDivExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineCeilDivExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineCeilDivExprGet(lhs, rhs)
```


Creates an affine ceildiv expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2925-L2929" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineConstantExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineConstantExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineConstantExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineConstantExprGet(ctx, constant)
```


Creates an affine constant expression with &#39;constant&#39; in the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2816-L2820" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineConstantExprGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineConstantExprGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineConstantExprGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineConstantExprGetValue(affineExpr)
```


Returns the value of the given affine constant expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2827-L2831" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineDimExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineDimExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineDimExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineDimExprGet(ctx, position)
```


Creates an affine dimension expression with &#39;position&#39; in the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2760-L2764" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineDimExprGetPosition-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineDimExprGetPosition-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineDimExprGetPosition</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineDimExprGetPosition(affineExpr)
```


Returns the position of the given affine dimension expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2769-L2773" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprCompose-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineExprCompose-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprCompose</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprCompose(affineExpr, affineMap)
```


Composes the given map with the given expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2707-L2711" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprDump(affineExpr)
```


Prints the affine expression to the standard error stream.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2645-L2649" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineExprEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprEqual(lhs, rhs)
```


Returns `true` if the two affine expressions are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2616-L2620" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprGetContext(affineExpr)
```


Gets the context that owns the affine expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2607-L2611" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprGetLargestKnownDivisor-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprGetLargestKnownDivisor-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprGetLargestKnownDivisor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprGetLargestKnownDivisor(affineExpr)
```


Returns the greatest known integral divisor of this affine expression. The result is always positive.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2672-L2676" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsAAdd-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsAAdd-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsAAdd</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsAAdd(affineExpr)
```


Checks whether the given affine expression is an add expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2836-L2840" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsABinary-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsABinary-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsABinary</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsABinary(affineExpr)
```


Checks whether the given affine expression is binary.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2936-L2940" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsACeilDiv-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsACeilDiv-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsACeilDiv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsACeilDiv(affineExpr)
```


Checks whether the given affine expression is an ceildiv expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2916-L2920" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsAConstant-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsAConstant-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsAConstant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsAConstant(affineExpr)
```


Checks whether the given affine expression is a constant expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2807-L2811" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsADim-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsADim-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsADim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsADim(affineExpr)
```


Checks whether the given affine expression is a dimension expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2751-L2755" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsAFloorDiv-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsAFloorDiv-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsAFloorDiv</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsAFloorDiv(affineExpr)
```


Checks whether the given affine expression is an floordiv expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2896-L2900" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsAMod-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsAMod-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsAMod</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsAMod(affineExpr)
```


Checks whether the given affine expression is an mod expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2876-L2880" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsAMul-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsAMul-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsAMul</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsAMul(affineExpr)
```


Checks whether the given affine expression is an mul expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2856-L2860" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsASymbol-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsASymbol-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsASymbol</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsASymbol(affineExpr)
```


Checks whether the given affine expression is a symbol expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2778-L2782" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsFunctionOfDim-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineExprIsFunctionOfDim-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsFunctionOfDim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsFunctionOfDim(affineExpr, position)
```


Checks whether the given affine expression involves AffineDimExpr &#39;position&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2692-L2696" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsMultipleOf-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineExprIsMultipleOf-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsMultipleOf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsMultipleOf(affineExpr, factor)
```


Checks whether the given affine expression is a multiple of &#39;factor&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2681-L2685" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsNull(affineExpr)
```


Returns `true` if the given affine expression is a null expression. Note constant zero is not a null expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2625-L2629" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsPureAffine-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsPureAffine-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsPureAffine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsPureAffine(affineExpr)
```


Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2663-L2667" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprIsSymbolicOrConstant-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineExprIsSymbolicOrConstant-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprIsSymbolicOrConstant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprIsSymbolicOrConstant(affineExpr)
```


Checks whether the given affine expression is made out of only symbols and constants.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2654-L2658" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAffineExprPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprPrint(affineExpr, callback, userData)
```


Prints an affine expression by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2634-L2638" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprShiftDims-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirAffineExprShiftDims-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprShiftDims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprShiftDims(affineExpr, numDims, shift, offset)
```


Replace dims[offset ... numDims) by dims[offset + shift ... shift + numDims).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2718-L2722" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineExprShiftSymbols-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirAffineExprShiftSymbols-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineExprShiftSymbols</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineExprShiftSymbols(affineExpr, numSymbols, shift, offset)
```


Replace symbols[offset ... numSymbols) by symbols[offset + shift ... shift + numSymbols).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2729-L2733" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineFloorDivExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineFloorDivExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineFloorDivExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineFloorDivExprGet(lhs, rhs)
```


Creates an affine floordiv expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2905-L2909" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapAttrGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapAttrGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapAttrGet(map)
```


Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3463-L3467" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirAffineMapAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapAttrGetTypeID()
```


Returns the typeID of an AffineMap attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3481-L3485" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapAttrGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapAttrGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapAttrGetValue(attr)
```


Returns the affine map wrapped in the given affine map attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3472-L3476" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapCompressUnusedSymbols-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirAffineMapCompressUnusedSymbols-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapCompressUnusedSymbols</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)
```


Returns the simplified affine map resulting from dropping the symbols that do not appear in any of the individual maps in `affineMaps`. Asserts that all maps in `affineMaps` are normalized to the same number of dims and symbols. Takes a callback `populateResult` to fill the `res` container with value `m` at entry `idx`. This allows returning without worrying about ownership considerations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3247-L3251" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapConstantGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapConstantGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapConstantGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapConstantGet(ctx, val)
```


Creates a single constant result affine map in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3045-L3049" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapDump(affineMap)
```


Prints the affine map to the standard error stream.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3001-L3005" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapEmptyGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapEmptyGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapEmptyGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapEmptyGet(ctx)
```


Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3010-L3014" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapEqual(a1, a2)
```


Checks if two affine maps are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2981-L2985" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirAffineMapGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
```


Creates an affine map with results defined by the given list of affine expressions. The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3030-L3034" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetContext(affineMap)
```


Gets the context that the given affine map was created with


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2963-L2967" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetMajorSubMap-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapGetMajorSubMap-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetMajorSubMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetMajorSubMap(affineMap, numResults)
```


Returns the affine map consisting of the most major `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3208-L3212" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetMinorSubMap-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapGetMinorSubMap-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetMinorSubMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetMinorSubMap(affineMap, numResults)
```


Returns the affine map consisting of the most minor `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3219-L3223" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetNumDims-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapGetNumDims-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetNumDims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetNumDims(affineMap)
```


Returns the number of dimensions of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3132-L3136" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetNumInputs-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapGetNumInputs-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetNumInputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetNumInputs(affineMap)
```


Returns the number of inputs (dimensions + symbols) of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3170-L3174" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetNumResults-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapGetNumResults-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetNumResults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetNumResults(affineMap)
```


Returns the number of results of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3150-L3154" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetNumSymbols-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapGetNumSymbols-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetNumSymbols</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetNumSymbols(affineMap)
```


Returns the number of symbols of the given affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3141-L3145" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetResult-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapGetResult-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetResult</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetResult(affineMap, pos)
```


Returns the result at the given position.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3159-L3163" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetSingleConstantResult-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapGetSingleConstantResult-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetSingleConstantResult</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetSingleConstantResult(affineMap)
```


Returns the constant result of the given affine map. The function asserts that the map has a single constant result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3123-L3127" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapGetSubMap-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapGetSubMap-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapGetSubMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapGetSubMap(affineMap, size, resultPos)
```


Returns the affine map consisting of the `resultPos` subset.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3197-L3201" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsEmpty-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsEmpty-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsEmpty</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsEmpty(affineMap)
```


Checks whether the given affine map is an empty affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3105-L3109" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsIdentity-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsIdentity-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsIdentity</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsIdentity(affineMap)
```


Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3087-L3091" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsMinorIdentity-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsMinorIdentity-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsMinorIdentity</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsMinorIdentity(affineMap)
```


Checks whether the given affine map is a minor identity affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3096-L3100" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsNull(affineMap)
```


Checks whether an affine map is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2972-L2976" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsPermutation-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsPermutation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsPermutation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsPermutation(affineMap)
```


Checks whether the given affine map represents a symbol-less permutation map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3188-L3192" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsProjectedPermutation-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsProjectedPermutation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsProjectedPermutation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsProjectedPermutation(affineMap)
```


Checks whether the given affine map represents a subset of a symbol-less permutation map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3179-L3183" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapIsSingleConstant-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineMapIsSingleConstant-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapIsSingleConstant</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapIsSingleConstant(affineMap)
```


Checks whether the given affine map is a single result constant affine map.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3114-L3118" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapMinorIdentityGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapMinorIdentityGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapMinorIdentityGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapMinorIdentityGet(ctx, dims, results)
```


Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3065-L3069" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapMultiDimIdentityGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapMultiDimIdentityGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapMultiDimIdentityGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapMultiDimIdentityGet(ctx, numDims)
```


Creates an affine map with &#39;numDims&#39; identity in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3054-L3058" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapPermutationGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapPermutationGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapPermutationGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapPermutationGet(ctx, size, permutation)
```


Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid permutation.) The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3076-L3080" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapPrint(affineMap, callback, userData)
```


Prints an affine map by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2990-L2994" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapReplace-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirAffineMapReplace-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapReplace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)
```


Apply AffineExpr::replace(`map`) to each of the results and return a new new AffineMap with the new results and the specified number of dims and symbols.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3230-L3234" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMapZeroResultGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAffineMapZeroResultGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMapZeroResultGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)
```


Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3019-L3023" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineModExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineModExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineModExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineModExprGet(lhs, rhs)
```


Creates an affine mod expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2885-L2889" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineMulExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineMulExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineMulExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineMulExprGet(lhs, rhs)
```


Creates an affine mul expression with &#39;lhs&#39; and &#39;rhs&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2865-L2869" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineSymbolExprGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAffineSymbolExprGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineSymbolExprGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineSymbolExprGet(ctx, position)
```


Creates an affine symbol expression with &#39;position&#39; in the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2787-L2791" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAffineSymbolExprGetPosition-Tuple{Any}' href='#Reactant.MLIR.API.mlirAffineSymbolExprGetPosition-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAffineSymbolExprGetPosition</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAffineSymbolExprGetPosition(affineExpr)
```


Returns the position of the given affine symbol expression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2798-L2802" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAnyQuantizedTypeGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirAnyQuantizedTypeGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAnyQuantizedTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)
```


Creates an instance of AnyQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7670-L7674" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirArrayAttrGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirArrayAttrGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirArrayAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirArrayAttrGet(ctx, numElements, elements)
```


Creates an array element containing the given list of elements in the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3499-L3503" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirArrayAttrGetElement-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirArrayAttrGetElement-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirArrayAttrGetElement</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirArrayAttrGetElement(attr, pos)
```


Returns pos-th element stored in the given array attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3519-L3523" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirArrayAttrGetNumElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirArrayAttrGetNumElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirArrayAttrGetNumElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirArrayAttrGetNumElements(attr)
```


Returns the number of elements stored in the given array attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3510-L3514" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirArrayAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirArrayAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirArrayAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirArrayAttrGetTypeID()
```


Returns the typeID of an Array attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3528-L3532" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAsmStateCreateForOperation-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAsmStateCreateForOperation-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAsmStateCreateForOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAsmStateCreateForOperation(op, flags)
```


Creates new AsmState, as with AsmState the IR should not be mutated in-between using this state. Must be freed with a call to [`mlirAsmStateDestroy`](/api/mlirc#Reactant.MLIR.API.mlirAsmStateDestroy-Tuple{Any})().


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1081-L1085" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAsmStateCreateForValue-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAsmStateCreateForValue-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAsmStateCreateForValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAsmStateCreateForValue(value, flags)
```


Creates new AsmState from value. Must be freed with a call to [`mlirAsmStateDestroy`](/api/mlirc#Reactant.MLIR.API.mlirAsmStateDestroy-Tuple{Any})().


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1092-L1096" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAsmStateDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirAsmStateDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAsmStateDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAsmStateDestroy(state)
```


Destroys printing flags created with mlirAsmStateCreate.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1103-L1107" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeDump(attr)
```


Prints the attribute to the standard error stream.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2442-L2446" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAttributeEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeEqual(a1, a2)
```


Checks if two attributes are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2422-L2426" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeGetContext(attribute)
```


Gets the context that an attribute was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2377-L2381" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeGetDialect-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeGetDialect-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeGetDialect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeGetDialect(attribute)
```


Gets the dialect of the attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2404-L2408" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeGetNull-Tuple{}' href='#Reactant.MLIR.API.mlirAttributeGetNull-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeGetNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeGetNull()
```


Returns an empty attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3441-L3445" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeGetType-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeGetType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeGetType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeGetType(attribute)
```


Gets the type of this attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2386-L2390" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeGetTypeID-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeGetTypeID-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeGetTypeID(attribute)
```


Gets the type id of the attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2395-L2399" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAAffineMap-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAAffineMap-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAAffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAAffineMap(attr)
```


Checks whether the given attribute is an affine map attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3454-L3458" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAArray-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAArray-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAArray</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAArray(attr)
```


Checks whether the given attribute is an array attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3490-L3494" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsABool-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsABool-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsABool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsABool(attr)
```


Checks whether the given attribute is a bool attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3700-L3704" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsADenseBoolArray-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsADenseBoolArray-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsADenseBoolArray</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsADenseBoolArray(attr)
```


Checks whether the given attribute is a dense array attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4074-L4078" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsADenseElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsADenseElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsADenseElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsADenseElements(attr)
```


Checks whether the given attribute is a dense elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4196-L4200" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsADictionary-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsADictionary-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsADictionary</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsADictionary(attr)
```


Checks whether the given attribute is a dictionary attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3537-L3541" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAElements(attr)
```


Checks whether the given attribute is an elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4030-L4034" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAFlatSymbolRef-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAFlatSymbolRef-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAFlatSymbolRef</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAFlatSymbolRef(attr)
```


Checks whether the given attribute is a flat symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3938-L3942" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAFloat-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAFloat-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAFloat</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAFloat(attr)
```


Checks whether the given attribute is a floating point attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3597-L3601" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAInteger-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAInteger-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAInteger</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAInteger(attr)
```


Checks whether the given attribute is an integer attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3646-L3650" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAIntegerSet-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAIntegerSet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAIntegerSet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAIntegerSet(attr)
```


Checks whether the given attribute is an integer set attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3727-L3731" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAOpaque-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAOpaque-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAOpaque</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAOpaque(attr)
```


Checks whether the given attribute is an opaque attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3763-L3767" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsASparseElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsASparseElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsASparseElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsASparseElements(attr)
```


Checks whether the given attribute is a sparse elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4786-L4790" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsASparseTensorEncodingAttr-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsASparseTensorEncodingAttr-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsASparseTensorEncodingAttr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsASparseTensorEncodingAttr(attr)
```


Checks whether the given attribute is a `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8208-L8212" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAString-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAString-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAString</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAString(attr)
```


Checks whether the given attribute is a string attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3814-L3818" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsASymbolRef-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsASymbolRef-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsASymbolRef</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsASymbolRef(attr)
```


Checks whether the given attribute is a symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3859-L3863" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAType-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAType(attr)
```


Checks whether the given attribute is a type attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3967-L3971" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsAUnit-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsAUnit-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsAUnit</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsAUnit(attr)
```


Checks whether the given attribute is a unit attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4003-L4007" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirAttributeIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeIsNull(attr)
```


Checks whether an attribute is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2413-L2417" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributeParseGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirAttributeParseGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributeParseGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributeParseGet(context, attr)
```


Parses an attribute. The attribute is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2366-L2370" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirAttributePrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirAttributePrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirAttributePrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirAttributePrint(attr, callback, userData)
```


Prints an attribute by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2431-L2435" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBF16TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirBF16TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBF16TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBF16TypeGet(ctx)
```


Creates a bf16 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5305-L5309" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBFloat16TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirBFloat16TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirBFloat16TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBFloat16TypeGetTypeID()
```


Returns the typeID of an BFloat16 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5287-L5291" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockAddArgument-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirBlockAddArgument-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockAddArgument</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockAddArgument(block, type, loc)
```


Appends an argument of the specified type to the block. Returns the newly added argument.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2013-L2017" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockAppendOwnedOperation-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBlockAppendOwnedOperation-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockAppendOwnedOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockAppendOwnedOperation(block, operation)
```


Takes an operation owned by the caller and appends it to the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1960-L1964" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockArgumentGetArgNumber-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockArgumentGetArgNumber-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockArgumentGetArgNumber</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockArgumentGetArgNumber(value)
```


Returns the position of the value in the argument list of its block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2109-L2113" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockArgumentGetOwner-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockArgumentGetOwner-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockArgumentGetOwner</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockArgumentGetOwner(value)
```


Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2100-L2104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockArgumentSetType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBlockArgumentSetType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockArgumentSetType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockArgumentSetType(value, type)
```


Sets the type of the block argument to the given type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2118-L2122" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockCreate-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirBlockCreate-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockCreate(nArgs, args, locs)
```


Creates a new empty block with the given argument types and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1868-L1872" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockDestroy(block)
```


Takes a block owned by the caller and destroys it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1879-L1883" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockDetach-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockDetach-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockDetach</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockDetach(block)
```


Detach a block from the owning region and assume ownership.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1888-L1892" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBlockEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockEqual(block, other)
```


Checks whether two blocks handles point to the same block. This does not perform deep comparison.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1906-L1910" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockEraseArgument-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBlockEraseArgument-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockEraseArgument</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockEraseArgument(block, index)
```


Erase the argument at &#39;index&#39; and remove it from the argument list.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2024-L2028" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetArgument-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBlockGetArgument-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetArgument</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetArgument(block, pos)
```


Returns `pos`-th argument of the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2044-L2048" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetFirstOperation-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockGetFirstOperation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetFirstOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetFirstOperation(block)
```


Returns the first operation in the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1942-L1946" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetNextInRegion-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockGetNextInRegion-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetNextInRegion</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetNextInRegion(block)
```


Returns the block immediately following the given block in its parent region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1933-L1937" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetNumArguments-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockGetNumArguments-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetNumArguments</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetNumArguments(block)
```


Returns the number of arguments of the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2004-L2008" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetParentOperation-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockGetParentOperation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetParentOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetParentOperation(arg1)
```


Returns the closest surrounding operation that contains this block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1915-L1919" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetParentRegion-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockGetParentRegion-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetParentRegion</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetParentRegion(block)
```


Returns the region that contains this block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1924-L1928" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockGetTerminator-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockGetTerminator-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockGetTerminator</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockGetTerminator(block)
```


Returns the terminator operation in the block or null if no terminator.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1951-L1955" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockInsertArgument-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirBlockInsertArgument-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockInsertArgument</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockInsertArgument(block, pos, type, loc)
```


Inserts an argument of the specified type at a specified index to the block. Returns the newly added argument.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2033-L2037" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockInsertOwnedOperation-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirBlockInsertOwnedOperation-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockInsertOwnedOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockInsertOwnedOperation(block, pos, operation)
```


Takes an operation owned by the caller and inserts it as `pos` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1971-L1975" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockInsertOwnedOperationAfter-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirBlockInsertOwnedOperationAfter-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockInsertOwnedOperationAfter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockInsertOwnedOperationAfter(block, reference, operation)
```


Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1982-L1986" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockInsertOwnedOperationBefore-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirBlockInsertOwnedOperationBefore-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockInsertOwnedOperationBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockInsertOwnedOperationBefore(block, reference, operation)
```


Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1993-L1997" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirBlockIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockIsNull(block)
```


Checks whether a block is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1897-L1901" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBlockPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirBlockPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBlockPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBlockPrint(block, callback, userData)
```


Prints a block by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2053-L2057" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBoolAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBoolAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBoolAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBoolAttrGet(ctx, value)
```


Creates a bool attribute in the given context with the given value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3709-L3713" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBoolAttrGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirBoolAttrGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBoolAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBoolAttrGetValue(attr)
```


Returns the value stored in the given bool attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3718-L3722" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBytecodeWriterConfigCreate-Tuple{}' href='#Reactant.MLIR.API.mlirBytecodeWriterConfigCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirBytecodeWriterConfigCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBytecodeWriterConfigCreate()
```


Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirBytecodeWriterConfigDestroy`](/api/mlirc#Reactant.MLIR.API.mlirBytecodeWriterConfigDestroy-Tuple{Any})().


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1208-L1212" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBytecodeWriterConfigDesiredEmitVersion-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirBytecodeWriterConfigDesiredEmitVersion-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBytecodeWriterConfigDesiredEmitVersion</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)
```


Sets the version to emit in the writer config.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1226-L1230" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirBytecodeWriterConfigDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirBytecodeWriterConfigDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirBytecodeWriterConfigDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirBytecodeWriterConfigDestroy(config)
```


Destroys printing flags created with [`mlirBytecodeWriterConfigCreate`](/api/mlirc#Reactant.MLIR.API.mlirBytecodeWriterConfigCreate-Tuple{}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1217-L1221" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirCalibratedQuantizedTypeGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirCalibratedQuantizedTypeGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirCalibratedQuantizedTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirCalibratedQuantizedTypeGet(expressedType, min, max)
```


Creates an instance of CalibratedQuantizedType with the given parameters in the same context as `expressedType` and returns it. The instance is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7935-L7939" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirCalibratedQuantizedTypeGetMax-Tuple{Any}' href='#Reactant.MLIR.API.mlirCalibratedQuantizedTypeGetMax-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirCalibratedQuantizedTypeGetMax</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirCalibratedQuantizedTypeGetMax(type)
```


Returns the max value of the given calibrated quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7955-L7959" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirCalibratedQuantizedTypeGetMin-Tuple{Any}' href='#Reactant.MLIR.API.mlirCalibratedQuantizedTypeGetMin-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirCalibratedQuantizedTypeGetMin</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirCalibratedQuantizedTypeGetMin(type)
```


Returns the min value of the given calibrated quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7946-L7950" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirComplexTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirComplexTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirComplexTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirComplexTypeGet(elementType)
```


Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5467-L5471" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirComplexTypeGetElementType-Tuple{Any}' href='#Reactant.MLIR.API.mlirComplexTypeGetElementType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirComplexTypeGetElementType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirComplexTypeGetElementType(type)
```


Returns the element type of the given complex type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5476-L5480" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirComplexTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirComplexTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirComplexTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirComplexTypeGetTypeID()
```


Returns the typeID of an Complex type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5449-L5453" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextAppendDialectRegistry-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextAppendDialectRegistry-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextAppendDialectRegistry</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextAppendDialectRegistry(ctx, registry)
```


Append the contents of the given dialect registry to the registry associated with the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L408-L412" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextAttachDiagnosticHandler-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirContextAttachDiagnosticHandler-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextAttachDiagnosticHandler</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
```


Attaches the diagnostic handler to the context. Handlers are invoked in the reverse order of attachment until one of them processes the diagnostic completely. When a handler is invoked it is passed the `userData` that was provided when it was attached. If non-NULL, `deleteUserData` is called once the system no longer needs to call the handler (for instance after the handler is detached or the context is destroyed). Returns an identifier that can be used to detach the handler.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6221-L6225" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextCreate-Tuple{}' href='#Reactant.MLIR.API.mlirContextCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirContextCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextCreate()
```


Creates an MLIR context and transfers its ownership to the caller. This sets the default multithreading option (enabled).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L323-L327" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextCreateWithRegistry-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextCreateWithRegistry-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextCreateWithRegistry</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextCreateWithRegistry(registry, threadingEnabled)
```


Creates an MLIR context, setting the multithreading setting explicitly and pre-loading the dialects from the provided DialectRegistry.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L341-L345" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextCreateWithThreading-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextCreateWithThreading-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextCreateWithThreading</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextCreateWithThreading(threadingEnabled)
```


Creates an MLIR context with an explicit setting of the multithreading setting and transfers its ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L332-L336" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextDestroy(context)
```


Takes an MLIR context owned by the caller and destroys it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L370-L374" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextDetachDiagnosticHandler-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextDetachDiagnosticHandler-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextDetachDiagnosticHandler</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextDetachDiagnosticHandler(context, id)
```


Detaches an attached diagnostic handler from the context given its identifier.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6235-L6239" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextEnableMultithreading-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextEnableMultithreading-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextEnableMultithreading</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextEnableMultithreading(context, enable)
```


Set threading mode (must be set to false to mlir-print-ir-after-all).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L439-L443" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextEqual(ctx1, ctx2)
```


Checks if two contexts are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L352-L356" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextGetAllowUnregisteredDialects-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextGetAllowUnregisteredDialects-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextGetAllowUnregisteredDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextGetAllowUnregisteredDialects(context)
```


Returns whether the context allows unregistered dialects.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L390-L394" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextGetNumLoadedDialects-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextGetNumLoadedDialects-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextGetNumLoadedDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextGetNumLoadedDialects(context)
```


Returns the number of dialects loaded by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L419-L423" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextGetNumRegisteredDialects-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextGetNumRegisteredDialects-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextGetNumRegisteredDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextGetNumRegisteredDialects(context)
```


Returns the number of dialects registered with the given context. A registered dialect will be loaded if needed by the parser.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L399-L403" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextGetNumThreads-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextGetNumThreads-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextGetNumThreads</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextGetNumThreads(context)
```


Gets the number of threads of the thread pool of the context when multithreading is enabled. Returns 1 if no multithreading.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L479-L483" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextGetOrLoadDialect-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextGetOrLoadDialect-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextGetOrLoadDialect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextGetOrLoadDialect(context, name)
```


Gets the dialect instance owned by the given context using the dialect namespace to identify it, loads (i.e., constructs the instance of) the dialect if necessary. If the dialect is not registered with the context, returns null. Use mlirContextLoad&lt;Name&gt;Dialect to load an unregistered dialect.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L428-L432" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextGetThreadPool-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextGetThreadPool-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextGetThreadPool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextGetThreadPool(context)
```


Gets the thread pool of the context when enabled multithreading, otherwise an assertion is raised.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L488-L492" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextIsNull(context)
```


Checks whether a context is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L361-L365" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextIsRegisteredOperation-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextIsRegisteredOperation-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextIsRegisteredOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextIsRegisteredOperation(context, name)
```


Returns whether the given fully-qualified operation (i.e. &#39;dialect.operation&#39;) is registered with the context. This will return true if the dialect is loaded and the operation is registered within the dialect.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L457-L461" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextLoadAllAvailableDialects-Tuple{Any}' href='#Reactant.MLIR.API.mlirContextLoadAllAvailableDialects-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextLoadAllAvailableDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextLoadAllAvailableDialects(context)
```


Eagerly loads all available dialects registered with a context, making them available for use for IR construction.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L448-L452" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextSetAllowUnregisteredDialects-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextSetAllowUnregisteredDialects-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextSetAllowUnregisteredDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextSetAllowUnregisteredDialects(context, allow)
```


Sets whether unregistered dialects are allowed in this context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L379-L383" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirContextSetThreadPool-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirContextSetThreadPool-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirContextSetThreadPool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirContextSetThreadPool(context, threadPool)
```


Sets the thread pool of the context explicitly, enabling multithreading in the process. This API should be used to avoid re-creating thread pools in long-running applications that perform multiple compilations, see the C++ documentation for MLIRContext for details.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L468-L472" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirCreateExternalPass-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirCreateExternalPass-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirCreateExternalPass</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)
```


Creates an external [`MlirPass`](@ref) that calls the supplied `callbacks` using the supplied `userData`. If `opName` is empty, the pass is a generic operation pass. Otherwise it is an operation pass specific to the specified pass name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8958-L8962" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseArrayGetNumElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirDenseArrayGetNumElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseArrayGetNumElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseArrayGetNumElements(attr)
```


Get the size of a dense array.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4154-L4158" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseBoolArrayGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDenseBoolArrayGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseBoolArrayGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseBoolArrayGet(ctx, size, values)
```


Create a dense array attribute with the given elements.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4107-L4111" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseBoolArrayGetElement-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDenseBoolArrayGetElement-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseBoolArrayGetElement</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseBoolArrayGetElement(attr, pos)
```


Get an element of a dense array.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4163-L4167" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseBoolResourceElementsAttrGetValue-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDenseBoolResourceElementsAttrGetValue-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseBoolResourceElementsAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseBoolResourceElementsAttrGetValue(attr, pos)
```


Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense resource elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4715-L4719" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrBoolGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrBoolGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrBoolGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
```


Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4313-L4317" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrGet(shapedType, numElements, elements)
```


Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4222-L4226" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrGetBoolValue-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrGetBoolValue-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrGetBoolValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrGetBoolValue(attr, pos)
```


Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4478-L4482" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrGetRawData-Tuple{Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrGetRawData-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrGetRawData</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrGetRawData(attr)
```


Returns the raw data of the given dense elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4561-L4565" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrGetSplatValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrGetSplatValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrGetSplatValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrGetSplatValue(attr)
```


Returns the single replicated value (splat) of a specific type contained by the given dense elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4427-L4431" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrIsSplat-Tuple{Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrIsSplat-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrIsSplat</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrIsSplat(attr)
```


Checks whether the given dense elements attribute contains a single replicated value (splat).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4418-L4422" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrRawBufferGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrRawBufferGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrRawBufferGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)
```


Creates a dense elements attribute with the given Shaped type and elements populated from a packed, row-major opaque buffer of contents.

The format of the raw buffer is a densely packed array of values that can be bitcast to the storage format of the element type specified. Types that are not byte aligned will be: - For bitwidth &gt; 1: Rounded up to the next byte. - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to the linear order of the shape type from MSB to LSB, padded to on the right.

A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255) will be interpreted as a splat. User code should be prepared for additional, conformant patterns to be identified as splats in the future.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4233-L4241" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrReshapeGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrReshapeGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrReshapeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrReshapeGet(attr, shapedType)
```


Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4407-L4411" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrSplatGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrSplatGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrSplatGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrSplatGet(shapedType, element)
```


Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4248-L4252" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseElementsAttrStringGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDenseElementsAttrStringGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseElementsAttrStringGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
```


Creates a dense elements attribute with the given shaped type from string elements.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4396-L4400" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDenseIntOrFPElementsAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirDenseIntOrFPElementsAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirDenseIntOrFPElementsAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDenseIntOrFPElementsAttrGetTypeID()
```


Returns the typeID of an DenseIntOrFPElements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4213-L4217" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDiagnosticGetLocation-Tuple{Any}' href='#Reactant.MLIR.API.mlirDiagnosticGetLocation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDiagnosticGetLocation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDiagnosticGetLocation(diagnostic)
```


Returns the location at which the diagnostic is reported.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6181-L6185" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDiagnosticGetNote-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDiagnosticGetNote-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDiagnosticGetNote</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDiagnosticGetNote(diagnostic, pos)
```


Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a valid zero-based index into the list of notes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6210-L6214" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDiagnosticGetNumNotes-Tuple{Any}' href='#Reactant.MLIR.API.mlirDiagnosticGetNumNotes-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDiagnosticGetNumNotes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDiagnosticGetNumNotes(diagnostic)
```


Returns the number of notes attached to the diagnostic.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6201-L6205" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDiagnosticGetSeverity-Tuple{Any}' href='#Reactant.MLIR.API.mlirDiagnosticGetSeverity-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDiagnosticGetSeverity</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDiagnosticGetSeverity(diagnostic)
```


Returns the severity of the diagnostic.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6190-L6194" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDiagnosticPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDiagnosticPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDiagnosticPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDiagnosticPrint(diagnostic, callback, userData)
```


Prints a diagnostic using the provided callback.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6170-L6174" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDialectEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectEqual(dialect1, dialect2)
```


Checks if two dialects that belong to the same context are equal. Dialects from different contexts will not compare equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L515-L519" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirDialectGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectGetContext(dialect)
```


Returns the context that owns the dialect.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L497-L501" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectGetNamespace-Tuple{Any}' href='#Reactant.MLIR.API.mlirDialectGetNamespace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectGetNamespace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectGetNamespace(dialect)
```


Returns the namespace of the given dialect.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L524-L528" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectHandleGetNamespace-Tuple{Any}' href='#Reactant.MLIR.API.mlirDialectHandleGetNamespace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectHandleGetNamespace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectHandleGetNamespace(arg1)
```


Returns the namespace associated with the provided dialect handle.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L533-L537" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectHandleInsertDialect-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDialectHandleInsertDialect-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectHandleInsertDialect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectHandleInsertDialect(arg1, arg2)
```


Inserts the dialect associated with the provided dialect handle into the provided dialect registry


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L542-L546" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectHandleLoadDialect-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDialectHandleLoadDialect-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectHandleLoadDialect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectHandleLoadDialect(arg1, arg2)
```


Loads the dialect associated with the provided dialect handle.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L564-L568" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectHandleRegisterDialect-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDialectHandleRegisterDialect-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectHandleRegisterDialect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectHandleRegisterDialect(arg1, arg2)
```


Registers the dialect associated with the provided dialect handle.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L553-L557" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirDialectIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectIsNull(dialect)
```


Checks if the dialect is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L506-L510" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectRegistryCreate-Tuple{}' href='#Reactant.MLIR.API.mlirDialectRegistryCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectRegistryCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectRegistryCreate()
```


Creates a dialect registry and transfers its ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L575-L579" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectRegistryDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirDialectRegistryDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectRegistryDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectRegistryDestroy(registry)
```


Takes a dialect registry owned by the caller and destroys it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L593-L597" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDialectRegistryIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirDialectRegistryIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDialectRegistryIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDialectRegistryIsNull(registry)
```


Checks if the dialect registry is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L584-L588" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDictionaryAttrGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirDictionaryAttrGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDictionaryAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDictionaryAttrGet(ctx, numElements, elements)
```


Creates a dictionary attribute containing the given list of elements in the provided context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3546-L3550" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDictionaryAttrGetElement-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDictionaryAttrGetElement-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDictionaryAttrGetElement</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDictionaryAttrGetElement(attr, pos)
```


Returns pos-th element of the given dictionary attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3566-L3570" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDictionaryAttrGetElementByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirDictionaryAttrGetElementByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDictionaryAttrGetElementByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDictionaryAttrGetElementByName(attr, name)
```


Returns the dictionary attribute element with the given name or NULL if the given name does not exist in the dictionary.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3577-L3581" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDictionaryAttrGetNumElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirDictionaryAttrGetNumElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDictionaryAttrGetNumElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDictionaryAttrGetNumElements(attr)
```


Returns the number of attributes contained in a dictionary attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3557-L3561" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDictionaryAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirDictionaryAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirDictionaryAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDictionaryAttrGetTypeID()
```


Returns the typeID of a Dictionary attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3588-L3592" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirDisctinctAttrCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirDisctinctAttrCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirDisctinctAttrCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirDisctinctAttrCreate(referencedAttr)
```


Creates a DisctinctAttr with the referenced attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3929-L3933" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirElementsAttrGetNumElements-Tuple{Any}' href='#Reactant.MLIR.API.mlirElementsAttrGetNumElements-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirElementsAttrGetNumElements</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirElementsAttrGetNumElements(attr)
```


Gets the total number of elements in the given elements attribute. In order to iterate over the attribute, obtain its type, which must be a statically shaped type and use its sizes to build a multi-dimensional index.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4061-L4065" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirElementsAttrGetValue-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirElementsAttrGetValue-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirElementsAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirElementsAttrGetValue(attr, rank, idxs)
```


Returns the element at the given rank-dimensional index.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4039-L4043" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirElementsAttrIsValidIndex-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirElementsAttrIsValidIndex-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirElementsAttrIsValidIndex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirElementsAttrIsValidIndex(attr, rank, idxs)
```


Checks whether the given rank-dimensional index is valid in the given elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4050-L4054" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirEmitError-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirEmitError-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirEmitError</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirEmitError(location, message)
```


Emits an error at the given location through the diagnostics engine. Used for testing purposes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6246-L6250" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirEnableGlobalDebug-Tuple{Any}' href='#Reactant.MLIR.API.mlirEnableGlobalDebug-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirEnableGlobalDebug</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirEnableGlobalDebug(enable)
```


Sets the global debugging flag.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6093-L6097" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineCreate-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirExecutionEngineCreate-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths, enableObjectDump)
```


Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is expected to be &quot;translatable&quot; to LLVM IR (only contains operations in dialects that implement the `LLVMTranslationDialectInterface`). The module ownership stays with the client and can be destroyed as soon as the call returns. `optLevel` is the optimization level to be used for transformation and code generation. LLVM passes at `optLevel` are run before code generation. The number and array of paths corresponding to shared libraries that will be loaded are specified via `numPaths` and `sharedLibPaths` respectively. TODO: figure out other options.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8534-L8538" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirExecutionEngineDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineDestroy(jit)
```


Destroy an ExecutionEngine instance.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8549-L8553" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineDumpToObjectFile-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirExecutionEngineDumpToObjectFile-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineDumpToObjectFile</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineDumpToObjectFile(jit, fileName)
```


Dump as an object in `fileName`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8611-L8615" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineInvokePacked-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirExecutionEngineInvokePacked-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineInvokePacked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineInvokePacked(jit, name, arguments)
```


Invoke a native function in the execution engine by name with the arguments and result of the invoked function passed as an array of pointers. The function must have been tagged with the `llvm.emit\_c\_interface` attribute. Returns a failure if the execution fails for any reason (the function name can&#39;t be resolved for instance).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8567-L8571" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirExecutionEngineIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineIsNull(jit)
```


Checks whether an execution engine is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8558-L8562" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineLookup-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirExecutionEngineLookup-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineLookup</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineLookup(jit, name)
```


Lookup a native function in the execution engine by name, returns nullptr if the name can&#39;t be looked-up.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8589-L8593" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineLookupPacked-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirExecutionEngineLookupPacked-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineLookupPacked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineLookupPacked(jit, name)
```


Lookup the wrapper of the native function in the execution engine with the given name, returns nullptr if the function can&#39;t be looked-up.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8578-L8582" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExecutionEngineRegisterSymbol-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirExecutionEngineRegisterSymbol-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExecutionEngineRegisterSymbol</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExecutionEngineRegisterSymbol(jit, name, sym)
```


Register a symbol with the jit: this symbol will be accessible to the jitted code.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8600-L8604" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirExternalPassSignalFailure-Tuple{Any}' href='#Reactant.MLIR.API.mlirExternalPassSignalFailure-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirExternalPassSignalFailure</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirExternalPassSignalFailure(pass)
```


This signals that the pass has failed. This is only valid to call during the `run` callback of [`MlirExternalPassCallbacks`](/api/mlirc#Reactant.MLIR.API.MlirExternalPassCallbacks). See Pass::signalPassFailure().


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8987-L8991" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirF16TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirF16TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirF16TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirF16TypeGet(ctx)
```


Creates an f16 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5332-L5336" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirF32TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirF32TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirF32TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirF32TypeGet(ctx)
```


Creates an f32 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5359-L5363" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirF64TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirF64TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirF64TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirF64TypeGet(ctx)
```


Creates a f64 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5386-L5390" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFlatSymbolRefAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirFlatSymbolRefAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFlatSymbolRefAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFlatSymbolRefAttrGet(ctx, symbol)
```


Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3947-L3951" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFlatSymbolRefAttrGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirFlatSymbolRefAttrGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFlatSymbolRefAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFlatSymbolRefAttrGetValue(attr)
```


Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3958-L3962" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat16TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat16TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat16TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat16TypeGetTypeID()
```


Returns the typeID of an Float16 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5314-L5318" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat32TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat32TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat32TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat32TypeGetTypeID()
```


Returns the typeID of an Float32 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5341-L5345" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat4E2M1FNTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat4E2M1FNTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat4E2M1FNTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat4E2M1FNTypeGet(ctx)
```


Creates an f4E2M1FN type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5008-L5012" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat4E2M1FNTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat4E2M1FNTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat4E2M1FNTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat4E2M1FNTypeGetTypeID()
```


Returns the typeID of an Float4E2M1FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4990-L4994" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat64TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat64TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat64TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat64TypeGetTypeID()
```


Returns the typeID of an Float64 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5368-L5372" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat6E2M3FNTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat6E2M3FNTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat6E2M3FNTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat6E2M3FNTypeGet(ctx)
```


Creates an f6E2M3FN type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5035-L5039" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat6E2M3FNTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat6E2M3FNTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat6E2M3FNTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat6E2M3FNTypeGetTypeID()
```


Returns the typeID of an Float6E2M3FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5017-L5021" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat6E3M2FNTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat6E3M2FNTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat6E3M2FNTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat6E3M2FNTypeGet(ctx)
```


Creates an f6E3M2FN type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5062-L5066" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat6E3M2FNTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat6E3M2FNTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat6E3M2FNTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat6E3M2FNTypeGetTypeID()
```


Returns the typeID of an Float6E3M2FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5044-L5048" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E3M4TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E3M4TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E3M4TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E3M4TypeGet(ctx)
```


Creates an f8E3M4 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5251-L5255" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E3M4TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E3M4TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E3M4TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E3M4TypeGetTypeID()
```


Returns the typeID of an Float8E3M4 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5233-L5237" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3B11FNUZTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E4M3B11FNUZTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3B11FNUZTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3B11FNUZTypeGet(ctx)
```


Creates an f8E4M3B11FNUZ type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5224-L5228" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3B11FNUZTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E4M3B11FNUZTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3B11FNUZTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3B11FNUZTypeGetTypeID()
```


Returns the typeID of an Float8E4M3B11FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5206-L5210" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3FNTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E4M3FNTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3FNTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3FNTypeGet(ctx)
```


Creates an f8E4M3FN type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5143-L5147" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3FNTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E4M3FNTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3FNTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3FNTypeGetTypeID()
```


Returns the typeID of an Float8E4M3FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5125-L5129" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3FNUZTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E4M3FNUZTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3FNUZTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3FNUZTypeGet(ctx)
```


Creates an f8E4M3FNUZ type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5197-L5201" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3FNUZTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E4M3FNUZTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3FNUZTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3FNUZTypeGetTypeID()
```


Returns the typeID of an Float8E4M3FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5179-L5183" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E4M3TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3TypeGet(ctx)
```


Creates an f8E4M3 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5116-L5120" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E4M3TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E4M3TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E4M3TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E4M3TypeGetTypeID()
```


Returns the typeID of an Float8E4M3 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5098-L5102" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E5M2FNUZTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E5M2FNUZTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E5M2FNUZTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E5M2FNUZTypeGet(ctx)
```


Creates an f8E5M2FNUZ type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5170-L5174" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E5M2FNUZTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E5M2FNUZTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E5M2FNUZTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E5M2FNUZTypeGetTypeID()
```


Returns the typeID of an Float8E5M2FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5152-L5156" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E5M2TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E5M2TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E5M2TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E5M2TypeGet(ctx)
```


Creates an f8E5M2 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5089-L5093" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E5M2TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E5M2TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E5M2TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E5M2TypeGetTypeID()
```


Returns the typeID of an Float8E5M2 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5071-L5075" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E8M0FNUTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloat8E8M0FNUTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E8M0FNUTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E8M0FNUTypeGet(ctx)
```


Creates an f8E8M0FNU type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5278-L5282" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloat8E8M0FNUTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloat8E8M0FNUTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloat8E8M0FNUTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloat8E8M0FNUTypeGetTypeID()
```


Returns the typeID of an Float8E8M0FNU type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5260-L5264" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloatAttrDoubleGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirFloatAttrDoubleGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloatAttrDoubleGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloatAttrDoubleGet(ctx, type, value)
```


Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3606-L3610" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloatAttrDoubleGetChecked-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirFloatAttrDoubleGetChecked-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloatAttrDoubleGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloatAttrDoubleGetChecked(loc, type, value)
```


Same as &quot;[`mlirFloatAttrDoubleGet`](/api/mlirc#Reactant.MLIR.API.mlirFloatAttrDoubleGet-Tuple{Any,%20Any,%20Any})&quot;, but if the type is not valid for a construction of a FloatAttr, returns a null [`MlirAttribute`](@ref).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3617-L3621" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloatAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloatAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloatAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloatAttrGetTypeID()
```


Returns the typeID of a Float attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3637-L3641" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloatAttrGetValueDouble-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloatAttrGetValueDouble-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloatAttrGetValueDouble</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloatAttrGetValueDouble(attr)
```


Returns the value stored in the given floating point attribute, interpreting the value as double.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3628-L3632" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloatTF32TypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFloatTF32TypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFloatTF32TypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloatTF32TypeGetTypeID()
```


Returns the typeID of a TF32 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5395-L5399" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFloatTypeGetWidth-Tuple{Any}' href='#Reactant.MLIR.API.mlirFloatTypeGetWidth-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFloatTypeGetWidth</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFloatTypeGetWidth(type)
```


Returns the bitwidth of a floating-point type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4981-L4985" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFreezeRewritePattern-Tuple{Any}' href='#Reactant.MLIR.API.mlirFreezeRewritePattern-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFreezeRewritePattern</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFreezeRewritePattern(op)
```


FrozenRewritePatternSet API


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9454-L9458" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFuncSetArgAttr-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirFuncSetArgAttr-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFuncSetArgAttr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFuncSetArgAttr(op, pos, name, attr)
```


Sets the argument attribute &#39;name&#39; of an argument at index &#39;pos&#39;. Asserts that the operation is a FuncOp.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6413-L6417" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFunctionTypeGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirFunctionTypeGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFunctionTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
```


Creates a function type, mapping a list of input types to result types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5995-L5999" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFunctionTypeGetInput-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirFunctionTypeGetInput-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFunctionTypeGetInput</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFunctionTypeGetInput(type, pos)
```


Returns the pos-th input type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6028-L6032" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFunctionTypeGetNumInputs-Tuple{Any}' href='#Reactant.MLIR.API.mlirFunctionTypeGetNumInputs-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFunctionTypeGetNumInputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFunctionTypeGetNumInputs(type)
```


Returns the number of input types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6010-L6014" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFunctionTypeGetNumResults-Tuple{Any}' href='#Reactant.MLIR.API.mlirFunctionTypeGetNumResults-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFunctionTypeGetNumResults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFunctionTypeGetNumResults(type)
```


Returns the number of result types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6019-L6023" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFunctionTypeGetResult-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirFunctionTypeGetResult-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirFunctionTypeGetResult</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFunctionTypeGetResult(type, pos)
```


Returns the pos-th result type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6037-L6041" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirFunctionTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirFunctionTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirFunctionTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirFunctionTypeGetTypeID()
```


Returns the typeID of an Function type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5977-L5981" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIRRewriterCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirIRRewriterCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIRRewriterCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIRRewriterCreate(context)
```


Create an IRRewriter and transfer ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9427-L9431" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIRRewriterCreateFromOp-Tuple{Any}' href='#Reactant.MLIR.API.mlirIRRewriterCreateFromOp-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIRRewriterCreateFromOp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIRRewriterCreateFromOp(op)
```


Create an IRRewriter and transfer ownership to the caller. Additionally set the insertion point before the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9436-L9440" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIRRewriterDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirIRRewriterDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIRRewriterDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIRRewriterDestroy(rewriter)
```


Takes an IRRewriter owned by the caller and destroys it. It is the responsibility of the user to only pass an IRRewriter class.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9445-L9449" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIdentifierEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIdentifierEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIdentifierEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIdentifierEqual(ident, other)
```


Checks whether two identifiers are the same.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2482-L2486" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIdentifierGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIdentifierGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIdentifierGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIdentifierGet(context, str)
```


Gets an identifier with the given string value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2462-L2466" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIdentifierGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirIdentifierGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIdentifierGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIdentifierGetContext(arg1)
```


Returns the context associated with this identifier


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2473-L2477" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIdentifierStr-Tuple{Any}' href='#Reactant.MLIR.API.mlirIdentifierStr-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIdentifierStr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIdentifierStr(ident)
```


Gets the string value of the identifier.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2491-L2495" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIndexTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirIndexTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIndexTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIndexTypeGet(ctx)
```


Creates an index type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4963-L4967" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIndexTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirIndexTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirIndexTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIndexTypeGetTypeID()
```


Returns the typeID of an Index type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4945-L4949" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirInferShapedTypeOpInterfaceInferReturnTypes-NTuple{11, Any}' href='#Reactant.MLIR.API.mlirInferShapedTypeOpInterfaceInferReturnTypes-NTuple{11, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirInferShapedTypeOpInterfaceInferReturnTypes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirInferShapedTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)
```


Infers the return shaped type components of the operation. Calls `callback` with the types of inferred arguments on success. Returns failure otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8707-L8711" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirInferShapedTypeOpInterfaceTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirInferShapedTypeOpInterfaceTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirInferShapedTypeOpInterfaceTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirInferShapedTypeOpInterfaceTypeID()
```


Returns the interface TypeID of the InferShapedTypeOpInterface.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8692-L8696" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirInferTypeOpInterfaceInferReturnTypes-NTuple{11, Any}' href='#Reactant.MLIR.API.mlirInferTypeOpInterfaceInferReturnTypes-NTuple{11, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirInferTypeOpInterfaceInferReturnTypes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)
```


Infers the return types of the operation identified by its canonical given the arguments that will be supplied to its generic builder. Calls `callback` with the types of inferred arguments, potentially several times, on success. Returns failure otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8659-L8663" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirInferTypeOpInterfaceTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirInferTypeOpInterfaceTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirInferTypeOpInterfaceTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirInferTypeOpInterfaceTypeID()
```


Returns the interface TypeID of the InferTypeOpInterface.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8644-L8648" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerAttrGet(type, value)
```


Creates an integer attribute of the given type with the given integer value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3655-L3659" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirIntegerAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerAttrGetTypeID()
```


Returns the typeID of an Integer attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3691-L3695" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerAttrGetValueInt-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerAttrGetValueInt-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerAttrGetValueInt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerAttrGetValueInt(attr)
```


Returns the value stored in the given integer attribute, assuming the value is of signless type and fits into a signed 64-bit integer.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3664-L3668" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerAttrGetValueSInt-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerAttrGetValueSInt-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerAttrGetValueSInt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerAttrGetValueSInt(attr)
```


Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3673-L3677" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerAttrGetValueUInt-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerAttrGetValueUInt-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerAttrGetValueUInt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerAttrGetValueUInt(attr)
```


Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3682-L3686" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetAttrGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetAttrGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetAttrGet(set)
```


Creates an integer set attribute wrapping the given set. The attribute belongs to the same context as the integer set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3736-L3740" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirIntegerSetAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetAttrGetTypeID()
```


Returns the typeID of an IntegerSet attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3754-L3758" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetAttrGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetAttrGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetAttrGetValue(attr)
```


Returns the integer set wrapped in the given integer set attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3745-L3749" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetDump(set)
```


Prints an integer set to the standard error stream.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3303-L3307" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetEmptyGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirIntegerSetEmptyGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetEmptyGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetEmptyGet(context, numDims, numSymbols)
```


Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3312-L3316" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerSetEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetEqual(s1, s2)
```


Checks if two integer set objects are equal. This is a &quot;shallow&quot; comparison of two objects. Only the sets with some small number of constraints are uniqued and compare equal here. Set objects that represent the same integer set with different constraints may be considered non-equal by this check. Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3283-L3287" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGet-NTuple{6, Any}' href='#Reactant.MLIR.API.mlirIntegerSetGet-NTuple{6, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGet(context, numDims, numSymbols, numConstraints, constraints, eqFlags)
```


Gets or creates a new integer set in the given context. The set is defined by a list of affine constraints, with the given number of input dimensions and symbols, which are treated as either equalities (eqFlags is 1) or inequalities (eqFlags is 0). Both `constraints` and `eqFlags` are expected to point to at least `numConstraint` consecutive values.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3323-L3327" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetConstraint-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetConstraint-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetConstraint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetConstraint(set, pos)
```


Returns `pos`-th constraint of the set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3421-L3425" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetContext(set)
```


Gets the context in which the given integer set lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3265-L3269" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetNumConstraints-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetNumConstraints-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetNumConstraints</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetNumConstraints(set)
```


Returns the number of constraints (equalities + inequalities) in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3394-L3398" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetNumDims-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetNumDims-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetNumDims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetNumDims(set)
```


Returns the number of dimensions in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3367-L3371" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetNumEqualities-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetNumEqualities-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetNumEqualities</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetNumEqualities(set)
```


Returns the number of equalities in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3403-L3407" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetNumInequalities-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetNumInequalities-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetNumInequalities</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetNumInequalities(set)
```


Returns the number of inequalities in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3412-L3416" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetNumInputs-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetNumInputs-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetNumInputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetNumInputs(set)
```


Returns the number of inputs (dimensions + symbols) in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3385-L3389" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetGetNumSymbols-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetGetNumSymbols-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetGetNumSymbols</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetGetNumSymbols(set)
```


Returns the number of symbols in the given set.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3376-L3380" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetIsCanonicalEmpty-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetIsCanonicalEmpty-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetIsCanonicalEmpty</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetIsCanonicalEmpty(set)
```


Checks whether the given set is a canonical empty set, e.g., the set returned by [`mlirIntegerSetEmptyGet`](/api/mlirc#Reactant.MLIR.API.mlirIntegerSetEmptyGet-Tuple{Any,%20Any,%20Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3358-L3362" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetIsConstraintEq-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerSetIsConstraintEq-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetIsConstraintEq</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetIsConstraintEq(set, pos)
```


Returns `true` of the `pos`-th constraint of the set is an equality constraint, `false` otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3432-L3436" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerSetIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetIsNull(set)
```


Checks whether an integer set is a null object.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3274-L3278" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirIntegerSetPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetPrint(set, callback, userData)
```


Prints an integer set by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3292-L3296" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerSetReplaceGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirIntegerSetReplaceGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerSetReplaceGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)
```


Gets or creates a new integer set in which the values and dimensions of the given set are replaced with the given affine expressions. `dimReplacements` and `symbolReplacements` are expected to point to at least as many consecutive expressions as the given set has dimensions and symbols, respectively. The new set will have `numResultDims` and `numResultSymbols` dimensions and symbols, respectively.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3341-L3345" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerTypeGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeGet(ctx, bitwidth)
```


Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4882-L4886" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirIntegerTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeGetTypeID()
```


Returns the typeID of an Integer type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4864-L4868" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeGetWidth-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerTypeGetWidth-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeGetWidth</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeGetWidth(type)
```


Returns the bitwidth of an integer type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4909-L4913" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeIsSigned-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerTypeIsSigned-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeIsSigned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeIsSigned(type)
```


Checks whether the given integer type is signed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4927-L4931" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeIsSignless-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerTypeIsSignless-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeIsSignless</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeIsSignless(type)
```


Checks whether the given integer type is signless.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4918-L4922" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeIsUnsigned-Tuple{Any}' href='#Reactant.MLIR.API.mlirIntegerTypeIsUnsigned-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeIsUnsigned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeIsUnsigned(type)
```


Checks whether the given integer type is unsigned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4936-L4940" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeSignedGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerTypeSignedGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeSignedGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeSignedGet(ctx, bitwidth)
```


Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4891-L4895" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIntegerTypeUnsignedGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirIntegerTypeUnsignedGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIntegerTypeUnsignedGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIntegerTypeUnsignedGet(ctx, bitwidth)
```


Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4900-L4904" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIsCurrentDebugType-Tuple{Any}' href='#Reactant.MLIR.API.mlirIsCurrentDebugType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirIsCurrentDebugType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIsCurrentDebugType(type)
```


Checks if `type` is set as the current debug type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6129-L6133" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirIsGlobalDebugEnabled-Tuple{}' href='#Reactant.MLIR.API.mlirIsGlobalDebugEnabled-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirIsGlobalDebugEnabled</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirIsGlobalDebugEnabled()
```


Retuns `true` if the global debugging flag is set, false otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6102-L6106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMArrayTypeGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMArrayTypeGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMArrayTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMArrayTypeGet(elementType, numElements)
```


Creates an llvm.array type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6556-L6560" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMArrayTypeGetElementType-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMArrayTypeGetElementType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMArrayTypeGetElementType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMArrayTypeGetElementType(type)
```


Returns the element type of the llvm.array type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6565-L6569" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMCConvAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMCConvAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMCConvAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMCConvAttrGet(ctx, cconv)
```


Creates a LLVM CConv attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6798-L6802" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMComdatAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMComdatAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMComdatAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMComdatAttrGet(ctx, comdat)
```


Creates a LLVM Comdat attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6817-L6821" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIAnnotationAttrGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirLLVMDIAnnotationAttrGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIAnnotationAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIAnnotationAttrGet(ctx, name, value)
```


Creates a LLVM DIAnnotation attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7197-L7201" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIBasicTypeAttrGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirLLVMDIBasicTypeAttrGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIBasicTypeAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIBasicTypeAttrGet(ctx, tag, name, sizeInBits, encoding)
```


Creates a LLVM DIBasicType attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6907-L6911" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDICompileUnitAttrGet-NTuple{8, Any}' href='#Reactant.MLIR.API.mlirLLVMDICompileUnitAttrGet-NTuple{8, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDICompileUnitAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDICompileUnitAttrGet(ctx, id, sourceLanguage, file, producer, isOptimized, emissionKind, nameTableKind)
```


Creates a LLVM DICompileUnit attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7067-L7071" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDICompositeTypeAttrGet-NTuple{18, Any}' href='#Reactant.MLIR.API.mlirLLVMDICompositeTypeAttrGet-NTuple{18, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDICompositeTypeAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDICompositeTypeAttrGet(ctx, recId, isRecSelf, tag, name, file, line, scope, baseType, flags, sizeInBits, alignInBits, nElements, elements, dataLocation, rank, allocated, associated)
```


Creates a LLVM DICompositeType attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6931-L6935" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDICompositeTypeAttrGetRecSelf-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDICompositeTypeAttrGetRecSelf-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDICompositeTypeAttrGetRecSelf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDICompositeTypeAttrGetRecSelf(recId)
```


Creates a self-referencing LLVM DICompositeType attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6922-L6926" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIDerivedTypeAttrGet-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirLLVMDIDerivedTypeAttrGet-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIDerivedTypeAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIDerivedTypeAttrGet(ctx, tag, name, baseType, sizeInBits, alignInBits, offsetInBits, dwarfAddressSpace, extraData)
```


Creates a LLVM DIDerivedType attribute. Note that `dwarfAddressSpace` is an optional field, where [`MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL`](@ref) indicates null and non-negative values indicate a value present.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6978-L6982" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIDerivedTypeAttrGetBaseType-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDIDerivedTypeAttrGetBaseType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIDerivedTypeAttrGetBaseType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIDerivedTypeAttrGetBaseType(diDerivedType)
```


Gets the base type from a LLVM DIDerivedType attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7031-L7035" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIExpressionAttrGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirLLVMDIExpressionAttrGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIExpressionAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIExpressionAttrGet(ctx, nOperations, operations)
```


Creates a LLVM DIExpression attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6873-L6877" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIExpressionElemAttrGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMDIExpressionElemAttrGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIExpressionElemAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIExpressionElemAttrGet(ctx, opcode, nArguments, arguments)
```


Creates a LLVM DIExpressionElem attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6862-L6866" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIFileAttrGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirLLVMDIFileAttrGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIFileAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIFileAttrGet(ctx, name, directory)
```


Creates a LLVM DIFileAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7042-L7046" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIFlagsAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMDIFlagsAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIFlagsAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIFlagsAttrGet(ctx, value)
```


Creates a LLVM DIFlags attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7087-L7091" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIImportedEntityAttrGet-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirLLVMDIImportedEntityAttrGet-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIImportedEntityAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIImportedEntityAttrGet(ctx, tag, scope, entity, file, line, name, nElements, elements)
```


Creates a LLVM DIImportedEntityAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7305-L7309" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDILexicalBlockAttrGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirLLVMDILexicalBlockAttrGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDILexicalBlockAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDILexicalBlockAttrGet(ctx, scope, file, line, column)
```


Creates a LLVM DILexicalBlock attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7096-L7100" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDILexicalBlockFileAttrGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMDILexicalBlockFileAttrGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDILexicalBlockFileAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDILexicalBlockFileAttrGet(ctx, scope, file, discriminator)
```


Creates a LLVM DILexicalBlockFile attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7111-L7115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDILocalVariableAttrGet-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirLLVMDILocalVariableAttrGet-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDILocalVariableAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDILocalVariableAttrGet(ctx, scope, name, diFile, line, arg, alignInBits, diType, flags)
```


Creates a LLVM DILocalVariableAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7122-L7126" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIModuleAttrGet-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirLLVMDIModuleAttrGet-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIModuleAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIModuleAttrGet(ctx, file, scope, name, configMacros, includePath, apinotes, line, isDecl)
```


Creates a LLVM DIModuleAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7284-L7288" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDIModuleAttrGetScope-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDIModuleAttrGetScope-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDIModuleAttrGetScope</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDIModuleAttrGetScope(diModule)
```


Gets the scope of this DIModuleAttr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7326-L7330" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDINullTypeAttrGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDINullTypeAttrGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDINullTypeAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDINullTypeAttrGet(ctx)
```


Creates a LLVM DINullType attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6853-L6857" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGet-NTuple{17, Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGet-NTuple{17, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGet(ctx, recId, isRecSelf, id, compileUnit, scope, name, linkageName, file, line, scopeLine, subprogramFlags, type, nRetainedNodes, retainedNodes, nAnnotations, annotations)
```


Creates a LLVM DISubprogramAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7152-L7156" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetCompileUnit-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetCompileUnit-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetCompileUnit</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetCompileUnit(diSubprogram)
```


Gets the compile unit from this DISubprogram.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7237-L7241" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetFile-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetFile-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetFile</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetFile(diSubprogram)
```


Gets the file from this DISubprogramAttr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7248-L7252" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetLine-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetLine-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetLine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetLine(diSubprogram)
```


Gets the line from this DISubprogramAttr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7219-L7223" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetRecSelf-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetRecSelf-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetRecSelf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetRecSelf(recId)
```


Creates a self-referencing LLVM DISubprogramAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7143-L7147" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetScope-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetScope-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetScope</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetScope(diSubprogram)
```


Gets the scope from this DISubprogramAttr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7208-L7212" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetScopeLine-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetScopeLine-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetScopeLine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetScopeLine(diSubprogram)
```


Gets the scope line from this DISubprogram.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7228-L7232" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetType-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubprogramAttrGetType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubprogramAttrGetType(diSubprogram)
```


Gets the type from this DISubprogramAttr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7259-L7263" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMDISubroutineTypeAttrGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMDISubroutineTypeAttrGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMDISubroutineTypeAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMDISubroutineTypeAttrGet(ctx, callingConvention, nTypes, types)
```


Creates a LLVM DISubroutineTypeAttr attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7270-L7274" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMFunctionTypeGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMFunctionTypeGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMFunctionTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)
```


Creates an llvm.func type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6574-L6578" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMFunctionTypeGetInput-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMFunctionTypeGetInput-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMFunctionTypeGetInput</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMFunctionTypeGetInput(type, pos)
```


Returns the pos-th input type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6597-L6601" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMFunctionTypeGetNumInputs-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMFunctionTypeGetNumInputs-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMFunctionTypeGetNumInputs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMFunctionTypeGetNumInputs(type)
```


Returns the number of input types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6588-L6592" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMFunctionTypeGetReturnType-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMFunctionTypeGetReturnType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMFunctionTypeGetReturnType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMFunctionTypeGetReturnType(type)
```


Returns the return type of the function type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6606-L6610" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMLinkageAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMLinkageAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMLinkageAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMLinkageAttrGet(ctx, linkage)
```


Creates a LLVM Linkage attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6842-L6846" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMPointerTypeGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMPointerTypeGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMPointerTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMPointerTypeGet(ctx, addressSpace)
```


Creates an llvm.ptr type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6520-L6524" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMPointerTypeGetAddressSpace-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMPointerTypeGetAddressSpace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMPointerTypeGetAddressSpace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMPointerTypeGetAddressSpace(pointerType)
```


Returns address space of llvm.ptr


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6538-L6542" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeGetElementType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeGetElementType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeGetElementType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeGetElementType(type, position)
```


Returns the `positions`-th field of the struct. Asserts if the struct is opaque, not yet initialized or if the position is out of range.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6642-L6646" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeGetIdentifier-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeGetIdentifier-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeGetIdentifier</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeGetIdentifier(type)
```


Returns the identifier of the identified struct. Asserts that the struct is identified, i.e., not literal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6662-L6666" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeGetNumElementTypes-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeGetNumElementTypes-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeGetNumElementTypes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeGetNumElementTypes(type)
```


Returns the number of fields in the struct. Asserts if the struct is opaque or not yet initialized.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6633-L6637" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeIdentifiedGet(ctx, name)
```


Creates an LLVM identified struct type with no body. If a struct type with this name already exists in the context, returns that type. Use [`mlirLLVMStructTypeIdentifiedNewGet`](/api/mlirc#Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedNewGet-NTuple{5,%20Any}) to create a fresh struct type, potentially renaming it. The body should be set separatelty by calling [`mlirLLVMStructTypeSetBody`](/api/mlirc#Reactant.MLIR.API.mlirLLVMStructTypeSetBody-NTuple{4,%20Any}), if it isn&#39;t set already.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6702-L6706" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedNewGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedNewGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeIdentifiedNewGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeIdentifiedNewGet(ctx, name, nFieldTypes, fieldTypes, isPacked)
```


Creates an LLVM identified struct type with no body and a name starting with the given prefix. If a struct with the exact name as the given prefix already exists, appends an unspecified suffix to the name so that the name is unique in context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6713-L6717" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeIsLiteral-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeIsLiteral-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeIsLiteral</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeIsLiteral(type)
```


Returns `true` if the type is a literal (unnamed) LLVM struct type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6624-L6628" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeIsOpaque-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeIsOpaque-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeIsOpaque</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeIsOpaque(type)
```


Returns `true` is the struct is explicitly opaque (will not have a body) or uninitialized (will eventually have a body).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6671-L6675" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeIsPacked-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeIsPacked-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeIsPacked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeIsPacked(type)
```


Returns `true` if the struct is packed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6653-L6657" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeLiteralGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeLiteralGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeLiteralGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)
```


Creates an LLVM literal (unnamed) struct type. This may assert if the fields have types not compatible with the LLVM dialect. For a graceful failure, use the checked version.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6680-L6684" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeLiteralGetChecked-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeLiteralGetChecked-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeLiteralGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeLiteralGetChecked(loc, nFieldTypes, fieldTypes, isPacked)
```


Creates an LLVM literal (unnamed) struct type if possible. Emits a diagnostic at the given location and returns null otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6691-L6695" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMStructTypeSetBody-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLLVMStructTypeSetBody-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMStructTypeSetBody</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMStructTypeSetBody(structType, nFieldTypes, fieldTypes, isPacked)
```


Sets the body of the identified struct if it hasn&#39;t been set yet. Returns whether the operation was successful.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6734-L6738" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLLVMVoidTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirLLVMVoidTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLLVMVoidTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLLVMVoidTypeGet(ctx)
```


Creates an llmv.void type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6547-L6551" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLinalgFillBuiltinNamedOpRegion-Tuple{Any}' href='#Reactant.MLIR.API.mlirLinalgFillBuiltinNamedOpRegion-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLinalgFillBuiltinNamedOpRegion</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLinalgFillBuiltinNamedOpRegion(mlirOp)
```


Apply the special region builder for the builtin named Linalg op. Assert that `mlirOp` is a builtin named Linalg op.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7335-L7339" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLlvmThreadPoolCreate-Tuple{}' href='#Reactant.MLIR.API.mlirLlvmThreadPoolCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLlvmThreadPoolCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLlvmThreadPoolCreate()
```


Create an LLVM thread pool. This is reexported here to avoid directly pulling in the LLVM headers directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L160-L164" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLlvmThreadPoolDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirLlvmThreadPoolDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLlvmThreadPoolDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLlvmThreadPoolDestroy(pool)
```


Destroy an LLVM thread pool.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L169-L173" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLoadIRDLDialects-Tuple{Any}' href='#Reactant.MLIR.API.mlirLoadIRDLDialects-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLoadIRDLDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLoadIRDLDialects(_module)
```


Loads all IRDL dialects in the provided module, registering the dialects in the module&#39;s associated context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6503-L6507" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationCallSiteGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLocationCallSiteGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationCallSiteGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationCallSiteGet(callee, caller)
```


Creates a call site location with a callee and a caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L714-L718" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationCallSiteGetCallee-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationCallSiteGetCallee-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationCallSiteGetCallee</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationCallSiteGetCallee(location)
```


Getter for callee of CallSite.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L725-L729" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationCallSiteGetCaller-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationCallSiteGetCaller-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationCallSiteGetCaller</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationCallSiteGetCaller(location)
```


Getter for caller of CallSite.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L734-L738" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationCallSiteGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirLocationCallSiteGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationCallSiteGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationCallSiteGetTypeID()
```


TypeID Getter for CallSite.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L743-L747" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLocationEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationEqual(l1, l2)
```


Checks if two locations are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L896-L900" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColGet(context, filename, line, col)
```


Creates an File/Line/Column location owned by the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L620-L624" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGet-NTuple{6, Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGet-NTuple{6, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGet(context, filename, start_line, start_col, end_line, end_col)
```


Creates an File/Line/Column range location owned by the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L631-L635" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGetEndColumn-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGetEndColumn-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGetEndColumn</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGetEndColumn(location)
```


Getter for end_column of FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L687-L691" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGetEndLine-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGetEndLine-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGetEndLine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGetEndLine(location)
```


Getter for end_line of FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L678-L682" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGetFilename-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGetFilename-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGetFilename</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGetFilename(location)
```


Getter for filename of FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L649-L653" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGetStartColumn-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGetStartColumn-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGetStartColumn</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGetStartColumn(location)
```


Getter for start_column of FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L669-L673" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGetStartLine-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGetStartLine-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGetStartLine</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGetStartLine(location)
```


Getter for start_line of FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L660-L664" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFileLineColRangeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirLocationFileLineColRangeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFileLineColRangeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFileLineColRangeGetTypeID()
```


TypeID Getter for FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L696-L700" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFromAttribute-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFromAttribute-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFromAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFromAttribute(attribute)
```


Creates a location from a location attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L611-L615" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFusedGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirLocationFusedGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFusedGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFusedGet(ctx, nLocations, locations, metadata)
```


Creates a fused location with an array of locations and metadata.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L761-L765" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFusedGetLocations-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirLocationFusedGetLocations-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFusedGetLocations</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFusedGetLocations(location, locationsCPtr)
```


Getter for locations of Fused. Requires pre-allocated memory of #fusedLocations X sizeof([`MlirLocation`](@ref)).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L784-L788" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFusedGetMetadata-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFusedGetMetadata-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFusedGetMetadata</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFusedGetMetadata(location)
```


Getter for metadata of Fused.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L795-L799" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFusedGetNumLocations-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationFusedGetNumLocations-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFusedGetNumLocations</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFusedGetNumLocations(location)
```


Getter for number of locations fused together.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L775-L779" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationFusedGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirLocationFusedGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationFusedGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationFusedGetTypeID()
```


TypeID Getter for Fused.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L804-L808" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationGetAttribute-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationGetAttribute-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationGetAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationGetAttribute(location)
```


Returns the underlying location attribute of this location.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L602-L606" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationGetContext(location)
```


Gets the context that a location was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L878-L882" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationIsACallSite-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationIsACallSite-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationIsACallSite</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationIsACallSite(location)
```


Checks whether the given location is an CallSite.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L752-L756" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationIsAFileLineColRange-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationIsAFileLineColRange-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationIsAFileLineColRange</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationIsAFileLineColRange(location)
```


Checks whether the given location is an FileLineColRange.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L705-L709" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationIsAFused-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationIsAFused-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationIsAFused</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationIsAFused(location)
```


Checks whether the given location is an Fused.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L813-L817" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationIsAName-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationIsAName-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationIsAName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationIsAName(location)
```


Checks whether the given location is an Name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L860-L864" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationIsNull(location)
```


Checks if the location is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L887-L891" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationNameGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirLocationNameGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationNameGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationNameGet(context, name, childLoc)
```


Creates a name location owned by the given context. Providing null location for childLoc is allowed and if childLoc is null location, then the behavior is the same as having unknown child location.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L822-L826" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationNameGetChildLoc-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationNameGetChildLoc-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationNameGetChildLoc</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationNameGetChildLoc(location)
```


Getter for childLoc of Name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L842-L846" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationNameGetName-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationNameGetName-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationNameGetName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationNameGetName(location)
```


Getter for name of Name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L833-L837" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationNameGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirLocationNameGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationNameGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationNameGetTypeID()
```


TypeID Getter for Name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L851-L855" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirLocationPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationPrint(location, callback, userData)
```


Prints a location by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L905-L909" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLocationUnknownGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirLocationUnknownGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLocationUnknownGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLocationUnknownGet(context)
```


Creates a location with unknown position owned by the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L869-L873" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLogicalResultFailure-Tuple{}' href='#Reactant.MLIR.API.mlirLogicalResultFailure-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLogicalResultFailure</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLogicalResultFailure()
```


Creates a logical result representing a failure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L151-L155" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLogicalResultIsFailure-Tuple{Any}' href='#Reactant.MLIR.API.mlirLogicalResultIsFailure-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLogicalResultIsFailure</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLogicalResultIsFailure(res)
```


Checks if the given logical result represents a failure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L133-L137" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLogicalResultIsSuccess-Tuple{Any}' href='#Reactant.MLIR.API.mlirLogicalResultIsSuccess-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirLogicalResultIsSuccess</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLogicalResultIsSuccess(res)
```


Checks if the given logical result represents a success.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L124-L128" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirLogicalResultSuccess-Tuple{}' href='#Reactant.MLIR.API.mlirLogicalResultSuccess-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirLogicalResultSuccess</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirLogicalResultSuccess()
```


Creates a logical result representing a success.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L142-L146" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeContiguousGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirMemRefTypeContiguousGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeContiguousGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)
```


Creates a MemRef type with the given rank, shape, memory space and element type in the same context as the element type. The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5835-L5839" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeContiguousGetChecked-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirMemRefTypeContiguousGetChecked-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeContiguousGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)
```


Same as &quot;[`mlirMemRefTypeContiguousGet`](/api/mlirc#Reactant.MLIR.API.mlirMemRefTypeContiguousGet-NTuple{4,%20Any})&quot; but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5846-L5850" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirMemRefTypeGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)
```


Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5804-L5808" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGetAffineMap-Tuple{Any}' href='#Reactant.MLIR.API.mlirMemRefTypeGetAffineMap-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGetAffineMap</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetAffineMap(type)
```


Returns the affine map of the given MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5892-L5896" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGetChecked-NTuple{6, Any}' href='#Reactant.MLIR.API.mlirMemRefTypeGetChecked-NTuple{6, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)
```


Same as &quot;[`mlirMemRefTypeGet`](/api/mlirc#Reactant.MLIR.API.mlirMemRefTypeGet-NTuple{5,%20Any})&quot; but returns a nullptr-wrapping [`MlirType`](@ref) o illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5819-L5823" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGetLayout-Tuple{Any}' href='#Reactant.MLIR.API.mlirMemRefTypeGetLayout-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGetLayout</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetLayout(type)
```


Returns the layout of the given MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5883-L5887" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGetMemorySpace-Tuple{Any}' href='#Reactant.MLIR.API.mlirMemRefTypeGetMemorySpace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGetMemorySpace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetMemorySpace(type)
```


Returns the memory space of the given MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5901-L5905" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGetStridesAndOffset-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirMemRefTypeGetStridesAndOffset-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGetStridesAndOffset</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetStridesAndOffset(type, strides, offset)
```


Returns the strides of the MemRef if the layout map is in strided form. Both strides and offset are out params. strides must point to pre-allocated memory of length equal to the rank of the memref.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5910-L5914" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMemRefTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirMemRefTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirMemRefTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMemRefTypeGetTypeID()
```


Returns the typeID of an MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5768-L5772" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirMergeSymbolsIntoFromClone-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirMergeSymbolsIntoFromClone-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirMergeSymbolsIntoFromClone</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirMergeSymbolsIntoFromClone(target, other)
```


Merge the symbols from `other` into `target`, potentially renaming them to avoid conflicts. Private symbols may be renamed during the merge, public symbols must have at most one declaration. A name conflict in public symbols is reported as an error before returning a failure.

Note that this clones the `other` operation unlike the C++ counterpart that takes ownership.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8513-L8519" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleCreateEmpty-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleCreateEmpty-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleCreateEmpty</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleCreateEmpty(location)
```


Creates a new, empty module and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L916-L920" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleCreateParse-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirModuleCreateParse-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleCreateParse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleCreateParse(context, _module)
```


Parses a module from the string and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L925-L929" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleCreateParseFromFile-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirModuleCreateParseFromFile-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleCreateParseFromFile</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleCreateParseFromFile(context, fileName)
```


Parses a module from file and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L936-L940" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleDestroy(_module)
```


Takes a module owned by the caller and deletes it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L974-L978" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleFromOperation-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleFromOperation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleFromOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleFromOperation(op)
```


Views the generic operation as a module. The returned module is null when the input operation was not a ModuleOp.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L992-L996" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleGetBody-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleGetBody-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleGetBody</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleGetBody(_module)
```


Gets the body of the module, i.e. the only block it contains.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L956-L960" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleGetContext(_module)
```


Gets the context that a module was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L947-L951" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleGetOperation-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleGetOperation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleGetOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleGetOperation(_module)
```


Views the module as a generic operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L983-L987" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirModuleIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirModuleIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirModuleIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirModuleIsNull(_module)
```


Checks whether a module is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L965-L969" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirNamedAttributeGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirNamedAttributeGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirNamedAttributeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirNamedAttributeGet(name, attr)
```


Associates an attribute with the name. Takes ownership of neither.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2451-L2455" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirNoneTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirNoneTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirNoneTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirNoneTypeGet(ctx)
```


Creates a None type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5440-L5444" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirNoneTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirNoneTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirNoneTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirNoneTypeGetTypeID()
```


Returns the typeID of an None type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5422-L5426" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpOperandGetNextUse-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpOperandGetNextUse-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpOperandGetNextUse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpOperandGetNextUse(opOperand)
```


Returns an op operand representing the next use of the value, or a null op operand if there is no next use.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2283-L2287" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpOperandGetOperandNumber-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpOperandGetOperandNumber-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpOperandGetOperandNumber</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpOperandGetOperandNumber(opOperand)
```


Returns the operand number of an op operand.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2274-L2278" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpOperandGetOwner-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpOperandGetOwner-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpOperandGetOwner</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpOperandGetOwner(opOperand)
```


Returns the owner operation of an op operand.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2265-L2269" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpOperandGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpOperandGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpOperandGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpOperandGetValue(opOperand)
```


Returns the value of an op operand.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2256-L2260" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpOperandIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpOperandIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpOperandIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpOperandIsNull(opOperand)
```


Returns whether the op operand is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2247-L2251" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPassManagerAddOwnedPass-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOpPassManagerAddOwnedPass-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPassManagerAddOwnedPass</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPassManagerAddOwnedPass(passManager, pass)
```


Add a pass and transfer ownership to the provided mlirOpPassManager. If the pass is not a generic operation pass or matching the type of the provided PassManager, a new OpPassManager is implicitly nested under the provided PassManager.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8887-L8891" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPassManagerAddPipeline-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirOpPassManagerAddPipeline-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPassManagerAddPipeline</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)
```


Parse a sequence of textual MLIR pass pipeline elements and add them to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8898-L8902" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPassManagerGetNestedUnder-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOpPassManagerGetNestedUnder-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPassManagerGetNestedUnder</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPassManagerGetNestedUnder(passManager, operationName)
```


Nest an OpPassManager under the provided OpPassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8865-L8869" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsAssumeVerified-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsAssumeVerified-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsAssumeVerified</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsAssumeVerified(flags)
```


Do not verify the operation when using custom operation printers.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1190-L1194" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsCreate-Tuple{}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsCreate()
```


Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirOpPrintingFlagsDestroy`](/api/mlirc#Reactant.MLIR.API.mlirOpPrintingFlagsDestroy-Tuple{Any})().


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1112-L1116" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsDestroy(flags)
```


Destroys printing flags created with [`mlirOpPrintingFlagsCreate`](/api/mlirc#Reactant.MLIR.API.mlirOpPrintingFlagsCreate-Tuple{}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1121-L1125" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsElideLargeElementsAttrs-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsElideLargeElementsAttrs-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsElideLargeElementsAttrs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
```


Enables the elision of large elements attributes by printing a lexically valid but otherwise meaningless form instead of the element data. The `largeElementLimit` is used to configure what is considered to be a &quot;large&quot; ElementsAttr by providing an upper limit to the number of elements.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1130-L1134" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsElideLargeResourceString-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsElideLargeResourceString-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsElideLargeResourceString</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsElideLargeResourceString(flags, largeResourceLimit)
```


Enables the elision of large resources strings by omitting them from the `dialect_resources` section. The `largeResourceLimit` is used to configure what is considered to be a &quot;large&quot; resource by providing an upper limit to the string size.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1141-L1145" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsEnableDebugInfo-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsEnableDebugInfo-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsEnableDebugInfo</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)
```


Enable or disable printing of debug information (based on `enable`). If &#39;prettyForm&#39; is set to true, debug information is printed in a more readable &#39;pretty&#39; form. Note: The IR generated with &#39;prettyForm&#39; is not parsable.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1152-L1156" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsPrintGenericOpForm-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsPrintGenericOpForm-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsPrintGenericOpForm</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsPrintGenericOpForm(flags)
```


Always print operations in the generic form.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1163-L1167" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsPrintNameLocAsPrefix-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsPrintNameLocAsPrefix-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsPrintNameLocAsPrefix</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsPrintNameLocAsPrefix(flags)
```


Print the name and location, if NamedLoc, as a prefix to the SSA ID.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1172-L1176" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsSkipRegions-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsSkipRegions-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsSkipRegions</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsSkipRegions(flags)
```


Skip printing regions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1199-L1203" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpPrintingFlagsUseLocalScope-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpPrintingFlagsUseLocalScope-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpPrintingFlagsUseLocalScope</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpPrintingFlagsUseLocalScope(flags)
```


Use local scope when printing the operation. This allows for using the printer in a more localized and thread-safe setting, but may not necessarily be identical to what the IR will look like when dumping the full module.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1181-L1185" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpResultGetOwner-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpResultGetOwner-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpResultGetOwner</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpResultGetOwner(value)
```


Returns an operation that produced this value as its result. Asserts if the value is not an op result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2127-L2131" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpResultGetResultNumber-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpResultGetResultNumber-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpResultGetResultNumber</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpResultGetResultNumber(value)
```


Returns the position of the value in the list of results of the operation that produced it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2136-L2140" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueAttrGet-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirOpaqueAttrGet-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
```


Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3772-L3776" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueAttrGetData-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpaqueAttrGetData-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueAttrGetData</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueAttrGetData(attr)
```


Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3796-L3800" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueAttrGetDialectNamespace-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpaqueAttrGetDialectNamespace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueAttrGetDialectNamespace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueAttrGetDialectNamespace(attr)
```


Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3787-L3791" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirOpaqueAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueAttrGetTypeID()
```


Returns the typeID of an Opaque attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3805-L3809" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueTypeGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOpaqueTypeGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)
```


Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6064-L6068" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueTypeGetData-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpaqueTypeGetData-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueTypeGetData</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueTypeGetData(type)
```


Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6084-L6088" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueTypeGetDialectNamespace-Tuple{Any}' href='#Reactant.MLIR.API.mlirOpaqueTypeGetDialectNamespace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueTypeGetDialectNamespace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueTypeGetDialectNamespace(type)
```


Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6075-L6079" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOpaqueTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirOpaqueTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirOpaqueTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOpaqueTypeGetTypeID()
```


Returns the typeID of an Opaque type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6046-L6050" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationClone-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationClone-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationClone</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationClone(op)
```


Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1261-L1265" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationCreate(state)
```


Creates an operation and transfers ownership to the caller. Note that caller owned child objects are transferred in this call and must not be further used. Particularly, this applies to any regions added to the state (the implementation may invalidate any such pointers).

This call can fail under the following conditions, in which case, it will return a null operation and emit diagnostics: - Result type inference is enabled and cannot be performed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1237-L1243" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationCreateParse-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationCreateParse-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationCreateParse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationCreateParse(context, sourceStr, sourceName)
```


Parses an operation, giving ownership to the caller. If parsing fails a null operation will be returned, and an error diagnostic emitted.

`sourceStr` may be either the text assembly format, or binary bytecode format. `sourceName` is used as the file name of the source; any IR without locations will get a `FileLineColLoc` location with `sourceName` as the file name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1248-L1254" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationDestroy(op)
```


Takes an operation owned by the caller and destroys it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1270-L1274" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationDump(op)
```


Prints an operation to stderr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1677-L1681" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationEqual(op, other)
```


Checks whether two operation handles point to the same operation. This does not perform deep comparison.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1297-L1301" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetAttribute-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetAttribute-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetAttribute(op, pos)
```


Return `pos`-th attribute of the operation. Deprecated, please use `mlirOperationGetInherentAttribute` or [`mlirOperationGetDiscardableAttribute`](/api/mlirc#Reactant.MLIR.API.mlirOperationGetDiscardableAttribute-Tuple{Any,%20Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1569-L1573" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetAttributeByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetAttributeByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetAttributeByName(op, name)
```


Returns an attribute attached to the operation given its name. Deprecated, please use [`mlirOperationGetInherentAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationGetInherentAttributeByName-Tuple{Any,%20Any}) or [`mlirOperationGetDiscardableAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationGetDiscardableAttributeByName-Tuple{Any,%20Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1580-L1584" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetBlock-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetBlock-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetBlock(op)
```


Gets the block that owns this operation, returning null if the operation is not owned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1342-L1346" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetContext(op)
```


Gets the context this operation is associated with


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1306-L1310" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetDiscardableAttribute-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetDiscardableAttribute-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetDiscardableAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetDiscardableAttribute(op, pos)
```


Return `pos`-th discardable attribute of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1516-L1520" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetDiscardableAttributeByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetDiscardableAttributeByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetDiscardableAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetDiscardableAttributeByName(op, name)
```


Returns a discardable attribute attached to the operation given its name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1527-L1531" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetFirstRegion-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetFirstRegion-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetFirstRegion</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetFirstRegion(op)
```


Returns first region attached to the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1841-L1845" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetInherentAttributeByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetInherentAttributeByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetInherentAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetInherentAttributeByName(op, name)
```


Returns an inherent attribute attached to the operation given its name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1485-L1489" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetLocation-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetLocation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetLocation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetLocation(op)
```


Gets the location of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1315-L1319" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetName-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetName-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetName(op)
```


Gets the name of the operation as an identifier.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1333-L1337" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNextInBlock-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNextInBlock-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNextInBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNextInBlock(op)
```


Returns an operation immediately following the given operation it its enclosing block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1378-L1382" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNumAttributes-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNumAttributes-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNumAttributes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNumAttributes(op)
```


Returns the number of attributes attached to the operation. Deprecated, please use `mlirOperationGetNumInherentAttributes` or [`mlirOperationGetNumDiscardableAttributes`](/api/mlirc#Reactant.MLIR.API.mlirOperationGetNumDiscardableAttributes-Tuple{Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1560-L1564" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNumDiscardableAttributes-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNumDiscardableAttributes-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNumDiscardableAttributes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNumDiscardableAttributes(op)
```


Returns the number of discardable attributes attached to the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1507-L1511" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNumOperands-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNumOperands-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNumOperands</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNumOperands(op)
```


Returns the number of operands of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1387-L1391" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNumRegions-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNumRegions-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNumRegions</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNumRegions(op)
```


Returns the number of regions attached to the given operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1360-L1364" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNumResults-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNumResults-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNumResults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNumResults(op)
```


Returns the number of results of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1427-L1431" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetNumSuccessors-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetNumSuccessors-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetNumSuccessors</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetNumSuccessors(op)
```


Returns the number of successor blocks of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1445-L1449" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetOperand-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetOperand-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetOperand</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetOperand(op, pos)
```


Returns `pos`-th operand of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1396-L1400" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetParentOperation-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetParentOperation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetParentOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetParentOperation(op)
```


Gets the operation that owns this operation, returning null if the operation is not owned.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1351-L1355" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetRegion-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetRegion-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetRegion</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetRegion(op, pos)
```


Returns `pos`-th region attached to the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1369-L1373" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetResult-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetResult-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetResult</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetResult(op, pos)
```


Returns `pos`-th result of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1436-L1440" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetSuccessor-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationGetSuccessor-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetSuccessor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetSuccessor(op, pos)
```


Returns `pos`-th successor of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1454-L1458" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationGetTypeID-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationGetTypeID-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationGetTypeID(op)
```


Gets the type id of the operation. Returns null if the operation does not have a registered operation description.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1324-L1328" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationHasInherentAttributeByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationHasInherentAttributeByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationHasInherentAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationHasInherentAttributeByName(op, name)
```


Returns true if this operation defines an inherent attribute with this name. Note: the attribute can be optional, so [`mlirOperationGetInherentAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationGetInherentAttributeByName-Tuple{Any,%20Any}) can still return a null attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1474-L1478" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationImplementsInterface-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationImplementsInterface-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationImplementsInterface</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationImplementsInterface(operation, interfaceTypeID)
```


Returns `true` if the given operation implements an interface identified by its TypeID.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8622-L8626" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationImplementsInterfaceStatic-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationImplementsInterfaceStatic-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationImplementsInterfaceStatic</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)
```


Returns `true` if the operation identified by its canonical string name implements the interface identified by its TypeID in the given context. Note that interfaces may be attached to operations in some contexts and not others.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8633-L8637" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationIsNull(op)
```


Checks whether the underlying operation is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1288-L1292" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationMoveAfter-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationMoveAfter-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationMoveAfter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationMoveAfter(op, other)
```


Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1695-L1699" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationMoveBefore-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationMoveBefore-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationMoveBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationMoveBefore(op, other)
```


Moves the given operation immediately before the other operation in its parent block. The given operation may be owner by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1704-L1708" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationPrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationPrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationPrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationPrint(op, callback, userData)
```


Prints an operation by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1613-L1617" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationPrintWithFlags-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirOperationPrintWithFlags-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationPrintWithFlags</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationPrintWithFlags(op, flags, callback, userData)
```


Same as [`mlirOperationPrint`](/api/mlirc#Reactant.MLIR.API.mlirOperationPrint-Tuple{Any,%20Any,%20Any}) but accepts flags controlling the printing behavior.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1624-L1628" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationPrintWithState-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirOperationPrintWithState-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationPrintWithState</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationPrintWithState(op, state, callback, userData)
```


Same as [`mlirOperationPrint`](/api/mlirc#Reactant.MLIR.API.mlirOperationPrint-Tuple{Any,%20Any,%20Any}) but accepts AsmState controlling the printing behavior as well as caching computed names.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1638-L1642" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationRemoveAttributeByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationRemoveAttributeByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationRemoveAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationRemoveAttributeByName(op, name)
```


Removes an attribute by name. Returns false if the attribute was not found and true if removed. Deprecated, please use `mlirOperationRemoveInherentAttributeByName` or [`mlirOperationRemoveDiscardableAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationRemoveDiscardableAttributeByName-Tuple{Any,%20Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1602-L1606" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationRemoveDiscardableAttributeByName-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationRemoveDiscardableAttributeByName-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationRemoveDiscardableAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationRemoveDiscardableAttributeByName(op, name)
```


Removes a discardable attribute by name. Returns false if the attribute was not found and true if removed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1549-L1553" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationRemoveFromParent-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationRemoveFromParent-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationRemoveFromParent</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationRemoveFromParent(op)
```


Removes the given operation from its parent block. The operation is not destroyed. The ownership of the operation is transferred to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1279-L1283" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationSetAttributeByName-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationSetAttributeByName-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationSetAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationSetAttributeByName(op, name, attr)
```


Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise. Deprecated, please use [`mlirOperationSetInherentAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationSetInherentAttributeByName-Tuple{Any,%20Any,%20Any}) or [`mlirOperationSetDiscardableAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationSetDiscardableAttributeByName-Tuple{Any,%20Any,%20Any}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1591-L1595" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationSetDiscardableAttributeByName-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationSetDiscardableAttributeByName-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationSetDiscardableAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationSetDiscardableAttributeByName(op, name, attr)
```


Sets a discardable attribute by name, replacing the existing if it exists or adding a new one otherwise. The new `attr` Attribute is not allowed to be null, use [`mlirOperationRemoveDiscardableAttributeByName`](/api/mlirc#Reactant.MLIR.API.mlirOperationRemoveDiscardableAttributeByName-Tuple{Any,%20Any}) to remove an Attribute instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1538-L1542" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationSetInherentAttributeByName-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationSetInherentAttributeByName-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationSetInherentAttributeByName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationSetInherentAttributeByName(op, name, attr)
```


Sets an inherent attribute by name, replacing the existing if it exists. This has no effect if &quot;name&quot; does not match an inherent attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1496-L1500" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationSetOperand-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationSetOperand-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationSetOperand</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationSetOperand(op, pos, newValue)
```


Sets the `pos`-th operand of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1405-L1409" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationSetOperands-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationSetOperands-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationSetOperands</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationSetOperands(op, nOperands, operands)
```


Replaces the operands of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1416-L1420" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationSetSuccessor-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationSetSuccessor-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationSetSuccessor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationSetSuccessor(op, pos, block)
```


Set `pos`-th successor of the operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1463-L1467" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationStateAddResults-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationStateAddResults-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationStateAddResults</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationStateAddResults(state, n, results)
```


Adds a list of components to the operation state.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1035-L1039" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationStateEnableResultTypeInference-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationStateEnableResultTypeInference-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationStateEnableResultTypeInference</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationStateEnableResultTypeInference(state)
```


Enables result type inference for the operation under construction. If enabled, then the caller must not have called [`mlirOperationStateAddResults`](/api/mlirc#Reactant.MLIR.API.mlirOperationStateAddResults-Tuple{Any,%20Any,%20Any})(). Note that if enabled, the [`mlirOperationCreate`](/api/mlirc#Reactant.MLIR.API.mlirOperationCreate-Tuple{Any})() call is failable: it will return a null operation on inference failure and will emit diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1070-L1074" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationStateGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirOperationStateGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationStateGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationStateGet(name, loc)
```


Constructs an operation state from a name and a location.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1024-L1028" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationVerify-Tuple{Any}' href='#Reactant.MLIR.API.mlirOperationVerify-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationVerify</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationVerify(op)
```


Verify the operation and return true if it passes, false if it fails.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1686-L1690" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationWalk-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirOperationWalk-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationWalk</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationWalk(op, callback, userData, walkOrder)
```


Walks operation `op` in `walkOrder` and calls `callback` on that operation. `*userData` is passed to the callback as well and can be used to tunnel some context or other data into the callback.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1740-L1744" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationWriteBytecode-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirOperationWriteBytecode-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationWriteBytecode</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationWriteBytecode(op, callback, userData)
```


Same as [`mlirOperationPrint`](/api/mlirc#Reactant.MLIR.API.mlirOperationPrint-Tuple{Any,%20Any,%20Any}) but writing the bytecode format.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1652-L1656" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirOperationWriteBytecodeWithConfig-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirOperationWriteBytecodeWithConfig-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirOperationWriteBytecodeWithConfig</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)
```


Same as [`mlirOperationWriteBytecode`](/api/mlirc#Reactant.MLIR.API.mlirOperationWriteBytecode-Tuple{Any,%20Any,%20Any}) but with writer config and returns failure only if desired bytecode could not be honored.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1663-L1667" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirParsePassPipeline-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirParsePassPipeline-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirParsePassPipeline</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirParsePassPipeline(passManager, pipeline, callback, userData)
```


Parse a textual MLIR pass pipeline and assign it to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8923-L8927" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerAddOwnedPass-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirPassManagerAddOwnedPass-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerAddOwnedPass</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerAddOwnedPass(passManager, pass)
```


Add a pass and transfer ownership to the provided top-level mlirPassManager. If the pass is not a generic operation pass or a ModulePass, a new OpPassManager is implicitly nested under the provided PassManager.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8876-L8880" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirPassManagerCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerCreate(ctx)
```


Create a new top-level PassManager with the default anchor.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8756-L8760" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerCreateOnOperation-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirPassManagerCreateOnOperation-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerCreateOnOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerCreateOnOperation(ctx, anchorOp)
```


Create a new top-level PassManager anchored on `anchorOp`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8765-L8769" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirPassManagerDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerDestroy(passManager)
```


Destroy the provided PassManager.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8776-L8780" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerEnableIRPrinting-NTuple{8, Any}' href='#Reactant.MLIR.API.mlirPassManagerEnableIRPrinting-NTuple{8, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerEnableIRPrinting</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerEnableIRPrinting(passManager, printBeforeAll, printAfterAll, printModuleScope, printAfterOnlyOnChange, printAfterOnlyOnFailure, flags, treePrintingPath)
```


Enable IR printing. The treePrintingPath argument is an optional path to a directory where the dumps will be produced. If it isn&#39;t provided then dumps are produced to stderr.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8816-L8820" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerEnableVerifier-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirPassManagerEnableVerifier-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerEnableVerifier</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerEnableVerifier(passManager, enable)
```


Enable / disable verify-each.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8843-L8847" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerGetAsOpPassManager-Tuple{Any}' href='#Reactant.MLIR.API.mlirPassManagerGetAsOpPassManager-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerGetAsOpPassManager</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerGetAsOpPassManager(passManager)
```


Cast a top-level PassManager to a generic OpPassManager.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8794-L8798" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerGetNestedUnder-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirPassManagerGetNestedUnder-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerGetNestedUnder</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerGetNestedUnder(passManager, operationName)
```


Nest an OpPassManager under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. To further nest more OpPassManager under the newly returned one, see `mlirOpPassManagerNest` below.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8854-L8858" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirPassManagerIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerIsNull(passManager)
```


Checks if a PassManager is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8785-L8789" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPassManagerRunOnOp-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirPassManagerRunOnOp-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPassManagerRunOnOp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPassManagerRunOnOp(passManager, op)
```


Run the provided `passManager` on the given `op`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8805-L8809" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirPrintPassPipeline-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirPrintPassPipeline-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirPrintPassPipeline</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirPrintPassPipeline(passManager, callback, userData)
```


Print a textual MLIR pass pipeline by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8912-L8916" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeCastExpressedToStorageType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeCastExpressedToStorageType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeCastExpressedToStorageType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeCastExpressedToStorageType(type, candidate)
```


Casts from a type based on the expressed type of the given quantized type to equivalent type based on storage type of the same quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7650-L7654" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeCastFromExpressedType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeCastFromExpressedType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeCastFromExpressedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeCastFromExpressedType(type, candidate)
```


Casts from a type based on the expressed type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7630-L7634" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeCastFromStorageType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeCastFromStorageType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeCastFromStorageType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeCastFromStorageType(type, candidate)
```


Casts from a type based on the storage type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7610-L7614" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeCastToExpressedType-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeCastToExpressedType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeCastToExpressedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeCastToExpressedType(type)
```


Casts from a type based on a quantized type to a corresponding typed based on the expressed type. Returns a null type if the cast is not valid.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7641-L7645" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeCastToStorageType-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeCastToStorageType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeCastToStorageType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeCastToStorageType(type)
```


Casts from a type based on a quantized type to a corresponding typed based on the storage type. Returns a null type if the cast is not valid.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7621-L7625" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetDefaultMaximumForInteger-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetDefaultMaximumForInteger-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetDefaultMaximumForInteger</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)
```


Returns the maximum possible value stored by a quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7516-L7520" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetDefaultMinimumForInteger-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetDefaultMinimumForInteger-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetDefaultMinimumForInteger</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)
```


Returns the minimum possible value stored by a quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7505-L7509" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetExpressedType-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetExpressedType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetExpressedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetExpressedType(type)
```


Gets the original type approximated by the given quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7527-L7531" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetFlags-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetFlags-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetFlags</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetFlags(type)
```


Gets the flags associated with the given quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7536-L7540" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetQuantizedElementType-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetQuantizedElementType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetQuantizedElementType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetQuantizedElementType(type)
```


Returns the element type of the given quantized type as another quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7601-L7605" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetSignedFlag-Tuple{}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetSignedFlag-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetSignedFlag</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetSignedFlag()
```


Returns the bit flag used to indicate signedness of a quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7496-L7500" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetStorageType-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetStorageType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetStorageType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetStorageType(type)
```


Returns the underlying type used to store the values.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7554-L7558" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeIntegralWidth-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeIntegralWidth-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeIntegralWidth</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetStorageTypeIntegralWidth(type)
```


Returns the integral bitwidth that the storage type of the given quantized type can represent exactly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7581-L7585" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeMax-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeMax-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeMax</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetStorageTypeMax(type)
```


Returns the maximum value that the storage type of the given quantized type can take.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7572-L7576" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeMin-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeMin-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeGetStorageTypeMin</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeGetStorageTypeMin(type)
```


Returns the minimum value that the storage type of the given quantized type can take.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7563-L7567" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeIsCompatibleExpressedType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeIsCompatibleExpressedType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeIsCompatibleExpressedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)
```


Returns `true` if the `candidate` type is compatible with the given quantized `type`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7590-L7594" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirQuantizedTypeIsSigned-Tuple{Any}' href='#Reactant.MLIR.API.mlirQuantizedTypeIsSigned-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirQuantizedTypeIsSigned</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirQuantizedTypeIsSigned(type)
```


Returns `true` if the given type is signed, `false` otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7545-L7549" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRankedTensorTypeGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirRankedTensorTypeGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRankedTensorTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRankedTensorTypeGet(rank, shape, elementType, encoding)
```


Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type. The type is owned by the context. Tensor types without any specific encoding field should assign [`mlirAttributeGetNull`](/api/mlirc#Reactant.MLIR.API.mlirAttributeGetNull-Tuple{})() to this parameter.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5713-L5717" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRankedTensorTypeGetChecked-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirRankedTensorTypeGetChecked-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRankedTensorTypeGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)
```


Same as &quot;[`mlirRankedTensorTypeGet`](/api/mlirc#Reactant.MLIR.API.mlirRankedTensorTypeGet-NTuple{4,%20Any})&quot; but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5724-L5728" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRankedTensorTypeGetEncoding-Tuple{Any}' href='#Reactant.MLIR.API.mlirRankedTensorTypeGetEncoding-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRankedTensorTypeGetEncoding</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRankedTensorTypeGetEncoding(type)
```


Gets the &#39;encoding&#39; attribute from the ranked tensor type, returning a null attribute if none.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5739-L5743" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRankedTensorTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirRankedTensorTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirRankedTensorTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRankedTensorTypeGetTypeID()
```


Returns the typeID of an RankedTensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5677-L5681" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionAppendOwnedBlock-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRegionAppendOwnedBlock-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionAppendOwnedBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionAppendOwnedBlock(region, block)
```


Takes a block owned by the caller and appends it to the given region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1799-L1803" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionCreate-Tuple{}' href='#Reactant.MLIR.API.mlirRegionCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionCreate()
```


Creates a new empty region and transfers ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1754-L1758" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirRegionDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionDestroy(region)
```


Takes a region owned by the caller and destroys it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1763-L1767" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRegionEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionEqual(region, other)
```


Checks whether two region handles point to the same region. This does not perform deep comparison.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1781-L1785" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionGetFirstBlock-Tuple{Any}' href='#Reactant.MLIR.API.mlirRegionGetFirstBlock-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionGetFirstBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionGetFirstBlock(region)
```


Gets the first block in the region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1790-L1794" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionGetNextInOperation-Tuple{Any}' href='#Reactant.MLIR.API.mlirRegionGetNextInOperation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionGetNextInOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionGetNextInOperation(region)
```


Returns the region immediately following the given region in its parent operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1850-L1854" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionInsertOwnedBlock-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRegionInsertOwnedBlock-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionInsertOwnedBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionInsertOwnedBlock(region, pos, block)
```


Takes a block owned by the caller and inserts it at `pos` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1808-L1812" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionInsertOwnedBlockAfter-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRegionInsertOwnedBlockAfter-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionInsertOwnedBlockAfter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionInsertOwnedBlockAfter(region, reference, block)
```


Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1819-L1823" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionInsertOwnedBlockBefore-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRegionInsertOwnedBlockBefore-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionInsertOwnedBlockBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionInsertOwnedBlockBefore(region, reference, block)
```


Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1830-L1834" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirRegionIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionIsNull(region)
```


Checks whether a region is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1772-L1776" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegionTakeBody-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRegionTakeBody-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegionTakeBody</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegionTakeBody(target, source)
```


Moves the entire content of the source region to the target region.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L1859-L1863" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegisterAllDialects-Tuple{Any}' href='#Reactant.MLIR.API.mlirRegisterAllDialects-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegisterAllDialects</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegisterAllDialects(registry)
```


Appends all upstream dialects and extensions to the dialect registry.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8996-L9000" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegisterAllLLVMTranslations-Tuple{Any}' href='#Reactant.MLIR.API.mlirRegisterAllLLVMTranslations-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRegisterAllLLVMTranslations</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegisterAllLLVMTranslations(context)
```


Register all translations to LLVM IR for dialects that can support it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9005-L9009" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRegisterAllPasses-Tuple{}' href='#Reactant.MLIR.API.mlirRegisterAllPasses-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirRegisterAllPasses</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRegisterAllPasses()
```


Register all compiler passes of MLIR.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9014-L9018" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseCancelOpModification-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseCancelOpModification-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseCancelOpModification</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseCancelOpModification(rewriter, op)
```


This method cancels a pending in-place modification. This can only be called on operations that were provided to a call to `startOpModification`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9338-L9342" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseClearInsertionPoint-Tuple{Any}' href='#Reactant.MLIR.API.mlirRewriterBaseClearInsertionPoint-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseClearInsertionPoint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseClearInsertionPoint(rewriter)
```


Reset the insertion point to no location. Creating an operation without a set insertion point is an error, but this can still be useful when the current insertion point a builder refers to is being removed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9048-L9052" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseClone-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseClone-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseClone</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseClone(rewriter, op)
```


Creates a deep copy of the specified operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9158-L9162" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseCloneRegionBefore-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseCloneRegionBefore-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseCloneRegionBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseCloneRegionBefore(rewriter, region, before)
```


Clone the blocks that belong to &quot;region&quot; before the given position in another region &quot;parent&quot;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9180-L9184" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseCloneWithoutRegions-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseCloneWithoutRegions-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseCloneWithoutRegions</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseCloneWithoutRegions(rewriter, op)
```


Creates a deep copy of this operation but keep the operation regions empty.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9169-L9173" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseCreateBlockBefore-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseCreateBlockBefore-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseCreateBlockBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseCreateBlockBefore(rewriter, insertBefore, nArgTypes, argTypes, locations)
```


Add new block with &#39;argTypes&#39; arguments and set the insertion point to the end of it. The block is placed before &#39;insertBefore&#39;. `locs` contains the locations of the inserted arguments, and should match the size of `argTypes`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9130-L9134" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseEraseBlock-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseEraseBlock-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseEraseBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseEraseBlock(rewriter, block)
```


Erases a block along with all operations inside it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9238-L9242" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseEraseOp-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseEraseOp-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseEraseOp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseEraseOp(rewriter, op)
```


Erases an operation that is known to have no uses.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9227-L9231" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseFinalizeOpModification-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseFinalizeOpModification-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseFinalizeOpModification</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseFinalizeOpModification(rewriter, op)
```


This method is used to signal the end of an in-place modification of the given operation. This can only be called on operations that were provided to a call to `startOpModification`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9327-L9331" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseGetBlock-Tuple{Any}' href='#Reactant.MLIR.API.mlirRewriterBaseGetBlock-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseGetBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseGetBlock(rewriter)
```


Returns the current block of the rewriter.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9121-L9125" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirRewriterBaseGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseGetContext(rewriter)
```


Get the MLIR context referenced by the rewriter.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9039-L9043" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseGetInsertionBlock-Tuple{Any}' href='#Reactant.MLIR.API.mlirRewriterBaseGetInsertionBlock-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseGetInsertionBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseGetInsertionBlock(rewriter)
```


Return the block the current insertion point belongs to. Note that the insertion point is not necessarily the end of the block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9112-L9116" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseInlineBlockBefore-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseInlineBlockBefore-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseInlineBlockBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseInlineBlockBefore(rewriter, source, op, nArgValues, argValues)
```


Inline the operations of block &#39;source&#39; before the operation &#39;op&#39;. The source block will be deleted and must have no uses. &#39;argValues&#39; is used to replace the block arguments of &#39;source&#39;

The source block must have no successors. Otherwise, the resulting IR would have unreachable operations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9249-L9255" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseInlineRegionBefore-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseInlineRegionBefore-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseInlineRegionBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseInlineRegionBefore(rewriter, region, before)
```


Move the blocks that belong to &quot;region&quot; before the given position in another region &quot;parent&quot;. The two regions must be different. The caller is responsible for creating or updating the operation transferring flow of control to the region and passing it the correct block arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9191-L9195" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseInsert-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseInsert-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseInsert</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseInsert(rewriter, op)
```


Insert the given operation at the current insertion point and return it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9147-L9151" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseMergeBlocks-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseMergeBlocks-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseMergeBlocks</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseMergeBlocks(rewriter, source, dest, nArgValues, argValues)
```


Inline the operations of block &#39;source&#39; into the end of block &#39;dest&#39;. The source block will be deleted and must have no uses. &#39;argValues&#39; is used to replace the block arguments of &#39;source&#39;

The dest block must have no successors. Otherwise, the resulting IR would have unreachable operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9266-L9272" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseMoveBlockBefore-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseMoveBlockBefore-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseMoveBlockBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseMoveBlockBefore(rewriter, block, existingBlock)
```


Unlink this block and insert it right before `existingBlock`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9305-L9309" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseMoveOpAfter-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseMoveOpAfter-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseMoveOpAfter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseMoveOpAfter(rewriter, op, existingOp)
```


Unlink this operation from its current block and insert it right after `existingOp` which may be in the same or another block in the same function.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9294-L9298" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseMoveOpBefore-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseMoveOpBefore-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseMoveOpBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseMoveOpBefore(rewriter, op, existingOp)
```


Unlink this operation from its current block and insert it right before `existingOp` which may be in the same or another block in the same function.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9283-L9287" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceAllOpUsesWithOperation-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceAllOpUsesWithOperation-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceAllOpUsesWithOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceAllOpUsesWithOperation(rewriter, from, to)
```


Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced) and that the `from` operation is about to be replaced.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9385-L9389" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceAllOpUsesWithValueRange-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceAllOpUsesWithValueRange-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceAllOpUsesWithValueRange</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceAllOpUsesWithValueRange(rewriter, from, nTo, to)
```


Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced) and that the `from` operation is about to be replaced.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9374-L9378" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceAllUsesExcept-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceAllUsesExcept-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceAllUsesExcept</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceAllUsesExcept(rewriter, from, to, exceptedUser)
```


Find uses of `from` and replace them with `to` except if the user is `exceptedUser`. Also notify the listener about every in-place op modification (for every use that was replaced).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9413-L9417" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceAllUsesWith-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceAllUsesWith-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceAllUsesWith</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceAllUsesWith(rewriter, from, to)
```


Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9349-L9353" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceAllValueRangeUsesWith-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceAllValueRangeUsesWith-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceAllValueRangeUsesWith</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, nValues, from, to)
```


Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9360-L9364" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceOpUsesWithinBlock-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceOpUsesWithinBlock-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceOpUsesWithinBlock</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceOpUsesWithinBlock(rewriter, op, nNewValues, newValues, block)
```


Find uses of `from` within `block` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced). The optional `allUsesReplaced` flag is set to &quot;true&quot; if all uses were replaced.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9396-L9400" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceOpWithOperation-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceOpWithOperation-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceOpWithOperation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceOpWithOperation(rewriter, op, newOp)
```


Replace the results of the given (original) operation with the specified new op (replacement). The result types of the two ops must match. The original op is erased.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9216-L9220" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseReplaceOpWithValues-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseReplaceOpWithValues-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseReplaceOpWithValues</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseReplaceOpWithValues(rewriter, op, nValues, values)
```


Replace the results of the given (original) operation with the specified list of values (replacements). The result types of the given op and the replacements must match. The original op is erased.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9202-L9206" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointAfter-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointAfter-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointAfter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseSetInsertionPointAfter(rewriter, op)
```


Sets the insertion point to the node after the specified operation, which will cause subsequent insertions to go right after it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9068-L9072" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointAfterValue-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointAfterValue-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointAfterValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseSetInsertionPointAfterValue(rewriter, value)
```


Sets the insertion point to the node after the specified value. If value has a defining operation, sets the insertion point to the node after such defining operation. This will cause subsequent insertions to go right after it. Otherwise, value is a BlockArgument. Sets the insertion point to the start of its block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9079-L9083" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointBefore-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointBefore-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointBefore</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseSetInsertionPointBefore(rewriter, op)
```


Sets the insertion point to the specified operation, which will cause subsequent insertions to go right before it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9057-L9061" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointToEnd-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointToEnd-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointToEnd</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseSetInsertionPointToEnd(rewriter, block)
```


Sets the insertion point to the end of the specified block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9101-L9105" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointToStart-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointToStart-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseSetInsertionPointToStart</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseSetInsertionPointToStart(rewriter, block)
```


Sets the insertion point to the start of the specified block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9090-L9094" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirRewriterBaseStartOpModification-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirRewriterBaseStartOpModification-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirRewriterBaseStartOpModification</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirRewriterBaseStartOpModification(rewriter, op)
```


This method is used to notify the rewriter that an in-place operation modification is about to happen. A call to this function _must_ be followed by a call to either `finalizeOpModification` or `cancelOpModification`. This is a minor efficiency win (it avoids creating a new operation and removing the old one) but also often allows simpler code in the client.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9316-L9320" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTAttrCheckBVCmpPredicate-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSMTAttrCheckBVCmpPredicate-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTAttrCheckBVCmpPredicate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTAttrCheckBVCmpPredicate(ctx, str)
```


Checks if the given string is a valid smt::BVCmpPredicate.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8114-L8118" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTAttrCheckIntPredicate-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSMTAttrCheckIntPredicate-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTAttrCheckIntPredicate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTAttrCheckIntPredicate(ctx, str)
```


Checks if the given string is a valid smt::IntPredicate.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8123-L8127" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTAttrGetBVCmpPredicate-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSMTAttrGetBVCmpPredicate-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTAttrGetBVCmpPredicate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTAttrGetBVCmpPredicate(ctx, str)
```


Creates a smt::BVCmpPredicateAttr with the given string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8152-L8156" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTAttrGetBitVector-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirSMTAttrGetBitVector-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTAttrGetBitVector</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTAttrGetBitVector(ctx, value, width)
```


Creates a smt::BitVectorAttr with the given value and width.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8141-L8145" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTAttrGetIntPredicate-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSMTAttrGetIntPredicate-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTAttrGetIntPredicate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTAttrGetIntPredicate(ctx, str)
```


Creates a smt::IntPredicateAttr with the given string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8163-L8167" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTAttrIsASMTAttribute-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTAttrIsASMTAttribute-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTAttrIsASMTAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTAttrIsASMTAttribute(attr)
```


Checks if the given attribute is a smt::SMTAttribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8132-L8136" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeGetArray-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirSMTTypeGetArray-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeGetArray</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeGetArray(ctx, domainType, rangeType)
```


Creates an array type with the given domain and range types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8003-L8007" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeGetBitVector-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSMTTypeGetBitVector-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeGetBitVector</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeGetBitVector(ctx, width)
```


Creates a smt::BitVectorType with the given width.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8023-L8027" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeGetBool-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeGetBool-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeGetBool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeGetBool(ctx)
```


Creates a smt::BoolType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8041-L8045" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeGetInt-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeGetInt-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeGetInt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeGetInt(ctx)
```


Creates a smt::IntType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8059-L8063" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeGetSMTFunc-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirSMTTypeGetSMTFunc-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeGetSMTFunc</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeGetSMTFunc(ctx, numberOfDomainTypes, domainTypes, rangeType)
```


Creates a smt::FuncType with the given domain and range types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8077-L8081" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeGetSort-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirSMTTypeGetSort-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeGetSort</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeGetSort(ctx, identifier, numberOfSortParams, sortParams)
```


Creates a smt::SortType with the given identifier and sort parameters.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8100-L8104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsAArray-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsAArray-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsAArray</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsAArray(type)
```


Checks if the given type is a smt::ArrayType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7994-L7998" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsABitVector-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsABitVector-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsABitVector</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsABitVector(type)
```


Checks if the given type is a smt::BitVectorType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8014-L8018" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsABool-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsABool-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsABool</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsABool(type)
```


Checks if the given type is a smt::BoolType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8032-L8036" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsAInt-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsAInt-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsAInt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsAInt(type)
```


Checks if the given type is a smt::IntType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8050-L8054" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsASMTFunc-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsASMTFunc-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsASMTFunc</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsASMTFunc(type)
```


Checks if the given type is a smt::FuncType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8068-L8072" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsASort-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsASort-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsASort</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsASort(type)
```


Checks if the given type is a smt::SortType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8091-L8095" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsAnyNonFuncSMTValueType-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsAnyNonFuncSMTValueType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsAnyNonFuncSMTValueType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsAnyNonFuncSMTValueType(type)
```


Checks if the given type is any non-func SMT value type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7976-L7980" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSMTTypeIsAnySMTValueType-Tuple{Any}' href='#Reactant.MLIR.API.mlirSMTTypeIsAnySMTValueType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSMTTypeIsAnySMTValueType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSMTTypeIsAnySMTValueType(type)
```


Checks if the given type is any SMT value type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7985-L7989" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSetGlobalDebugType-Tuple{Any}' href='#Reactant.MLIR.API.mlirSetGlobalDebugType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSetGlobalDebugType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSetGlobalDebugType(type)
```


Sets the current debug type, similarly to `-debug-only=type` in the command-line tools. Note that global debug should be enabled for any output to be produced.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6111-L6115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSetGlobalDebugTypes-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSetGlobalDebugTypes-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSetGlobalDebugTypes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSetGlobalDebugTypes(types, n)
```


Sets multiple current debug types, similarly to `-debug-only=type1,type2&quot; in the command-line tools. Note that global debug should be enabled for any output to be produced.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6120-L6124" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeGetDimSize-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirShapedTypeGetDimSize-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeGetDimSize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeGetDimSize(type, dim)
```


Returns the dim-th dimension of the given ranked shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5539-L5543" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeGetDynamicSize-Tuple{}' href='#Reactant.MLIR.API.mlirShapedTypeGetDynamicSize-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeGetDynamicSize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeGetDynamicSize()
```


Returns the value indicating a dynamic size in a shaped type. Prefer [`mlirShapedTypeIsDynamicSize`](/api/mlirc#Reactant.MLIR.API.mlirShapedTypeIsDynamicSize-Tuple{Any}) to direct comparisons with this value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5557-L5561" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeGetDynamicStrideOrOffset-Tuple{}' href='#Reactant.MLIR.API.mlirShapedTypeGetDynamicStrideOrOffset-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeGetDynamicStrideOrOffset</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeGetDynamicStrideOrOffset()
```


Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`mlirShapedTypeGetDynamicStrideOrOffset`](/api/mlirc#Reactant.MLIR.API.mlirShapedTypeGetDynamicStrideOrOffset-Tuple{}) to direct comparisons with this value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5575-L5579" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeGetElementType-Tuple{Any}' href='#Reactant.MLIR.API.mlirShapedTypeGetElementType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeGetElementType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeGetElementType(type)
```


Returns the element type of the shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5494-L5498" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeGetRank-Tuple{Any}' href='#Reactant.MLIR.API.mlirShapedTypeGetRank-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeGetRank</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeGetRank(type)
```


Returns the rank of the given ranked shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5512-L5516" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeHasRank-Tuple{Any}' href='#Reactant.MLIR.API.mlirShapedTypeHasRank-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeHasRank</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeHasRank(type)
```


Checks whether the given shaped type is ranked.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5503-L5507" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeHasStaticShape-Tuple{Any}' href='#Reactant.MLIR.API.mlirShapedTypeHasStaticShape-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeHasStaticShape</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeHasStaticShape(type)
```


Checks whether the given shaped type has a static shape.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5521-L5525" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeIsDynamicDim-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirShapedTypeIsDynamicDim-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeIsDynamicDim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeIsDynamicDim(type, dim)
```


Checks wither the dim-th dimension of the given shaped type is dynamic.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5530-L5534" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeIsDynamicSize-Tuple{Any}' href='#Reactant.MLIR.API.mlirShapedTypeIsDynamicSize-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeIsDynamicSize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeIsDynamicSize(size)
```


Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5548-L5552" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirShapedTypeIsDynamicStrideOrOffset-Tuple{Any}' href='#Reactant.MLIR.API.mlirShapedTypeIsDynamicStrideOrOffset-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirShapedTypeIsDynamicStrideOrOffset</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirShapedTypeIsDynamicStrideOrOffset(val)
```


Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5566-L5570" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSimplifyAffineExpr-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirSimplifyAffineExpr-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSimplifyAffineExpr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSimplifyAffineExpr(expr, numDims, numSymbols)
```


Simplify an affine expression by flattening and some amount of simple analysis. This has complexity linear in the number of nodes in &#39;expr&#39;. Returns the simplified expression, which is the same as the input expression if it can&#39;t be simplified. When `expr` is semi-affine, a simplified semi-affine expression is constructed in the sorted order of dimension and symbol positions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2740-L2744" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseElementsAttrGetIndices-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseElementsAttrGetIndices-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseElementsAttrGetIndices</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseElementsAttrGetIndices(attr)
```


Returns the dense elements attribute containing 64-bit integer indices of non-null elements in the given sparse elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4806-L4810" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseElementsAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirSparseElementsAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseElementsAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseElementsAttrGetTypeID()
```


Returns the typeID of a SparseElements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4824-L4828" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseElementsAttrGetValues-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseElementsAttrGetValues-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseElementsAttrGetValues</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseElementsAttrGetValues(attr)
```


Returns the dense elements attribute containing the non-null elements in the given sparse elements attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4815-L4819" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseElementsAttribute-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirSparseElementsAttribute-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseElementsAttribute</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
```


Creates a sparse elements attribute of the given shape from a list of indices and a list of associated values. Both lists are expected to be dense elements attributes with the same number of elements. The list of indices is expected to contain 64-bit integers. The attribute is created in the same context as the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4795-L4799" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGet-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGet-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGet(ctx, lvlRank, lvlTypes, dimToLvl, lvlTodim, posWidth, crdWidth, explicitVal, implicitVal)
```


Creates a `sparse\_tensor.encoding` attribute with the given parameters.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8217-L8221" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetCrdWidth-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetCrdWidth-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetCrdWidth</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetCrdWidth(attr)
```


Returns the coordinate bitwidth of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8300-L8304" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetDimToLvl-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetDimToLvl-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetDimToLvl</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetDimToLvl(attr)
```


Returns the dimension-to-level mapping of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8269-L8273" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetExplicitVal-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetExplicitVal-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetExplicitVal</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetExplicitVal(attr)
```


Returns the explicit value of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8309-L8313" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetImplicitVal-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetImplicitVal-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetImplicitVal</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetImplicitVal(attr)
```


Returns the implicit value of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8320-L8324" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlFmt-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlFmt-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlFmt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetLvlFmt(attr, lvl)
```


Returns a specified level-format of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8258-L8262" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlToDim-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlToDim-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlToDim</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetLvlToDim(attr)
```


Returns the level-to-dimension mapping of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8280-L8284" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetLvlType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)
```


Returns a specified level-type of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8247-L8251" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetPosWidth-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetPosWidth-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingAttrGetPosWidth</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingAttrGetPosWidth(attr)
```


Returns the position bitwidth of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8291-L8295" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSparseTensorEncodingGetLvlRank-Tuple{Any}' href='#Reactant.MLIR.API.mlirSparseTensorEncodingGetLvlRank-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSparseTensorEncodingGetLvlRank</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSparseTensorEncodingGetLvlRank(attr)
```


Returns the level-rank of the `sparse\_tensor.encoding` attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8238-L8242" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStridedLayoutAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirStridedLayoutAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirStridedLayoutAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStridedLayoutAttrGetTypeID()
```


Returns the typeID of a StridedLayout attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4855-L4859" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringAttrGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirStringAttrGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirStringAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringAttrGet(ctx, str)
```


Creates a string attribute in the given context containing the given string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3823-L3827" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirStringAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirStringAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringAttrGetTypeID()
```


Returns the typeID of a String attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3850-L3854" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringAttrGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirStringAttrGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirStringAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringAttrGetValue(attr)
```


Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3841-L3845" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringAttrTypedGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirStringAttrTypedGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirStringAttrTypedGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringAttrTypedGet(type, str)
```


Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3832-L3836" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringRefCreate-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirStringRefCreate-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirStringRefCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringRefCreate(str, length)
```


Constructs a string reference from the pointer and length. The pointer need not reference to a null-terminated string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L80-L84" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringRefCreateFromCString-Tuple{Any}' href='#Reactant.MLIR.API.mlirStringRefCreateFromCString-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirStringRefCreateFromCString</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringRefCreateFromCString(str)
```


Constructs a string reference from a null-terminated C string. Prefer [`mlirStringRefCreate`](/api/mlirc#Reactant.MLIR.API.mlirStringRefCreate-Tuple{Any,%20Any}) if the length of the string is known.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L89-L93" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirStringRefEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirStringRefEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirStringRefEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirStringRefEqual(string, other)
```


Returns true if two string references are equal, false otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L98-L102" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolRefAttrGet-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirSymbolRefAttrGet-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolRefAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
```


Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3868-L3872" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolRefAttrGetLeafReference-Tuple{Any}' href='#Reactant.MLIR.API.mlirSymbolRefAttrGetLeafReference-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolRefAttrGetLeafReference</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolRefAttrGetLeafReference(attr)
```


Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3891-L3895" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolRefAttrGetNestedReference-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSymbolRefAttrGetNestedReference-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolRefAttrGetNestedReference</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolRefAttrGetNestedReference(attr, pos)
```


Returns pos-th reference nested in the given symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3909-L3913" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolRefAttrGetNumNestedReferences-Tuple{Any}' href='#Reactant.MLIR.API.mlirSymbolRefAttrGetNumNestedReferences-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolRefAttrGetNumNestedReferences</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolRefAttrGetNumNestedReferences(attr)
```


Returns the number of references nested in the given symbol reference attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3900-L3904" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolRefAttrGetRootReference-Tuple{Any}' href='#Reactant.MLIR.API.mlirSymbolRefAttrGetRootReference-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolRefAttrGetRootReference</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolRefAttrGetRootReference(attr)
```


Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3882-L3886" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolRefAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirSymbolRefAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolRefAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolRefAttrGetTypeID()
```


Returns the typeID of an SymbolRef attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3920-L3924" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirSymbolTableCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableCreate(operation)
```


Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2518-L2522" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirSymbolTableDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableDestroy(symbolTable)
```


Destroys the symbol table created with [`mlirSymbolTableCreate`](/api/mlirc#Reactant.MLIR.API.mlirSymbolTableCreate-Tuple{Any}). This does not affect the operations in the table.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2536-L2540" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableErase-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSymbolTableErase-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableErase</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableErase(symbolTable, operation)
```


Removes the given operation from the symbol table and erases it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2567-L2571" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableGetSymbolAttributeName-Tuple{}' href='#Reactant.MLIR.API.mlirSymbolTableGetSymbolAttributeName-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableGetSymbolAttributeName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableGetSymbolAttributeName()
```


Returns the name of the attribute used to store symbol names compatible with symbol tables.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2500-L2504" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableGetVisibilityAttributeName-Tuple{}' href='#Reactant.MLIR.API.mlirSymbolTableGetVisibilityAttributeName-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableGetVisibilityAttributeName</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableGetVisibilityAttributeName()
```


Returns the name of the attribute used to store symbol visibility.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2509-L2513" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableInsert-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSymbolTableInsert-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableInsert</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableInsert(symbolTable, operation)
```


Inserts the given operation into the given symbol table. The operation must have the symbol trait. If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness. Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately. Returns the name of the symbol after insertion.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2556-L2560" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirSymbolTableIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableIsNull(symbolTable)
```


Returns true if the symbol table is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2527-L2531" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableLookup-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirSymbolTableLookup-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableLookup</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableLookup(symbolTable, name)
```


Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol. If the symbol cannot be found, returns a null operation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2545-L2549" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableReplaceAllSymbolUses-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirSymbolTableReplaceAllSymbolUses-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableReplaceAllSymbolUses</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)
```


Attempt to replace all uses that are nested within the given operation of the given symbol &#39;oldSymbol&#39; with the provided &#39;newSymbol&#39;. This does not traverse into nested symbol tables. Will fail atomically if there are any unknown operations that may be potential symbol tables.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2578-L2582" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirSymbolTableWalkSymbolTables-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirSymbolTableWalkSymbolTables-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirSymbolTableWalkSymbolTables</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)
```


Walks all symbol table operations nested within, and including, `op`. For each symbol table operation, the provided callback is invoked with the op and a boolean signifying if the symbols within that symbol table can be treated as if all uses within the IR are visible to the caller. `allSymUsesVisible` identifies whether all of the symbol uses of symbols within `op` are visible.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2589-L2593" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTF32TypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirTF32TypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTF32TypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTF32TypeGet(ctx)
```


Creates a TF32 type in the given context. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5413-L5417" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformApplyNamedSequence-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirTransformApplyNamedSequence-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformApplyNamedSequence</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformApplyNamedSequence(payload, transformRoot, transformModule, transformOptions)
```


Applies the transformation script starting at the given transform root operation to the given payload operation. The module containing the transform root as well as the transform options should be provided. The transform operation must implement TransformOpInterface and the module must be a ModuleOp. Returns the status of the application.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8497-L8501" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformOptionsCreate-Tuple{}' href='#Reactant.MLIR.API.mlirTransformOptionsCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformOptionsCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformOptionsCreate()
```


Creates a default-initialized transform options object.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8435-L8439" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformOptionsDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirTransformOptionsDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformOptionsDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformOptionsDestroy(transformOptions)
```


Destroys a transform options object previously created by [`mlirTransformOptionsCreate`](/api/mlirc#Reactant.MLIR.API.mlirTransformOptionsCreate-Tuple{}).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8488-L8492" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformOptionsEnableExpensiveChecks-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTransformOptionsEnableExpensiveChecks-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformOptionsEnableExpensiveChecks</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformOptionsEnableExpensiveChecks(transformOptions, enable)
```


Enables or disables expensive checks in transform options.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8444-L8448" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformOptionsEnforceSingleTopLevelTransformOp-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTransformOptionsEnforceSingleTopLevelTransformOp-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformOptionsEnforceSingleTopLevelTransformOp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformOptionsEnforceSingleTopLevelTransformOp(transformOptions, enable)
```


Enables or disables the enforcement of the top-level transform op being single in transform options.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8466-L8470" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformOptionsGetEnforceSingleTopLevelTransformOp-Tuple{Any}' href='#Reactant.MLIR.API.mlirTransformOptionsGetEnforceSingleTopLevelTransformOp-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformOptionsGetEnforceSingleTopLevelTransformOp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(transformOptions)
```


Returns true if the enforcement of the top-level transform op being single is enabled in transform options.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8477-L8481" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTransformOptionsGetExpensiveChecksEnabled-Tuple{Any}' href='#Reactant.MLIR.API.mlirTransformOptionsGetExpensiveChecksEnabled-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTransformOptionsGetExpensiveChecksEnabled</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTransformOptionsGetExpensiveChecksEnabled(transformOptions)
```


Returns true if expensive checks are enabled in transform options.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L8455-L8459" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTranslateModuleToLLVMIR-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTranslateModuleToLLVMIR-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTranslateModuleToLLVMIR</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTranslateModuleToLLVMIR(_module, context)
```


Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context. This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.

**Returns**

the generated LLVM IR Module from the translated MLIR module, it is owned by the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9754-L9761" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTranslateModuleToSMTLIB-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirTranslateModuleToSMTLIB-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTranslateModuleToSMTLIB</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTranslateModuleToSMTLIB(arg1, arg2, userData, inlineSingleUseValues, indentLetBody)
```


Emits SMTLIB for the specified module using the provided callback and user data


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9477-L9481" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTupleTypeGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirTupleTypeGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTupleTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTupleTypeGet(ctx, numElements, elements)
```


Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5948-L5952" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTupleTypeGetNumTypes-Tuple{Any}' href='#Reactant.MLIR.API.mlirTupleTypeGetNumTypes-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTupleTypeGetNumTypes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTupleTypeGetNumTypes(type)
```


Returns the number of types contained in a tuple.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5959-L5963" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTupleTypeGetType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTupleTypeGetType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTupleTypeGetType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTupleTypeGetType(type, pos)
```


Returns the pos-th type in the tuple type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5968-L5972" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTupleTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirTupleTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirTupleTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTupleTypeGetTypeID()
```


Returns the typeID of an Tuple type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5930-L5934" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeAttrGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeAttrGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeAttrGet(type)
```


Creates a type attribute wrapping the given type in the same context as the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3976-L3980" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirTypeAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeAttrGetTypeID()
```


Returns the typeID of a Type attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3994-L3998" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeAttrGetValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeAttrGetValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeAttrGetValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeAttrGetValue(attr)
```


Returns the type stored in the given type attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L3985-L3989" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeDump(type)
```


Prints the type to the standard error stream.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2357-L2361" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTypeEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeEqual(t1, t2)
```


Checks if two types are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2337-L2341" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeFromLLVMIRTranslatorCreate(ctx)
```


Create an LLVM::TypeFromLLVMIRTranslator and transfer ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9772-L9776" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeFromLLVMIRTranslatorDestroy(translator)
```


Takes an LLVM::TypeFromLLVMIRTranslator owned by the caller and destroys it. It is the responsibility of the user to only pass an LLVM::TypeFromLLVMIRTranslator class.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9783-L9787" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorTranslateType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorTranslateType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeFromLLVMIRTranslatorTranslateType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeFromLLVMIRTranslatorTranslateType(translator, llvmType)
```


Translates the given LLVM IR type to the MLIR LLVM dialect.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9794-L9798" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeGetContext(type)
```


Gets the context that a type was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2301-L2305" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeGetDialect-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeGetDialect-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeGetDialect</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeGetDialect(type)
```


Gets the dialect a type belongs to.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2319-L2323" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeGetTypeID-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeGetTypeID-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeGetTypeID(type)
```


Gets the type ID of the type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2310-L2314" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDAllocatorAllocateTypeID-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIDAllocatorAllocateTypeID-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDAllocatorAllocateTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDAllocatorAllocateTypeID(allocator)
```


Allocates a type id that is valid for the lifetime of the allocator


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L232-L236" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDAllocatorCreate-Tuple{}' href='#Reactant.MLIR.API.mlirTypeIDAllocatorCreate-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDAllocatorCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDAllocatorCreate()
```


Creates a type id allocator for dynamic type id creation


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L214-L218" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDAllocatorDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIDAllocatorDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDAllocatorDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDAllocatorDestroy(allocator)
```


Deallocates the allocator and all allocated type ids


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L223-L227" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIDCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDCreate(ptr)
```


`ptr` must be 8 byte aligned and unique to a type valid for the duration of the returned type id&#39;s usage


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L178-L182" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTypeIDEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDEqual(typeID1, typeID2)
```


Checks if two type ids are equal.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L196-L200" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDHashValue-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIDHashValue-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDHashValue</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDHashValue(typeID)
```


Returns the hash value of the type id.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L205-L209" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIDIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIDIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIDIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIDIsNull(typeID)
```


Checks whether a type id is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L187-L191" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAAnyQuantizedType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAAnyQuantizedType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAAnyQuantizedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAAnyQuantizedType(type)
```


Returns `true` if the given type is an AnyQuantizedType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7661-L7665" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsABF16-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsABF16-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsABF16</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsABF16(type)
```


Checks whether the given type is a bf16 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5296-L5300" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsACalibratedQuantizedType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsACalibratedQuantizedType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsACalibratedQuantizedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsACalibratedQuantizedType(type)
```


Returns `true` if the given type is a CalibratedQuantizedType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7926-L7930" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAComplex-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAComplex-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAComplex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAComplex(type)
```


Checks whether the given type is a Complex type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5458-L5462" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAF16-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAF16-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAF16</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAF16(type)
```


Checks whether the given type is an f16 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5323-L5327" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAF32-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAF32-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAF32</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAF32(type)
```


Checks whether the given type is an f32 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5350-L5354" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAF64-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAF64-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAF64</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAF64(type)
```


Checks whether the given type is an f64 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5377-L5381" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat(type)
```


Checks whether the given type is a floating-point type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4972-L4976" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat4E2M1FN-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat4E2M1FN-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat4E2M1FN</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat4E2M1FN(type)
```


Checks whether the given type is an f4E2M1FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4999-L5003" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat6E2M3FN-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat6E2M3FN-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat6E2M3FN</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat6E2M3FN(type)
```


Checks whether the given type is an f6E2M3FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5026-L5030" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat6E3M2FN-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat6E3M2FN-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat6E3M2FN</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat6E3M2FN(type)
```


Checks whether the given type is an f6E3M2FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5053-L5057" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E3M4-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E3M4-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E3M4</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E3M4(type)
```


Checks whether the given type is an f8E3M4 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5242-L5246" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E4M3-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E4M3-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E4M3</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E4M3(type)
```


Checks whether the given type is an f8E4M3 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5107-L5111" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E4M3B11FNUZ-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E4M3B11FNUZ-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E4M3B11FNUZ</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E4M3B11FNUZ(type)
```


Checks whether the given type is an f8E4M3B11FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5215-L5219" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E4M3FN-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E4M3FN-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E4M3FN</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E4M3FN(type)
```


Checks whether the given type is an f8E4M3FN type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5134-L5138" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E4M3FNUZ-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E4M3FNUZ-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E4M3FNUZ</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E4M3FNUZ(type)
```


Checks whether the given type is an f8E4M3FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5188-L5192" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E5M2-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E5M2-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E5M2</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E5M2(type)
```


Checks whether the given type is an f8E5M2 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5080-L5084" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E5M2FNUZ-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E5M2FNUZ-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E5M2FNUZ</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E5M2FNUZ(type)
```


Checks whether the given type is an f8E5M2FNUZ type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5161-L5165" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFloat8E8M0FNU-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFloat8E8M0FNU-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFloat8E8M0FNU</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFloat8E8M0FNU(type)
```


Checks whether the given type is an f8E8M0FNU type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5269-L5273" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAFunction-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAFunction-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAFunction</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAFunction(type)
```


Checks whether the given type is a function type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5986-L5990" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAIndex-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAIndex-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAIndex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAIndex(type)
```


Checks whether the given type is an index type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4954-L4958" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAInteger-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAInteger-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAInteger</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAInteger(type)
```


Checks whether the given type is an integer type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4873-L4877" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsALLVMPointerType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsALLVMPointerType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsALLVMPointerType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsALLVMPointerType(type)
```


Returns `true` if the type is an LLVM dialect pointer type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6529-L6533" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsALLVMStructType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsALLVMStructType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsALLVMStructType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsALLVMStructType(type)
```


Returns `true` if the type is an LLVM dialect struct type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6615-L6619" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAMemRef-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAMemRef-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAMemRef</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAMemRef(type)
```


Checks whether the given type is a MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5777-L5781" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsANone-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsANone-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsANone</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsANone(type)
```


Checks whether the given type is a None type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5431-L5435" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAOpaque-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAOpaque-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAOpaque</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAOpaque(type)
```


Checks whether the given type is an opaque type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L6055-L6059" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAQuantizedType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAQuantizedType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAQuantizedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAQuantizedType(type)
```


Returns `true` if the given type is a quantization dialect type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7487-L7491" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsARankedTensor-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsARankedTensor-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsARankedTensor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsARankedTensor(type)
```


Checks whether the given type is a ranked tensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5686-L5690" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAShaped-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAShaped-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAShaped</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAShaped(type)
```


Checks whether the given type is a Shaped type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5485-L5489" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsATF32-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsATF32-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsATF32</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsATF32(type)
```


Checks whether the given type is an TF32 type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5404-L5408" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsATensor-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsATensor-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsATensor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsATensor(type)
```


Checks whether the given type is a Tensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5668-L5672" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsATuple-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsATuple-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsATuple</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsATuple(type)
```


Checks whether the given type is a tuple type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5939-L5943" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAUniformQuantizedPerAxisType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAUniformQuantizedPerAxisType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAUniformQuantizedPerAxisType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAUniformQuantizedPerAxisType(type)
```


Returns `true` if the given type is a UniformQuantizedPerAxisType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7742-L7746" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAUniformQuantizedSubChannelType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAUniformQuantizedSubChannelType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAUniformQuantizedSubChannelType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAUniformQuantizedSubChannelType(type)
```


Returns `true` if the given type is a UniformQuantizedSubChannel.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7831-L7835" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAUniformQuantizedType-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAUniformQuantizedType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAUniformQuantizedType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAUniformQuantizedType(type)
```


Returns `true` if the given type is a UniformQuantizedType.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7687-L7691" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAUnrankedMemRef-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAUnrankedMemRef-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAUnrankedMemRef</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAUnrankedMemRef(type)
```


Checks whether the given type is an UnrankedMemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5795-L5799" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAUnrankedTensor-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAUnrankedTensor-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAUnrankedTensor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAUnrankedTensor(type)
```


Checks whether the given type is an unranked tensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5704-L5708" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsAVector-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsAVector-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsAVector</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsAVector(type)
```


Checks whether the given type is a Vector type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5593-L5597" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeIsNull(type)
```


Checks whether a type is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2328-L2332" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeParseGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTypeParseGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeParseGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeParseGet(context, type)
```


Parses a type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2292-L2296" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypePrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirTypePrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypePrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypePrint(type, callback, userData)
```


Prints a location by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2346-L2350" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorCreate-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorCreate-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorCreate</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeToLLVMIRTranslatorCreate(ctx)
```


Create an LLVM::TypeToLLVMIRTranslator and transfer ownership to the caller.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9809-L9813" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorDestroy-Tuple{Any}' href='#Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorDestroy-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorDestroy</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeToLLVMIRTranslatorDestroy(translator)
```


Takes an LLVM::TypeToLLVMIRTranslator owned by the caller and destroys it. It is the responsibility of the user to only pass an LLVM::TypeToLLVMIRTranslator class.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9820-L9824" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorTranslateType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorTranslateType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirTypeToLLVMIRTranslatorTranslateType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirTypeToLLVMIRTranslatorTranslateType(translator, mlirType)
```


Translates the given MLIR LLVM dialect to the LLVM IR type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L9831-L9835" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGet-NTuple{9, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGet-NTuple{9, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)
```


Creates an instance of UniformQuantizedPerAxisType with the given parameters in the same context as `storageType` and returns it. `scales` and `zeroPoints` point to `nDims` number of elements. The instance is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7751-L7755" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetNumDims-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetNumDims-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetNumDims</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedPerAxisTypeGetNumDims(type)
```


Returns the number of axes in the given quantized per-axis type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7780-L7784" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)
```


Returns the index of the quantized dimension in the given quantized per-axis type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7811-L7815" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetScale-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetScale-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetScale</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedPerAxisTypeGetScale(type, pos)
```


Returns `pos`-th scale of the given quantized per-axis type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7789-L7793" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetZeroPoint-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetZeroPoint-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeGetZeroPoint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)
```


Returns `pos`-th zero point of the given quantized per-axis type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7800-L7804" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeIsFixedPoint-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeIsFixedPoint-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedPerAxisTypeIsFixedPoint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)
```


Returns `true` if the given uniform quantized per-axis type is fixed-point.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7822-L7826" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGet-NTuple{10, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGet-NTuple{10, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedSubChannelTypeGet(flags, storageType, expressedType, scalesAttr, zeroPointsAttr, blockSizeInfoLength, quantizedDimensions, blockSizes, storageTypeMin, storageTypeMax)
```


Creates a UniformQuantizedSubChannelType with the given parameters.

The type is owned by the context. `scalesAttr` and `zeroPointsAttr` must be DenseElementsAttrs. `quantizedDimensions` and `blockSizes` point to `blockSizeInfoLength` number of elements, describing respectively the quantization axis and corresponding block size.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7840-L7846" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetBlockSize-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetBlockSize-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetBlockSize</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedSubChannelTypeGetBlockSize(type, pos)
```


Returns the block size at the given position.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7895-L7899" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetNumBlockSizes-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetNumBlockSizes-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetNumBlockSizes</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type)
```


Returns the number of block sizes provided in type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7873-L7877" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetQuantizedDimension-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetQuantizedDimension-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetQuantizedDimension</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type, pos)
```


Returns the quantized dimension at the given position.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7884-L7888" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetScales-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetScales-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetScales</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedSubChannelTypeGetScales(type)
```


Returns the scales of the quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7906-L7910" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetZeroPoints-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetZeroPoints-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedSubChannelTypeGetZeroPoints</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedSubChannelTypeGetZeroPoints(type)
```


Returns the zero-points of the quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7915-L7919" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedTypeGet-NTuple{7, Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedTypeGet-NTuple{7, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)
```


Creates an instance of UniformQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7696-L7700" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedTypeGetScale-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedTypeGetScale-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedTypeGetScale</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedTypeGetScale(type)
```


Returns the scale of the given uniform quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7715-L7719" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedTypeGetZeroPoint-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedTypeGetZeroPoint-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedTypeGetZeroPoint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedTypeGetZeroPoint(type)
```


Returns the zero point of the given uniform quantized type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7724-L7728" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUniformQuantizedTypeIsFixedPoint-Tuple{Any}' href='#Reactant.MLIR.API.mlirUniformQuantizedTypeIsFixedPoint-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUniformQuantizedTypeIsFixedPoint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUniformQuantizedTypeIsFixedPoint(type)
```


Returns `true` if the given uniform quantized type is fixed-point.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L7733-L7737" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnitAttrGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirUnitAttrGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnitAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnitAttrGet(ctx)
```


Creates a unit attribute in the given context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4012-L4016" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnitAttrGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirUnitAttrGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirUnitAttrGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnitAttrGetTypeID()
```


Returns the typeID of a Unit attribute.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4021-L4025" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnmanagedDenseResourceElementsAttrGet-NTuple{8, Any}' href='#Reactant.MLIR.API.mlirUnmanagedDenseResourceElementsAttrGet-NTuple{8, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnmanagedDenseResourceElementsAttrGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnmanagedDenseResourceElementsAttrGet(shapedType, name, data, dataLength, dataAlignment, dataIsMutable, deleter, userData)
```


Unlike the typed accessors below, constructs the attribute with a raw data buffer and no type/alignment checking. Use a more strongly typed accessor if possible. If dataIsMutable is false, then an immutable AsmResourceBlob will be created and that passed data contents will be treated as const. If the deleter is non NULL, then it will be called when the data buffer can no longer be accessed (passing userData to it).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L4574-L4578" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedMemRefTypeGet-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirUnrankedMemRefTypeGet-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedMemRefTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedMemRefTypeGet(elementType, memorySpace)
```


Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5861-L5865" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedMemRefTypeGetChecked-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirUnrankedMemRefTypeGetChecked-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedMemRefTypeGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)
```


Same as &quot;[`mlirUnrankedMemRefTypeGet`](/api/mlirc#Reactant.MLIR.API.mlirUnrankedMemRefTypeGet-Tuple{Any,%20Any})&quot; but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5872-L5876" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedMemRefTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirUnrankedMemRefTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedMemRefTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedMemRefTypeGetTypeID()
```


Returns the typeID of an UnrankedMemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5786-L5790" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedMemrefGetMemorySpace-Tuple{Any}' href='#Reactant.MLIR.API.mlirUnrankedMemrefGetMemorySpace-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedMemrefGetMemorySpace</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedMemrefGetMemorySpace(type)
```


Returns the memory spcae of the given Unranked MemRef type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5921-L5925" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedTensorTypeGet-Tuple{Any}' href='#Reactant.MLIR.API.mlirUnrankedTensorTypeGet-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedTensorTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedTensorTypeGet(elementType)
```


Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5748-L5752" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedTensorTypeGetChecked-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirUnrankedTensorTypeGetChecked-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedTensorTypeGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedTensorTypeGetChecked(loc, elementType)
```


Same as &quot;[`mlirUnrankedTensorTypeGet`](/api/mlirc#Reactant.MLIR.API.mlirUnrankedTensorTypeGet-Tuple{Any})&quot; but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5757-L5761" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirUnrankedTensorTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirUnrankedTensorTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirUnrankedTensorTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirUnrankedTensorTypeGetTypeID()
```


Returns the typeID of an UnrankedTensor type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5695-L5699" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueDump-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueDump-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueDump</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueDump(value)
```


Prints the value to the standard error stream.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2163-L2167" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueEqual-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirValueEqual-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueEqual</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueEqual(value1, value2)
```


Returns 1 if two values are equal, 0 otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2073-L2077" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueGetContext-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueGetContext-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueGetContext</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueGetContext(v)
```


Gets the context that a value was created with.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2238-L2242" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueGetFirstUse-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueGetFirstUse-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueGetFirstUse</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueGetFirstUse(value)
```


Returns an op operand representing the first use of the value, or a null op operand if there are no uses.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2197-L2201" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueGetLocation-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueGetLocation-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueGetLocation</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueGetLocation(v)
```


Gets the location of the value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2229-L2233" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueGetType-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueGetType-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueGetType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueGetType(value)
```


Returns the type of the value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2145-L2149" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueIsABlockArgument-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueIsABlockArgument-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueIsABlockArgument</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueIsABlockArgument(value)
```


Returns 1 if the value is a block argument, 0 otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2082-L2086" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueIsAOpResult-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueIsAOpResult-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueIsAOpResult</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueIsAOpResult(value)
```


Returns 1 if the value is an operation result, 0 otherwise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2091-L2095" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueIsNull-Tuple{Any}' href='#Reactant.MLIR.API.mlirValueIsNull-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueIsNull</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueIsNull(value)
```


Returns whether the value is null.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2064-L2068" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValuePrint-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirValuePrint-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValuePrint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValuePrint(value, callback, userData)
```


Prints a value by sending chunks of the string representation and forwarding `userData to`callback`. Note that the callback may be called several times with consecutive chunks of the string.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2172-L2176" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValuePrintAsOperand-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirValuePrintAsOperand-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValuePrintAsOperand</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValuePrintAsOperand(value, state, callback, userData)
```


Prints a value as an operand (i.e., the ValueID).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2183-L2187" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueReplaceAllUsesExcept-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirValueReplaceAllUsesExcept-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueReplaceAllUsesExcept</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueReplaceAllUsesExcept(of, with, numExceptions, exceptions)
```


Replace all uses of &#39;of&#39; value with &#39;with&#39; value, updating anything in the IR that uses &#39;of&#39; to use &#39;with&#39; instead, except if the user is listed in &#39;exceptions&#39;. The &#39;exceptions&#39; parameter is an array of [`MlirOperation`](@ref) pointers with a length of &#39;numExceptions&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2215-L2219" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueReplaceAllUsesOfWith-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirValueReplaceAllUsesOfWith-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueReplaceAllUsesOfWith</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueReplaceAllUsesOfWith(of, with)
```


Replace all uses of &#39;of&#39; value with the &#39;with&#39; value, updating anything in the IR that uses &#39;of&#39; to use the other value instead. When this returns there are zero uses of &#39;of&#39;.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2206-L2210" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirValueSetType-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirValueSetType-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirValueSetType</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirValueSetType(value, type)
```


Set the type of the value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L2154-L2158" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeGet-Tuple{Any, Any, Any}' href='#Reactant.MLIR.API.mlirVectorTypeGet-Tuple{Any, Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeGet</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeGet(rank, shape, elementType)
```


Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5602-L5606" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeGetChecked-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirVectorTypeGetChecked-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeGetChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeGetChecked(loc, rank, shape, elementType)
```


Same as &quot;[`mlirVectorTypeGet`](/api/mlirc#Reactant.MLIR.API.mlirVectorTypeGet-Tuple{Any,%20Any,%20Any})&quot; but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5613-L5617" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeGetScalable-NTuple{4, Any}' href='#Reactant.MLIR.API.mlirVectorTypeGetScalable-NTuple{4, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeGetScalable</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeGetScalable(rank, shape, scalable, elementType)
```


Creates a scalable vector type with the shape identified by its rank and dimensions. A subset of dimensions may be marked as scalable via the corresponding flag list, which is expected to have as many entries as the rank of the vector. The vector is created in the same context as the element type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5624-L5628" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeGetScalableChecked-NTuple{5, Any}' href='#Reactant.MLIR.API.mlirVectorTypeGetScalableChecked-NTuple{5, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeGetScalableChecked</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeGetScalableChecked(loc, rank, shape, scalable, elementType)
```


Same as &quot;[`mlirVectorTypeGetScalable`](/api/mlirc#Reactant.MLIR.API.mlirVectorTypeGetScalable-NTuple{4,%20Any})&quot; but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5635-L5639" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeGetTypeID-Tuple{}' href='#Reactant.MLIR.API.mlirVectorTypeGetTypeID-Tuple{}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeGetTypeID</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeGetTypeID()
```


Returns the typeID of an Vector type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5584-L5588" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeIsDimScalable-Tuple{Any, Any}' href='#Reactant.MLIR.API.mlirVectorTypeIsDimScalable-Tuple{Any, Any}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeIsDimScalable</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeIsDimScalable(type, dim)
```


Checks whether the &quot;dim&quot;-th dimension of the given vector is scalable.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5659-L5663" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.API.mlirVectorTypeIsScalable-Tuple{Any}' href='#Reactant.MLIR.API.mlirVectorTypeIsScalable-Tuple{Any}'><span class="jlbinding">Reactant.MLIR.API.mlirVectorTypeIsScalable</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mlirVectorTypeIsScalable(type)
```


Checks whether the given vector type is scalable, i.e., has at least one scalable dimension.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/libMLIR_h.jl#L5650-L5654" target="_blank" rel="noreferrer">source</a></Badge>

</details>


# Other Functions {#Other-Functions}
