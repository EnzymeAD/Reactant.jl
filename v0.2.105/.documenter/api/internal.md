


# Internal API {#Internal-API}

::: danger Private

These functions are not part of the public API and are subject to change at any time.

:::
<details class='jldocstring custom-block' >
<summary><a id='Reactant.REDUB_ARGUMENTS_NAME' href='#Reactant.REDUB_ARGUMENTS_NAME'><span class="jlbinding">Reactant.REDUB_ARGUMENTS_NAME</span></a> <Badge type="info" class="jlObjectType jlConstant" text="Constant" /></summary>



```julia
Reactant.REDUB_ARGUMENTS_NAME
```


The variable name bound to `call_with_reactant`&#39;s tuple of arguments in its `@generated` method definition.

This binding can be used to manually reference/destructure `call_with_reactants` arguments

This is required because user arguments could have a name which clashes with whatever name we choose for our argument. Thus we gensym to create it.

This originates from https://github.com/JuliaLabs/Cassette.jl/blob/c29b237c1ec0deda3a1037ec519eebe216952bfe/src/overdub.jl#L154


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/utils.jl#L17-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.codegen_unflatten!' href='#Reactant.Compiler.codegen_unflatten!'><span class="jlbinding">Reactant.Compiler.codegen_unflatten!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
codegen_unflatten!
```


Generate Julia code to wrap the XLA buffers back into the output result datatypes. The name is due to its similarity to the `unflatten` function in `jax.tree_util.register_pytree_node`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L2477-L2482" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.codegen_flatten!' href='#Reactant.Compiler.codegen_flatten!'><span class="jlbinding">Reactant.Compiler.codegen_flatten!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
codegen_flatten!
```


Generate Julia code to extract the XLA buffers from input arguments. The name is due to its similarity to the `flatten` function in `jax.tree_util.register_pytree_node`.

**Arguments**
- `linear_args`: A list of arguments to be flattened.
  

**Returns**
- `flatten_names`: A list of `Symbol`s representing the names of the flattened arguments.
  
- `flatten_code`: A list of `Expr`s to extract the XLA buffers from the input arguments.
  

**Note**

The _linearized arguments_ do not directly refer to the  are the arguments that have been flattened into a single list.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L2163-L2181" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.codegen_xla_call' href='#Reactant.Compiler.codegen_xla_call'><span class="jlbinding">Reactant.Compiler.codegen_xla_call</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
codegen_xla_call
```


Generate Julia code to call the XLA executable.

**Arguments**
- `flatten_names`: A list of `Symbol`s representing the names of the flattened linear arguments.
  
- `nresults`: The number of results to expect.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L2692-L2701" target="_blank" rel="noreferrer">source</a></Badge>

</details>

