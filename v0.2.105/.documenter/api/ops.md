


# `Reactant.Ops` API {#Reactant.Ops-API}

`Reactant.Ops` module provides a high-level API to construct MLIR operations without having to directly interact with the different dialects.

Currently we haven&#39;t documented all the functions in `Reactant.Ops`.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.gather_getindex-Union{Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}}} where {T, N}' href='#Reactant.Ops.gather_getindex-Union{Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}}} where {T, N}'><span class="jlbinding">Reactant.Ops.gather_getindex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
gather_getindex(src, gather_indices)
```


Uses [`MLIR.Dialects.stablehlo.gather`](/api/dialects/stablehlo#Reactant.MLIR.Dialects.stablehlo.gather-Tuple{Reactant.MLIR.IR.Value,%20Reactant.MLIR.IR.Value}) to get the values of `src` at the indices specified by `gather_indices`. If the indices are contiguous it is recommended to directly use [`MLIR.Dialects.stablehlo.dynamic_slice`](/api/dialects/stablehlo#Reactant.MLIR.Dialects.stablehlo.dynamic_slice-Tuple{Reactant.MLIR.IR.Value,%20Vector{Reactant.MLIR.IR.Value}}) instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L1791-L1797" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.hlo_call-Tuple{Any, Vararg{Any}}' href='#Reactant.Ops.hlo_call-Tuple{Any, Vararg{Any}}'><span class="jlbinding">Reactant.Ops.hlo_call</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
Ops.hlo_call(mlir_code::String, args::Vararg{AnyTracedRArray}...; func_name::String="main") -> NTuple{N, AnyTracedRArray}
```


Given a MLIR module given as a string, calls the function identified by the `func_name` keyword parameter (default &quot;main&quot;) with the provided arguments and return a tuple for each result of the call.

```julia
julia> Reactant.@jit(
          Ops.hlo_call(
              """
              module {
                func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
                  %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
                  return %0 : tensor<3xf32>
                }
              }
              """,
              Reactant.to_rarray(Float32[1, 2, 3]),
              Reactant.to_rarray(Float32[1, 2, 3]),
          )
       )
(ConcretePJRTArray{Float32, 1}(Float32[2.0, 4.0, 6.0]),)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L1582-L1605" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.mesh-Tuple{Reactant.Sharding.Mesh}' href='#Reactant.Ops.mesh-Tuple{Reactant.Sharding.Mesh}'><span class="jlbinding">Reactant.Ops.mesh</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mesh(
    mesh::Reactant.Sharding.Mesh; mod::MLIR.IR.Module=MLIR.IR.mmodule(),
    sym_name::String="mesh",
    location=mlir_stacktrace("mesh", @__FILE__, @__LINE__)
)
mesh(
    mesh_axes::Vector{<:Pair{<:Union{String,Symbol},Int64}},
    logical_device_ids::Vector{Int64};
    sym_name::String="mesh",
    mod::MLIR.IR.Module=MLIR.IR.mmodule(),
    location=mlir_stacktrace("mesh", @__FILE__, @__LINE__)
)
```


Produces a [`Reactant.MLIR.Dialects.sdy.mesh`](/api/dialects/shardy#Reactant.MLIR.Dialects.sdy.mesh-Tuple{}) operation with the given `mesh` and `logical_device_ids`.

Based on the provided `sym_name``, we generate a unique name for the mesh in the module's`SymbolTable`. Note that users shouldn't use this sym_name directly, instead they should use the returned`sym_name` to refer to the mesh in the module.

::: warning Warning

The `logical_device_ids` argument are the logical device ids, not the physical device ids. For example, if the physical device ids are `[2, 4, 123, 293]`, the corresponding logical device ids are `[0, 1, 2, 3]`.

:::

**Returned Value**

We return a NamedTuple with the following fields:
- `sym_name`: The unique name of the mesh in the module&#39;s `SymbolTable`.
  
- `mesh_attr`: `sdy::mlir::MeshAttr` representing the mesh.
  
- `mesh_op`: The `sdy.mesh` operation.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L2467-L2501" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.randexp-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T' href='#Reactant.Ops.randexp-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T'><span class="jlbinding">Reactant.Ops.randexp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
randexp(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
)
```


Generate a random array of type `T` with the given shape and seed from an exponential distribution with rate 1. Returns a NamedTuple with the following fields:
- `output_state`: The state of the random number generator after the operation.
  
- `output`: The generated array.
  

**Arguments**
- `T`: The type of the generated array.
  
- `seed`: The seed for the random number generator.
  
- `shape`: The shape of the generated array.
  
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L1440-L1462" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.randn-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T' href='#Reactant.Ops.randn-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T'><span class="jlbinding">Reactant.Ops.randn</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
randn(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
)
```


Generate a random array of type `T` with the given shape and seed from a standard normal distribution of mean 0 and standard deviation 1. Returns a NamedTuple with the following fields:
- `output_state`: The state of the random number generator after the operation.
  
- `output`: The generated array.
  

**Arguments**
- `T`: The type of the generated array.
  
- `seed`: The seed for the random number generator.
  
- `shape`: The shape of the generated array.
  
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L1397-L1420" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.reduce-Union{Tuple{T}, Tuple{Reactant.TracedRArray{T}, Reactant.TracedRNumber{T}, Vector{Int64}, Function}, Tuple{Reactant.TracedRArray{T}, Reactant.TracedRNumber{T}, Vector{Int64}, Function, Any}} where T' href='#Reactant.Ops.reduce-Union{Tuple{T}, Tuple{Reactant.TracedRArray{T}, Reactant.TracedRNumber{T}, Vector{Int64}, Function}, Tuple{Reactant.TracedRArray{T}, Reactant.TracedRNumber{T}, Vector{Int64}, Function, Any}} where T'><span class="jlbinding">Reactant.Ops.reduce</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
reduce(
    x::TracedRArray{T},
    init_values::TracedRNumber{T},
    dimensions::Vector{Int},
    fn::Function,
    location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
)
```


Applies a reduction function `fn` along the specified `dimensions` of input `x`, starting from `init_values`.

**Arguments**
- `x`: The input array.
  
- `init_values`: The initial value.
  
- `dimensions`: The dimensions to reduce along.
  
- `fn`: A binary operator.
  

::: warning Warning

This reduction operation follows StableHLO semantics. The key difference between this operation and Julia&#39;s built-in `reduce` is explained below:
- The function `fn` and the initial value `init_values` must form a **monoid**, meaning:
  - `fn` must be an **associative** binary operation.
    
  - `init_values` must be the **identity element** associated with `fn`.
    
  
- This constraint ensures consistent results across all implementations.
  

If `init_values` is not the identity element of `fn`, the results may vary between CPU and GPU executions. For example:

```julia
A = [1 3; 2 4;;; 5 7; 6 8;;; 9 11; 10 12]
init_values = 2
dimensions = [1, 3]
```

- **CPU version &amp; Julia&#39;s `reduce`**:
  - Reduce along dimension 1 → `[(15) (21); (18) (24)]`
    
  - Reduce along dimension 3 → `[(33 + 2)  (45 + 2)]` → `[35 47]`
    
  
- **GPU version**:
  - Reduce along dimension 1 → `[(15 + 2) (21 + 2); (18 + 2) (24 + 2)]`
    
  - Reduce along dimension 3 → `[37 49]`
    
  

:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L2620-L2661" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.rng_bit_generator-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T<:Integer' href='#Reactant.Ops.rng_bit_generator-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T<:Integer'><span class="jlbinding">Reactant.Ops.rng_bit_generator</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
rng_bit_generator(
    ::Type{T},
    seed::TracedRArray{UInt64,1},
    shape;
    algorithm::String="DEFAULT",
    location=mlir_stacktrace("rand", @__FILE__, @__LINE__),
)
```


Generate a random array of type `T` with the given shape and seed from a uniform random distribution between 0 and 1 (for floating point types). Returns a NamedTuple with the following fields:
- `output_state`: The state of the random number generator after the operation.
  
- `output`: The generated array.
  

**Arguments**
- `T`: The type of the generated array.
  
- `seed`: The seed for the random number generator.
  
- `shape`: The shape of the generated array.
  
- `algorithm`: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L1320-L1343" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.scatter_setindex-Union{Tuple{T2}, Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}, Reactant.TracedRArray{T2, 1}}} where {T, N, T2}' href='#Reactant.Ops.scatter_setindex-Union{Tuple{T2}, Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}, Reactant.TracedRArray{T2, 1}}} where {T, N, T2}'><span class="jlbinding">Reactant.Ops.scatter_setindex</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
scatter_setindex(dest, scatter_indices, updates)
```


Uses [`MLIR.Dialects.stablehlo.scatter`](/api/dialects/stablehlo#Reactant.MLIR.Dialects.stablehlo.scatter-Tuple{Vector{Reactant.MLIR.IR.Value},%20Reactant.MLIR.IR.Value,%20Vector{Reactant.MLIR.IR.Value}}) to set the values of `dest` at the indices specified by `scatter_indices` to the values in `updates`. If the indices are contiguous it is recommended to directly use [`MLIR.Dialects.stablehlo.dynamic_update_slice`](/api/dialects/stablehlo#Reactant.MLIR.Dialects.stablehlo.dynamic_update_slice-Tuple{Reactant.MLIR.IR.Value,%20Reactant.MLIR.IR.Value,%20Vector{Reactant.MLIR.IR.Value}}) instead.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L1700-L1707" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Ops.sharding_constraint-Tuple{Union{Number, AbstractArray}, Reactant.Sharding.AbstractSharding}' href='#Reactant.Ops.sharding_constraint-Tuple{Union{Number, AbstractArray}, Reactant.Sharding.AbstractSharding}'><span class="jlbinding">Reactant.Ops.sharding_constraint</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
sharding_constraint(
    input::Union{TracedRArray,TracedRNumber},
    sharding::Reactant.Sharding.AbstractSharding;
    location=mlir_stacktrace("sharding_constraint", @__FILE__, @__LINE__)
)
```


Produces a [`Reactant.MLIR.Dialects.sdy.sharding_constraint`](/api/dialects/shardy#Reactant.MLIR.Dialects.sdy.sharding_constraint-Tuple{Reactant.MLIR.IR.Value}) operation with the given `input` and `sharding`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Ops.jl#L2579-L2588" target="_blank" rel="noreferrer">source</a></Badge>

</details>

