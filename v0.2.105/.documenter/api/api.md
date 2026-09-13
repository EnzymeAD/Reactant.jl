


# Core Reactant API {#Core-Reactant-API}

## Compile API {#Compile-API}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.@compile' href='#Reactant.Compiler.@compile'><span class="jlbinding">Reactant.Compiler.@compile</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@compile [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L2017-L2019" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.@jit' href='#Reactant.Compiler.@jit'><span class="jlbinding">Reactant.Compiler.@jit</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@jit [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)
```


Run @compile f(args..) then immediately execute it


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L2041-L2045" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## ReactantCore API {#ReactantCore-API}
<details class='jldocstring custom-block' >
<summary><a id='ReactantCore.within_compile' href='#ReactantCore.within_compile'><span class="jlbinding">ReactantCore.within_compile</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
within_compile()
```


Returns true if this function is executed in a Reactant compilation context, otherwise false.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/lib/ReactantCore/src/ReactantCore.jl#L35-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='ReactantCore.@trace' href='#ReactantCore.@trace'><span class="jlbinding">ReactantCore.@trace</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@trace <expr>
```


Converts certain expressions like control flow into a Reactant friendly form. Importantly, if no traced value is found inside the expression, then there is no overhead.

**Currently Supported**
- `if` conditions (with `elseif` and other niceties) (`@trace if ...`)
  
- `if` statements with a preceeding assignment (`@trace a = if ...`) (note the positioning of the macro needs to be before the assignment and not before the `if`)
  
- `for` statements with a single induction variable iterating over a syntactic `StepRange` of integers.
  

**Special Considerations**
- Apply `@trace` only at the outermost `if`. Nested `if` statements will be automatically expanded into the correct form.
  

**Extended Help**

**Caveats (Deviations from Core Julia Semantics)**

**New variables introduced**

```julia
@trace if x > 0
    y = x + 1
    p = 1
else
    y = x - 1
end
```


In the outer scope `p` is not defined if `x ≤ 0`. However, for the traced version, it is defined and set to a dummy value.

**Short Circuiting Operations**

```julia
@trace if x > 0 && z > 0
    y = x + 1
else
    y = x - 1
end
```


`&&` and `||` are short circuiting operations. In the traced version, we replace them with `&` and `|` respectively.

**Type-Unstable Branches**

```julia
@trace if x > 0
    y = 1.0f0
else
    y = 1.0
end
```


This will not compile since `y` is a `Float32` in one branch and a `Float64` in the other. You need to ensure that all branches have the same type.

Another example is the following for loop which changes the type of `x` between iterations.

```julia
x = ... # ConcreteRArray{Int64, 1}
for i in 1f0:0.5f0:10f0
    x = x .+ i # ConcreteRArray{Float32, 1}
end
```


**Certain Symbols are Reserved**

Symbols like [:(:), :nothing, :missing, :Inf, :Inf16, :Inf32, :Inf64, :Base, :Core] are not allowed as variables in `@trace` expressions. While certain cases might work but these are not guaranteed to work. For example, the following will not work:

```julia
function fn(x)
    nothing = sum(x)
    @trace if nothing > 0
        y = 1.0
    else
        y = 2.0
    end
    return y, nothing
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/lib/ReactantCore/src/ReactantCore.jl#L43-L130" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Inspect Generated HLO {#Inspect-Generated-HLO}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.@code_hlo' href='#Reactant.Compiler.@code_hlo'><span class="jlbinding">Reactant.Compiler.@code_hlo</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@code_hlo [optimize = ...] [no_nan = <true/false>] f(args...)
```


See also [`@code_xla`](/api/api#Reactant.Compiler.@code_xla), [`@code_mhlo`](/api/api#Reactant.Compiler.@code_mhlo).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L1909-L1913" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.@code_mhlo' href='#Reactant.Compiler.@code_mhlo'><span class="jlbinding">Reactant.Compiler.@code_mhlo</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@code_mhlo [optimize = ...] [no_nan = <true/false>] f(args...)
```


Similar to `@code_hlo`, but prints the module after running the XLA compiler.

See also [`@code_xla`](/api/api#Reactant.Compiler.@code_xla), [`@code_hlo`](/api/api#Reactant.Compiler.@code_hlo).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L1943-L1949" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Compiler.@code_xla' href='#Reactant.Compiler.@code_xla'><span class="jlbinding">Reactant.Compiler.@code_xla</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@code_xla [optimize = ...] [no_nan = <true/false>] f(args...)
```


Similar to `@code_hlo`, but prints the HLO module.

See also [`@code_mhlo`](/api/api#Reactant.Compiler.@code_mhlo), [`@code_hlo`](/api/api#Reactant.Compiler.@code_hlo).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Compiler.jl#L1979-L1985" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Profile XLA {#Profile-XLA}

Reactant can hook into XLA&#39;s profiler to generate compilation and execution traces. See the [profiling tutorial](/tutorials/profiling#profiling) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.Profiler.with_profiler' href='#Reactant.Profiler.with_profiler'><span class="jlbinding">Reactant.Profiler.with_profiler</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
with_profiler(f, trace_output_dir::String; trace_device=true, trace_host=true, create_perfetto_link=false)
```


Runs the provided function under a profiler for XLA (similar to [JAX&#39;s profiler](https://jax.readthedocs.io/en/latest/profiling.html)). The traces will be exported in the provided folder and can be seen using tools like [perfetto.dev](https://ui.perfetto.dev). It will return the return values from the function. The `create_perfetto_link` parameter can be used to automatically generate a perfetto url to visualize the trace.

```julia
compiled_func = with_profiler("./traces") do
    @compile sync=true myfunc(x, y, z)
end

with_profiler("./traces/") do
    compiled_func(x, y, z)
end
```


::: tip Note

When profiling compiled functions make sure to [`Reactant.Compiler.@compile`](/api/api#Reactant.Compiler.@compile) with the `sync=true` option so that the compiled execution is captured by the profiler.

:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Profiler.jl#L6-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Profiler.annotate' href='#Reactant.Profiler.annotate'><span class="jlbinding">Reactant.Profiler.annotate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
annotate(f, name, [level=TRACE_ME_LEVEL_CRITICAL])
```


Generate an annotation in the current trace.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Profiler.jl#L70-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.Profiler.@annotate' href='#Reactant.Profiler.@annotate'><span class="jlbinding">Reactant.Profiler.@annotate</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@annotate [name] function foo(a, b, c)
    ...
end
```


The created function will generate an annotation in the captured XLA profiles.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Profiler.jl#L86-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Devices {#Devices}
<details class='jldocstring custom-block' >
<summary><a id='Reactant.devices' href='#Reactant.devices'><span class="jlbinding">Reactant.devices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
devices(backend::String)
devices(backend::XLA.AbstractClient = XLA.default_backend())
```


Return a list of devices available for the given client.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Devices.jl#L1-L6" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.addressable_devices' href='#Reactant.addressable_devices'><span class="jlbinding">Reactant.addressable_devices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
addressable_devices(backend::String)
addressable_devices(backend::XLA.AbstractClient = XLA.default_backend())
```


Return a list of addressable devices available for the given client.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/Devices.jl#L11-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Internal utils {#Internal-utils}
<details class='jldocstring custom-block' >
<summary><a id='ReactantCore.materialize_traced_array' href='#ReactantCore.materialize_traced_array'><span class="jlbinding">ReactantCore.materialize_traced_array</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
materialize_traced_array(AbstractArray{<:TracedRNumber})::TracedRArray
```


Given an AbstractArray{TracedRNumber}, return or create an equivalent TracedRArray.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/lib/ReactantCore/src/ReactantCore.jl#L532-L537" target="_blank" rel="noreferrer">source</a></Badge>

</details>

