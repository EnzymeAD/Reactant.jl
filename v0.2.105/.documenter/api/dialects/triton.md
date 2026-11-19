


# Triton Dialect {#Triton-Dialect}

Refer to the [official documentation](https://triton-lang.org/main/dialects/TritonDialect.html) for more details.
<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.assert-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.assert-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.assert</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`assert`

`tt.assert` takes a condition tensor and a message string. If the condition is false, the message is printed, and the program is aborted.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L229-L234" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.atomic_cas-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.atomic_cas-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.atomic_cas</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`atomic_cas`

compare cmp with data old at location ptr,

if old == cmp, store val to ptr,

else store old to ptr,

return old


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L254-L264" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.atomic_rmw' href='#Reactant.MLIR.Dialects.tt.atomic_rmw'><span class="jlbinding">Reactant.MLIR.Dialects.tt.atomic_rmw</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`atomic_rmw`

load data at ptr, do rmw_op with val, and store result to ptr.

return old value at ptr


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L286-L292" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.broadcast-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.broadcast-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.broadcast</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`broadcast`

For a given tensor, broadcast changes one or more dimensions with size 1 to a new size, e.g. tensor&lt;1x32x1xf32&gt; -&gt; tensor&lt;2x32x4xf32&gt;.  You cannot change the size of a non-1 dimension.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L345-L351" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.call-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.call-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.call</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`call`

The `tt.call` operation represents a direct call to a function that is within the same symbol scope as the call. The operands and result types of the call must match the specified function type. The callee is encoded as a symbol reference attribute named &quot;callee&quot;.

**Example**

```mlir
%2 = tt.call @my_add(%0, %1) : (f32, f32) -> f32
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L16-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.clampf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.clampf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.clampf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`clampf`

Clamp operation for floating point types.

The operation takes three arguments: x, min, and max. It returns a tensor of the same shape as x with its values clamped to the range [min, max].


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L390-L396" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.descriptor_gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.descriptor_gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.descriptor_gather</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`descriptor_gather`

The `tt.descriptor_gather` op will be lowered to NVIDIA TMA gather operations on targets that support it.

`desc_ptr` is a pointer to the TMA descriptor allocated in global memory. The descriptor block must have 1 row and the indices must be a 1D tensor. Accordingly, the result is a 2D tensor multiple rows.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L424-L433" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.descriptor_load-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.descriptor_load-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.descriptor_load</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`descriptor_load`

This operation will be lowered to Nvidia TMA load operation on targets supporting it. `desc` is a tensor descriptor object. The destination tensor type and shape must match the descriptor otherwise the result is undefined.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L455-L461" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.descriptor_reduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.descriptor_reduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.descriptor_reduce</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`descriptor_reduce`

This operation will be lowered to Nvidia TMA store operation on targets supporting it. `desc` is a tensor descriptor object. The shape and types of `src` must match the descriptor otherwise the result is undefined.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L490-L496" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.descriptor_scatter-NTuple{4, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.descriptor_scatter-NTuple{4, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.descriptor_scatter</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`descriptor_scatter`

The `tt.descriptor_scatter` op will be lowered to NVIDIA TMA scatter operations on targets that support it.

`desc_ptr` is a pointer to the TMA descriptor allocated in global memory. The descriptor block must have 1 row and the indices must be a 1D tensor. Accordingly, the result is a 2D tensor multiple rows.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L518-L527" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.descriptor_store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.descriptor_store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.descriptor_store</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`descriptor_store`

This operation will be lowered to Nvidia TMA store operation on targets supporting it. `desc` is a tensor descriptor object. The shape and types of `src` must match the descriptor otherwise the result is undefined.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L549-L555" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.dot</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`dot`

$$d = matrix_multiply($$

a, b) + c. inputPrecision describes how to exercise the TC when the inputs are f32. It can be one of: tf32, tf32x3, ieee. tf32: use TC with tf32 ops. tf32x3: implement the 3xTF32 trick. For more info see the pass in F32DotTC.cpp ieee: don&#39;t use TC, implement dot in software. If the GPU does not have Tensor cores or the inputs are not f32, this flag is ignored.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L577-L586" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.dot_scaled' href='#Reactant.MLIR.Dialects.tt.dot_scaled'><span class="jlbinding">Reactant.MLIR.Dialects.tt.dot_scaled</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



`dot_scaled`

$$d = matrix_multiply(scale($$

a, $a_scale), scale($b, b_scale)) + c. Where scale(x, s) is a function that applies the scale per block following microscaling spec.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L619-L624" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.elementwise_inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.elementwise_inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.elementwise_inline_asm</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`elementwise_inline_asm`

Runs an inline asm block to generate one or more tensors.

The asm block is given `packed_element` elements at a time.  Exactly which elems it receives is unspecified.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L677-L684" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.extern_elementwise-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.extern_elementwise-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.extern_elementwise</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`extern_elementwise`

call an external function $symbol implemented in $libpath/$libname with $args return $libpath/$libname:$symbol($args...)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L812-L817" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.fp_to_fp-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.fp_to_fp-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.fp_to_fp</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`fp_to_fp`

Floating point casting for custom types (F8), and non-default rounding modes.

F8 &lt;-&gt; FP16, BF16, FP32, FP64


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L850-L856" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.func-Tuple{}' href='#Reactant.MLIR.Dialects.tt.func-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.func</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`func`

Operations within the function cannot implicitly capture values defined outside of the function, i.e. Functions are `IsolatedFromAbove`. All external references must use function arguments or attributes that establish a symbolic connection (e.g. symbols referenced by name via a string attribute like SymbolRefAttr). An external function declaration (used when referring to a function declared in some other module) has no body. While the MLIR textual form provides a nice inline syntax for function arguments, they are internally represented as “block arguments” to the first block in the region.

Only dialect attribute names may be specified in the attribute dictionaries for function arguments, results, or the function itself.

**Example**

```mlir
// External function definitions.
tt.func @abort()
tt.func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// A function that returns its argument twice:
tt.func @count(%x: i64) -> (i64, i64)
  attributes {fruit: "banana"} {
  return %x, %x: i64, i64
}

// A function with an argument attribute
tt.func @example_fn_arg(%x: i32 {swift.self = unit})

// A function with a result attribute
tt.func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// A function with an attribute
tt.func @example_fn_attr() attributes {dialectName.attrName = false}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L58-L96" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.gather</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`gather`

Gather elements from the input tensor using the indices tensor along a single specified axis. The output tensor has the same shape as the indices tensor. The input and indices tensors must have the same number of dimension, and each dimension of the indices tensor that is not the gather dimension cannot be greater than the corresponding dimension in the input tensor.

The `efficient_layout` attribute is set when the compiler has determined an optimized layout for the operation, indicating that it should not be changed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L877-L890" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.histogram-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.histogram-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.histogram</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`histogram`

Return the histogram of the input tensor. The number of bins is equal to the dimension of the output tensor. Each bins has a width of 1 and bins start at 0.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L962-L968" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.join-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.join-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.join</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`join`

For example, if the two input tensors are 4x8xf32, returns a tensor of shape 4x8x2xf32.

Because Triton tensors always have a power-of-two number of elements, the two input tensors must have the same shape.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1007-L1015" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.make_range-Tuple{}' href='#Reactant.MLIR.Dialects.tt.make_range-Tuple{}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.make_range</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`make_range`

Returns an 1D int32 tensor.

Values span from start to $end (exclusive), with step = 1


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1084-L1090" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.make_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.make_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.make_tensor_descriptor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`make_tensor_descriptor`

`tt.make_tensor_descriptor` takes both meta information of the parent tensor and the block size, and returns a descriptor object which can be used to load/store from the tensor in global memory.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1110-L1115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.make_tensor_ptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.make_tensor_ptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.make_tensor_ptr</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`make_tensor_ptr`

`tt.make_tensor_ptr` takes both meta information of the parent tensor and the block tensor, then it returns a pointer to the block tensor, e.g. returns a type of `tt.ptr<tensor<8x8xf16>>`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1141-L1146" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.mulhiui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.mulhiui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.mulhiui</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`mulhiui`

Most significant N bits of the 2N-bit product of two integers.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1174-L1178" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.precise_divf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.precise_divf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.precise_divf</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`precise_divf`

Precise div for floating point types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1201-L1205" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.precise_sqrt-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.precise_sqrt-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.precise_sqrt</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`precise_sqrt`

Precise sqrt for floating point types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1228-L1232" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.print-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.print-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.print</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`print`

`tt.print` takes a literal string prefix and an arbitrary number of scalar or tensor arguments that should be printed. format are generated automatically from the arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1253-L1258" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.reinterpret_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.reinterpret_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.reinterpret_tensor_descriptor</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reinterpret_tensor_descriptor`

This Op exists to help the transition from untyped raw TMA objects to typed Tensor descriptor objects. Ideally, we can remove this once the APIs are fully fleshed out.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L130-L135" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.reshape-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.reshape-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.reshape</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`reshape`

reinterpret a tensor to a different shape.

If allow_reorder is set the compiler is free to change the order of elements to generate more efficient code.

If efficient_layout is set, this is a hint that the destination layout should be kept for performance reason. The compiler is still free to change it for better performance.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1345-L1355" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}' href='#Reactant.MLIR.Dialects.tt.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.return_</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`return_`

The `tt.return` operation represents a return operation within a function. The operation takes variable number of operands and produces no results. The operand number and types must match the signature of the function that contains the operation.

**Example**

```mlir
tt.func @foo() : (i32, f8) {
  ...
  tt.return %0, %1 : i32, f8
}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L155-L171" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.split-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.split-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.split</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`split`

The input must be a tensor whose last dimension has size 2.  Returns two tensors, src[..., 0] and src[..., 1].

For example, if the input shape is 4x8x2xf32, returns two tensors of shape 4x8xf32.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1451-L1459" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Reactant.MLIR.Dialects.tt.trans-Tuple{Reactant.MLIR.IR.Value}' href='#Reactant.MLIR.Dialects.tt.trans-Tuple{Reactant.MLIR.IR.Value}'><span class="jlbinding">Reactant.MLIR.Dialects.tt.trans</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



`trans`

For example, given a tensor x with shape [1,2,4], transpose(x) with order=[2,0,1] rearranges the tensor to have shape [4,1,2].

Although this op is called &quot;trans&quot;, it implements both tl.trans() and tl.permute().  (&quot;permute&quot; might be a better name, but it&#39;s called &quot;trans&quot; because originally it only supported 2D tensors.)

**Implementation note on encodings:**

In the TritonGPU dialect (and probably others), an encoding is chosen for this op&#39;s output so it&#39;s a nop from the perspective of code generation.

For example, suppose tensor x has an encoding such that GPU thread [i,j,k] has a register containing element [i,j,k] of the tensor.  Now we transpose x with order [2,1,0], i.e. we reverse the order of its dimensions.  In TritonGPU, we will choose a layout for the output of the transpose so that GPU thread [i,j,k] has element [k,j,i] of transpose(x).  But this is the same element it had before!  All we&#39;ve done is &quot;rename&quot; the element that thread [i,j,k] has.

The &quot;real&quot; transpose – i.e. moving data between GPU threads – occurs in convertLayout ops that appear before and/or after the operation.

We do this so that you can chain multiple data-movement ops (e.g. transpose+reshape+concat) without going to shared memory after each one.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/EnzymeAD/Reactant.jl/blob/c1a1e1dc3b6985fead24f05e7d04139ed0a37df0/src/mlir/Dialects/Triton.jl#L1518-L1546" target="_blank" rel="noreferrer">source</a></Badge>

</details>

