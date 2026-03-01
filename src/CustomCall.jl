"""
    CustomCall

Module providing a framework for calling Julia functions from XLA during execution
via XLA FFI custom calls. Currently supports CPU (Host) callbacks only.

# Overview

The framework registers a generic XLA FFI handler (`"reactant_julia_callback"`) on the
Host platform. At trace time, `custom_call` emits a `stablehlo.custom_call` op targeting
this handler. The function pointer to the Julia callback is passed as a backend config
attribute.

At execution time, the callback receives fully constructed Julia `Array` views over the
XLA-allocated buffers (zero-copy). The function signature is
`f(output_arrays..., input_arrays...)` — N output arrays followed by M input arrays.

# Usage

```julia
using Reactant

# Define a callback: 1 output array, 2 input arrays
function my_add!(out, x, y)
    out .= x .+ y
    return nothing
end

# Use in a compiled function
function f(x, y)
    return Reactant.CustomCall.custom_call(
        my_add!,
        ((Float32, (4,)),),  # one output: Float32 tensor of shape (4,)
        x, y;
    )
end
```
"""
module CustomCall

using ..Reactant: Reactant, TracedRArray, TracedRNumber, RArray, RNumber, unwrapped_eltype
using ..MLIR: MLIR
using ..MLIR.Dialects: stablehlo
using ..Ops: Ops, mlir_type, mlir_stacktrace

# ============================================================================
# XLA FFI element type encoding (matches xla::PrimitiveType / xla::ffi::DataType)
# ============================================================================

"""Mapping from Julia types to XLA PrimitiveType integer codes."""
const JULIA_TO_XLA_TYPE = Dict{Type,Int32}(
    Bool => Int32(1),   # PRED
    Int8 => Int32(2),   # S8
    Int16 => Int32(3),   # S16
    Int32 => Int32(4),   # S32
    Int64 => Int32(5),   # S64
    UInt8 => Int32(6),   # U8
    UInt16 => Int32(7),   # U16
    UInt32 => Int32(8),   # U32
    UInt64 => Int32(9),   # U64
    Float16 => Int32(10),  # F16
    Float32 => Int32(11),  # F32
    Float64 => Int32(12),  # F64
    ComplexF32 => Int32(15),  # C64
    # BFloat16 => Int32(16), # BF16  (add when available)
    ComplexF64 => Int32(18),  # C128
)

"""Mapping from XLA PrimitiveType integer codes back to Julia types."""
const XLA_TYPE_TO_JULIA = Dict{Int32,Type}(v => k for (k, v) in JULIA_TO_XLA_TYPE)

"""
    xla_element_type(::Type{T}) -> Int32

Return the XLA `PrimitiveType` integer code for a Julia element type.
"""
function xla_element_type(::Type{T}) where {T}
    return get(JULIA_TO_XLA_TYPE, T) do
        error("Unsupported element type for XLA FFI custom call: $T")
    end
end

"""
    julia_element_type(code::Int32) -> Type

Return the Julia type corresponding to an XLA `PrimitiveType` integer code.
"""
function julia_element_type(code::Int32)
    return get(XLA_TYPE_TO_JULIA, code) do
        error("Unknown XLA PrimitiveType code: $code")
    end
end

# ============================================================================
# Callback registry – prevent GC from collecting cfunction pointers
# ============================================================================

"""
Global set of registered callbacks. Storing them here prevents Julia's GC from
collecting the `@cfunction` pointers while XLA might still call them.
"""
const _CALLBACK_REGISTRY = Dict{UInt,Any}()
const _CALLBACK_LOCK = ReentrantLock()

function _register_callback!(ptr::Ptr{Cvoid}, closure)
    key = UInt(ptr)
    lock(_CALLBACK_LOCK) do
        _CALLBACK_REGISTRY[key] = closure
    end
    return nothing
end

# ============================================================================
# The low-level C-compatible callback adapter
#
# Key design: all type/shape information is captured at trace time when
# _make_c_callback is called. The trampoline closure uses these captured
# values directly, avoiding any Dict lookups, ntuple with runtime N,
# or other dynamic dispatch that could trigger JIT/GC on a non-Julia thread.
# ============================================================================

"""
    _wrap_buffers(ptrs, specs, n)

Internal. Construct Julia `Array` views for `n` buffers given a pointer to
the data-pointer array and a tuple of `(Type, shape)` specs captured at trace
time.
"""
@inline function _wrap_buffers(
    data_ptrs::Ptr{Ptr{Cvoid}}, specs::NTuple{N,Tuple{DataType,Tuple}}, ::Val{N}
) where {N}
    return ntuple(Val(N)) do i
        T, shape = @inbounds specs[i]
        ptr = unsafe_load(data_ptrs, i)
        if length(shape) == 0
            unsafe_wrap(Array, Ptr{T}(ptr), (1,))
        else
            unsafe_wrap(Array, Ptr{T}(ptr), shape)
        end
    end
end

"""
    _make_c_callback(f, output_specs, input_specs) -> Ptr{Cvoid}

Create a C-callable function pointer that, when called by the C++ FFI handler,
constructs Julia `Array` views using the types and shapes captured at trace time,
then calls `f(output_arrays..., input_arrays...)`.

`output_specs` and `input_specs` are `NTuple{N, Tuple{DataType, Tuple}}` of
`(ElementType, shape)` pairs.
"""
function _make_c_callback(
    f::F, output_specs::NTuple{N_out}, input_specs::NTuple{N_in}
) where {F,N_out,N_in}
    # Capture everything at compile time via Val
    out_val = Val(N_out)
    in_val = Val(N_in)

    trampoline = function (
        inputs_ptr::Ptr{Ptr{Cvoid}},
        _input_types::Ptr{Int32},
        _input_ranks::Ptr{Int32},
        _input_dims_ptrs::Ptr{Ptr{Int64}},
        _num_inputs::Int64,
        outputs_ptr::Ptr{Ptr{Cvoid}},
        _output_types::Ptr{Int32},
        _output_ranks::Ptr{Int32},
        _output_dims_ptrs::Ptr{Ptr{Int64}},
        _num_outputs::Int64,
    )
        # Use captured specs — no dynamic dispatch here
        output_arrays = _wrap_buffers(outputs_ptr, output_specs, out_val)
        input_arrays = _wrap_buffers(inputs_ptr, input_specs, in_val)
        f(output_arrays..., input_arrays...)
        return nothing
    end

    cfunc = @cfunction(
        $trampoline,
        Cvoid,
        (
            Ptr{Ptr{Cvoid}},  # inputs
            Ptr{Int32},       # input_types
            Ptr{Int32},       # input_ranks
            Ptr{Ptr{Int64}},  # input_dims
            Int64,            # num_inputs
            Ptr{Ptr{Cvoid}},  # outputs
            Ptr{Int32},       # output_types
            Ptr{Int32},       # output_ranks
            Ptr{Ptr{Int64}},  # output_dims
            Int64,            # num_outputs
        ),
    )

    # Extract raw pointer from Base.CFunction
    ptr = Base.unsafe_convert(Ptr{Cvoid}, cfunc)
    # Store the CFunction (and closure) in the registry to prevent GC
    _register_callback!(ptr, (f, trampoline, cfunc))
    return ptr
end

# ============================================================================
# User-facing API
# ============================================================================

"""
    custom_call(
        f,
        result_specs::Tuple,
        args::Union{TracedRArray, TracedRNumber}...;
        has_side_effect::Bool = true,
        result_alias = nothing,
        location = Ops.mlir_stacktrace("custom_call", @__FILE__, @__LINE__),
    )

Emit a `stablehlo.custom_call` that will call back into Julia function `f` at
execution time. **CPU-only** for now.

# Arguments

- `f`: A Julia function with signature `f(output_arrays..., input_arrays...)`.
  The function receives N output arrays followed by M input arrays as positional
  arguments. Each array is a zero-copy Julia `Array` view over the XLA-allocated
  buffer with the correct element type and shape. Write results into the output
  arrays. **The function must not throw.**

- `result_specs`: A tuple of `(ElementType, shape)` pairs describing each output.
  - `ElementType` is a Julia type (e.g., `Float32`, `Int64`).
  - `shape` is a tuple of integers (e.g., `(3, 4)`). Use `()` for scalars.
  - Example: `((Float32, (4,)), (Int64, ()))` for one 4-element vector and one scalar.

- `args...`: Traced inputs (`TracedRArray` or `TracedRNumber`).

# Keyword Arguments

- `has_side_effect::Bool = true`: Whether the custom call has side effects. Set to `true`
  if the callback modifies global state or the ordering matters. Default is `true` for
  safety; set to `false` if the call is a pure function of its inputs for better
  optimization.

- `result_alias`: Optional output operand aliasing specification.

# Returns

A tuple of `TracedRArray` / `TracedRNumber` corresponding to the `result_specs`.
If there is exactly one result, returns it unwrapped (not in a tuple).

# Example

```julia
function my_scale!(out, x)
    out .= 2.0f0 .* x
    return nothing
end

function traced_fn(x)
    return Reactant.CustomCall.custom_call(
        my_scale!,
        ((Float32, (4,)),),
        x,
    )
end
```
"""
function custom_call(
    f,
    result_specs::Tuple,
    args::Union{TracedRArray,TracedRNumber}...;
    has_side_effect::Bool=true,
    result_alias=nothing,
    location=mlir_stacktrace("custom_call", @__FILE__, @__LINE__),
)
    # Build specs tuples for the callback (captured at trace time)
    output_specs = Tuple((T, Tuple(shape)) for (T, shape) in result_specs)
    input_specs = Tuple((unwrapped_eltype(typeof(arg)), Tuple(size(arg))) for arg in args)

    # Get the C function pointer for the callback
    callback_ptr = _make_c_callback(f, output_specs, input_specs)
    callback_ptr_int = Int64(UInt(callback_ptr))

    # Build MLIR input values
    input_values = [arg.mlir_data for arg in args]

    # Build MLIR result types
    result_types = MLIR.IR.Type[]
    for (T, shape) in result_specs
        shape_vec = collect(Int, shape)
        push!(result_types, MLIR.IR.TensorType(shape_vec, MLIR.IR.Type(T)))
    end

    # Build backend_config as a dictionary with the callback pointer
    backend_config = Dict("callback_ptr" => MLIR.IR.Attribute(callback_ptr_int))

    # Emit the custom call op
    op = stablehlo.custom_call(
        input_values;
        result_0=result_types,
        call_target_name="reactant_julia_callback",
        has_side_effect=MLIR.IR.Attribute(has_side_effect),
        backend_config,
        api_version=Int32(4),
        output_operand_aliases=result_alias,
        location,
    )

    # Wrap results  
    results = []
    for (i, (T, shape)) in enumerate(result_specs)
        res = MLIR.IR.result(op, i)
        if shape == () || shape == (1,) && length(shape) == 0
            push!(results, TracedRNumber{T}((), res))
        else
            shape_tuple = Tuple(shape)
            N = length(shape)
            push!(results, TracedRArray{T,N}((), res, shape_tuple))
        end
    end

    # Unwrap single results for convenience
    if length(results) == 1
        return results[1]
    end
    return Tuple(results)
end

"""
    custom_call(
        f,
        result_spec::Pair{Type, <:Tuple},
        args::Union{TracedRArray, TracedRNumber}...;
        kwargs...
    )

Convenience method accepting a single `Pair{Type, Shape}` instead of a tuple of specs.

# Example

```julia
Reactant.CustomCall.custom_call(my_fn, Float32 => (4,), x, y)
```
"""
function custom_call(
    f,
    result_spec::Pair{<:Type,<:Tuple},
    args::Union{TracedRArray,TracedRNumber}...;
    kwargs...,
)
    return custom_call(f, ((result_spec.first, result_spec.second),), args...; kwargs...)
end

end # module CustomCall
