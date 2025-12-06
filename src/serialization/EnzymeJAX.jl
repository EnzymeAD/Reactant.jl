module EnzymeJAX

using ..Reactant: Reactant, Compiler, MLIR, Serialization

"""
    export_to_enzymeax(
        f,
        args...;
        output_dir::String=".",
        function_name::String="exported_function",
    )

Export a Julia function to EnzymeJAX format for use in Python/JAX.

This function:
1. Compiles the function to StableHLO via `Reactant.@code_hlo`
2. Saves the MLIR/StableHLO code to a `.mlir` file
3. Saves input arrays to `.npy` files (transposed to account for row-major vs column-major)
4. Generates a Python script with the function wrapped for EnzymeJAX's `hlo_call`

## Arguments

  - `f`: The Julia function to export
  - `args...`: The arguments to the function (used to infer types and shapes)

## Keyword Arguments

  - `output_dir::String="."`: Directory where output files will be saved
  - `function_name::String="exported_function"`: Base name for generated files

## Returns

A tuple `(mlir_path, python_path, input_paths)` containing paths to:
  - The generated `.mlir` file
  - The generated `.py` file
  - A vector of paths to input `.npy` files

## Example

```julia
using Reactant

# Define a simple function
function my_function(x, y::NamedTuple)
    return x .+ y.x .- y.y
end

# Create some example inputs
x = Reactant.to_rarray(Float32[1, 2, 3])
y = (; x=Reactant.to_rarray(Float32[4, 5, 6]), y=Reactant.to_rarray(Float32[7, 8, 9]))

# Export to EnzymeJAX
python_file_path = Reactant.Serialization.export_to_enzymeax(my_function, x, y)
```

Then in Python:
```python
# Run the generated Python script
from exported.my_function import run_my_function
import jax

result = jax.jit(run_my_function)(*inputs)
```
"""
function export_to_enzymeax(
    f, args...; output_dir::Union{String,Nothing}=nothing, function_name::String=string(f)
)
    if output_dir === nothing
        output_dir = mktempdir(; cleanup=false)
        @info "Output directory is $(output_dir)"
    else
        mkpath(output_dir)
    end

    # Generate the StableHLO/MLIR code using compile_mlir
    # This returns compilation result with traced argument information
    mod, mlir_fn_res = Compiler.compile_mlir(f, args)
    hlo_code = string(mod)

    # Save MLIR code
    fnid = 0
    while isfile(joinpath(output_dir, "$(function_name)_$(fnid).mlir"))
        fnid += 1
    end
    mlir_path = joinpath(output_dir, "$(function_name)_$(fnid).mlir")
    write(mlir_path, hlo_code)

    # Process and save inputs based on the linearized arguments
    # seen_args is an OrderedIdDict where keys are concrete args and values are traced args
    # linear_args contains only the arguments that need to be passed to the function
    # We iterate over seen_args which preserves the order, and only save those in linear_args
    input_paths = String[]
    input_info = []
    input_idx = 1
    for (concrete_arg, traced_arg) in mlir_fn_res.seen_args
        # Only process arguments that are in linear_args (skip computed values)
        if traced_arg in mlir_fn_res.linear_args
            # Save the input (transposed for row-major Python/NumPy)
            input_path = joinpath(
                output_dir, "$(function_name)_$(fnid)_input_$(input_idx).npy"
            )
            _save_transposed_array(input_path, _to_array(concrete_arg))
            push!(input_paths, input_path)
            push!(input_info, (shape=size(concrete_arg), dtype=eltype(concrete_arg)))
            input_idx += 1
        end
    end

    # Generate Python script
    python_path = joinpath(output_dir, "$(function_name).py")
    _generate_python_script(python_path, function_name, mlir_path, input_paths, input_info)
    return python_path
end

_to_array(x::Reactant.ConcreteRArray) = Array(x)
_to_array(x::Reactant.ConcreteRNumber) = Number(x)

# Save an array to a .npy file, transposing to account for row-major vs
# column-major ordering.
function _save_transposed_array(path::String, arr::AbstractArray)
    # For multi-dimensional arrays, we need to reverse the dimensions for Python/NumPy
    transposed = permutedims(arr, reverse(1:ndims(arr)))

    # Use a simple .npy writer
    # NPY format v1.0: magic (6 bytes) + version (2 bytes) + header_len (2 bytes) +
    #                  header + data
    open(path, "w") do io
        # Magic number for .npy format
        write(io, UInt8[0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
        # Version 1.0
        write(io, UInt8[0x01, 0x00])

        # Prepare header
        dtype_str = _numpy_dtype_string(eltype(arr))
        shape_str = join(size(transposed), ", ")
        header = "{'descr': '$(dtype_str)', 'fortran_order': False, 'shape': ($(shape_str),)}"

        # Pad header to be aligned on 64 bytes (16-byte alignment for v1.0)
        # Total size needs to be divisible by 16
        header_len = length(header) + 1  # +1 for newline
        total_len = 10 + header_len  # 10 = magic(6) + version(2) + header_len(2)
        padding = (16 - (total_len % 16)) % 16
        header = header * " "^padding * "\n"
        header_len = length(header)

        # Write header length (little-endian UInt16)
        write(io, UInt16(header_len))
        # Write header
        write(io, header)
        # Write data
        write(io, vec(transposed))
    end
    return nothing
end

# TODO: use a proper package for this
_numpy_dtype_string(::Type{Bool}) = "|b1"
_numpy_dtype_string(::Type{Int8}) = "|i1"
_numpy_dtype_string(::Type{UInt8}) = "|u1"
_numpy_dtype_string(::Type{Int16}) = "<i2"
_numpy_dtype_string(::Type{UInt16}) = "<u2"
_numpy_dtype_string(::Type{Int32}) = "<i4"
_numpy_dtype_string(::Type{UInt32}) = "<u4"
_numpy_dtype_string(::Type{Int64}) = "<i8"
_numpy_dtype_string(::Type{UInt64}) = "<u8"
_numpy_dtype_string(::Type{Float16}) = "<f2"
_numpy_dtype_string(::Type{Float32}) = "<f4"
_numpy_dtype_string(::Type{Float64}) = "<f8"
_numpy_dtype_string(::Type{ComplexF32}) = "<c8"
_numpy_dtype_string(::Type{ComplexF64}) = "<c16"

function _generate_python_script(
    python_path::String,
    function_name::String,
    mlir_path::String,
    input_paths::Vector{String},
    input_info::Vector,
)
    # Get relative paths for the Python script
    output_dir = dirname(python_path)
    mlir_rel = relpath(mlir_path, output_dir)
    input_rels = [relpath(p, output_dir) for p in input_paths]

    # Generate input loading code
    input_loads = join(
        [
            "    inputs.append(np.load(os.path.join(_script_dir, \"$rel\")))" for
            rel in input_rels
        ],
        "\n",
    )

    # Generate argument list and documentation
    arg_names = ["arg$i" for i in 1:length(input_paths)]
    arg_list = join(arg_names, ", ")

    # Generate docstring for arguments
    arg_docs = join(
        [
            "        $(arg_names[i]): Array of shape $(reverse(info.shape)) and dtype $(Serialization.NUMPY_SIMPLE_TYPES[info.dtype])"
            for (i, info) in enumerate(input_info)
        ],
        "\n",
    )

    # Build the complete Python script
    script = """
    \"\"\"
    Auto-generated Python script for calling exported Julia/Reactant function via EnzymeJAX.

    This script was generated by Reactant.Serialization.export_to_enzymeax().
    \"\"\"

    from enzyme_ad.jax import hlo_call
    import jax
    import jax.numpy as jnp
    import numpy as np
    import os

    # Get the directory of this script
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the MLIR/StableHLO code
    with open(os.path.join(_script_dir, \"$(mlir_rel)\"), \"r\") as f:
        _hlo_code = f.read()

    def load_inputs():
        \"\"\"Load the example inputs that were exported from Julia.\"\"\"
        inputs = []
    $input_loads
        return tuple(inputs)

    def run_$(function_name)($(arg_list)):
        \"\"\"
        Call the exported Julia function via EnzymeJAX.

        Args:
    $arg_docs

        Returns:
            The result of calling the exported function.

        Note:
            All inputs must be in row-major (Python/NumPy) order. If you're passing
            arrays from Julia, make sure to transpose them first using:
            \`permutedims(arr, reverse(1:ndims(arr)))\`
        \"\"\"
        return hlo_call(
            $(arg_list),
            source=_hlo_code,
        )

    if __name__ == \"__main__\":
        # Load the example inputs
        inputs = load_inputs()

        # Run the function (with JIT compilation)
        print(\"Running $(function_name) with JIT compilation...\")
        result = jax.jit(run_$(function_name))(*inputs)
        print(\"Result:\", result)
        print(\"Result shape:\", result.shape if hasattr(result, 'shape') else 'scalar')
        print(\"Result dtype:\", result.dtype if hasattr(result, 'dtype') else type(result))
    """

    write(python_path, strip(script) * "\n")
    return nothing
end

end  # module
