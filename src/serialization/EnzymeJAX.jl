module EnzymeJAX

using ..Reactant: Reactant, Compiler, MLIR

const NUMPY_SIMPLE_TYPES = Dict(
    Bool => "np.bool_",
    Int8 => "np.int8",
    Int16 => "np.int16",
    Int32 => "np.int32",
    Int64 => "np.int64",
    UInt8 => "np.uint8",
    UInt16 => "np.uint16",
    UInt32 => "np.uint32",
    UInt64 => "np.uint64",
    Float16 => "np.float16",
    Float32 => "np.float32",
    Float64 => "np.float64",
    ComplexF32 => "np.complex64",
    ComplexF64 => "np.complex128",
)

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
function my_function(x, y)
    return x .+ y
end

# Create some example inputs
x = Reactant.to_rarray(Float32[1, 2, 3])
y = Reactant.to_rarray(Float32[4, 5, 6])

# Export to EnzymeJAX
mlir_path, python_path, input_paths = Reactant.Serialization.export_to_enzymeax(
    my_function, x, y;
    output_dir="/tmp/exported",
    function_name="my_function"
)
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
    f,
    args...;
    output_dir::String=".",
    function_name::String="exported_function",
)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Generate the StableHLO/MLIR code using compile_mlir directly
    mod, mlir_fn_res = Compiler.compile_mlir(
        f, args;
        shardy_passes=:none
    )
    hlo_code = string(mod)
    
    # Save MLIR code
    mlir_path = joinpath(output_dir, "$(function_name).mlir")
    write(mlir_path, hlo_code)
    
    # Process and save inputs
    input_paths = String[]
    input_info = []
    
    for (i, arg) in enumerate(args)
        # Convert to array if needed
        arr = _to_array(arg)
        
        # Save the input (transposed for row-major Python/NumPy)
        input_path = joinpath(output_dir, "$(function_name)_input_$(i).npy")
        _save_transposed_array(input_path, arr)
        push!(input_paths, input_path)
        
        # Store shape and dtype info (in Julia's column-major ordering)
        push!(input_info, (shape=size(arr), dtype=eltype(arr)))
    end
    
    # Generate Python script
    python_path = joinpath(output_dir, "$(function_name).py")
    _generate_python_script(python_path, function_name, mlir_path, input_paths, input_info)
    
    return (mlir_path, python_path, input_paths)
end

"""
Convert Reactant types to regular Julia arrays for saving.
"""
function _to_array(x::Reactant.ConcreteRArray)
    return Array(x)
end

function _to_array(x::Reactant.ConcreteRNumber)
    return [x.data]
end

function _to_array(x::AbstractArray)
    return Array(x)
end

function _to_array(x::Number)
    return [x]
end

function _to_array(x::Tuple)
    error("Tuple arguments are not yet supported. Please flatten your arguments.")
end

function _to_array(x::NamedTuple)
    error("NamedTuple arguments are not yet supported. Please flatten your arguments.")
end

"""
Save an array to a .npy file, transposing to account for row-major vs column-major ordering.
"""
function _save_transposed_array(path::String, arr::AbstractArray)
    # For multi-dimensional arrays, we need to reverse the dimensions for Python/NumPy
    # Julia: column-major (fastest changing index first)
    # Python: row-major (fastest changing index last)
    transposed = permutedims(arr, reverse(1:ndims(arr)))
    
    # Use a simple .npy writer
    # NPY format v1.0: magic (6 bytes) + version (2 bytes) + header_len (2 bytes) + header + data
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

"""
Get NumPy dtype string for a Julia type.
"""
function _numpy_dtype_string(::Type{Bool})
    return "|b1"
end

function _numpy_dtype_string(::Type{Int8})
    return "|i1"
end

function _numpy_dtype_string(::Type{UInt8})
    return "|u1"
end

function _numpy_dtype_string(::Type{Int16})
    return "<i2"
end

function _numpy_dtype_string(::Type{UInt16})
    return "<u2"
end

function _numpy_dtype_string(::Type{Int32})
    return "<i4"
end

function _numpy_dtype_string(::Type{UInt32})
    return "<u4"
end

function _numpy_dtype_string(::Type{Int64})
    return "<i8"
end

function _numpy_dtype_string(::Type{UInt64})
    return "<u8"
end

function _numpy_dtype_string(::Type{Float16})
    return "<f2"
end

function _numpy_dtype_string(::Type{Float32})
    return "<f4"
end

function _numpy_dtype_string(::Type{Float64})
    return "<f8"
end

function _numpy_dtype_string(::Type{ComplexF32})
    return "<c8"
end

function _numpy_dtype_string(::Type{ComplexF64})
    return "<c16"
end

"""
Generate a Python script that uses EnzymeJAX to call the exported function.
"""
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
    
    # Build the Python script without leading indentation
    lines = String[]
    
    # Header
    push!(lines, "\"\"\"")
    push!(lines, "Auto-generated Python script for calling exported Julia/Reactant function via EnzymeJAX.")
    push!(lines, "")
    push!(lines, "This script was generated by Reactant.Serialization.export_to_enzymeax().")
    push!(lines, "\"\"\"")
    push!(lines, "")
    push!(lines, "from enzyme_ad.jax import hlo_call")
    push!(lines, "import jax")
    push!(lines, "import jax.numpy as jnp")
    push!(lines, "import numpy as np")
    push!(lines, "import os")
    push!(lines, "")
    push!(lines, "# Get the directory of this script")
    push!(lines, "_script_dir = os.path.dirname(os.path.abspath(__file__))")
    push!(lines, "")
    push!(lines, "# Load the MLIR/StableHLO code")
    push!(lines, "with open(os.path.join(_script_dir, \"$(mlir_rel)\"), \"r\") as f:")
    push!(lines, "    _hlo_code = f.read()")
    push!(lines, "")
    
    # Function to load inputs
    push!(lines, "def load_inputs():")
    push!(lines, "    \"\"\"Load the example inputs that were exported from Julia.\"\"\"")
    push!(lines, "    inputs = []")
    for input_rel in input_rels
        push!(lines, "    inputs.append(np.load(os.path.join(_script_dir, \"$(input_rel)\")))")
    end
    push!(lines, "    return tuple(inputs)")
    push!(lines, "")
    
    # Main function
    arg_names = ["arg$i" for i in 1:length(input_paths)]
    arg_list = join(arg_names, ", ")
    
    push!(lines, "def run_$(function_name)($(arg_list)):")
    push!(lines, "    \"\"\"")
    push!(lines, "    Call the exported Julia function via EnzymeJAX.")
    push!(lines, "    ")
    push!(lines, "    Args:")
    
    for (i, info) in enumerate(input_info)
        # Note: shapes are already transposed for Python
        python_shape = reverse(info.shape)
        push!(lines, "        $(arg_names[i]): Array of shape $(python_shape) and dtype $(NUMPY_SIMPLE_TYPES[info.dtype])")
    end
    
    push!(lines, "    ")
    push!(lines, "    Returns:")
    push!(lines, "        The result of calling the exported function.")
    push!(lines, "    ")
    push!(lines, "    Note:")
    push!(lines, "        All inputs must be in row-major (Python/NumPy) order. If you're passing")
    push!(lines, "        arrays from Julia, make sure to transpose them first using:")
    push!(lines, "        `permutedims(arr, reverse(1:ndims(arr)))`")
    push!(lines, "    \"\"\"")
    push!(lines, "    return hlo_call(")
    push!(lines, "        $(arg_list),")
    push!(lines, "        source=_hlo_code,")
    push!(lines, "    )")
    push!(lines, "")
    
    # Main block
    push!(lines, "if __name__ == \"__main__\":")
    push!(lines, "    # Load the example inputs")
    push!(lines, "    inputs = load_inputs()")
    push!(lines, "    ")
    push!(lines, "    # Run the function (with JIT compilation)")
    push!(lines, "    print(\"Running $(function_name) with JIT compilation...\")")
    push!(lines, "    result = jax.jit(run_$(function_name))(*inputs)")
    push!(lines, "    print(\"Result:\", result)")
    push!(lines, "    print(\"Result shape:\", result.shape if hasattr(result, 'shape') else 'scalar')")
    push!(lines, "    print(\"Result dtype:\", result.dtype if hasattr(result, 'dtype') else type(result))")
    
    # Write the script
    write(python_path, join(lines, "\n") * "\n")
    return nothing
end

end  # module
