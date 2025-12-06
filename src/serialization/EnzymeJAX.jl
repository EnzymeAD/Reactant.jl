module EnzymeJAX

using ..Reactant: Reactant, Compiler, MLIR, Serialization

"""
    export_to_enzymejax(
        f,
        args...;
        output_dir::Union{String,Nothing}=nothing,
        function_name::String=string(f)
    )

Export a Julia function to EnzymeJAX format for use in Python/JAX.

This function:
1. Compiles the function to StableHLO via `Reactant.@code_hlo`
2. Saves the MLIR/StableHLO code to a `.mlir` file
3. Saves all input arrays to a single compressed `.npz` file (transposed to account for
   row-major vs column-major)
4. Generates a Python script with the function wrapped for EnzymeJAX's `hlo_call`

## Requirements

- **NPZ.jl**: Must be loaded with `using NPZ` for compression support

## Arguments

  - `f`: The Julia function to export
  - `args...`: The arguments to the function (used to infer types and shapes)

## Keyword Arguments

  - `output_dir::Union{String,Nothing}`: Directory where output files will be saved. If
    `nothing`, uses a temporary directory and prints the path.
  - `function_name::String`: Base name for generated files
  - `preserve_sharding::Bool`: Whether to preserve sharding information in the exported
    function. Defaults to `true`.

## Returns

The path to the generated Python script as a `String`.

## Files Generated

  - `{function_name}.mlir`: The StableHLO/MLIR module
  - `{function_name}_{id}_inputs.npz`: Compressed NPZ file containing all input arrays
  - `{function_name}.py`: Python script with the function wrapped for EnzymeJAX

## Example

```julia
using Reactant, NPZ

# Define a simple function
function my_function(x::AbstractArray, y::NamedTuple, z::Number)
    return x .+ y.x .- y.y .+ z
end

# Create some example inputs
x = Reactant.to_rarray(reshape(collect(Float32, 1:6), 2, 3))
y = (;
    x=Reactant.to_rarray(reshape(collect(Float32, 7:12), 2, 3)),
    y=Reactant.to_rarray(reshape(collect(Float32, 13:18), 2, 3))
)
z = Reactant.to_rarray(10.0f0; track_numbers=true)

# Export to EnzymeJAX
python_file_path = Reactant.Serialization.export_to_enzymejax(my_function, x, y, z)
```

Then in Python:
```python
# Run the generated Python script
from exported.my_function import run_my_function
import jax

result = run_my_function(*inputs)
```
"""
function export_to_enzymejax(
    f,
    args...;
    output_dir::Union{String,Nothing}=nothing,
    function_name::String=string(f),
    preserve_sharding::Bool=true,
)
    if output_dir === nothing
        output_dir = mktempdir(; cleanup=false)
        @info "Output directory is $(output_dir)"
    else
        mkpath(output_dir)
    end

    # Generate the StableHLO/MLIR code using compile_mlir
    # This returns compilation result with traced argument information
    argprefix = gensym("exportarg")
    mod, mlir_fn_res = Compiler.compile_mlir(
        f, args; argprefix, drop_unsupported_attributes=true
    )
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
    input_data = Dict{String,Union{AbstractArray,Number}}()
    input_info = []
    input_idx = 1
    for (concrete_arg, traced_arg) in mlir_fn_res.seen_args
        path = Reactant.TracedUtils.get_idx(traced_arg, argprefix)[2:end]

        # Store input data for the single NPZ file
        arr_key = "arr_$input_idx"
        input_data[arr_key] = _to_array(concrete_arg)
        push!(
            input_info,
            (
                shape=size(concrete_arg),
                dtype=Reactant.unwrapped_eltype(concrete_arg),
                path="arg." * join(string.(path), "."),
                key=arr_key,
            ),
        )
        input_idx += 1
    end

    # Save all inputs to a single NPZ file
    input_path = joinpath(output_dir, "$(function_name)_$(fnid)_inputs.npz")
    save_inputs_npz(input_path, input_data)

    # Generate Python script
    python_path = joinpath(output_dir, "$(function_name).py")
    _generate_python_script(python_path, function_name, mlir_path, input_path, input_info)
    return python_path
end

_to_array(x::Reactant.ConcreteRArray) = Array(x)
_to_array(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)

function save_inputs_npz(
    output_path::String, inputs::Dict{String,<:Union{AbstractArray,Number}}
)
    if !Serialization.serialization_supported(Val(:NPZ))
        error("`NPZ.jl` is required for saving compressed arrays. Please load it with \
               `using NPZ` and try again.")
    end
    return save_inputs_npz_impl(output_path, inputs)
end

function save_inputs_npz_impl end

function _generate_python_script(
    python_path::String,
    function_name::String,
    mlir_path::String,
    input_path::String,
    input_info::Vector,
)
    # Get relative paths for the Python script
    output_dir = dirname(python_path)
    mlir_rel = relpath(mlir_path, output_dir)
    input_rel = relpath(input_path, output_dir)

    # Generate argument list and documentation
    arg_names = ["arg$i" for i in 1:length(input_info)]
    arg_list = join(arg_names, ", ")

    # Generate docstring for arguments
    arg_docs = join(
        [
            "        $(arg_names[i]): Array of shape $(reverse(info.shape)) and dtype $(Serialization.NUMPY_SIMPLE_TYPES[info.dtype]). Path: $(info.path)"
            for (i, info) in enumerate(input_info)
        ],
        "\n",
    )

    arg_size_checks = [
        "assert $(arg_names[i]).shape == $(reverse(info.shape)), f\"Expected shape of $(arg_names[i]) to be $(reverse(info.shape)). Got {$(arg_names[i]).shape} (path: $(info.path))\""
        for (i, info) in enumerate(input_info)
    ]
    arg_dtype_checks = [
        "assert $(arg_names[i]).dtype == np.dtype('$(Serialization.NUMPY_SIMPLE_TYPES[info.dtype])'), f\"Expected dtype of $(arg_names[i]) to be $(Serialization.NUMPY_SIMPLE_TYPES[info.dtype]). Got {$(arg_names[i]).dtype} (path: $(info.path))\""
        for (i, info) in enumerate(input_info)
    ]

    load_inputs = ["npz_data['$(info.key)']" for info in input_info]

    # Build the complete Python script
    script = """
    \"\"\"
    Auto-generated Python script for calling exported Julia/Reactant function via EnzymeJAX.

    This script was generated by Reactant.Serialization.export_to_enzymejax().
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
        npz_data = np.load(os.path.join(_script_dir, \"$(input_rel)\"))
        inputs = [$(join(load_inputs, ", "))]
        return tuple(inputs)

    @jax.jit
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
        $(join(arg_dtype_checks, "\n    "))
        $(join(arg_size_checks, "\n    "))
        return hlo_call(
            $(arg_list),
            source=_hlo_code,
        )

    if __name__ == \"__main__\":
        # Load the example inputs
        inputs = load_inputs()

        # Run the function (with JIT compilation)
        print(\"Running $(function_name) with JIT compilation...\")
        result = run_$(function_name)(*inputs)
        print(\"Result:\", result)
    """

    write(python_path, strip(script) * "\n")
    return nothing
end

end  # module
