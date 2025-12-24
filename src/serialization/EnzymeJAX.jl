module EnzymeJAX

using ..Reactant: Reactant, Compiler, Serialization

"""
    export_to_enzymejax(
        f,
        args...;
        output_dir::Union{String,Nothing}=nothing,
        function_name::String=string(f),
        preserve_sharding::Bool=true,
        compile_options=Reactant.Compiler.CompileOptions(),
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
  - `compile_options`: Compilation options passed to `Reactant.Compiler.compile_mlir`. See
    [`Reactant.Compiler.CompileOptions`](@ref) for more details.

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
    compile_options=Reactant.Compiler.CompileOptions(),
)
    function_name = replace(function_name, "!" => "_")

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
        f,
        args;
        argprefix,
        drop_unsupported_attributes=true,
        compile_options,
        # to support older jax versions which don't support shardy
        shardy_passes=:to_mhlo_shardings,
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

        # Extract sharding information if available and if preserve_sharding is true
        sharding_info = nothing
        if preserve_sharding && _has_sharding_info(concrete_arg)
            sharding_info = _extract_sharding_info(concrete_arg)
        end

        push!(
            input_info,
            (
                shape=size(concrete_arg),
                dtype=Reactant.unwrapped_eltype(concrete_arg),
                path="arg." * join(string.(path), "."),
                key=arr_key,
                sharding=sharding_info,
            ),
        )
        input_idx += 1
    end

    # Save all inputs to a single NPZ file
    input_path = joinpath(output_dir, "$(function_name)_$(fnid)_inputs.npz")
    save_inputs_npz(input_path, input_data)

    # Generate Python script
    python_path = joinpath(output_dir, "$(function_name).py")
    _generate_python_script(
        python_path, function_name, mlir_path, input_path, input_info; preserve_sharding
    )
    return python_path
end

_to_array(x::Reactant.ConcreteRArray) = Array(x)
_to_array(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)

_has_sharding_info(x::Reactant.ConcreteRArray) = Reactant.Sharding.is_sharded(x.sharding)
_has_sharding_info(x) = false

function _extract_sharding_info(x::Reactant.ConcreteRArray)
    sharding = x.sharding
    if sharding isa Reactant.Sharding.ShardInfo
        inner_sharding = sharding.sharding
        if inner_sharding isa Reactant.Sharding.NamedSharding
            # TODO: we need to export is_closed, priority, and subaxes at some point
            return (;
                type="NamedSharding",
                mesh=inner_sharding.mesh,
                partition_spec=inner_sharding.partition_spec,
            )
        elseif inner_sharding isa Reactant.Sharding.Replicated
            return (; type="Replicated", mesh=inner_sharding.mesh)
        elseif inner_sharding isa Reactant.Sharding.NoSharding
            return (; type="NoSharding")
        else
            error("Unsupported sharding type: $(typeof(inner_sharding))")
        end
    end
    return (; type="NoSharding")
end

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
    input_info::Vector;
    preserve_sharding::Bool=true,
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

    # Generate sharding annotations if available
    has_any_sharding =
        preserve_sharding && any(info.sharding !== nothing for info in input_info)

    device_put_calls = String[]
    if has_any_sharding
        inserted_meshes = IdDict()
        counter = 0
        for (i, info) in enumerate(input_info)
            if info.sharding !== nothing
                if haskey(inserted_meshes, info.sharding.mesh)
                    pymesh = inserted_meshes[info.sharding.mesh]
                else
                    pymesh = "mesh$counter"
                    counter += 1
                    inserted_meshes[info.sharding.mesh] = pymesh
                    axis_sizes = join(string.(reverse(info.sharding.mesh.axis_sizes)), ", ")
                    mesh_axes = join(
                        reverse(["'$(string(x))'" for x in info.sharding.mesh.axis_names]),
                        ", ",
                    )

                    push!(
                        device_put_calls,
                        "$(pymesh) = jax.make_mesh(($(axis_sizes)), ($(mesh_axes)))",
                    )
                end

                push!(
                    device_put_calls,
                    "# Set up sharding for $(arg_names[i]): $(info.sharding.type)",
                )

                # Create device_put call with NamedSharding
                if info.sharding.type == "NoSharding"
                    device_put_calls_str = "$(arg_names[i]) = jnp.asarray($(arg_names[i]))"
                elseif info.sharding.type == "NamedSharding"
                    pstrings = [
                        if length(p) == 1
                            p[1] isa Nothing ? "None" : "'$(string(p[1]))'"
                        else
                            join(string.(reverse(p)), ", ")
                        end for p in info.sharding.partition_spec
                    ]
                    partition_spec = join(reverse(pstrings), ", ")
                    device_put_calls_str = "$(arg_names[i]) = jax.device_put($(arg_names[i]), jax.sharding.NamedSharding($(pymesh), P($(partition_spec))))"
                else
                    error("Unsupported sharding type: $(info.sharding.type)")
                end
                push!(device_put_calls, device_put_calls_str)
            end
        end
    end

    if has_any_sharding
        inputs_to_jax_arrays = """# Apply sharding to inputs using device_put and NamedSharding
            $(join(device_put_calls, "\n    "))
        """
    else
        convert_str_list = join(
            ["    $(argname) = jnp.asarray($(argname))" for argname in arg_names], "\n"
        )
        inputs_to_jax_arrays = """
        # Convert inputs to jax arrays
        $(convert_str_list)
        """
    end

    load_inputs = ["npz_data['$(info.key)']" for info in input_info]

    # Build the complete Python script
    script = """
    \"\"\"
    Auto-generated Python script for calling exported Julia/Reactant function via EnzymeJAX.

    This script was generated by Reactant.Serialization.export_to_enzymejax().
    \"\"\"

    from enzyme_ad.jax import hlo_call
    import jax
    from jax.sharding import PartitionSpec as P
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
        ($(arg_list),) = load_inputs()
        $(inputs_to_jax_arrays)
        # Run the function
        print(\"Running $(function_name)...\")
        result = run_$(function_name)($(arg_list))
        print(\"Result:\", result)
    """

    write(python_path, strip(script) * "\n")
    return nothing
end

end  # module
