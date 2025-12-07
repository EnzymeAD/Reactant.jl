module ReactantExport

using ..Reactant: Reactant, Compiler, Serialization

"""
    export_to_reactant_script(
        f,
        args...;
        output_dir::Union{String,Nothing}=nothing,
        function_name::String=string(f)
    )

Export a Julia function to a standalone Reactant script.

This function:
1. Compiles the function to StableHLO via `Reactant.@code_hlo`
2. Saves the MLIR/StableHLO code to a `.mlir` file
3. Saves all input arrays to a single compressed `.npz` file (transposed to account for
   row-major vs column-major)
4. Generates a Julia script that only depends on Reactant for loading and executing

## Requirements

- **NPZ.jl**: Must be loaded with `using NPZ` for compression support

## Arguments

  - `f`: The Julia function to export
  - `args...`: The arguments to the function (used to infer types and shapes)

## Keyword Arguments

  - `output_dir::Union{String,Nothing}`: Directory where output files will be saved. If
    `nothing`, uses a temporary directory and prints the path.
  - `function_name::String`: Base name for generated files

## Returns

The path to the generated Julia script as a `String`.

## Files Generated

  - `{function_name}.mlir`: The StableHLO/MLIR module
  - `{function_name}_{id}_inputs.npz`: Compressed NPZ file containing all input arrays
  - `{function_name}.jl`: Julia script that loads and executes the exported function

## Example

```julia
using Reactant, NPZ

# Define a simple function
function my_function(x::AbstractArray, y::AbstractArray)
    return x .+ y
end

# Create some example inputs
x = Reactant.to_rarray(rand(Float32, 2, 3))
y = Reactant.to_rarray(rand(Float32, 2, 3))

# Export to Reactant script
julia_file_path = Reactant.Serialization.export_to_reactant_script(my_function, x, y)
```

Then in Julia:
```julia
# Run the generated Julia script
include(julia_file_path)
```
"""
function export_to_reactant_script(
    f,
    args...;
    output_dir::Union{String,Nothing}=nothing,
    function_name::String=string(f),
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
        f,
        args;
        argprefix,
        drop_unsupported_attributes=true,
        shardy_passes=:none,
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

    # Generate Julia script
    julia_path = joinpath(output_dir, "$(function_name).jl")
    _generate_julia_script(julia_path, function_name, mlir_path, input_path, input_info)
    return julia_path
end

_to_array(x::Reactant.ConcreteRArray) = Array(x)
_to_array(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)

function save_inputs_npz(
    output_path::String, inputs::Dict{String,<:Union{AbstractArray,Number}}
)
    if !Serialization.serialization_supported(Val(:NPZ))
        error(
            "`NPZ.jl` is required for saving compressed arrays. Please load it with \
               `using NPZ` and try again.",
        )
    end
    return save_inputs_npz_impl(output_path, inputs)
end

function save_inputs_npz_impl end

function _generate_julia_script(
    julia_path::String,
    function_name::String,
    mlir_path::String,
    input_path::String,
    input_info::Vector,
)
    # Get relative paths for the Julia script
    output_dir = dirname(julia_path)
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

    load_inputs = ["npz_data[\"$(info.key)\"]" for info in input_info]

    # Build the complete Julia script
    script = """
    \"\"\"
    Auto-generated Julia script for calling exported Reactant function.

    This script was generated by Reactant.Serialization.export_to_reactant_script().
    \"\"\"

    using Reactant
    using NPZ

    # Get the directory of this script
    const SCRIPT_DIR = @__DIR__

    # Load the MLIR/StableHLO code
    const HLO_CODE = read(joinpath(SCRIPT_DIR, "$(mlir_rel)"), String)

    function load_inputs()
        \"\"\"Load the example inputs that were exported from Julia.\"\"\"
        npz_data = npzread(joinpath(SCRIPT_DIR, "$(input_rel)"))
        # Transpose back from Python/NumPy (row-major) to Julia (column-major)
        inputs = [
            $(join(["let arr = " * load * "; arr isa Number ? arr : permutedims(arr, reverse(1:ndims(arr))) end" for load in load_inputs], ",\n            "))
        ]
        return tuple(inputs...)
    end

    function run_$(function_name)($(arg_list))
        \"\"\"
        Execute the exported Julia function using Reactant.

        Args:
    $arg_docs

        Returns:
            The result of calling the exported function.
        \"\"\"
        # Load HLO module from string
        # TODO: This will use Reactant's HLO execution API once available
        # For now, we document that this is a placeholder that will be implemented
        error("Direct HLO execution from loaded IR is not yet implemented in Reactant. " *
              "This script serves as a template for future functionality.")
    end

    # Main execution when script is run directly
    if abspath(PROGRAM_FILE) == @__FILE__
        # Load the example inputs
        ($(arg_list),) = load_inputs()
        
        # Convert to RArrays
        $(join(["$arg = Reactant.to_rarray($arg)" for arg in arg_names], "\n    "))
        
        # Run the function
        println("Running $(function_name)...")
        result = run_$(function_name)($(arg_list))
        println("Result: ", result)
    end
    """

    write(julia_path, strip(script) * "\n")
    return nothing
end

end  # module
