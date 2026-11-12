module EnzymeJAX

using ..Reactant: Reactant, Compiler, Serialization, MLIR

"""
    export_to_enzymejax(
        f,
        args...;
        output_dir::Union{String,Nothing}=nothing,
        function_name::String=string(f),
        preserve_sharding::Bool=true,
        export_data_as_npz::Bool=true,
        compile_options=Reactant.Compiler.CompileOptions(),
    )

Export a Julia function to EnzymeJAX format for use in Python/JAX.

This function:
1. Compiles the function to StableHLO via `Reactant.code_hlo`
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
  - `function_name::String`: Base name for generated files.
  - `preserve_sharding::Bool`: Whether to preserve sharding information in the exported
    function. Defaults to `true`.
  - `export_data_as_npz::Bool`: Whether to export the input data as an NPZ file. Defaults to
    `true`. This requires `NPZ.jl` to be loaded. If `false`, the Python script will
    initialize the inputs with random data.
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
    export_data_as_npz::Bool=true,
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

    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        mod, mlir_fn_res = Compiler.compile_mlir(
            ctx,
            f,
            args;
            argprefix,
            drop_unsupported_attributes=true,
            compile_options,
            # to support older jax versions which don't support shardy
            shardy_passes=:to_mhlo_shardings,
        )
        hlo_code = try
            string(mod)
        finally
            MLIR.IR.dispose(mod)
        end

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
        if export_data_as_npz
            save_inputs_npz(input_path, input_data)
        end

        # Generate Python script
        python_path = joinpath(output_dir, "$(function_name).py")
        _generate_python_script(
            python_path,
            function_name,
            mlir_path,
            input_path,
            input_info;
            export_data_as_npz,
            preserve_sharding,
        )
        return python_path
    end
end

_to_array(x::Reactant.ConcreteRArray) = Array(x)
_to_array(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)

_has_sharding_info(x::Reactant.ConcreteRArray) = Reactant.Sharding.is_sharded(x.sharding)
_has_sharding_info(x) = false

function _extract_sharding_info(x::Reactant.ConcreteRArray)
    sharding = x.sharding
    if sharding isa Reactant.Sharding.ShardInfo
        return _extract_inner_sharding_info(sharding.sharding, ndims(x))
    end
    return (; type="NoSharding")
end

function _extract_inner_sharding_info(
    inner_sharding::Reactant.Sharding.NamedSharding, ::Int
)
    return (;
        type="NamedSharding",
        mesh=inner_sharding.mesh,
        partition_spec=inner_sharding.partition_spec,
        is_closed=inner_sharding.is_closed,
        priority=inner_sharding.priority,
        subaxes=inner_sharding.subaxes,
    )
end

function _extract_inner_sharding_info(
    inner_sharding::Reactant.Sharding.HloSharding, ndims_x::Int
)
    if inner_sharding.parent_sharding isa Reactant.Sharding.NamedSharding
        parent = inner_sharding.parent_sharding
        return (;
            type="NamedSharding",
            mesh=inner_sharding.mesh,
            partition_spec=parent.partition_spec,
            is_closed=inner_sharding.is_closed,
            priority=inner_sharding.priority,
            subaxes=parent.subaxes,
        )
    elseif inner_sharding.parent_sharding !== nothing
        return _extract_inner_sharding_info(inner_sharding.parent_sharding, ndims_x)
    end
    return error("Unsupported sharding type: $(typeof(inner_sharding))")
end

function _extract_inner_sharding_info(
    inner_sharding::Reactant.Sharding.DimsSharding, ndims_x::Int
)
    return _extract_inner_sharding_info(
        Reactant.Sharding.NamedSharding(inner_sharding, ndims_x), ndims_x
    )
end

function _extract_inner_sharding_info(inner_sharding::Reactant.Sharding.Replicated, ::Int)
    return (; type="Replicated", mesh=inner_sharding.mesh)
end

function _extract_inner_sharding_info(::Reactant.Sharding.NoSharding, ::Int)
    return (; type="NoSharding")
end

function _extract_inner_sharding_info(inner_sharding, ::Int)
    return error("Unsupported sharding type: $(typeof(inner_sharding))")
end

function _py_tuple(items::AbstractVector{String})
    isempty(items) && return "()"
    length(items) == 1 && return "($(items[1]),)"
    return "($(join(items, ", ")))"
end

_py_bool(b::Bool) = b ? "True" : "False"

function _compute_mesh_axis_split_bounds(mesh::Reactant.Sharding.Mesh, input_info::Vector)
    n_axes = length(mesh.axis_names)
    axis_bounds = Vector{Vector{Int}}(undef, n_axes)
    for k in 1:n_axes
        ax = mesh.axis_names[k]
        ax_size = Int(mesh.axis_sizes[k])
        bounds = Int[1, ax_size]
        for idx in 1:length(input_info)
            sharding_info = input_info[idx].sharding
            if sharding_info !== nothing &&
                sharding_info.type == "NamedSharding" &&
                sharding_info.mesh === mesh
                for d in 1:length(sharding_info.partition_spec)
                    pspec_d = sharding_info.partition_spec[d]
                    subaxes_d = sharding_info.subaxes[d]
                    for j in 1:length(pspec_d)
                        sa = subaxes_d[j]
                        if pspec_d[j] === ax && sa !== nothing
                            pre, sz = sa
                            push!(bounds, Int(pre))
                            push!(bounds, Int(pre * sz))
                        end
                    end
                end
            end
        end
        unique!(sort!(bounds))
        axis_bounds[k] = bounds
    end
    return axis_bounds
end

function _is_default_iota_mesh(mesh::Reactant.Sharding.Mesh)
    n = length(mesh)
    (length(mesh.device_ids) == n && length(mesh.logical_device_ids) == n) || return false
    for idx in 1:n
        (mesh.device_ids[idx] == idx - 1 && mesh.logical_device_ids[idx] == idx - 1) ||
            return false
    end
    return true
end

function _build_mesh_code(
    pymesh::String, mesh::Reactant.Sharding.Mesh, axis_bounds::Vector{Vector{Int}}
)
    effective_names = String[]
    effective_sizes = String[]
    for k in 1:length(mesh.axis_names)
        ax_str = String(mesh.axis_names[k])
        ax_size = Int(mesh.axis_sizes[k])
        bounds = axis_bounds[k]
        if length(bounds) <= 2
            push!(effective_names, "'$(ax_str)'")
            push!(effective_sizes, string(ax_size))
        else
            for m in 1:(length(bounds) - 1)
                @assert bounds[m + 1] % bounds[m] == 0
                push!(effective_names, "'$(ax_str)_sub_$(m - 1)'")
                push!(effective_sizes, string(div(bounds[m + 1], bounds[m])))
            end
        end
    end
    sizes_tuple = _py_tuple(effective_sizes)
    names_tuple = _py_tuple(effective_names)
    axis_types_expr = "(jax.sharding.AxisType.Auto,) * $(length(effective_names))"
    if _is_default_iota_mesh(mesh)
        return "$(pymesh) = jax.make_mesh($(sizes_tuple), $(names_tuple), $(axis_types_expr))"
    end
    ord_devs = String[]
    for lid in mesh.logical_device_ids
        push!(ord_devs, "jax.devices()[$(mesh.device_ids[lid + 1])]")
    end
    return "$(pymesh) = jax.sharding.Mesh(np.array([$(join(ord_devs, ", "))]).reshape($(sizes_tuple)), $(names_tuple), axis_types=$(axis_types_expr))"
end

function _resolve_axis_to_mesh_names(
    ax::Symbol,
    subaxis::Union{Nothing,Dims{2}},
    mesh::Reactant.Sharding.Mesh,
    axis_bounds::Vector{Vector{Int}},
)
    k = findfirst(==(ax), mesh.axis_names)
    @assert k !== nothing "Axis $(ax) not found in mesh $(mesh.axis_names)"
    ax_str = String(ax)
    bounds = axis_bounds[k]
    if length(bounds) <= 2
        return String["'$(ax_str)'"]
    end
    names = String[]
    if subaxis === nothing
        for m in 1:(length(bounds) - 1)
            push!(names, "'$(ax_str)_sub_$(m - 1)'")
        end
    else
        pre, sz = subaxis
        target_end = pre * sz
        for m in 1:(length(bounds) - 1)
            if bounds[m] >= pre && bounds[m + 1] <= target_end
                push!(names, "'$(ax_str)_sub_$(m - 1)'")
            end
        end
    end
    return names
end

function _format_pspec_dim_for_device_put(
    pspec_d::Vector{Union{Nothing,Symbol}},
    subaxes_d::Vector{Union{Nothing,Dims{2}}},
    mesh::Reactant.Sharding.Mesh,
    axis_bounds::Vector{Vector{Int}},
)
    resolved = String[]
    for j in 1:length(pspec_d)
        ax = pspec_d[j]
        ax === nothing && continue
        append!(resolved, _resolve_axis_to_mesh_names(ax, subaxes_d[j], mesh, axis_bounds))
    end
    isempty(resolved) && return "None"
    length(resolved) == 1 && return resolved[1]
    return _py_tuple(resolved)
end

function _format_pspec_dim_raw(pspec_d::Vector{Union{Nothing,Symbol}})
    axes = String[]
    for ax in pspec_d
        ax === nothing && continue
        push!(axes, "'$(String(ax))'")
    end
    isempty(axes) && return "None"
    length(axes) == 1 && return axes[1]
    return _py_tuple(axes)
end

function _format_subaxes_dim_py(subaxes_d::Vector{Union{Nothing,Dims{2}}})
    items = String[]
    for sa in subaxes_d
        if sa === nothing
            push!(items, "None")
        else
            push!(items, "($(sa[1]), $(sa[2]))")
        end
    end
    return _py_tuple(items)
end

function _format_sdy_sharding_str(sharding_info)
    nd = length(sharding_info.partition_spec)
    dim_strs = Vector{String}(undef, nd)
    for d in 1:nd
        rev_d = nd - d + 1
        pspec_d = sharding_info.partition_spec[rev_d]
        subaxes_d = sharding_info.subaxes[rev_d]
        closed = sharding_info.is_closed[rev_d]
        prio = sharding_info.priority[rev_d]
        parts = String[]
        for j in 1:length(pspec_d)
            ax = pspec_d[j]
            ax === nothing && continue
            sa = subaxes_d[j]
            if sa === nothing
                push!(parts, "\\\"$(String(ax))\\\"")
            else
                push!(parts, "\\\"$(String(ax))\\\":($(sa[1]))$(sa[2])")
            end
        end
        if !closed
            push!(parts, "?")
        end
        prio_str = prio >= 0 ? "p$(prio)" : ""
        dim_strs[d] = "{" * join(parts, ", ") * "}" * prio_str
    end
    return "#sdy.sharding<@mesh, [" * join(dim_strs, ", ") * "]>"
end

function _format_mesh_axes_py(mesh::Reactant.Sharding.Mesh)
    pairs = String[]
    for k in 1:length(mesh.axis_names)
        push!(pairs, "('$(String(mesh.axis_names[k]))', $(Int(mesh.axis_sizes[k])))")
    end
    return _py_tuple(pairs)
end

function _format_sharding_metadata_entry(arg_name::String, sharding_info)
    if sharding_info === nothing || sharding_info.type == "NoSharding"
        return "\"$(arg_name)\": {\"type\": \"NoSharding\"}"
    elseif sharding_info.type == "Replicated"
        mesh_axes_py = _format_mesh_axes_py(sharding_info.mesh)
        return "\"$(arg_name)\": {\"type\": \"Replicated\", \"mesh_axes\": $(mesh_axes_py)}"
    elseif sharding_info.type == "NamedSharding"
        nd = length(sharding_info.partition_spec)
        pspec_items = String[]
        closed_items = String[]
        prio_items = String[]
        subaxes_items = String[]
        for d in nd:-1:1
            push!(pspec_items, _format_pspec_dim_raw(sharding_info.partition_spec[d]))
            push!(closed_items, _py_bool(sharding_info.is_closed[d]))
            push!(prio_items, string(sharding_info.priority[d]))
            push!(subaxes_items, _format_subaxes_dim_py(sharding_info.subaxes[d]))
        end
        mesh_axes_py = _format_mesh_axes_py(sharding_info.mesh)
        sdy_str = _format_sdy_sharding_str(sharding_info)
        return "\"$(arg_name)\": {\"type\": \"NamedSharding\", \"mesh_axes\": $(mesh_axes_py), \"partition_spec\": $(_py_tuple(pspec_items)), \"is_closed\": $(_py_tuple(closed_items)), \"priority\": $(_py_tuple(prio_items)), \"subaxes\": $(_py_tuple(subaxes_items)), \"sdy_sharding\": \"$(sdy_str)\"}"
    else
        error("Unsupported sharding type: $(sharding_info.type)")
    end
end

function save_inputs_npz(
    output_path::String, inputs::Dict{String,<:Union{AbstractArray,Number}}
)
    if !Serialization.serialization_supported(Val(:NPZ))
        error("`NPZ.jl` is required for saving compressed arrays. Please load it with \
               `using NPZ` and try again. If saving inputs is not required, set \
                `export_data_as_npz=false`.")
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
    export_data_as_npz::Bool=true,
    preserve_sharding::Bool=true,
)
    # Get relative paths for the Python script
    output_dir = dirname(python_path)
    mlir_rel = relpath(mlir_path, output_dir)
    input_rel = relpath(input_path, output_dir)

    # Generate argument list and documentation
    n_inputs = length(input_info)
    arg_names = Vector{String}(undef, n_inputs)
    arg_doc_lines = Vector{String}(undef, n_inputs)
    arg_size_checks = Vector{String}(undef, n_inputs)
    arg_dtype_checks = Vector{String}(undef, n_inputs)
    for i in 1:n_inputs
        info = input_info[i]
        arg_name = "arg$i"
        arg_names[i] = arg_name
        rev_shape = reverse(info.shape)
        np_dtype = Serialization.NUMPY_SIMPLE_TYPES[info.dtype]
        arg_doc_lines[i] = "        $(arg_name): Array of shape $(rev_shape) and dtype $(np_dtype). Path: $(info.path)"
        arg_size_checks[i] = "assert $(arg_name).shape == $(rev_shape), f\"Expected shape of $(arg_name) to be $(rev_shape). Got {$(arg_name).shape} (path: $(info.path))\""
        arg_dtype_checks[i] = "assert $(arg_name).dtype == np.dtype('$(np_dtype)'), f\"Expected dtype of $(arg_name) to be $(np_dtype). Got {$(arg_name).dtype} (path: $(info.path))\""
    end
    arg_list = join(arg_names, ", ")
    arg_docs = join(arg_doc_lines, "\n")

    # Generate sharding annotations if available
    has_any_sharding = false
    if preserve_sharding
        for i in 1:n_inputs
            if input_info[i].sharding !== nothing
                has_any_sharding = true
                break
            end
        end
    end

    device_put_calls = String[]
    sharding_metadata_block = ""
    if has_any_sharding
        metadata_entries = Vector{String}(undef, n_inputs)
        for i in 1:n_inputs
            metadata_entries[i] = _format_sharding_metadata_entry(
                arg_names[i], input_info[i].sharding
            )
        end
        sharding_metadata_block =
            "\n    SHARDING_METADATA = {\n        " *
            join(metadata_entries, ",\n        ") *
            ",\n    }\n"

        inserted_meshes = IdDict{Any,Tuple{String,Vector{Vector{Int}}}}()
        counter = 0
        for i in 1:n_inputs
            info = input_info[i]
            arg_name = arg_names[i]
            if info.sharding === nothing || info.sharding.type == "NoSharding"
                push!(device_put_calls, "$(arg_name) = jnp.asarray($(arg_name))")
                continue
            end

            if haskey(inserted_meshes, info.sharding.mesh)
                pymesh, axis_bounds = inserted_meshes[info.sharding.mesh]
            else
                pymesh = "mesh$counter"
                counter += 1
                axis_bounds = _compute_mesh_axis_split_bounds(
                    info.sharding.mesh, input_info
                )
                inserted_meshes[info.sharding.mesh] = (pymesh, axis_bounds)
                push!(
                    device_put_calls,
                    _build_mesh_code(pymesh, info.sharding.mesh, axis_bounds),
                )
            end

            push!(
                device_put_calls, "# Set up sharding for $(arg_name): $(info.sharding.type)"
            )

            if info.sharding.type == "Replicated"
                device_put_calls_str = "$(arg_name) = jax.device_put($(arg_name), jax.sharding.NamedSharding($(pymesh), P()))"
            elseif info.sharding.type == "NamedSharding"
                nd = length(info.sharding.partition_spec)
                pstrings = Vector{String}(undef, nd)
                for d in 1:nd
                    rev_d = nd - d + 1
                    pstrings[d] = _format_pspec_dim_for_device_put(
                        info.sharding.partition_spec[rev_d],
                        info.sharding.subaxes[rev_d],
                        info.sharding.mesh,
                        axis_bounds,
                    )
                end
                partition_spec = join(pstrings, ", ")
                device_put_calls_str = "$(arg_name) = jax.device_put($(arg_name), jax.sharding.NamedSharding($(pymesh), P($(partition_spec))))"
            else
                error("Unsupported sharding type: $(info.sharding.type)")
            end
            push!(device_put_calls, device_put_calls_str)
        end
    end

    if has_any_sharding
        inputs_to_jax_arrays = """# Apply sharding to inputs using device_put and NamedSharding
            $(join(device_put_calls, "\n    "))
        """
    else
        convert_lines = Vector{String}(undef, n_inputs)
        for i in 1:n_inputs
            convert_lines[i] = "    $(arg_names[i]) = jnp.asarray($(arg_names[i]))"
        end
        convert_str_list = join(convert_lines, "\n")
        inputs_to_jax_arrays = """
        # Convert inputs to jax arrays
        $(convert_str_list)
        """
    end

    if export_data_as_npz
        load_inputs = Vector{String}(undef, n_inputs)
        for i in 1:n_inputs
            load_inputs[i] = "npz_data['$(input_info[i].key)']"
        end

        load_inputs_fn = """
    def load_inputs():
        \"\"\"Load the example inputs that were exported from Julia.\"\"\"
        npz_data = np.load(os.path.join(_script_dir, \"$(input_rel)\"))
        inputs = [$(join(load_inputs, ", "))]
        return tuple(inputs)
        """
    else
        load_inputs_fn = """
    def generate_random_inputs(key, shape, dtype):
        key1, key2 = jax.random.split(key)
        return jax.random.normal(key1, shape, dtype), key2

    def load_inputs(prng_key = None):
        \"\"\"Load random inputs for the exported function.\"\"\"
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)

        inputs = []"""

        for i in 1:n_inputs
            info = input_info[i]
            load_inputs_fn *= """\n    tmp, prng_key = generate_random_inputs(prng_key, $(reverse(info.shape)), jnp.dtype('$(Serialization.NUMPY_SIMPLE_TYPES[info.dtype])'))\n    inputs.append(tmp)"""
        end

        load_inputs_fn *= """\n    return tuple(inputs)"""
    end

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
    $(sharding_metadata_block)
    $(load_inputs_fn)

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
