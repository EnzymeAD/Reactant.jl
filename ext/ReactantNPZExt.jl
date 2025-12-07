module ReactantNPZExt

using NPZ: npzwrite
using Reactant.Serialization: Serialization, EnzymeJAX

Serialization.serialization_supported(::Val{:NPZ}) = true

# Helper function to save all input data to a single NPZ file
function EnzymeJAX.save_inputs_npz_impl(
    output_path::String, inputs::Dict{String,<:Union{AbstractArray,Number}}
)
    # Transpose arrays for Python/NumPy (row-major vs column-major)
    transposed_inputs = Dict{String,Union{AbstractArray,Number}}()
    for (name, arr) in inputs
        transposed_inputs[name] =
            arr isa Number ? arr : permutedims(arr, reverse(1:ndims(arr)))
    end

    # Save all inputs to a single NPZ file with compression
    npzwrite(output_path, transposed_inputs)
    return output_path
end

end  # module
