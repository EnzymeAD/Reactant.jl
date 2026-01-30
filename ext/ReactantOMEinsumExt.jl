module ReactantOMEinsumExt

using Reactant
using Reactant: @reactant_overlay, looped_any, use_overlayed_version, @opcall
using OMEinsum
using OMEinsum: _analyze_binary_input

@reactant_overlay @noinline function OMEinsum.get_output_array(xs, size, fillzero)
    # we ignore fillzero here, as it's easier for us to zero-initialize arrays
    if looped_any(use_overlayed_version, xs)
        T = promote_type(map(eltype, xs)...)
        return @opcall fill(zero(T), size)
    else
        return Reactant.call_with_native(OMEinsum.get_output_array, xs, size, fillzero)
    end
end

@reactant_overlay @noinline function OMEinsum.einsum!(ixs, iy, @nospecialize(xs::NTuple{1,Any}), @nospecialize(y), sx, sy, size_dict)
    if looped_any(use_overlayed_version, xs)
        @assert use_overlayed_version(y)
        # TODO
        error("unary einsum support not implemented yet")
    else
        return Reactant.call_with_native(OMEinsum.einsum!, ixs, iy, xs, y, sx, sy, size_dict)
    end
end

@reactant_overlay @noinline function OMEinsum.einsum!(ixs, iy, @nospecialize(xs::NTuple{2,Any}), @nospecialize(y), sx, sy, size_dict)
    if looped_any(use_overlayed_version, xs)
        @assert use_overlayed_version(y)

        a, b = xs
        ia, ib = ixs
        inner, _, _, batch = _analyze_binary_input(ia, ib, iy)

        contracting_dimensions = (
            [findfirst(==(i), ia) for i in inner],
            [findfirst(==(i), ib) for i in inner],
        )
        batching_dimensions = (
            [findfirst(==(i), ia) for i in batch],
            [findfirst(==(i), ib) for i in batch],
        )

        c = @opcall dot_general(a, b; contracting_dimensions, batching_dimensions)
        @assert size(c) == size(y)
        @assert eltype(c) == eltype(y)

        y.mlir_data = c.mlir_data
        return y
    else
        return Reactant.call_with_native(OMEinsum.einsum!, ixs, iy, xs, y, sx, sy, size_dict)
    end
end

end
