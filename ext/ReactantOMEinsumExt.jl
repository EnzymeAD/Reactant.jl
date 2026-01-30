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

@reactant_overlay @noinline function OMEinsum.tensorpermute!(
    C::AbstractArray{T,N}, A::AbstractArray{T,N}, perm, sx, sy
) where {T,N}
    if use_overlayed_version(A)
        @assert use_overlayed_version(C)
        permv = collect(perm)
        res = sy * C + sx * @opcall transpose(A, permv)
        C.mlir_data = res.mlir_data
        return C
    else
        return Reactant.call_with_native(OMEinsum.tensorpermute!, C, A, perm, sx, sy)
    end
end

@reactant_overlay @noinline function OMEinsum.einsum!(
    ixs, iy, @nospecialize(xs::NTuple{2,Any}), @nospecialize(y), sx, sy, size_dict
)
    if looped_any(use_overlayed_version, xs)
        @assert use_overlayed_version(y)

        # shortcut for scalar multiplication
        if looped_any(x -> x isa Number, xs)
            c = sy * y + sx * xs[1] * xs[2]
            y.mlir_data = c.mlir_data
            return y
        end

        LT = keytype(size_dict)
        a, b = xs
        ia, ib = collect.(LT, ixs)
        iyv = collect(LT, iy)
        inner, a_outer, b_outer, batch = _analyze_binary_input(ia, ib, iyv)

        contracting_dimensions = (
            Int[findfirst(==(i), ia) for i in inner],
            Int[findfirst(==(i), ib) for i in inner],
        )
        batching_dimensions = (
            Int[findfirst(==(i), ia) for i in batch],
            Int[findfirst(==(i), ib) for i in batch],
        )

        c = @opcall dot_general(a, b; contracting_dimensions, batching_dimensions)

        # permute dims to match iy
        ic = vcat(batch, a_outer, b_outer)
        perm = Int[findfirst(==(i), ic) for i in iyv]
        c = @opcall transpose(c, perm)
        @assert size(c) == size(y)
        @assert eltype(c) == eltype(y)

        # just like GEMM, we do: y = sy * y + sx * c
        c = sy * y + sx * c
        y.mlir_data = c.mlir_data
        return y
    else
        return Reactant.call_with_native(
            OMEinsum.einsum!, ixs, iy, xs, y, sx, sy, size_dict
        )
    end
end

end
