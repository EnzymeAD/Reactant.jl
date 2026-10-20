module ReactantStridedViewsExt

using Reactant
using ReactantCore
using Reactant: @opcall, @reactant_overlay, use_overlayed_version, call_with_native
using StridedViews: StridedViews as SV, StridedView

@reactant_overlay function Base.getindex(
    x::StridedView{T,N}, I::Vararg{SV.SliceIndex,N}
) where {T,N}
    if use_overlayed_version(x)
        y = ReactantCore.materialize_traced_array(x)

        start_indices = zeros(Int, N)
        limit_indices = zeros(Int, N)
        _strides = ones(Int, N)

        for (d, sliceind) in enumerate(I)
            start_indices[d] = first(sliceind)
            limit_indices[d] = last(sliceind)
            if sliceind isa AbstractRange
                _strides[d] = Base.step(sliceind)
            end
        end

        z = @opcall slice(y, start_indices, limit_indices; strides=_strides)
        return StridedView(z, size(z), strides(z), 0, x.op)
    else
        call_with_native(Base.getindex, x, I...)
    end
end

function ReactantCore.materialize_traced_array(x::StridedView)
    xp = ReactantCore.materialize_traced_array(parent(x))

    isview = length(x) != length(xp)
    isreshape = size(x) != size(xp)
    isperm = !issorted(strides(x))

    if isreshape && !isview && !isperm
        xp_shape = zeros(Int, ndims(x))
        for d in 1:(ndims(x) - 1)
            xp_shape[d] = stride(x, d + 1) ÷ stride(x, d)
        end
        xp_shape[end] = length(x) ÷ stride(x, ndims(x))
        xp = @opcall reshape(xp, xp_shape)
    elseif isperm && !isview
        xp_strides = collect(strides(x))
        sp = invperm(sortperm(xp_strides))
        xp = permutedims(xp, sp)
    end

    # x.offset is "0-indexed"
    offset = collect(Base._to_subscript_indices(xp, SV.offset(x) + 1))

    start_indices = offset
    limit_indices = collect(size(x)) + offset .- 1
    y = @opcall slice(xp, start_indices, limit_indices)
    return y
end

# write through the view: store every element at the position of the parent it addresses
function Reactant.TracedUtils.set_mlir_data!(
    x::StridedView{<:Reactant.TracedRNumber,N,<:Reactant.TracedRArray}, data
) where {N}
    x.op === identity ||
        x.op === conj ||
        throw(ArgumentError("cannot write through a StridedView with op $(x.op)"))
    v = Reactant.TracedRArray{Reactant.unwrapped_eltype(x),N}((), data, size(x))
    x.op === conj && (v = @opcall conj(v))
    # 1-based linear index into the parent of each element of the view
    idx = broadcast(
        +,
        SV.offset(x) + 1,
        ntuple(N) do d
            reshape((0:(size(x, d) - 1)) .* stride(x, d), ntuple(k -> k == d ? size(x, d) : 1, N))
        end...,
    )
    parent(x)[vec(collect(idx))] = vec(v)
    return x
end

end
