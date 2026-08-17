module ReactantStridedViewsExt

using Reactant
using ReactantCore
using Reactant: @opcall, @reactant_overlay, use_overlayed_version, call_with_native
using StridedViews: StridedViews as SV, StridedView

@reactant_overlay function SV.sreshape(x::StridedView, newsize)
    if use_overlayed_version(x)
        y = ReactantCore.materialize_traced_array(x)
        z = @opcall reshape(y, collect(newsize))
        return StridedView(z, size(z), strides(z), 0, x.op)
    else
        call_with_native(SV.sreshape, x, newsize)
    end
end

@reactant_overlay function Base.getindex(x::StridedView{T,N}, I::Vararg{SV.SliceIndex,N}) where {T,N}
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
        call_with_native(SV.sreshape, x, newsize)
    end
end

function ReactantCore.materialize_traced_array(x::StridedView)
    xp = ReactantCore.materialize_traced_array(parent(x))
    
    isview = length(x) != length(xp)
    isreshape = size(x) != size(xp)
    isperm = !issorted(strides(x))

    if isreshape && !isview && !isperm
        xp_shape = zeros(Int, ndims(x))
        for d in 1:ndims(x)-1
            xp_shape[d] = stride(x,d+1) ÷ stride(x,d)
        end
        xp_shape[end] = length(x) ÷ stride(x,ndims(x))
        xp = @opcall reshape(xp, xp_shape)
    elseif isperm || isview && isreshape
        error("Not implemented")
    end

    # x.offset is "0-indexed"
    offset = collect(Base._to_subscript_indices(xp, SV.offset(x) + 1))

    start_indices = offset
    limit_indices = collect(size(x)) + offset .- 1
    _strides = collect(strides(x))
    y = @opcall slice(xp, start_indices, limit_indices; strides=_strides)
    return y
end

end
