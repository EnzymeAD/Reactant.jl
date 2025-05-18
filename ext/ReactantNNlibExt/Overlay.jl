@reactant_overlay @noinline function NNlib.conv!(y, x, w, cdims::DenseConvDims; kwargs...)
    if any(Reactant.use_overlayed_version, (y, x, w))
        overloaded_conv!(y, x, w, cdims; kwargs...)
    else
        Base.inferencebarrier(NNlib.conv!)(y, x, w, cdims; kwargs...)
    end
end

@reactant_overlay @noinline function NNlib.maxpool!(y, x, pdims::NNlib.PoolDims; kwargs...)
    if any(Reactant.use_overlayed_version, (y, x))
        overloaded_maxpool!(y, x, pdims; kwargs...)
    else
        Base.inferencebarrier(NNlib.maxpool!)(y, x, pdims; kwargs...)
    end
end

@reactant_overlay @noinline function NNlib.meanpool!(y, x, pdims::NNlib.PoolDims; kwargs...)
    if any(Reactant.use_overlayed_version, (y, x))
        overloaded_meanpool!(y, x, pdims; kwargs...)
    else
        Base.inferencebarrier(NNlib.meanpool!)(y, x, pdims; kwargs...)
    end
end

@reactant_overlay @noinline function NNlib.∇conv_filter!(
    dw, x, dy, cdims::NNlib.DenseConvDims; kwargs...
)
    if any(Reactant.use_overlayed_version, (dw, x, dy))
        overloaded_∇conv_filter!(dw, x, dy, cdims; kwargs...)
    else
        Base.inferencebarrier(NNlib.∇conv_filter!)(dw, x, dy, cdims; kwargs...)
    end
end

@reactant_overlay @noinline function NNlib.∇conv_data!(
    dx, dy, w, cdims::NNlib.DenseConvDims; kwargs...
)
    if any(Reactant.use_overlayed_version, (dx, dy, w))
        overloaded_∇conv_data!(dx, dy, w, cdims; kwargs...)
    else
        Base.inferencebarrier(NNlib.∇conv_data!)(dx, dy, w, cdims; kwargs...)
    end
end
