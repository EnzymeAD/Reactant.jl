@reactant_overlay function NNlib.conv!(y, x, w, cdims::DenseConvDims; kwargs...)
    if any(Reactant.use_overlayed_version, (y, x, w))
        overloaded_conv!(y, x, w, cdims; kwargs...)
    else
        Reactant.call_with_native(NNlib.conv!, y, x, w, cdims; kwargs...)
    end
end

@reactant_overlay function NNlib.maxpool!(y, x, pdims::NNlib.PoolDims; kwargs...)
    if any(Reactant.use_overlayed_version, (y, x))
        overloaded_maxpool!(y, x, pdims; kwargs...)
    else
        Reactant.call_with_native(NNlib.maxpool!, y, x, pdims; kwargs...)
    end
end

@reactant_overlay function NNlib.meanpool!(y, x, pdims::NNlib.PoolDims; kwargs...)
    if any(Reactant.use_overlayed_version, (y, x))
        overloaded_meanpool!(y, x, pdims; kwargs...)
    else
        Reactant.call_with_native(NNlib.meanpool!, y, x, pdims; kwargs...)
    end
end

@reactant_overlay function NNlib.∇conv_filter!(
    dw, x, dy, cdims::NNlib.DenseConvDims; kwargs...
)
    if any(Reactant.use_overlayed_version, (dw, x, dy))
        overloaded_∇conv_filter!(dw, x, dy, cdims; kwargs...)
    else
        Reactant.call_with_native(NNlib.∇conv_filter!, dw, x, dy, cdims; kwargs...)
    end
end

@reactant_overlay function NNlib.∇conv_data!(
    dx, dy, w, cdims::NNlib.DenseConvDims; kwargs...
)
    if any(Reactant.use_overlayed_version, (dx, dy, w))
        overloaded_∇conv_data!(dx, dy, w, cdims; kwargs...)
    else
        Reactant.call_with_native(NNlib.∇conv_data!, dx, dy, w, cdims; kwargs...)
    end
end

@reactant_overlay function NNlib.grid_sample(
    input::AbstractArray, grid; padding_mode=:zeros
)
    if any(Reactant.use_overlayed_version, (input, grid))
        overloaded_grid_sample(input, grid; padding_mode)
    else
        Reactant.call_with_native(NNlib.grid_sample, input, grid; padding_mode)
    end
end

@reactant_overlay function NNlib.grid_sample!(
    output::AbstractArray, input::AbstractArray, grid, padding_mode=:zeros
)
    if any(Reactant.use_overlayed_version, (output, input, grid))
        overloaded_grid_sample!(output, input, grid, padding_mode)
    else
        Reactant.call_with_native(NNlib.grid_sample!, output, input, grid, padding_mode)
    end
end

@reactant_overlay function NNlib.∇grid_sample(
    Δ::AbstractArray, input::AbstractArray, grid; padding_mode=:zeros
)
    if any(Reactant.use_overlayed_version, (Δ, input, grid))
        overloaded_∇grid_sample(Δ, input, grid; padding_mode)
    else
        Reactant.call_with_native(NNlib.∇grid_sample, Δ, input, grid; padding_mode)
    end
end

@reactant_overlay function NNlib.∇grid_sample!(
    dx::AbstractArray,
    dgrid::AbstractArray,
    Δ::AbstractArray,
    input::AbstractArray,
    grid::AbstractArray,
    padding_mode,
)
    if any(Reactant.use_overlayed_version, (dx, dgrid, Δ, input, grid))
        overloaded_∇grid_sample!(dx, dgrid, Δ, input, grid, padding_mode)
    else
        Reactant.call_with_native(
            NNlib.∇grid_sample!, dx, dgrid, Δ, input, grid, padding_mode
        )
    end
end
