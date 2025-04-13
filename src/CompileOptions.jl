# TODO: make the other optimize options into a struct as well
@kwdef struct OptimizeCommunicationOptions
    periodic_concat::Int = 0
    rotate_comm::Int = 0
    rotate_to_pad_comm::Int = 1
    wrap_comm::Int = 0
    extend_comm::Int = 0
    dus_to_pad_manual_comp_comm::Int = 2
    dus_to_pad_comm::Int = 1
    concat_two_operands_comm::Int = 0
    concat_to_pad_comm::Int = 1
    extend_to_pad_comm::Int = 1
    wrap_to_pad_comm::Int = 1
end

function Base.String(options::OptimizeCommunicationOptions)
    return (
        "optimize-communication{" *
        join(["$(f)=$(getfield(options, f))" for f in fieldnames(typeof(options))], " ") *
        "}"
    )
end
