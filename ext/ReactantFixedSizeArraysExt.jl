module ReactantFixedSizeArraysExt

using FixedSizeArrays
using Reactant
using Reactant: TracedRArray, TracedRNumber, Ops
using ReactantCore: ReactantCore

function Reactant.traced_type_inner(
    @nospecialize(_::Type{FixedSizeArrays.FixedSizeArray{T, N, Memory{I}}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T, N, I}
    T2 = Reactant.TracedRNumber{T}
    I2 = Reactant.TracedRNumber{I}
    return FixedSizeArrays.FixedSizeArray{T2, N, Memory{I2}}
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::FixedSizeArrays.FixedSizeArray{T, N, Memory{I}}),
    @nospecialize(path),
    mode; kwargs...
) where {T, N, I}
    return FixedSizeArrays.FixedSizeArray(
        Reactant.make_tracer(
            seen, parent(prev), (path..., 1), mode; kwargs..., track_numbers=Number
        )
    )
end
    
end
