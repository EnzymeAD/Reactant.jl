module ReactantFixedSizeArraysExt

using FixedSizeArrays
using Reactant
using Reactant: TracedRArray, TracedRNumber, Ops
using ReactantCore: ReactantCore

function Reactant.traced_type_inner(
    @nospecialize(_::Type{FixedSizeArrays.FixedSizeArrayDefault{T,N}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N}
    T2 = Reactant.TracedRNumber{T}
    return FixedSizeArrays.FixedSizeArrayDefault{T2,N}
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::FixedSizeArrays.FixedSizeArrayDefault{T,N}),
    @nospecialize(path),
    mode;
    kwargs...,
) where {T,N}
    shape = size(prev)
    return reshape(Reactant.make_tracer(
        seen, parent(prev), (path..., 1), mode; kwargs..., track_numbers=Number
    ), shape)
end

end
