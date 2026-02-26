# Bindings to automatically convert Dates.jl types to traced version

reactant_type_prefix = "TracedR"

for T in (
    :Year,
    :Quarter,
    :Month,
    :Week,
    :Day,
    :Hour,
    :Minute,
    :Second,
    :Millisecond,
    :Microsecond,
    :Nanosecond,
    :DateTime,
    :Date,
    :Time,
)
    reactant_type = Symbol(reactant_type_prefix, T)

    @eval function Reactant.make_tracer(
        seen,
        @nospecialize(prev::$T),
        @nospecialize(path),
        mode;
        @nospecialize(track_numbers::Type = Union{}),
        @nospecialize(sharding = Reactant.Sharding.NoSharding()),
        @nospecialize(runtime),
        kwargs...,
    )
        RT = Reactant.traced_type($T, Val(mode), track_numbers, sharding, runtime)
        return RT(prev)
    end

    @eval function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:$T}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
        # all Dates types are hard coded to have Int64 fields
        NF = Reactant.traced_type_inner(Int64, seen, mode, track_numbers, ndevices, runtime)
        return $(reactant_type){NF}
    end
end
