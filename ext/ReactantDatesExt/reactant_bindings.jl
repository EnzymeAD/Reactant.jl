# Bindings to automatically convert Dates.jl types to traced version

reactant_type_prefix = "Reactant"

# no converts for now

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
        @nospecialize(prev::Dates.$T),
        @nospecialize(path),
        mode;
        @nospecialize(track_numbers::Type = Union{}),
        @nospecialize(sharding = Reactant.Sharding.NoSharding()),
        @nospecialize(runtime),
        kwargs...,
    )
        if mode == Reactant.ArrayToConcrete
            RT = Reactant.traced_type(Dates.$T, Val(mode), track_numbers, sharding, runtime)
            return RT(prev)
        else
            return prev
        end
    end

    @eval function Reactant.traced_type_inner(
        @nospecialize(T::Type{<:Dates.$T}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
        if mode == Reactant.ArrayToConcrete
            # all Dates types are hard coded to have Int64 fields
            NF = Reactant.traced_type_inner(Int64, seen, mode, track_numbers, ndevices, runtime)
            return $(reactant_type){NF}
        else
            return T
        end
    end
end
