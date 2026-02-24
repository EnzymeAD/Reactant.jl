module ReactantDatesExt

using Dates:
    Dates,
    value,
    DatePeriod,
    TimePeriod,
    AbstractDateTime,
    TimeType,
    UTInstant,
    Year,
    Quarter,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    DateTime,
    Date,
    Time
using Reactant: Reactant

include("types.jl")
include("accessors.jl")
include("arithmetic.jl")
include("conversions.jl")
include("reactant_bindings.jl")
end
