module ReactantDatesExt

using Dates:
    Dates,
    value,
    DatePeriod,
    TimePeriod,
    AbstractDateTime,
    TimeType,
    UTInstant,
    DateTime,
    Date,
    Time
using Reactant: Reactant, @trace
using Adapt: Adapt

include("types.jl")
include("accessors.jl")
include("adjusters.jl")
include("io.jl")
include("arithmetic.jl")
include("conversions.jl")
include("adapt.jl")
include("reactant_bindings.jl")
end
