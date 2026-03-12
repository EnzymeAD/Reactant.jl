import Base: ==

# Define custom DatePeriod and TimePeriod types that are parametric (and therefore tracable by Reactant)
for T in (:ReactantYear, :ReactantQuarter, :ReactantMonth, :ReactantWeek, :ReactantDay)
    @eval struct $T{I} <: DatePeriod
        value::I
        $T(v::Number) = new{typeof(v)}(v)
    end
end
for T in (
    :ReactantHour,
    :ReactantMinute,
    :ReactantSecond,
    :ReactantMillisecond,
    :ReactantMicrosecond,
    :ReactantNanosecond,
)
    @eval struct $T{I} <: TimePeriod
        value::I
        $T(v::Number) = new{typeof(v)}(v)
    end
end

# custom DateTime, Date, Time types analogues to those defined in Dates.jl
struct ReactantDateTime{I} <: AbstractDateTime
    instant::UTInstant{ReactantMillisecond{I}}
end

struct ReactantDate{I} <: TimeType
    instant::UTInstant{ReactantDay{I}}
end

struct ReactantTime{I} <: TimeType
    instant::ReactantNanosecond{I}
end

# value accessor for Reactant* period types (Period already defines value(x::Period) = x.value,
# which works since Reactant* periods <: Period, but we define these for Reactant* TimeTypes)
Dates.value(dt::ReactantDateTime) = dt.instant.periods.value
Dates.value(dt::ReactantDate) = dt.instant.periods.value
Dates.value(t::ReactantTime) = t.instant.value

# Traits
Base.isfinite(::Union{Type{T},T}) where {T<:ReactantDateTime} = true
Base.isfinite(::Union{Type{T},T}) where {T<:ReactantDate} = true
Base.isfinite(::Union{Type{T},T}) where {T<:ReactantTime} = true

# eps
Base.eps(::Type{ReactantDateTime}) = ReactantMillisecond(1)
Base.eps(::Type{ReactantDate}) = ReactantDay(1)
Base.eps(::Type{ReactantTime}) = ReactantNanosecond(1)
Base.eps(::T) where {T<:ReactantDateTime} = eps(T)
Base.eps(::T) where {T<:ReactantDate} = eps(T)
Base.eps(::T) where {T<:ReactantTime} = eps(T)

# zero
Base.zero(::Type{ReactantDateTime}) = ReactantMillisecond(0)
Base.zero(::Type{ReactantDate}) = ReactantDay(0)
Base.zero(::Type{ReactantTime}) = ReactantNanosecond(0)
Base.zero(::T) where {T<:ReactantDateTime} = zero(T)
Base.zero(::T) where {T<:ReactantDate} = zero(T)
Base.zero(::T) where {T<:ReactantTime} = zero(T)

# isless and == for Reactant* TimeTypes
Base.isless(x::ReactantDateTime, y::ReactantDateTime) = isless(value(x), value(y))
Base.isless(x::ReactantDate, y::ReactantDate) = isless(value(x), value(y))
Base.isless(x::ReactantTime, y::ReactantTime) = isless(value(x), value(y))
(==)(x::ReactantDateTime, y::ReactantDateTime) = (==)(value(x), value(y))
(==)(x::ReactantDate, y::ReactantDate) = (==)(value(x), value(y))
(==)(x::ReactantTime, y::ReactantTime) = (==)(value(x), value(y))

# ReactantDate-ReactantDateTime promotion
Base.promote_rule(::Type{ReactantDate}, ::Type{ReactantDateTime}) = ReactantDateTime
