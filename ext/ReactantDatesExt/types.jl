import Base: ==

# Define custom DatePeriod and TimePeriod types that are parametric (and therefore tracable by Reactant)
for T in (:TracedRYear, :TracedRQuarter, :TracedRMonth, :TracedRWeek, :TracedRDay)
    @eval struct $T{I} <: DatePeriod
        value::I
        $T(v::Number) = new{typeof(v)}(v)
    end
end
for T in (
        :TracedRHour,
        :TracedRMinute,
        :TracedRSecond,
        :TracedRMillisecond,
        :TracedRMicrosecond,
        :TracedRNanosecond,
    )
    @eval struct $T{I} <: TimePeriod
        value::I
        $T(v::Number) = new{typeof(v)}(v)
    end
end

# custom DateTime, Date, Time types analogues to those defined in Dates.jl
struct TracedRDateTime{I} <: AbstractDateTime
    instant::UTInstant{TracedRMillisecond{I}}
end

struct TracedRDate{I} <: TimeType
    instant::UTInstant{TracedRDay{I}}
end

struct TracedRTime{I} <: TimeType
    instant::TracedRNanosecond{I}
end

# value accessor for TracedR* period types (Period already defines value(x::Period) = x.value,
# which works since TracedR* periods <: Period, but we define these for TracedR* TimeTypes)
Dates.value(dt::TracedRDateTime) = dt.instant.periods.value
Dates.value(dt::TracedRDate) = dt.instant.periods.value
Dates.value(t::TracedRTime) = t.instant.value

# Traits
Base.isfinite(::Union{Type{T}, T}) where {T <: TracedRDateTime} = true
Base.isfinite(::Union{Type{T}, T}) where {T <: TracedRDate} = true
Base.isfinite(::Union{Type{T}, T}) where {T <: TracedRTime} = true

# eps
Base.eps(::Type{TracedRDateTime}) = TracedRMillisecond(1)
Base.eps(::Type{TracedRDate}) = TracedRDay(1)
Base.eps(::Type{TracedRTime}) = TracedRNanosecond(1)
Base.eps(::T) where {T <: TracedRDateTime} = eps(T)
Base.eps(::T) where {T <: TracedRDate} = eps(T)
Base.eps(::T) where {T <: TracedRTime} = eps(T)

# zero
Base.zero(::Type{TracedRDateTime}) = TracedRMillisecond(0)
Base.zero(::Type{TracedRDate}) = TracedRDay(0)
Base.zero(::Type{TracedRTime}) = TracedRNanosecond(0)
Base.zero(::T) where {T <: TracedRDateTime} = zero(T)
Base.zero(::T) where {T <: TracedRDate} = zero(T)
Base.zero(::T) where {T <: TracedRTime} = zero(T)

# isless and == for TracedR* TimeTypes
Base.isless(x::TracedRDateTime, y::TracedRDateTime) = isless(value(x), value(y))
Base.isless(x::TracedRDate, y::TracedRDate) = isless(value(x), value(y))
Base.isless(x::TracedRTime, y::TracedRTime) = isless(value(x), value(y))
(==)(x::TracedRDateTime, y::TracedRDateTime) = (==)(value(x), value(y))
(==)(x::TracedRDate, y::TracedRDate) = (==)(value(x), value(y))
(==)(x::TracedRTime, y::TracedRTime) = (==)(value(x), value(y))

# TracedRDate-TracedRDateTime promotion
Base.promote_rule(::Type{TracedRDate}, ::Type{TracedRDateTime}) = TracedRDateTime
