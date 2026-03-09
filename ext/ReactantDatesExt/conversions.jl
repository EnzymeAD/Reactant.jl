# Straight conversions from Dates.jl types to TracedR* types

# Period conversions
for (S, T) in (
    (:Year, :TracedRYear),
    (:Quarter, :TracedRQuarter),
    (:Month, :TracedRMonth),
    (:Week, :TracedRWeek),
    (:Day, :TracedRDay),
    (:Hour, :TracedRHour),
    (:Minute, :TracedRMinute),
    (:Second, :TracedRSecond),
    (:Millisecond, :TracedRMillisecond),
    (:Microsecond, :TracedRMicrosecond),
    (:Nanosecond, :TracedRNanosecond),
)
    @eval Base.convert(::Type{$T}, x::Dates.$S) = $T(value(x))
    @eval Base.convert(::Type{$T{Int64}}, x::Dates.$S) = $T(value(x))
    @eval Base.convert(::Type{$T{I}}, x::Dates.$S) where {I} = $T(convert(I, value(x)))
    @eval Base.convert(::Type{Dates.$S}, x::$T) = Dates.$S(value(x))
    @eval $T(x::Dates.$S) = $T(value(x))

    # e.g. for conversions from Int64 to ConcretePJRTNumber
    @eval Base.convert(::Type{$T{I}}, x::$T{J}) where {I,J} = $T(convert(I, value(x)))
end

# Cross-period conversions: Dates.Source → TracedRTarget (where Source ≠ Target)
# Uses Dates' own conversion logic first, then converts to the TracedR type.
# This covers all pairs that Dates.convert supports natively.
const _CROSS_PERIOD_PAIRS = (
    # DatePeriod → DatePeriod
    (:Year, :Quarter, :TracedRQuarter),
    (:Year, :Month, :TracedRMonth),
    (:Quarter, :Month, :TracedRMonth),
    (:Week, :Day, :TracedRDay),
    # DatePeriod → TimePeriod
    (:Week, :Hour, :TracedRHour),
    (:Week, :Minute, :TracedRMinute),
    (:Week, :Second, :TracedRSecond),
    (:Week, :Millisecond, :TracedRMillisecond),
    (:Week, :Microsecond, :TracedRMicrosecond),
    (:Week, :Nanosecond, :TracedRNanosecond),
    (:Day, :Hour, :TracedRHour),
    (:Day, :Minute, :TracedRMinute),
    (:Day, :Second, :TracedRSecond),
    (:Day, :Millisecond, :TracedRMillisecond),
    (:Day, :Microsecond, :TracedRMicrosecond),
    (:Day, :Nanosecond, :TracedRNanosecond),
    # TimePeriod → TimePeriod
    (:Hour, :Minute, :TracedRMinute),
    (:Hour, :Second, :TracedRSecond),
    (:Hour, :Millisecond, :TracedRMillisecond),
    (:Hour, :Microsecond, :TracedRMicrosecond),
    (:Hour, :Nanosecond, :TracedRNanosecond),
    (:Minute, :Second, :TracedRSecond),
    (:Minute, :Millisecond, :TracedRMillisecond),
    (:Minute, :Microsecond, :TracedRMicrosecond),
    (:Minute, :Nanosecond, :TracedRNanosecond),
    (:Second, :Millisecond, :TracedRMillisecond),
    (:Second, :Microsecond, :TracedRMicrosecond),
    (:Second, :Nanosecond, :TracedRNanosecond),
    (:Millisecond, :Microsecond, :TracedRMicrosecond),
    (:Millisecond, :Nanosecond, :TracedRNanosecond),
    (:Microsecond, :Nanosecond, :TracedRNanosecond),
)

for (S, T, TR) in _CROSS_PERIOD_PAIRS
    @eval function Base.convert(::Type{$TR}, x::Dates.$S)
        return convert($TR, convert(Dates.$T, x))
    end
    @eval function Base.convert(::Type{$TR{I}}, x::Dates.$S) where {I}
        return convert($TR{I}, convert(Dates.$T, x))
    end
end

# DateTime conversion
function Base.convert(::Type{TracedRDateTime}, dt::Dates.DateTime)
    return TracedRDateTime(UTInstant(TracedRMillisecond(value(dt))))
end
function Base.convert(::Type{DateTime}, x::TracedRDateTime)
    return DateTime(UTInstant(Dates.Millisecond(value(x))))
end
TracedRDateTime(dt::DateTime) = convert(TracedRDateTime, dt)
function TracedRDateTime{I}(dt::DateTime) where {I}
    return TracedRDateTime{I}(UTInstant(TracedRMillisecond(convert(I, value(dt)))))
end
function Base.convert(::Type{TracedRDateTime{I}}, dt::Dates.DateTime) where {I}
    return TracedRDateTime{I}(dt)
end

# Date conversion
function Base.convert(::Type{TracedRDate}, dt::Dates.Date)
    return TracedRDate(UTInstant(TracedRDay(value(dt))))
end
function Base.convert(::Type{Date}, x::TracedRDate)
    return Date(UTInstant(Dates.Day(value(x))))
end
TracedRDate(dt::Date) = convert(TracedRDate, dt)
function TracedRDate{I}(dt::Date) where {I}
    return TracedRDate{I}(UTInstant(TracedRDay(convert(I, value(dt)))))
end
function Base.convert(::Type{TracedRDate{I}}, dt::Dates.Date) where {I}
    return TracedRDate{I}(dt)
end

# Time conversion
function Base.convert(::Type{TracedRTime}, t::Time)
    return TracedRTime(TracedRNanosecond(value(t)))
end
Base.convert(::Type{Time}, x::TracedRTime) = Dates.Time(Dates.Nanosecond(value(x)))
TracedRTime(t::Time) = convert(TracedRTime, t)
TracedRTime{I}(t::Time) where {I} = TracedRTime{I}(TracedRNanosecond(convert(I, value(t))))
Base.convert(::Type{TracedRTime{I}}, t::Dates.Time) where {I} = TracedRTime{I}(t)

# converting e.g. Int64 to ConcretePJRTNumber
function Base.convert(
    ::Type{UTInstant{P}}, uti::UTInstant{Q}
) where {P<:TracedRMillisecond,Q<:TracedRMillisecond}
    return UTInstant{P}(uti.periods)
end

Dates.datetime2julian(dt::TracedRDateTime) = (value(dt) - Dates.JULIANEPOCH) / 86400000.0
