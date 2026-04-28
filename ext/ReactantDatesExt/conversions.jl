# Straight conversions from Dates.jl types to Reactant* types

# Period conversions
for (S, T) in (
    (:Year, :ReactantYear),
    (:Quarter, :ReactantQuarter),
    (:Month, :ReactantMonth),
    (:Week, :ReactantWeek),
    (:Day, :ReactantDay),
    (:Hour, :ReactantHour),
    (:Minute, :ReactantMinute),
    (:Second, :ReactantSecond),
    (:Millisecond, :ReactantMillisecond),
    (:Microsecond, :ReactantMicrosecond),
    (:Nanosecond, :ReactantNanosecond),
)
    @eval Base.convert(::Type{$T}, x::Dates.$S) = $T(value(x))
    @eval Base.convert(::Type{$T{Int64}}, x::Dates.$S) = $T(value(x))
    @eval Base.convert(::Type{$T{I}}, x::Dates.$S) where {I} = $T(convert(I, value(x)))
    @eval Base.convert(::Type{Dates.$S}, x::$T) = Dates.$S(value(x))
    @eval $T(x::Dates.$S) = $T(value(x))

    # e.g. for conversions from Int64 to ConcretePJRTNumber
    @eval Base.convert(::Type{$T{I}}, x::$T{J}) where {I,J} = $T(convert(I, value(x)))
end

# Cross-period conversions: Dates.Source → ReactantTarget (where Source ≠ Target)
# Uses Dates' own conversion logic first, then converts to the Reactant type.
# This covers all pairs that Dates.convert supports natively.
const _CROSS_PERIOD_PAIRS = (
    # DatePeriod → DatePeriod (downward: larger → smaller)
    (:Year, :Quarter, :ReactantQuarter),
    (:Year, :Month, :ReactantMonth),
    (:Quarter, :Month, :ReactantMonth),
    (:Week, :Day, :ReactantDay),
    # DatePeriod → DatePeriod (upward: smaller → larger)
    (:Month, :Quarter, :ReactantQuarter),
    (:Month, :Year, :ReactantYear),
    (:Quarter, :Year, :ReactantYear),
    (:Day, :Week, :ReactantWeek),
    # DatePeriod → TimePeriod (downward)
    (:Week, :Hour, :ReactantHour),
    (:Week, :Minute, :ReactantMinute),
    (:Week, :Second, :ReactantSecond),
    (:Week, :Millisecond, :ReactantMillisecond),
    (:Week, :Microsecond, :ReactantMicrosecond),
    (:Week, :Nanosecond, :ReactantNanosecond),
    (:Day, :Hour, :ReactantHour),
    (:Day, :Minute, :ReactantMinute),
    (:Day, :Second, :ReactantSecond),
    (:Day, :Millisecond, :ReactantMillisecond),
    (:Day, :Microsecond, :ReactantMicrosecond),
    (:Day, :Nanosecond, :ReactantNanosecond),
    # TimePeriod → DatePeriod (upward)
    (:Hour, :Day, :ReactantDay),
    (:Hour, :Week, :ReactantWeek),
    (:Minute, :Day, :ReactantDay),
    (:Minute, :Week, :ReactantWeek),
    (:Second, :Day, :ReactantDay),
    (:Second, :Week, :ReactantWeek),
    (:Millisecond, :Day, :ReactantDay),
    (:Millisecond, :Week, :ReactantWeek),
    (:Microsecond, :Day, :ReactantDay),
    (:Microsecond, :Week, :ReactantWeek),
    (:Nanosecond, :Day, :ReactantDay),
    (:Nanosecond, :Week, :ReactantWeek),
    # TimePeriod → TimePeriod (downward: larger → smaller)
    (:Hour, :Minute, :ReactantMinute),
    (:Hour, :Second, :ReactantSecond),
    (:Hour, :Millisecond, :ReactantMillisecond),
    (:Hour, :Microsecond, :ReactantMicrosecond),
    (:Hour, :Nanosecond, :ReactantNanosecond),
    (:Minute, :Second, :ReactantSecond),
    (:Minute, :Millisecond, :ReactantMillisecond),
    (:Minute, :Microsecond, :ReactantMicrosecond),
    (:Minute, :Nanosecond, :ReactantNanosecond),
    (:Second, :Millisecond, :ReactantMillisecond),
    (:Second, :Microsecond, :ReactantMicrosecond),
    (:Second, :Nanosecond, :ReactantNanosecond),
    (:Millisecond, :Microsecond, :ReactantMicrosecond),
    (:Millisecond, :Nanosecond, :ReactantNanosecond),
    (:Microsecond, :Nanosecond, :ReactantNanosecond),
    # TimePeriod → TimePeriod (upward: smaller → larger)
    (:Nanosecond, :Microsecond, :ReactantMicrosecond),
    (:Nanosecond, :Millisecond, :ReactantMillisecond),
    (:Nanosecond, :Second, :ReactantSecond),
    (:Nanosecond, :Minute, :ReactantMinute),
    (:Nanosecond, :Hour, :ReactantHour),
    (:Microsecond, :Millisecond, :ReactantMillisecond),
    (:Microsecond, :Second, :ReactantSecond),
    (:Microsecond, :Minute, :ReactantMinute),
    (:Microsecond, :Hour, :ReactantHour),
    (:Millisecond, :Second, :ReactantSecond),
    (:Millisecond, :Minute, :ReactantMinute),
    (:Millisecond, :Hour, :ReactantHour),
    (:Second, :Minute, :ReactantMinute),
    (:Second, :Hour, :ReactantHour),
    (:Minute, :Hour, :ReactantHour),
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
function Base.convert(::Type{ReactantDateTime}, dt::Dates.DateTime)
    return ReactantDateTime(UTInstant(ReactantMillisecond(value(dt))))
end
function Base.convert(::Type{DateTime}, x::ReactantDateTime)
    return DateTime(UTInstant(Dates.Millisecond(value(x))))
end
ReactantDateTime(dt::DateTime) = convert(ReactantDateTime, dt)
function ReactantDateTime{I}(dt::DateTime) where {I}
    return ReactantDateTime{I}(UTInstant(ReactantMillisecond(convert(I, value(dt)))))
end
function Base.convert(::Type{ReactantDateTime{I}}, dt::Dates.DateTime) where {I}
    return ReactantDateTime{I}(dt)
end
function Base.convert(::Type{ReactantDateTime}, x::Dates.Millisecond)
    return ReactantDateTime(Dates.UTInstant(ReactantMillisecond(Dates.value(x))))
end
function Base.convert(::Type{Dates.Millisecond}, dt::ReactantDateTime)
    return Dates.Millisecond(Dates.value(dt))
end

# Date conversion
function Base.convert(::Type{ReactantDate}, dt::Dates.Date)
    return ReactantDate(UTInstant(ReactantDay(value(dt))))
end
function Base.convert(::Type{Date}, x::ReactantDate)
    return Date(UTInstant(Dates.Day(value(x))))
end
ReactantDate(dt::Date) = convert(ReactantDate, dt)
function ReactantDate{I}(dt::Date) where {I}
    return ReactantDate{I}(UTInstant(ReactantDay(convert(I, value(dt)))))
end
function Base.convert(::Type{ReactantDate{I}}, dt::Dates.Date) where {I}
    return ReactantDate{I}(dt)
end
function Base.convert(::Type{ReactantDate}, x::Dates.Day)
    return ReactantDate(Dates.UTInstant(ReactantDay(Dates.value(x))))
end
Base.convert(::Type{Dates.Day}, dt::ReactantDate) = Dates.Day(Dates.value(dt))

# Time conversion
function Base.convert(::Type{ReactantTime}, t::Time)
    return ReactantTime(ReactantNanosecond(value(t)))
end
Base.convert(::Type{Time}, x::ReactantTime) = Dates.Time(Dates.Nanosecond(value(x)))
ReactantTime(t::Time) = convert(ReactantTime, t)
function ReactantTime{I}(t::Time) where {I}
    return ReactantTime{I}(ReactantNanosecond(convert(I, value(t))))
end
Base.convert(::Type{ReactantTime{I}}, t::Dates.Time) where {I} = ReactantTime{I}(t)

# converting e.g. Int64 to ConcretePJRTNumber
function Base.convert(
    ::Type{UTInstant{P}}, uti::UTInstant{Q}
) where {P<:ReactantMillisecond,Q<:ReactantMillisecond}
    return UTInstant{P}(uti.periods)
end

Dates.datetime2julian(dt::ReactantDateTime) = (value(dt) - Dates.JULIANEPOCH) / 86400000.0

# conversions within ReactantDatesExt types
function Base.convert(::Type{ReactantDateTime}, dt::ReactantDate)
    return ReactantDateTime(
        Dates.UTInstant(ReactantMillisecond(Dates.value(dt) * 86400000))
    )
end
function Base.convert(::Type{ReactantDate}, dt::ReactantDateTime)
    return ReactantDate(Dates.UTInstant(ReactantDay(div(Dates.value(dt), 86400000))))
end
function Base.convert(::Type{ReactantTime}, dt::ReactantDateTime)
    return ReactantTime(ReactantNanosecond((Dates.value(dt) % 86400000) * 1000000))
end
