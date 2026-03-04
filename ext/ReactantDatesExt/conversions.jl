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
    @eval Base.convert(::Type{Dates.$S}, x::$T) = $S(value(x))
    @eval $T(x::Dates.$S) = $T(value(x))

    # e.g. for conversions from Int64 to ConcretePJRTNumber
    @eval Base.convert(::Type{$T{I}}, x::$T{J}) where {I,J} = $T(convert(I, value(x)))
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
    return TracedRDateTime{I}(UTInstant(TracedRMillisecond(value(dt))))
end

# Date conversion
function Base.convert(::Type{TracedRDate}, dt::Dates.Date)
    return TracedRDate(UTInstant(TracedRDay(value(dt))))
end
function Base.convert(::Type{Date}, x::TracedRDate)
    return Date(UTInstant(Dates.Day(value(x))))
end
TracedRDate(dt::Date) = convert(TracedRDate, dt)
TracedRDate{I}(dt::Date) where {I} = TracedRDate{I}(UTInstant(TracedRDay(value(dt))))

# Time conversion
function Base.convert(::Type{TracedRTime}, t::Time)
    return TracedRTime(TracedRNanosecond(value(t)))
end
Base.convert(::Type{Time}, x::TracedRTime) = Dates.Time(Dates.Nanosecond(value(x)))
TracedRTime(t::Time) = convert(TracedRTime, t)
TracedRTime{I}(t::Time) where {I} = TracedRTime{I}(TracedRNanosecond(value(t)))

# converting e.g. Int64 to ConcretePJRTNumber
function Base.convert(
    ::Type{UTInstant{P}}, uti::UTInstant{Q}
) where {P<:TracedRMillisecond,Q<:TracedRMillisecond}
    return UTInstant{P}(uti.periods)
end

Dates.datetime2julian(dt::TracedRDateTime) = (value(dt) - Dates.JULIANEPOCH) / 86400000.0
