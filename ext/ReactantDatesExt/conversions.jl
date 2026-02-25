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
    @eval Base.convert(::Type{Dates.$S}, x::$T) = Dates.$S(value(x))
    @eval $T(x::Dates.$S) = $T(value(x))
end

# DateTime conversion
function Base.convert(::Type{TracedRDateTime}, dt::Dates.DateTime)
    return TracedRDateTime(UTInstant(TracedRMillisecond(value(dt))))
end
function Base.convert(::Type{Dates.DateTime}, x::TracedRDateTime)
    return Dates.DateTime(UTInstant(Dates.Millisecond(value(x))))
end
TracedRDateTime(dt::Dates.DateTime) = convert(TracedRDateTime, dt)
TracedRDateTime{I}(dt::Dates.DateTime) where {I} =
    TracedRDateTime{I}(UTInstant(TracedRMillisecond(value(dt))))

# Date conversion
function Base.convert(::Type{TracedRDate}, dt::Dates.Date)
    return TracedRDate(UTInstant(TracedRDay(value(dt))))
end
Base.convert(::Type{Dates.Date}, x::TracedRDate) =
    Dates.Date(UTInstant(Dates.Day(value(x))))
TracedRDate(dt::Dates.Date) = convert(TracedRDate, dt)
TracedRDate{I}(dt::Dates.Date) where {I} = TracedRDate{I}(UTInstant(TracedRDay(value(dt))))

# Time conversion
function Base.convert(::Type{TracedRTime}, t::Dates.Time)
    return TracedRTime(TracedRNanosecond(value(t)))
end
Base.convert(::Type{Dates.Time}, x::TracedRTime) = Dates.Time(Dates.Nanosecond(value(x)))
TracedRTime(t::Dates.Time) = convert(TracedRTime, t)
TracedRTime{I}(t::Dates.Time) where {I} = TracedRTime{I}(TracedRNanosecond(value(t)))
