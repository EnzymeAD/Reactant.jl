Base.print(io::IO, dt::ReactantDate) = print(io, convert(Dates.Date, dt))
Base.print(io::IO, dt::ReactantDateTime) = print(io, convert(Dates.DateTime, dt))

Base.string(t::ReactantTime) = string(convert(Dates.Time, t))
Base.print(io::IO, t::ReactantTime) = print(io, string(t))
Base.show(io::IO, ::MIME"text/plain", t::ReactantTime) = print(io, t)

function Base.show(io::IO, t::ReactantTime)
    return if get(io, :compact, false)::Bool
        print(io, t)
    else
        t_dates = convert(Dates.Time, t)
        values = [
            Dates.hour(t_dates)
            Dates.minute(t_dates)
            Dates.second(t_dates)
            Dates.millisecond(t_dates)
            Dates.microsecond(t_dates)
            Dates.nanosecond(t_dates)
        ]
        index = something(findlast(!iszero, values), 1)

        print(io, ReactantTime, "(")
        for i in 1:index
            show(io, values[i])
            i != index && print(io, ", ")
        end
        print(io, ")")
    end
end

for date_type in (:ReactantDate, :ReactantDateTime)
    # Human readable output (i.e. "2012-01-01")
    @eval Base.show(io::IO, ::MIME"text/plain", dt::$date_type) = print(io, dt)
    # Parsable output (i.e. ReactantDate("2012-01-01"))
    @eval Base.show(io::IO, dt::$date_type) = print(io, typeof(dt), "(\"", dt, "\")")
end

# _units is defined per concrete Period type in Dates for printing periods
for (T, unit) in (
    (:ReactantYear, "years"),
    (:ReactantMonth, "months"),
    (:ReactantDay, "days"),
    (:ReactantHour, "hours"),
    (:ReactantMinute, "minutes"),
    (:ReactantSecond, "seconds"),
    (:ReactantMillisecond, "milliseconds"),
    (:ReactantMicrosecond, "microseconds"),
    (:ReactantNanosecond, "nanoseconds"),
)
    @eval Dates._units(::$T) = " " * $unit
end
