import Base: +, -, fld, mod, div
import Dates:
    Period,
    yearmonthday,
    daysinmonth,
    hour,
    minute,
    second,
    millisecond,
    toms,
    tons,
    days,
    monthwrap,
    yearwrap,
    totaldays

# toms (to milliseconds) for traced period types
toms(c::ReactantNanosecond) = div(value(c), 1000000, RoundNearest)
toms(c::ReactantMicrosecond) = div(value(c), 1000, RoundNearest)
toms(c::ReactantMillisecond) = value(c)
toms(c::ReactantSecond) = 1000 * value(c)
toms(c::ReactantMinute) = 60000 * value(c)
toms(c::ReactantHour) = 3600000 * value(c)
toms(c::ReactantDay) = 86400000 * value(c)
toms(c::ReactantWeek) = 604800000 * value(c)

# tons (to nanoseconds) for traced period types
tons(c::ReactantNanosecond) = value(c)
tons(c::ReactantMicrosecond) = value(c) * 1000
tons(c::ReactantMillisecond) = value(c) * 1000000
tons(c::ReactantSecond) = value(c) * 1000000000
tons(c::ReactantMinute) = value(c) * 60000000000
tons(c::ReactantHour) = value(c) * 3600000000000

# days for traced period types
days(c::ReactantDay) = value(c)
days(c::ReactantWeek) = 7 * value(c)

# ReactantDateTime/ReactantDate-ReactantYear arithmetic
function (+)(dt::ReactantDateTime, y::ReactantYear)
    oy, m, d = yearmonthday(dt)
    ny = oy + value(y)
    ld = daysinmonth(ny, m)
    return ReactantDateTime(
        UTInstant(
            ReactantMillisecond(
                value(dt) - toms(ReactantDay(totaldays(oy, m, d))) +
                toms(ReactantDay(totaldays(ny, m, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (+)(dt::ReactantDate, y::ReactantYear)
    oy, m, d = yearmonthday(dt)
    ny = oy + value(y)
    ld = daysinmonth(ny, m)
    return ReactantDate(UTInstant(ReactantDay(totaldays(ny, m, d <= ld ? d : ld))))
end
function (-)(dt::ReactantDateTime, y::ReactantYear)
    oy, m, d = yearmonthday(dt)
    ny = oy - value(y)
    ld = daysinmonth(ny, m)
    return ReactantDateTime(
        UTInstant(
            ReactantMillisecond(
                value(dt) - toms(ReactantDay(totaldays(oy, m, d))) +
                toms(ReactantDay(totaldays(ny, m, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (-)(dt::ReactantDate, y::ReactantYear)
    oy, m, d = yearmonthday(dt)
    ny = oy - value(y)
    ld = daysinmonth(ny, m)
    return ReactantDate(UTInstant(ReactantDay(totaldays(ny, m, d <= ld ? d : ld))))
end

# ReactantDateTime/ReactantDate-ReactantMonth arithmetic
function (+)(dt::ReactantDateTime, z::ReactantMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, value(z))
    mm = monthwrap(m, value(z))
    ld = daysinmonth(ny, mm)
    return ReactantDateTime(
        UTInstant(
            ReactantMillisecond(
                value(dt) - toms(ReactantDay(totaldays(y, m, d))) +
                toms(ReactantDay(totaldays(ny, mm, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (+)(dt::ReactantDate, z::ReactantMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, value(z))
    mm = monthwrap(m, value(z))
    ld = daysinmonth(ny, mm)
    return ReactantDate(UTInstant(ReactantDay(totaldays(ny, mm, d <= ld ? d : ld))))
end
function (-)(dt::ReactantDateTime, z::ReactantMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, -value(z))
    mm = monthwrap(m, -value(z))
    ld = daysinmonth(ny, mm)
    return ReactantDateTime(
        UTInstant(
            ReactantMillisecond(
                value(dt) - toms(ReactantDay(totaldays(y, m, d))) +
                toms(ReactantDay(totaldays(ny, mm, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (-)(dt::ReactantDate, z::ReactantMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, -value(z))
    mm = monthwrap(m, -value(z))
    ld = daysinmonth(ny, mm)
    return ReactantDate(UTInstant(ReactantDay(totaldays(ny, mm, d <= ld ? d : ld))))
end

# ReactantDateTime/ReactantDate-ReactantQuarter arithmetic (delegates to Month)
(+)(x::ReactantDate, y::ReactantQuarter) = x + ReactantMonth(3 * value(y))
(-)(x::ReactantDate, y::ReactantQuarter) = x - ReactantMonth(3 * value(y))
(+)(x::ReactantDateTime, y::ReactantQuarter) = x + ReactantMonth(3 * value(y))
(-)(x::ReactantDateTime, y::ReactantQuarter) = x - ReactantMonth(3 * value(y))

# ReactantDate-ReactantWeek/ReactantDay arithmetic
function (+)(x::ReactantDate, y::ReactantWeek)
    return ReactantDate(UTInstant(ReactantDay(value(x) + 7 * value(y))))
end
function (-)(x::ReactantDate, y::ReactantWeek)
    return ReactantDate(UTInstant(ReactantDay(value(x) - 7 * value(y))))
end
function (+)(x::ReactantDate, y::ReactantDay)
    return ReactantDate(UTInstant(ReactantDay(value(x) + value(y))))
end
function (-)(x::ReactantDate, y::ReactantDay)
    return ReactantDate(UTInstant(ReactantDay(value(x) - value(y))))
end

# ReactantDateTime + any Period (via toms)
function (+)(x::ReactantDateTime, y::Period)
    return ReactantDateTime(UTInstant(ReactantMillisecond(value(x) + toms(y))))
end
function (-)(x::ReactantDateTime, y::Period)
    return ReactantDateTime(UTInstant(ReactantMillisecond(value(x) - toms(y))))
end

# ReactantTime + any TimePeriod (via tons)
(+)(x::ReactantTime, y::TimePeriod) = ReactantTime(ReactantNanosecond(value(x) + tons(y)))
(-)(x::ReactantTime, y::TimePeriod) = ReactantTime(ReactantNanosecond(value(x) - tons(y)))

# Commutativity
(+)(y::Period, x::ReactantDateTime) = x + y
(+)(y::Period, x::ReactantDate) = x + y
(+)(y::TimePeriod, x::ReactantTime) = x + y

# CONCRETE TimeType ± TRACED period.
#
# The mirror of `ReactantDateTime ± Period` above: there the datetime is traced and the period may be
# concrete; here the datetime is concrete and the period is traced. That combination arises whenever
# a wall-clock epoch is advanced by a traced offset, e.g. `epoch + Millisecond(round(Int, 1000t))`
# for a traced elapsed time `t`.
#
# Without these, Base's `+(::DateTime, ::Period)` builds a
# `UTInstant{ReactantMillisecond{TracedRNumber}}` and then tries to pour it into a plain `DateTime`,
# whose instant must hold a concrete `Int64`:
#
#     MethodError: no method matching Int64(::UTInstant{ReactantMillisecond{TracedRNumber{Int64}}})
#
# Promoting the datetime to its Reactant counterpart first is the only representable answer.
const ReactantDatePeriod = Union{
    ReactantYear,ReactantQuarter,ReactantMonth,ReactantWeek,ReactantDay
}
const ReactantTimePeriod = Union{
    ReactantHour,
    ReactantMinute,
    ReactantSecond,
    ReactantMillisecond,
    ReactantMicrosecond,
    ReactantNanosecond,
}
const ReactantPeriod = Union{ReactantDatePeriod,ReactantTimePeriod}

(+)(x::DateTime, y::ReactantPeriod) = ReactantDateTime(x) + y
(-)(x::DateTime, y::ReactantPeriod) = ReactantDateTime(x) - y
(+)(y::ReactantPeriod, x::DateTime) = x + y

(+)(x::Date, y::ReactantDatePeriod) = ReactantDate(x) + y
(-)(x::Date, y::ReactantDatePeriod) = ReactantDate(x) - y
(+)(y::ReactantDatePeriod, x::Date) = x + y

(+)(x::Time, y::ReactantTimePeriod) = ReactantTime(x) + y
(-)(x::Time, y::ReactantTimePeriod) = ReactantTime(x) - y
(+)(y::ReactantTimePeriod, x::Time) = x + y
