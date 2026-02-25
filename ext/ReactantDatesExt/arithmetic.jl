import Base: +, -, fld, mod, div, mod1
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
toms(c::TracedRNanosecond) = div(value(c), 1000000, RoundNearest)
toms(c::TracedRMicrosecond) = div(value(c), 1000, RoundNearest)
toms(c::TracedRMillisecond) = value(c)
toms(c::TracedRSecond) = 1000 * value(c)
toms(c::TracedRMinute) = 60000 * value(c)
toms(c::TracedRHour) = 3600000 * value(c)
toms(c::TracedRDay) = 86400000 * value(c)
toms(c::TracedRWeek) = 604800000 * value(c)

# tons (to nanoseconds) for traced period types
tons(c::TracedRNanosecond) = value(c)
tons(c::TracedRMicrosecond) = value(c) * 1000
tons(c::TracedRMillisecond) = value(c) * 1000000
tons(c::TracedRSecond) = value(c) * 1000000000
tons(c::TracedRMinute) = value(c) * 60000000000
tons(c::TracedRHour) = value(c) * 3600000000000

# days for traced period types
days(c::TracedRDay) = value(c)
days(c::TracedRWeek) = 7 * value(c)

# TracedRDateTime/TracedRDate-TracedRYear arithmetic
function (+)(dt::TracedRDateTime, y::TracedRYear)
    oy, m, d = yearmonthday(dt)
    ny = oy + value(y)
    ld = daysinmonth(ny, m)
    return TracedRDateTime(
        UTInstant(
            TracedRMillisecond(
                value(dt) - toms(TracedRDay(totaldays(oy, m, d))) +
                toms(TracedRDay(totaldays(ny, m, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (+)(dt::TracedRDate, y::TracedRYear)
    oy, m, d = yearmonthday(dt)
    ny = oy + value(y)
    ld = daysinmonth(ny, m)
    return TracedRDate(UTInstant(TracedRDay(totaldays(ny, m, d <= ld ? d : ld))))
end
function (-)(dt::TracedRDateTime, y::TracedRYear)
    oy, m, d = yearmonthday(dt);
    ny = oy - value(y);
    ld = daysinmonth(ny, m)
    return TracedRDateTime(
        UTInstant(
            TracedRMillisecond(
                value(dt) - toms(TracedRDay(totaldays(oy, m, d))) +
                toms(TracedRDay(totaldays(ny, m, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (-)(dt::TracedRDate, y::TracedRYear)
    oy, m, d = yearmonthday(dt);
    ny = oy - value(y);
    ld = daysinmonth(ny, m)
    return TracedRDate(UTInstant(TracedRDay(totaldays(ny, m, d <= ld ? d : ld))))
end

# TracedRDateTime/TracedRDate-TracedRMonth arithmetic
function (+)(dt::TracedRDateTime, z::TracedRMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, value(z))
    mm = monthwrap(m, value(z));
    ld = daysinmonth(ny, mm)
    return TracedRDateTime(
        UTInstant(
            TracedRMillisecond(
                value(dt) - toms(TracedRDay(totaldays(y, m, d))) +
                toms(TracedRDay(totaldays(ny, mm, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (+)(dt::TracedRDate, z::TracedRMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, value(z))
    mm = monthwrap(m, value(z));
    ld = daysinmonth(ny, mm)
    return TracedRDate(UTInstant(TracedRDay(totaldays(ny, mm, d <= ld ? d : ld))))
end
function (-)(dt::TracedRDateTime, z::TracedRMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, -value(z))
    mm = monthwrap(m, -value(z));
    ld = daysinmonth(ny, mm)
    return TracedRDateTime(
        UTInstant(
            TracedRMillisecond(
                value(dt) - toms(TracedRDay(totaldays(y, m, d))) +
                toms(TracedRDay(totaldays(ny, mm, d <= ld ? d : ld))),
            ),
        ),
    )
end
function (-)(dt::TracedRDate, z::TracedRMonth)
    y, m, d = yearmonthday(dt)
    ny = yearwrap(y, m, -value(z))
    mm = monthwrap(m, -value(z));
    ld = daysinmonth(ny, mm)
    return TracedRDate(UTInstant(TracedRDay(totaldays(ny, mm, d <= ld ? d : ld))))
end

# TracedRDateTime/TracedRDate-TracedRQuarter arithmetic (delegates to Month)
(+)(x::TracedRDate, y::TracedRQuarter) = x + TracedRMonth(3 * value(y))
(-)(x::TracedRDate, y::TracedRQuarter) = x - TracedRMonth(3 * value(y))
(+)(x::TracedRDateTime, y::TracedRQuarter) = x + TracedRMonth(3 * value(y))
(-)(x::TracedRDateTime, y::TracedRQuarter) = x - TracedRMonth(3 * value(y))

# TracedRDate-TracedRWeek/TracedRDay arithmetic
(+)(x::TracedRDate, y::TracedRWeek) =
    TracedRDate(UTInstant(TracedRDay(value(x) + 7 * value(y))))
(-)(x::TracedRDate, y::TracedRWeek) =
    TracedRDate(UTInstant(TracedRDay(value(x) - 7 * value(y))))
(+)(x::TracedRDate, y::TracedRDay) = TracedRDate(UTInstant(TracedRDay(value(x) + value(y))))
(-)(x::TracedRDate, y::TracedRDay) = TracedRDate(UTInstant(TracedRDay(value(x) - value(y))))

# TracedRDateTime + any Period (via toms)
(+)(x::TracedRDateTime, y::Period) =
    TracedRDateTime(UTInstant(TracedRMillisecond(value(x) + toms(y))))
(-)(x::TracedRDateTime, y::Period) =
    TracedRDateTime(UTInstant(TracedRMillisecond(value(x) - toms(y))))

# TracedRTime + any TimePeriod (via tons)
(+)(x::TracedRTime, y::TimePeriod) = TracedRTime(TracedRNanosecond(value(x) + tons(y)))
(-)(x::TracedRTime, y::TimePeriod) = TracedRTime(TracedRNanosecond(value(x) - tons(y)))

# Commutativity
(+)(y::Period, x::TracedRDateTime) = x + y
(+)(y::Period, x::TracedRDate) = x + y
(+)(y::TimePeriod, x::TracedRTime) = x + y
