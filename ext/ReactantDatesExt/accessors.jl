import Dates:
    year,
    quarter,
    month,
    week,
    day,
    hour,
    minute,
    second,
    millisecond,
    microsecond,
    nanosecond,
    yearmonth,
    monthday,
    yearmonthday,
    dayofmonth,
    days

# days accessor
days(dt::TracedRDate) = value(dt)
days(dt::TracedRDateTime) = fld(value(dt), 86400000)

# Calendar accessors dispatching on TracedR* TimeTypes
year(dt::Union{TracedRDate,TracedRDateTime}) = Dates.year(days(dt))
quarter(dt::Union{TracedRDate,TracedRDateTime}) = Dates.quarter(days(dt))
month(dt::Union{TracedRDate,TracedRDateTime}) = Dates.month(days(dt))
week(dt::Union{TracedRDate,TracedRDateTime}) = Dates.week(days(dt))
day(dt::Union{TracedRDate,TracedRDateTime}) = Dates.day(days(dt))

# DateTime time-of-day accessors
hour(dt::TracedRDateTime) = mod(fld(value(dt), 3600000), 24)
minute(dt::TracedRDateTime) = mod(fld(value(dt), 60000), 60)
second(dt::TracedRDateTime) = mod(fld(value(dt), 1000), 60)
millisecond(dt::TracedRDateTime) = mod(value(dt), 1000)

# Time accessors
hour(t::TracedRTime) = mod(fld(value(t), 3600000000000), Int64(24))
minute(t::TracedRTime) = mod(fld(value(t), 60000000000), Int64(60))
second(t::TracedRTime) = mod(fld(value(t), 1000000000), Int64(60))
millisecond(t::TracedRTime) = mod(fld(value(t), Int64(1000000)), Int64(1000))
microsecond(t::TracedRTime) = mod(fld(value(t), Int64(1000)), Int64(1000))
nanosecond(t::TracedRTime) = mod(value(t), Int64(1000))

dayofmonth(dt::Union{TracedRDate,TracedRDateTime}) = day(dt)

# Compound accessors
yearmonth(dt::Union{TracedRDate,TracedRDateTime}) = Dates.yearmonth(days(dt))
monthday(dt::Union{TracedRDate,TracedRDateTime}) = Dates.monthday(days(dt))
yearmonthday(dt::Union{TracedRDate,TracedRDateTime}) = Dates.yearmonthday(days(dt))
