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
days(dt::ReactantDate) = value(dt)
days(dt::ReactantDateTime) = fld(value(dt), 86400000)

# Calendar accessors dispatching on Reactant* TimeTypes
year(dt::Union{ReactantDate,ReactantDateTime}) = Dates.year(days(dt))
quarter(dt::Union{ReactantDate,ReactantDateTime}) = Dates.quarter(days(dt))
month(dt::Union{ReactantDate,ReactantDateTime}) = Dates.month(days(dt))
week(dt::Union{ReactantDate,ReactantDateTime}) = Dates.week(days(dt))
day(dt::Union{ReactantDate,ReactantDateTime}) = Dates.day(days(dt))

# DateTime time-of-day accessors
hour(dt::ReactantDateTime) = mod(fld(value(dt), 3600000), 24)
minute(dt::ReactantDateTime) = mod(fld(value(dt), 60000), 60)
second(dt::ReactantDateTime) = mod(fld(value(dt), 1000), 60)
millisecond(dt::ReactantDateTime) = mod(value(dt), 1000)

# Time accessors
hour(t::ReactantTime) = mod(fld(value(t), 3600000000000), Int64(24))
minute(t::ReactantTime) = mod(fld(value(t), 60000000000), Int64(60))
second(t::ReactantTime) = mod(fld(value(t), 1000000000), Int64(60))
millisecond(t::ReactantTime) = mod(fld(value(t), Int64(1000000)), Int64(1000))
microsecond(t::ReactantTime) = mod(fld(value(t), Int64(1000)), Int64(1000))
nanosecond(t::ReactantTime) = mod(value(t), Int64(1000))

dayofmonth(dt::Union{ReactantDate,ReactantDateTime}) = day(dt)

# Compound accessors
yearmonth(dt::Union{ReactantDate,ReactantDateTime}) = Dates.yearmonth(days(dt))
monthday(dt::Union{ReactantDate,ReactantDateTime}) = Dates.monthday(days(dt))
yearmonthday(dt::Union{ReactantDate,ReactantDateTime}) = Dates.yearmonthday(days(dt))
