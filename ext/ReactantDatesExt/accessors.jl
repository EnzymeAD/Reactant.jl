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
days(rm::ReactantMillisecond) = div(value(rm), 86400000)

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

# The algorithm is identical to Dates/src/accessors.jl (proleptic Gregorian calendar)
# These versions replace ternary operators with `flds` and `divs`
function Dates.yearmonthday(days::Reactant.TracedRNumber)
    z = days + 306;
    h = 100z - 25;
    a = fld(h, 3652425);
    b = a - fld(a, 4)
    y = fld(100b + h, 36525);
    c = b + z - 365y - fld(y, 4);
    m = div(5c + 456, 153)
    d = c - div(153m - 457, 5)
    overflow = fld(m - 1, 12)
    return (y + overflow, m - 12overflow, d)
end

function Dates.yearmonth(days::Reactant.TracedRNumber)
    z = days + 306;
    h = 100z - 25;
    a = fld(h, 3652425);
    b = a - fld(a, 4)
    y = fld(100b + h, 36525);
    c = b + z - 365y - fld(y, 4);
    m = div(5c + 456, 153)
    overflow = fld(m - 1, 12)
    return (y + overflow, m - 12overflow)
end

function Dates.monthday(days::Reactant.TracedRNumber)
    z = days + 306;
    h = 100z - 25;
    a = fld(h, 3652425);
    b = a - fld(a, 4)
    y = fld(100b + h, 36525);
    c = b + z - 365y - fld(y, 4);
    m = div(5c + 456, 153)
    d = c - div(153m - 457, 5)
    overflow = fld(m - 1, 12)
    return (m - 12overflow, d)
end

# Replaces the MONTHDAYS[m] tuple lookup with a
# branching computation over the cumulative month-day offsets
function Dates.dayofyear(
    y::Reactant.TracedRNumber, m::Reactant.TracedRNumber, d::Reactant.TracedRNumber
)
    @trace if m == 1
        monthdays = zero(m)
    elseif m == 2
        monthdays = oftype(m, 31)
    elseif m == 3
        monthdays = oftype(m, 59)
    elseif m == 4
        monthdays = oftype(m, 90)
    elseif m == 5
        monthdays = oftype(m, 120)
    elseif m == 6
        monthdays = oftype(m, 151)
    elseif m == 7
        monthdays = oftype(m, 181)
    elseif m == 8
        monthdays = oftype(m, 212)
    elseif m == 9
        monthdays = oftype(m, 243)
    elseif m == 10
        monthdays = oftype(m, 273)
    elseif m == 11
        monthdays = oftype(m, 304)
    else
        monthdays = oftype(m, 334)
    end
    leap_correction = ifelse((m > 2) & Dates.isleapyear(y), one(m), zero(m))
    return monthdays + d + leap_correction
end

function Dates.dayofyear(dt::Union{ReactantDate,ReactantDateTime})
    y, m, d = Dates.yearmonthday(dt)
    return Dates.dayofyear(y, m, d)
end
