# TracedRDate: same output as Base.print(io::IO, dt::Date) in stdlib Dates
function Base.print(io::IO, dt::TracedRDate)
    y, m, d = yearmonthday(dt)
    yy = y < 0 ? string("-", lpad(-y, 4, "0")) : lpad(y, 4, "0")
    mm = lpad(m, 2, "0")
    dd = lpad(d, 2, "0")
    return print(io, "$yy-$mm-$dd")
end

# TracedRDateTime: same output as Base.print(io::IO, dt::DateTime) in stdlib Dates
function Base.print(io::IO, dt::TracedRDateTime)
    y, m, d = yearmonthday(dt)
    yy = y < 0 ? string("-", lpad(-y, 4, "0")) : lpad(y, 4, "0")
    mm = lpad(m, 2, "0")
    dd = lpad(d, 2, "0")
    hh = lpad(hour(dt), 2, "0")
    mii = lpad(minute(dt), 2, "0")
    ss = lpad(second(dt), 2, "0")
    ms = millisecond(dt)
    return if ms == 0
        print(io, "$yy-$mm-$dd", 'T', "$hh:$mii:$ss")
    else
        msss = lpad(ms, 3, "0")
        print(io, "$yy-$mm-$dd", 'T', "$hh:$mii:$ss.$msss")
    end
end

# TracedRTime: same output as Base.string(t::Time) in stdlib Dates
function Base.string(t::TracedRTime)
    h, mi, s = hour(t), minute(t), second(t)
    hh = lpad(h, 2, "0")
    mii = lpad(mi, 2, "0")
    ss = lpad(s, 2, "0")
    nss = millisecond(t) * 1000000 + microsecond(t) * 1000 + nanosecond(t)
    ns = nss == 0 ? "" : "." * rstrip(lpad(nss, 9, "0"), '0')
    return "$hh:$mii:$ss$ns"
end

Base.show(io::IO, ::MIME"text/plain", t::TracedRTime) = print(io, t)
Base.print(io::IO, t::TracedRTime) = print(io, string(t))

function Base.show(io::IO, t::TracedRTime)
    return if get(io, :compact, false)::Bool
        print(io, t)
    else
        values = [
            hour(t)
            minute(t)
            second(t)
            millisecond(t)
            microsecond(t)
            nanosecond(t)
        ]
        index = something(findlast(!iszero, values), 1)

        print(io, TracedRTime, "(")
        for i in 1:index
            show(io, values[i])
            i != index && print(io, ", ")
        end
        print(io, ")")
    end
end

for date_type in (:TracedRDate, :TracedRDateTime)
    # Human readable output (i.e. "2012-01-01")
    @eval Base.show(io::IO, ::MIME"text/plain", dt::$date_type) = print(io, dt)
    # Parsable output (i.e. TracedRDate("2012-01-01"))
    @eval Base.show(io::IO, dt::$date_type) = print(io, typeof(dt), "(\"", dt, "\")")
end
