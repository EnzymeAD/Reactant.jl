function Dates.firstdayofmonth(dt::ReactantDate{I}) where {I}
    return ReactantDate{I}(Date(Dates.year(dt), Dates.month(dt)))
end
function Dates.firstdayofmonth(dt::ReactantDateTime{I}) where {I}
    return ReactantDateTime{I}(DateTime(Dates.year(dt), Dates.month(dt)))
end
