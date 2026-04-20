Dates.firstdayofmonth(dt::ReactantDatesExt.ReactantDate{I}) where {I} =
    ReactantDatesExt.ReactantDate{I}(Date(Dates.year(dt), Dates.month(dt)))
Dates.firstdayofmonth(dt::ReactantDatesExt.ReactantDateTime{I}) where {I} =
    ReactantDatesExt.ReactantDateTime{I}(DateTime(Dates.year(dt), Dates.month(dt)))
