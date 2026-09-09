function Adapt.adapt_structure(to, dt::ReactantDateTime)
    return ReactantDateTime(UTInstant(ReactantMillisecond(Adapt.adapt(to, value(dt)))))
end

function Adapt.adapt_structure(to, d::ReactantDate)
    return ReactantDate(UTInstant(ReactantDay(Adapt.adapt(to, value(d)))))
end

function Adapt.adapt_structure(to, t::ReactantTime)
    return ReactantTime(ReactantNanosecond(Adapt.adapt(to, value(t))))
end

# The period types are single-field wrappers, so adapting them is just adapting the value.
for (_, T) in _PERIOD_PAIRS
    @eval Adapt.adapt_structure(to, p::$T) = $T(Adapt.adapt(to, value(p)))
end
