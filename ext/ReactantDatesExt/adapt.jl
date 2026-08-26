# Adapt support, so Reactant date/time types can be passed as GPU KERNEL ARGUMENTS.
#
# Kernel arguments are walked by `Adapt`, which replaces `TracedRNumber`/`TracedRArray` with their
# device-side `CuTracedRNumber`/`CuTracedArray` counterparts. `Adapt` only recurses into types that
# say how; a type with no method is treated as opaque and its fields are left alone.
#
# `ReactantDateTime` and friends wrap exactly one traced number, so without these methods that number
# survives into the kernel still HOST-side. Every operation on it then tries to emit MLIR instead of
# computing, and Reactant's own tracing machinery is dragged into device code:
#
#     Reason: unsupported dynamic function invocation (call to Reactant.MLIR.IR.Location)
#       @ Reactant/src/Ops.jl:153
#
# Reactant's own kernel-argument checker diagnoses this and prescribes the fix:
#
#     GPU kernel argument of type $T contains an unadapted traced value at field: $bad
#     … some struct in the hierarchy is missing `Adapt.@adapt_structure`, so its fields were not
#     recursed into during GPU adaptation.
#
# These cannot be a plain `Adapt.@adapt_structure`, because the traced number is not a direct field:
# it sits behind `UTInstant` and the period wrapper (`ReactantDateTime.instant.periods.value`), so
# the value has to be adapted and the wrappers rebuilt around it. This mirrors what Reactant already
# does for `TracedStepRangeLen` and `Base.TwicePrecision` in ReactantCUDAExt.
#
# Written against a generic adaptor rather than the CUDA one: adapting a wrapper by adapting its
# payload is correct for ANY adaptor, and it keeps this extension free of a CUDA dependency.

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
