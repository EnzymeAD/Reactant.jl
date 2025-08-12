function ReactantCore.traced_if(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args; track_numbers=Number
) where {TFn,FFn}
    return Ops.if_condition(cond, true_fn, false_fn, args...; track_numbers)
end

function ReactantCore.traced_call(f::Function, args...)
    return Ops.call(f, args...)
end

function ReactantCore.traced_while(
    cond_fn::CFn,
    body_fn::BFn,
    args;
    track_numbers=Number,
    verify_arg_names=nothing,
    checkpointing=false,
    mincut=false,
) where {CFn,BFn}
    return Ops.while_loop(
        cond_fn, body_fn, args...; track_numbers, verify_arg_names, checkpointing, mincut
    )
end
