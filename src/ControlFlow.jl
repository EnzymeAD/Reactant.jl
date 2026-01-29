function ReactantCore.traced_if(
    cond::TracedRNumber{Bool}, true_fn::TFn, false_fn::FFn, args; track_numbers=Number
) where {TFn,FFn}
    return @opcall if_condition(cond, true_fn, false_fn, args...; track_numbers)
end

function ReactantCore.traced_call(f::Function, args...; tessera_op=nothing)
    if isnothing(tessera_op)
        tessera_op = Reactant.Compiler.get_tessera_op(f)
    end
    return @opcall call(f, args...; tessera_op)
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
    return @opcall while_loop(
        cond_fn, body_fn, args...; track_numbers, verify_arg_names, checkpointing, mincut
    )
end
