using ..Reactant: MLIR, TracedUtils, Ops, TracedRArray
import ..Reactant: promote_to

"""
    process_probprog_function(f, args_with_rng, op_name)

This function handles the probprog argument convention where:
- **Index 1**: RNG state
- **Index 2**: Function `f` (when wrapped)
- **Index 3+**: Remaining arguments

This wrapper ensures the RNG state is threaded through as the first result,
followed by the actual function results.
"""
function process_probprog_function(f, args_with_rng, op_name)
    argprefix = gensym(op_name * "arg")
    resprefix = gensym(op_name * "result")
    resargprefix = gensym(op_name * "resarg")

    wrapper_fn = (all_args...) -> begin
        res = f(all_args...)
        (all_args[1], (res isa Tuple ? res : (res,))...)
    end

    mlir_fn_res = TracedUtils.make_mlir_fn(
        wrapper_fn,
        args_with_rng,
        (),
        string(f),
        false;
        do_transpose=false,
        args_in_result=:result,
        argprefix,
        resprefix,
        resargprefix,
    )

    return mlir_fn_res, argprefix, resprefix, resargprefix
end

"""
    process_probprog_inputs(linear_args, f, args_with_rng, fnwrap, argprefix)

This function handles the probprog argument convention where:
- **Index 1**: RNG state
- **Index 2**: Function `f` (when `fnwrap` is true)
- **Index 3+**: Other arguments
"""
function process_probprog_inputs(linear_args, f, args_with_rng, fnwrap, argprefix)
    inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 2 && fnwrap
            TracedUtils.push_val!(inputs, f, path[3:end])
        else
            if fnwrap && idx > 1
                idx -= 1
            end
            TracedUtils.push_val!(inputs, args_with_rng[idx], path[3:end])
        end
    end
    return inputs
end

"""
    process_probprog_outputs(op, linear_results, result, f, args_with_rng, fnwrap, resprefix, argprefix, start_idx=0, rng_only=false)

This function handles the probprog argument convention where:
- **Index 1**: RNG state
- **Index 2**: Function `f` (when `fnwrap` is true)
- **Index 3+**: Other arguments

When setting results, the function checks:
1. If result path matches `resprefix`, store in `result`
2. If result path matches `argprefix`, store in `args_with_rng` (adjust indices for wrapped function)

`start_idx` varies depending on the ProbProg operation:
- `sample` and `untraced_call` return only function outputs:
  Use `start_idx=0`: `linear_results[i]` corresponds to `op.result[i]`
- `simulate` and `generate` return trace, weight, then outputs:
  Use `start_idx=2`: `linear_results[i]` corresponds to `op.result[i+2]`
- `mh` and `regenerate` return trace, accepted/weight, rng_state (no model outputs):
  Use `start_idx=2, rng_only=true`: only process first result (rng_state)

`rng_only`: When true, only process the first result (RNG state), skipping model outputs
"""
function process_probprog_outputs(
    op,
    linear_results,
    result,
    f,
    args_with_rng,
    fnwrap,
    resprefix,
    argprefix,
    start_idx=0,
    rng_only=false,
)
    num_to_process = rng_only ? 1 : length(linear_results)

    for i in 1:num_to_process
        res = linear_results[i]
        resv = MLIR.IR.result(op, i + start_idx)

        if TracedUtils.has_idx(res, resprefix)
            path = TracedUtils.get_idx(res, resprefix)
            TracedUtils.set!(result, path[2:end], resv)
        end

        if TracedUtils.has_idx(res, argprefix)
            idx, path = TracedUtils.get_argidx(res, argprefix)
            if fnwrap && idx == 2
                TracedUtils.set!(f, path[3:end], resv)
            else
                if fnwrap && idx > 2
                    idx -= 1
                end
                TracedUtils.set!(args_with_rng[idx], path[3:end], resv)
            end
        end

        if !TracedUtils.has_idx(res, resprefix) && !TracedUtils.has_idx(res, argprefix)
            TracedUtils.set!(res, (), resv)
        end
    end
end

to_trace_tensor(t::ProbProgTrace) = promote_to(TracedRArray{UInt64,0}, t)

function from_trace_tensor(trace_tensor)
    while !isready(trace_tensor)
        yield()
    end
    return unsafe_pointer_to_objref(Ptr{Any}(Array(trace_tensor)[1]))::ProbProgTrace
end

function promote_to(::Type{TracedRArray{UInt64,0}}, t::ProbProgTrace)
    ptr = reinterpret(UInt64, pointer_from_objref(t))
    return Ops.fill(ptr, Int64[])
end

to_constraint_tensor(c::Constraint) = promote_to(TracedRArray{UInt64,0}, c)

function from_constraint_tensor(constraint_tensor)
    while !isready(constraint_tensor)
        yield()
    end
    return unsafe_pointer_to_objref(Ptr{Any}(Array(constraint_tensor)[1]))::Constraint
end

function promote_to(::Type{TracedRArray{UInt64,0}}, c::Constraint)
    ptr = reinterpret(UInt64, pointer_from_objref(c))
    return Ops.fill(ptr, Int64[])
end
