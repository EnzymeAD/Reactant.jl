using ..Reactant: ConcreteRNumber
using ..Compiler: @compile

function metropolis_hastings(
    trace::ProbProgTrace,
    sel::Selection;
    compiled_cache::Union{Nothing,CompiledFnCache}=nothing,
)
    if trace.fn === nothing || trace.rng === nothing
        error("MH requires a trace with fn and rng recorded (use generate to create trace)")
    end

    constraints = Dict{Symbol,Any}()
    constrained_symbols = Set{Symbol}()

    for (sym, val) in trace.choices
        if !(sym in sel)
            constraints[sym] = val
            push!(constrained_symbols, sym)
        end
    end

    cache_key = (typeof(trace.fn), constrained_symbols)

    compiled_fn = nothing
    if compiled_cache !== nothing
        compiled_fn = get(compiled_cache, cache_key, nothing)
    end

    if compiled_fn === nothing
        function wrapper_fn(rng, constraint_ptr, args...)
            return generate_internal(
                rng, trace.fn, args...; constraint_ptr, constrained_symbols
            )
        end

        constraint_ptr = ConcreteRNumber(
            reinterpret(UInt64, pointer_from_objref(constraints))
        )

        compiled_fn = @compile optimize = :probprog wrapper_fn(
            trace.rng, constraint_ptr, trace.args...
        )

        if compiled_cache !== nothing
            compiled_cache[cache_key] = compiled_fn
        end
    end

    constraint_ptr = ConcreteRNumber(reinterpret(UInt64, pointer_from_objref(constraints)))

    old_gc_state = GC.enable(false)
    new_trace_ptr = nothing
    try
        new_trace_ptr, _, _ = compiled_fn(trace.rng, constraint_ptr, trace.args...)
    finally
        GC.enable(old_gc_state)
    end

    new_trace = unsafe_pointer_to_objref(Ptr{Any}(Array(new_trace_ptr)[1]))

    new_trace.fn = trace.fn
    new_trace.args = trace.args
    new_trace.rng = trace.rng

    log_alpha = new_trace.weight - trace.weight

    if log(rand()) < log_alpha
        return (new_trace, true)
    else
        return (trace, false)
    end
end