using Base: ReentrantLock

mutable struct ProbProgTrace
    fn::Union{Nothing,Function}
    args::Union{Nothing,Tuple}
    choices::Dict{Symbol,Any}
    retval::Any
    weight::Any
    subtraces::Dict{Symbol,Any}
    rng::Union{Nothing,AbstractRNG}

    function ProbProgTrace(fn::Function, args::Tuple)
        return new(
            fn, args, Dict{Symbol,Any}(), nothing, nothing, Dict{Symbol,Any}(), nothing
        )
    end

    function ProbProgTrace()
        return new(
            nothing, (), Dict{Symbol,Any}(), nothing, nothing, Dict{Symbol,Any}(), nothing
        )
    end
end

const Constraint = Dict{Symbol,Any}
const Selection = Set{Symbol}
const CompiledFnCache = Dict{Tuple{Type,Set{Symbol}},Any}

const _trace_ref_lock = ReentrantLock()
const _trace_refs = Vector{Any}()

function _keepalive!(tr::ProbProgTrace)
    lock(_trace_ref_lock)
    try
        push!(_trace_refs, tr)
    finally
        unlock(_trace_ref_lock)
    end
    return tr
end

get_choices(trace::ProbProgTrace) = trace.choices
select(syms::Symbol...) = Set(syms)
choicemap() = Constraint()

function with_compiled_cache(f)
    cache = CompiledFnCache()
    return f(cache)
end