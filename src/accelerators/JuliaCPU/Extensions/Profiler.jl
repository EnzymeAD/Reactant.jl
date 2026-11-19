mutable struct JuliaProfiler <: Profiler
    const options::Reactant.Proto.tensorflow.ProfileOptions
end

function create_profiler(::JuliaPlugin, options::Reactant.Proto.tensorflow.ProfileOptions)
    return JuliaProfiler(options)
end

# XLA runs one profiling session at a time, so there is at most one profiler recording the activity of the plugin
const active_profiler = Ref{Union{Nothing,JuliaProfiler}}(nothing)

# the profiler is created in every profiling session, but it only records if device tracing is enabled
function start!(profiler::JuliaProfiler)
    profiler.options.device_tracer_level > 0 || return nothing
    active_profiler[] = profiler
    return nothing
end

function stop!(profiler::JuliaProfiler)
    active_profiler[] === profiler && (active_profiler[] = nothing)
    return nothing
end
