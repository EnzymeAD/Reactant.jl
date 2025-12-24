module XProfUtils

using CondaPkg: CondaPkg
using JSON3: JSON3
using PythonCall: PythonCall, Py, pyconvert, pyimport
using Reactant: Reactant, @compile
using Setfield: @set!
using Sockets: Sockets

const xprof_pywrap_profiler_plugin = Ref{Union{Py,Nothing}}(nothing)
const xprof_convert_raw_to_tool_data = Ref{Union{Py,Nothing}}(nothing)

function get_free_port()
    # Bind to port 0 to let the OS assign a free port
    server = Sockets.listen(Sockets.IPv4(0), 0)
    _, port = Sockets.getsockname(server)
    close(server)
    return port
end

function __init__()
    xprof_pywrap_profiler_plugin[] = pyimport("xprof.convert._pywrap_profiler_plugin")
    xprof_convert_raw_to_tool_data[] = pyimport("xprof.convert.raw_to_tool_data")

    grpc_port = get_free_port()
    @debug "Using port $(grpc_port) for XProfUtils.jl"
    xprof_pywrap_profiler_plugin[].initialize_stubs("0.0.0.0:$(grpc_port)")
    xprof_pywrap_profiler_plugin[].start_grpc_server(grpc_port)

    return nothing
end

function profile_with_xprof(
    fn::F,
    args...;
    nrepeat::Int=1,
    warmup::Int=1,
    profile_dir::Union{String,Nothing}=nothing,
    compile_options::Reactant.CompileOptions=Reactant.CompileOptions(),
    kwargs...,
) where {F}
    @set! compile_options.sync = true
    compiled_fn = @compile compile_options = compile_options fn(args...; kwargs...)
    return profile_with_xprof(compiled_fn, args...; kwargs..., nrepeat, warmup, profile_dir)
end

# If passing in a compiled function we assume sync=true
# we can update the Thunk to have a sync field
function profile_with_xprof(
    fn::Reactant.Compiler.Thunk,
    args...;
    nrepeat::Int=1,
    warmup::Int=1,
    profile_dir::Union{String,Nothing}=nothing,
    kwargs...,
)
    profile_dir === nothing && (profile_dir = joinpath(tempdir(), "reactant_profile"))
    mkpath(profile_dir)

    # warmup
    for _ in 1:warmup
        fn(args...; kwargs...)
    end

    # profile
    Reactant.with_profiler(profile_dir) do
        for i in 1:nrepeat
            id = Reactant.Profiler.profiler_activity_start(
                "bench",
                Reactant.Profiler.TRACE_ME_LEVEL_CRITICAL,
                "step_num" => i,
                "_r" => 1,
            )
            try
                fn(args...; kwargs...)
            finally
                Reactant.Profiler.profiler_activity_end(id)
            end
        end
    end

    trace_output_dir = joinpath(profile_dir, "plugins", "profile")
    date = maximum(readdir(trace_output_dir))
    traces_path = joinpath(trace_output_dir, date)
    filename = first(f for f in readdir(traces_path) if endswith(f, ".xplane.pb"))
    @assert filename !== nothing "No xplane file found in $traces_path"
    xplane_file = joinpath(traces_path, filename)

    return extract_mean_step_time(xplane_file, nrepeat)
end

function extract_mean_step_time(xplane_file::String, nrepeat::Int)
    @assert xprof_convert_raw_to_tool_data[] !== nothing "xprof is not installed."
    @assert ispath(xplane_file) "xplane_file $xplane_file does not exist."

    # try from overview_page first
    try
        res = extract_mean_step_time_from_overview_page(xplane_file, nrepeat)
        res !== nothing && return res
    catch
    end
    @debug "Failed to extract mean step time from overview_page"
    return extract_mean_step_time_from_hlo_op_profile(xplane_file, nrepeat)
end

function extract_mean_step_time_from_overview_page(xplane_file::String, nrepeat::Int)
    xplane_files = PythonCall.pylist([xplane_file])
    pydata = xprof_convert_raw_to_tool_data[].xspace_to_tool_data(
        xplane_files, "overview_page", PythonCall.pydict()
    )[0]

    overview_data = JSON3.read(pyconvert(Vector{UInt8}, pydata))

    step_table = overview_data[2]
    cols = step_table["cols"]
    rows = step_table["rows"]

    col_indices = Dict{String,Int}(col["id"] => i for (i, col) in enumerate(cols))
    step_time_idx = get(col_indices, "stepTimeMs", nothing)

    (step_time_idx === nothing || length(rows) == 0) && return nothing

    step_times_ms = Float64[]
    for row in rows
        cells = get(row, "c", [])
        if step_time_idx <= length(cells)
            step_time = get(cells[step_time_idx], "v", 0.0)
            if step_time > 0
                push!(step_times_ms, step_time)
            end
        end
    end

    length(step_times_ms) == 0 && return nothing
    return sum(step_times_ms) / length(step_times_ms)
end

function extract_mean_step_time_from_hlo_op_profile(xplane_file::String, nrepeat::Int)
    xplane_files = PythonCall.pylist([xplane_file])

    pydata = xprof_convert_raw_to_tool_data[].xspace_to_tool_data(
        xplane_files, "op_profile", PythonCall.pydict()
    )[0]

    jl_data = pyconvert(Vector{UInt8}, pydata)
    picosec = JSON3.read(jl_data)["byProgram"]["metrics"]["normalizedTimePs"]
    return (picosec / 1e9) / nrepeat
end

export extract_mean_step_time, profile_with_xprof

end
