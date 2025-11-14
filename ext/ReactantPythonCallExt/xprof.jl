# Currently prototyping with xprof via python. we should instead add this into
# the C++ API.
function xspace_to_tools_data(filename::String, tool_name::String)
    if !XPROF_PROFILER_SUPPORTED[]
        error("xprof is not supported...")
    end

    return String(
        pyconvert(
            Vector{UInt8},
            xprofconvertptr[].xspace_to_tool_data(pylist([filename]), tool_name, pydict())[0],
        ),
    )
end
