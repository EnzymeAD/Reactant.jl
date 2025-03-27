using Documenter, JSON
using OffsetArrays

DocMeta.setdocmeta!(OffsetArrays, :DocTestSetup, :(using OffsetArrays); recursive=true)

makedocs(
    sitename = "OffsetArrays",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md", "internals.md", "reference.md"],
    modules = [OffsetArrays],
    doctestfilters = [r"at \./.*", r"at /home.*", r"top-level scope.*", r"\[\d*\]\s*$"],   # for backtraces
)

# a workdaround to github action that only push preview when PR has "push_preview" labels
# issue: https://github.com/JuliaDocs/Documenter.jl/issues/1225
function should_push_preview(event_path = get(ENV, "GITHUB_EVENT_PATH", nothing))
    event_path === nothing && return false
    event = JSON.parsefile(event_path)
    haskey(event, "pull_request") || return false
    labels = [x["name"] for x in event["pull_request"]["labels"]]
    return "push_preview" in labels
 end

deploydocs(
    repo = "github.com:JuliaArrays/OffsetArrays.jl.git",
    push_preview = should_push_preview()
)
