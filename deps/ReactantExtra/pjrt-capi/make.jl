using Clang.Generators
using Clang: CLLinkageSpec, children

# Recurse into `extern "C"` blocks in case a header is parsed in C++ mode (same as `make.jl`).
function Generators.collect_top_level_nodes!(
    nodes::Vector{ExprNode}, cursor::CLLinkageSpec, options
)
    for child in children(cursor)
        Generators.collect_top_level_nodes!(nodes, child, options)
    end
    return nodes
end

const CORE_HEADER = joinpath("xla", "pjrt", "c", "pjrt_c_api.h")

function xla_include_root(headers)
    idx = findfirst(h -> endswith(h, CORE_HEADER), headers)
    isnothing(idx) && error("`$(CORE_HEADER)` must be one of the input headers")
    return chopsuffix(headers[idx], CORE_HEADER)
end
