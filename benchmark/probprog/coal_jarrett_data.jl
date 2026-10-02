# Loader for the Jarrett coal-mining disaster series used by the coal MH
# harnesses under COAL_DATA=jarrett (Impulse, Gen.jl). Dependency-free on
# purpose: gen_env has no JSON package. The file and its provenance are
# documented in pplbench/pplbench/models/real_data/README.md.
#
# Returns (T, tidx, counts): T = 111 years (1851..1961), so the switch ranges
# over s = 0..T; tidx = 0-based year index of each OBSERVED year (the two
# missing years, 1890 and 1934, are dropped and contribute no likelihood term;
# every other year keeps its calendar position); counts = the 109 observed
# counts in year order.
const COAL_JARRETT_JSON = joinpath(
    @__DIR__, "pplbench", "pplbench", "models", "real_data", "coal_mining_jarrett.json"
)

function load_coal_jarrett(path::AbstractString=COAL_JARRETT_JSON)
    txt = read(path, String)
    m = match(r"\"counts\"\s*:\s*\[([^\]]*)\]", txt)
    m === nothing && error("no counts array in $path")
    toks = strip.(split(m.captures[1], ","))
    T = length(toks)
    tidx = Int[]
    counts = Int[]
    for (i, tok) in enumerate(toks)
        tok == "null" && continue
        push!(tidx, i - 1)
        push!(counts, parse(Int, tok))
    end
    return T, tidx, counts
end
