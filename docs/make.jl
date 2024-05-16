pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Enzyme to environment stack

using Reactant
using Documenter

DocMeta.setdocmeta!(Reactant, :DocTestSetup, :(using Reactant); recursive=true)

# Generate examples

using Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

examples = Pair{String,String}[]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(;
    modules=[Reactant],
    authors="William Moses <wsmoses@illinois.edu>, Valentin Churavy <vchuravy@mit.edu>",
    repo="https://github.com/EnzymeAD/Reactant.jl/blob/{commit}{path}#{line}",
    sitename="Reactant.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://enzymead.github.io/Reactant.jl/",
        assets = [
            asset("https://plausible.io/js/plausible.js",
                    class=:js,
                    attributes=Dict(Symbol("data-domain") => "enzyme.mit.edu", :defer => "")
                )
	    ],
    ),
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
    doctest = true,
    strict = true,
)

deploydocs(;
    repo="github.com/EnzymeAD/Reactant.jl",
    devbranch = "main",
    push_preview = true,
)
