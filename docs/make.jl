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
    Literate.markdown(example_filepath, OUTPUT_DIR; documenter=true)
end

examples = [
    title => joinpath("generated", string(name, ".md")) for (title, name) in examples
]

gh = Documenter.Remotes.GitHub("EnzymeAD", "Reactant.jl")

makedocs(;
    modules=[
        Reactant,
        Reactant.XLA,
        Reactant.MLIR,
        Reactant.MLIR.API,
        Reactant.MLIR.IR,
        Reactant.MLIR.Dialects.chlo,
        Reactant.MLIR.Dialects.vhlo,
        Reactant.MLIR.Dialects.stablehlo,
        Reactant.MLIR.Dialects.enzyme,
        Reactant.MLIR.Dialects.arith,
        Reactant.MLIR.Dialects.func,
        Reactant.MLIR.Dialects.affine,
        Reactant.MLIR.Dialects.builtin,
    ],
    authors="William Moses <wsmoses@illinois.edu>, Valentin Churavy <vchuravy@mit.edu>",
    sitename="Reactant.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://enzymead.github.io/Reactant.jl/",
        size_threshold_ignore=["api.md"],
        assets=[
            asset(
                "https://plausible.io/js/plausible.js";
                class=:js,
                attributes=Dict(Symbol("data-domain") => "enzyme.mit.edu", :defer => ""),
            ),
        ],
    ),
    pages=["Home" => "index.md", "API reference" => "api.md"],
    doctest=true,
    warnonly=true,
)

deploydocs(; repo="github.com/EnzymeAD/Reactant.jl", devbranch="main", push_preview=true)
