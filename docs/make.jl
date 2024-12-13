pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../lib/ReactantCore/"))

using Reactant, ReactantCore
using Documenter, DocumenterVitepress

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

pages = [
    "Reactant.jl" => "index.md",
    "Introduction" => ["Getting Started" => "introduction/index.md"],
    "Tutorials" => ["Overview" => "tutorials/index.md"],
    "API Reference" => [
        "Reactant API" => "api/api.md",
        "Ops" => "api/ops.md",
        "Dialects" => [
            "ArithOps" => "api/arith.md",
            "Affine" => "api/affine.md",
            "Builtin" => "api/builtin.md",
            "Chlo" => "api/chlo.md",
            "Enzyme" => "api/enzyme.md",
            "Func" => "api/func.md",
            "StableHLO" => "api/stablehlo.md",
            "VHLO" => "api/vhlo.md",
        ],
        "MLIR API" => "api/mlirc.md",
        "XLA" => "api/xla.md",
    ],
]

makedocs(;
    modules=[
        Reactant,
        ReactantCore,
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
    format=MarkdownVitepress(;
        repo="github.com/EnzymeAD/Reactant.jl",
        devbranch="main",
        devurl="dev",
        # md_output_path=".",    # Uncomment for local testing
        # build_vitepress=false, # Uncomment for local testing
    ),
    # clean=false, # Uncomment for local testing
    pages=pages,
    doctest=true,
    warnonly=[:cross_references],
)

deploydocs(; repo="github.com/EnzymeAD/Reactant.jl", devbranch="main", push_preview=true)
