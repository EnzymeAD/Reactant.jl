using Reactant, ReactantCore
using Documenter, DocumenterVitepress

DocMeta.setdocmeta!(Reactant, :DocTestSetup, :(using Reactant); recursive=true)

# Helper functions
function first_letter_uppercase(str)
    return uppercase(str[1]) * str[2:end]
end

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
    "Introduction" => [
        "Getting Started" => "introduction/index.md",
        "Configuration" => "introduction/configuration.md",
    ],
    "Tutorials" =>
        ["Overview" => "tutorials/index.md", "Profiling" => "tutorials/profiling.md"],
    "API Reference" => [
        "Reactant API" => "api/api.md",
        "Ops" => "api/ops.md",
        "Dialects" => sort!(
            [
                first_letter_uppercase(first(splitext(basename(file)))) =>
                    joinpath("api/dialects", file) for
                file in readdir(joinpath(@__DIR__, "src/api/dialects")) if
                splitext(file)[2] == ".md"
            ];
            by=first,
        ),
        "MLIR API" => "api/mlirc.md",
        "XLA" => "api/xla.md",
        "Internal API" => "api/internal.md",
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
        filter(
            Base.Fix2(isa, Module),
            [
                getproperty(Reactant.MLIR.Dialects, x) for
                x in names(Reactant.MLIR.Dialects; all=true) if x != :Dialects
            ],
        )...,
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
