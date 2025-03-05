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
    "Introduction" => [
        "Getting Started" => "introduction/index.md",
        "Configuration" => "introduction/configuration.md",
    ],
    "Tutorials" =>
        ["Overview" => "tutorials/index.md", "Profiling" => "tutorials/profiling.md"],
    "API Reference" => [
        "Reactant API" => "api/api.md",
        "Ops" => "api/ops.md",
        "Dialects" => [
            "ArithOps" => "api/dialects/arith.md",
            "Affine" => "api/dialects/affine.md",
            "Builtin" => "api/dialects/builtin.md",
            "Chlo" => "api/dialects/chlo.md",
            "Enzyme" => "api/dialects/enzyme.md",
            "Func" => "api/dialects/func.md",
            "StableHLO" => "api/dialects/stablehlo.md",
            "VHLO" => "api/dialects/vhlo.md",
            "GPU" => "api/dialects/gpu.md",
            "LLVM" => "api/dialects/llvm.md",
            "NVVM" => "api/dialects/nvvm.md",
            "TPU" => "api/dialects/tpu.md",
            "Triton" => "api/dialects/triton.md",
            "Shardy" => "api/dialects/shardy.md",
            "MPI" => "api/dialects/mpi.md",
            "MemRef" => "api/dialects/memref.md",
        ],
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
