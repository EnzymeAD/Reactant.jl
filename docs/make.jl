using Reactant, ReactantCore
using Documenter, DocumenterVitepress

DocMeta.setdocmeta!(Reactant, :DocTestSetup, :(using Reactant); recursive=true)

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
    doctest=true,
    warnonly=[:cross_references],
)

deploydocs(; repo="github.com/EnzymeAD/Reactant.jl", devbranch="main", push_preview=true)
