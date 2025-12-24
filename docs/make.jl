using Reactant, ReactantCore, EnzymeCore
using Documenter, DocumenterVitepress

DocMeta.setdocmeta!(Reactant, :DocTestSetup, :(using Reactant); recursive=true)

makedocs(;
    modules=[
        Reactant,
        ReactantCore,
        EnzymeCore,
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
    authors=join(
        [
            "William Moses <wmoses@mit.edu>",
            "Valentin Churavy <vchuravy@mit.edu>",
            "Sergio Sánchez Ramírez <sergio.sanchez.ramirez@bsc.es>",
            "Paul Berg <paul@plutojl.org>",
            "Avik Pal <avikpal@mit.edu>",
            "Mosè Giordano <mose@gnu.org>",
        ],
        ", ",
    ),
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

DocumenterVitepress.deploydocs(;
    repo="github.com/EnzymeAD/Reactant.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
