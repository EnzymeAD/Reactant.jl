#!/usr/bin/env julia
"""
    run_mlir.jl — CLI wrapper for Reactant.Serialization.generate_mlir_runner

Usage:
    julia --project=Reactant.jl scripts/run_mlir.jl [file1.mlir] [file2.mlir ...] [output.jl]

Accepts N MLIR files and an optional output path (any .jl argument).
Marshaling between sequential modules is driven automatically by
`tf.aliasing_output` attributes in the MLIR IR.
"""

using Reactant

mlir_files = filter(f -> endswith(f, ".mlir"), ARGS)
out_idx = findfirst(f -> endswith(f, ".jl"), ARGS)
output_path = out_idx !== nothing ? ARGS[out_idx] : "execute.jl"

isempty(mlir_files) && error("No .mlir files provided. Usage: run_mlir.jl [file1.mlir ...] [output.jl]")

println("=== run_mlir.jl — MLIR → script generator ===")
for (i, f) in enumerate(mlir_files)
    println("  Module $i: $f")
end
println("  Output:   $output_path")

Reactant.Serialization.generate_mlir_runner(mlir_files; output_path)

println("Done. Run with:  julia --project=Reactant.jl $output_path [--cpu]")
