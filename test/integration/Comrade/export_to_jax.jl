using Pkg;
Pkg.activate(@__DIR__);
ENV["XLA_REACTANT_GPU_PREALLOCATE"] = "false"
using Reactant, NPZ, PythonCall

include("comimager.jl")

logdensity_python_file_path = Reactant.Serialization.export_to_enzymejax(
    logdensityof, tpostr, xr; output_dir=joinpath(@__DIR__, "Sampling/Serialized/Fwd")
)

println("Exported to: $logdensity_python_file_path")

grad_logdensity_python_file_path = Reactant.Serialization.export_to_enzymejax(
    gl, tpostr, xr; output_dir=joinpath(@__DIR__, "Sampling/Serialized/Bwd")
)

println("Exported to: $grad_logdensity_python_file_path")
