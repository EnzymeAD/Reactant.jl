using Reactant, SafeTestsets, Test

if lowercase(get(ENV, "REACTANT_BACKEND_GROUP", "all")) == "gpu"
    Reactant.set_default_backend("gpu")
end

include("integration/cuda.jl")
