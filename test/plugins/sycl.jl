using Reactant

original_backend = Reactant.XLA.default_backend()
Reactant.set_default_backend("sycl")

include("common.jl")

Reactant.set_default_backend(original_backend)
