module ReactantPythonCallExt

using PythonCall: PythonCall, Py, pyconvert, pyimport
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay
using Reactant.Ops: @opcall
using Reactant.Serialization: NUMPY_SIMPLE_TYPES

const jaxptr = Ref{Py}()
const jnpptr = Ref{Py}()
const JAX_LOADED = Ref{Bool}(false)

function __init__()
    JAX_LOADED[] = false
    return nothing
end

function ensure_jax_loaded!()
    JAX_LOADED[] && return nothing
    try
        jax = pyimport("jax")
        jnp = pyimport("jax.numpy")
        jaxptr[] = jax
        jnpptr[] = jnp
        JAX_LOADED[] = true
    catch
        error(
            "Failed to import `jax`, which is required to trace Python functions " *
            "invoked with `PythonCall.pycall`. Install it in your Python environment, " *
            "e.g. with `using CondaPkg; CondaPkg.add_pip(\"jax\")`.",
        )
    end
    return nothing
end

include("overlays.jl")
include("pycall.jl")

end
