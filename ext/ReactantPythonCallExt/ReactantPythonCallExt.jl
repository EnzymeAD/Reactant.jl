module ReactantPythonCallExt

using PythonCall: PythonCall, Py, pyconvert, pydict, pyfunc, pyimport, pylist
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay
using Reactant.Ops: @opcall
using Reactant.Serialization: NUMPY_SIMPLE_TYPES

const jaxptr = Ref{Py}()
const jnpptr = Ref{Py}()
const JAX_LOADED = Ref{Bool}(false)

const tfptr = Ref{Py}()
const tf2xlaptr = Ref{Py}()
const npptr = Ref{Py}()
const SAVED_MODEL_LOADED = Ref{Bool}(false)

function __init__()
    JAX_LOADED[] = false
    SAVED_MODEL_LOADED[] = false
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

function ensure_tf_loaded!()
    SAVED_MODEL_LOADED[] && return nothing
    try
        tf = pyimport("tensorflow")
        tf.config.set_visible_devices(pylist(); device_type="GPU")
        tf2xla = pyimport("tensorflow.compiler.tf2xla.python.xla")
        np = pyimport("numpy")
        tfptr[] = tf
        tf2xlaptr[] = tf2xla
        npptr[] = np
        SAVED_MODEL_LOADED[] = true
    catch
        error(
            "Failed to import `tensorflow` and `numpy`, which are required to export " *
            "Reactant compiled functions as a TensorFlow SavedModel. Install them in " *
            "your Python environment, e.g. with `using CondaPkg; " *
            "CondaPkg.add_pip(\"tensorflow\"); CondaPkg.add_pip(\"numpy\")` " *
            "(if your Python version lacks TensorFlow wheels, also cap Python via " *
            "`CondaPkg.add(\"python\"; version=\"<3.14\")`).",
        )
    end
    return nothing
end

include("overlays.jl")
include("pycall.jl")
include("saved_model.jl")

end
