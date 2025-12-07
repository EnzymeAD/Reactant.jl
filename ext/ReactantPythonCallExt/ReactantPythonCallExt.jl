module ReactantPythonCallExt

using PythonCall: PythonCall, Py, pyconvert, pydict, pyfunc, pyimport, pylist
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay
using Reactant.Ops: @opcall
using Reactant.Serialization: NUMPY_SIMPLE_TYPES

const jaxptr = Ref{Py}()
const jnpptr = Ref{Py}()

const JAX_TRACING_SUPPORTED = Ref{Bool}(false)

const tfptr = Ref{Py}()
const tf2xlaptr = Ref{Py}()
const npptr = Ref{Py}()

const SAVED_MODEL_EXPORT_SUPPORTED = Ref{Bool}(false)

function __init__()
    try
        jaxptr[] = pyimport("jax")
        jnpptr[] = pyimport("jax.numpy")
        JAX_TRACING_SUPPORTED[] = true
    catch err
        @warn "Failed to import jax. Tracing jax functions invoked with pycall won't \
               be supported." exception = (err, catch_backtrace())
    end

    try
        tfptr[] = pyimport("tensorflow")
        tfptr[].config.set_visible_devices(pylist(); device_type="GPU")
        tf2xlaptr[] = pyimport("tensorflow.compiler.tf2xla.python.xla")
        npptr[] = pyimport("numpy")
        SAVED_MODEL_EXPORT_SUPPORTED[] = true
    catch err
        @warn "Failed to import tensorflow. Exporting Reactant compiled functions as \
               tensorflow SavedModel will not be \
               supported." exception = (err, catch_backtrace())
    end
    return nothing
end

include("overlays.jl")
include("pycall.jl")
include("saved_model.jl")

end
