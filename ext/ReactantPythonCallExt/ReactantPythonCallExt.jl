module ReactantPythonCallExt

using PythonCall:
    PythonCall, Py, pyconvert, pyeval, pyexec, pyimport, pyisinstance, pylist, pytuple
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay
using Reactant.Ops: @opcall
using Reactant.Serialization: NUMPY_SIMPLE_TYPES

const jaxptr = Ref{Py}()
const jnpptr = Ref{Py}()

const JAX_TRACING_SUPPORTED = Ref{Bool}(false)

const torchptr = Ref{Py}()
const npptr = Ref{Py}()
# Python helper that exports a torch module to StableHLO with its parameters and
# buffers as arguments and its buffer updates as results, using a named lowering
# backend (see pytorch.jl).
const torch_export_with_state = Ref{Py}()

const TORCH_EXPORT_SUPPORTED = Ref{Bool}(false)

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
        torchptr[] = pyimport("torch")
        npptr[] = pyimport("numpy")
        # The capture stage needs only torch; each lowering registers itself in
        # `_LOWERINGS` (see pytorch.jl). torchax is currently the only lowering.
        pyexec(TORCH_CAPTURE_PY, @__MODULE__)
        pyexec(TORCHAX_LOWERING_PY, @__MODULE__)
        torch_export_with_state[] = pyeval("_torch_export_with_state", @__MODULE__)
        TORCH_EXPORT_SUPPORTED[] = true
    catch err
        @warn "Failed to import torch/torchax. Importing PyTorch modules invoked with \
               pycall won't be supported." exception = (err, catch_backtrace())
    end
    return nothing
end

include("overlays.jl")
include("pycall.jl")
include("pytorch.jl")

end
