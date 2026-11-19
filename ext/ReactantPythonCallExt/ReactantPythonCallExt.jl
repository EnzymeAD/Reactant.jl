module ReactantPythonCallExt

using PythonCall:
    PythonCall,
    Py,
    pyconvert,
    pycontains,
    pyeval,
    pyexec,
    pygetattr,
    pyimport,
    pyisinstance,
    pylist,
    pytuple
using PrettyTables: PrettyTables
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
const TORCH_INIT_ATTEMPTED = Ref{Bool}(false)

function __init__()
    try
        jaxptr[] = pyimport("jax")
        jnpptr[] = pyimport("jax.numpy")
        JAX_TRACING_SUPPORTED[] = true
    catch err
        @warn "Failed to import jax. Tracing jax functions invoked with pycall won't \
               be supported." exception = (err, catch_backtrace())
    end
    return nothing
end

# torch is set up on first use rather than in `__init__`, so code that never touches
# PyTorch does not pay for importing it, and torch's OpenMP runtime is not loaded next
# to jax's (on Windows the two can abort the process). With `require=false` this only
# sets up torch when the user has already imported it: an object can only be a torch
# module if torch is in `sys.modules`.
function _torch_ready(require::Bool=true)
    TORCH_EXPORT_SUPPORTED[] && return true
    TORCH_INIT_ATTEMPTED[] && return false
    require || pycontains(pyimport("sys").modules, "torch") || return false
    TORCH_INIT_ATTEMPTED[] = true
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
        @warn "Failed to import torch/torchax. Importing PyTorch modules won't be \
               supported." exception = (err, catch_backtrace())
    end
    return TORCH_EXPORT_SUPPORTED[]
end

include("overlays.jl")
include("pycall.jl")
include("pytorch.jl")

end
