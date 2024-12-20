module ReactantPythonCallExt

using PythonCall
using Reactant: Reactant, TracedRArray, AnyTracedRArray, MLIR, TracedRNumber
using ReactantCore: @trace

using PythonCall

const jax = pyimport("jax")
const numpy = jax.numpy

function PythonCall.pycall(f::Py, args::Reactant.TracedRArray...; kwargs...)
    inputs = map(args) do arg
        trules = PythonCall.pyconvert_rules_cache(eltype(arg))
        numpy.array(size(arg), dtype=trules)
    end
    lowered = jax.jit(f).lower(inputs...)
    return Ops.hlo_call(
        pyconvert(String, lowered.as_text()),
        args...
    )
end

end # module ReactantPythonCallExt
