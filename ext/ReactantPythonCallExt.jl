module ReactantPythonCallExt

using PythonCall
using Reactant: Reactant, TracedRArray, AnyTracedRArray, MLIR, TracedRNumber
using ReactantCore: @trace

using PythonCall

const jaxptr = Ref{Py}()

function PythonCall.pycall(
    f::Py, arg0::Reactant.TracedRArray, argNs::Reactant.TracedRArray...; kwargs...
)
    jax = jaxptr[]
    numpy = jax.numpy
    inputs = map((arg0, argNs...)) do arg
        JT = eltype(arg)
        PT = nothing
        for (CPT, CJT) in PythonCall.Convert.NUMPY_SIMPLE_TYPES
            if JT == CJT
                PT = CPT
                break
            end
        end
        numpy.zeros(size(arg); dtype=getproperty(numpy, Symbol(PT)))
    end
    lowered = jax.jit(f).lower(inputs...)
    txt = pyconvert(String, lowered.as_text())
    res = Reactant.Ops.hlo_call(txt, arg0, argNs...)
    if length(res) == 0
        return nothing
    else
        return res[1]
    end
end

function __init__()
    return jaxptr[] = pyimport("jax")
end

end # module ReactantPythonCallExt
