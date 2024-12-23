module ReactantPythonCallExt

using PythonCall
using Reactant: Reactant, TracedRArray, AnyTracedRArray, MLIR, TracedRNumber
using ReactantCore: @trace

using PythonCall

const jaxptr = Ref{Py}()


const NUMPY_SIMPLE_TYPES = (
    ("bool_", Bool),
    ("int8", Int8),
    ("int16", Int16),
    ("int32", Int32),
    ("int64", Int64),
    ("uint8", UInt8),
    ("uint16", UInt16),
    ("uint32", UInt32),
    ("uint64", UInt64),
    ("float16", Float16),
    ("float32", Float32),
    ("float64", Float64),
    ("complex32", ComplexF16),
    ("complex64", ComplexF32),
    ("complex128", ComplexF64),
)

function PythonCall.pycall(
    f::Py, arg0::Reactant.TracedRArray, argNs::Reactant.TracedRArray...; kwargs...
)
    jax = jaxptr[]
    numpy = jax.numpy
    inputs = map((arg0, argNs...)) do arg
        JT = eltype(arg)
        PT = nothing
        for (CPT, CJT) in NUMPY_SIMPLE_TYPES
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
