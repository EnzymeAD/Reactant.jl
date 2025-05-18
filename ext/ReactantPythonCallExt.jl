module ReactantPythonCallExt

using PythonCall
using Reactant: Reactant, TracedRArray

const jaxptr = Ref{Py}()

const NUMPY_SIMPLE_TYPES = Dict(
    Bool => :bool_,
    Int8 => :int8,
    Int16 => :int16,
    Int32 => :int32,
    Int64 => :int64,
    UInt8 => :uint8,
    UInt16 => :uint16,
    UInt32 => :uint32,
    UInt64 => :uint64,
    Float16 => :float16,
    Float32 => :float32,
    Float64 => :float64,
    ComplexF16 => :complex32,
    ComplexF32 => :complex64,
    ComplexF64 => :complex128,
)

function PythonCall.pycall(f::Py, arg0::TracedRArray, argNs::TracedRArray...; kwargs...)
    jax = jaxptr[]
    numpy = jax.numpy
    inputs = map((arg0, argNs...)) do arg
        numpy.zeros(
            size(arg);
            dtype=getproperty(numpy, NUMPY_SIMPLE_TYPES[Reactant.unwrapped_eltype(arg)]),
        )
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
