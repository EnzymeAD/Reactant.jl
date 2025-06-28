module ReactantPythonCallExt

using PythonCall
using Reactant: Reactant, TracedRArray

const jaxptr = Ref{Py}()
const jnpptr = Ref{Py}()

const JAX_TRACING_SUPPORTED = Ref{Bool}(false)

const tfptr = Ref{Py}()
const tf2xlaptr = Ref{Py}()

const SAVED_MODEL_EXPORT_SUPPORTED = Ref{Bool}(false)

const NUMPY_SIMPLE_TYPES = Dict(
    Bool => :bool,
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
        tf2xlaptr[] = pyimport("tensorflow.compiler.tf2xla.python.xla")
        SAVED_MODEL_EXPORT_SUPPORTED[] = true
    catch err
        @warn "Failed to import tensorflow. Exporting Reactant compiled functions as \
               tensorflow SavedModel will not be \
               supported." exception = (err, catch_backtrace())
    end
    return nothing
end

include("pycall.jl")

end
