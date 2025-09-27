module ReactantPythonCallExt

using PythonCall:
    PythonCall, Py, pyconvert, pydict, pyfunc, pyimport, pylist, pyisinstance, pytuple
using Reactant: Reactant, TracedRArray, TracedRNumber, @reactant_overlay
using Reactant.Ops: @opcall
using Reactant_jll: Reactant_jll

const jaxptr = Ref{Py}()
const jnpptr = Ref{Py}()

const JAX_TRACING_SUPPORTED = Ref{Bool}(false)

const tritonptr = Ref{Py}()

const TRITON_COMPILE_SUPPORTED = Ref{Bool}(false)

const tfptr = Ref{Py}()
const tf2xlaptr = Ref{Py}()
const npptr = Ref{Py}()

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
    ComplexF16 => :complex16,
    ComplexF32 => :complex32,
    ComplexF64 => :complex64,
)

const MLIR_TYPE_STRING = Dict(
    Float64 => "fp64",
    Float32 => "fp32",
    Float16 => "fp16",
    Int64 => "i64",
    Int32 => "i32",
    Int16 => "i16",
    Int8 => "i8",
    UInt64 => "ui64",
    UInt32 => "ui32",
    UInt16 => "ui16",
    UInt8 => "ui8",
    Bool => "i1",
    Reactant.F8E4M3FN => "fp8e4nv",
    Reactant.F8E5M2FNUZ => "fp8e5b16",
    Reactant.F8E4M3FNUZ => "fp8e4b8",
    Reactant.F8E5M2 => "fp8e5",
)
if isdefined(Core, :BFloat16)
    MLIR_TYPE_STRING[Core.BFloat16] = "bf16"
end

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
        tritonptr[] = pyimport("triton")
        TRITON_COMPILE_SUPPORTED[] = true
    catch err
        @warn "Failed to import triton. Compiling jax functions with triton won't be \
               supported." exception = (err, catch_backtrace())
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
