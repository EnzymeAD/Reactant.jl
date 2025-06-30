module TFSavedModel

using ..Serialization: serialization_supported
using ..Reactant: AbstractConcreteArray, AbstractConcreteNumber, Compiler, MLIR

# https://github.com/openxla/stablehlo/blob/955fa7e6e3b0a6411edc8ff6fcce1e644440acbd/stablehlo/integrations/python/stablehlo/savedmodel/stablehlo_to_tf_saved_model.py

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

struct VariableSignature
    shape::Vector{Int}
    dtype::Symbol
end

abstract type VariableType end

struct InputArgument <: VariableType
    position::Int
end

struct Parameter <: VariableType
    name::String
end

struct ReactantFunctionSpec
    input_signature::Vector{VariableSignature}
    output_signature::Vector{VariableSignature}
    input_locations::Vector{<:VariableType}
    bytecode::Base.CodeUnits{UInt8,String}
    state_dict::Dict
end

function export_as_saved_model(
    thunk::Compiler.Thunk,
    saved_model_path::String,
    target_version::VersionNumber,
    input_locations::Vector,
    state_dict::Dict,
)
    isempty(thunk.module_string) && error(
        "To export a thunk, ensure that it has been compiled with `serializable=true`."
    )

    if !serialization_supported(Val(:SavedModel))
        error("Serialization to SavedModel is not supported. This might happen if \
               PythonCall hasn't been installed and loaded.")
    end

    mlir_mod = MLIR.IR.with_context() do ctx
        parse(MLIR.IR.Module, thunk.module_string)
    end

    ftype = MLIR.IR.FunctionType(first(MLIR.IR.body(mlir_mod)))

    input_signature = [
        VariableSignature(
            reverse(collect(Int64, size(MLIR.IR.input(ftype, i)))),
            # collect(Int64, size(MLIR.IR.input(ftype, i))),
            NUMPY_SIMPLE_TYPES[MLIR.IR.julia_type(eltype(MLIR.IR.input(ftype, i)))],
        ) for i in 1:MLIR.IR.ninputs(ftype)
    ]

    output_signature = [
        VariableSignature(
            reverse(collect(Int64, size(MLIR.IR.result(ftype, i)))),
            # collect(Int64, size(MLIR.IR.result(ftype, i))),
            NUMPY_SIMPLE_TYPES[MLIR.IR.julia_type(eltype(MLIR.IR.result(ftype, i)))],
        ) for i in 1:MLIR.IR.nresults(ftype)
    ]

    if isempty(input_locations)
        input_locations = [InputArgument(i) for i in 1:length(input_signature)]
    end

    c_print_callback = @cfunction(
        MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any)
    )
    ref = Ref(IOBuffer())
    result = MLIR.IR.LogicalResult(
        MLIR.API.stablehloSerializePortableArtifactFromModule(
            mlir_mod, string(target_version), c_print_callback, ref, true
        ),
    )
    MLIR.IR.isfailure(result) && throw("Couldn't serialize the module")
    serialized_module = codeunits(String(take!(ref[])))

    return to_tf_saved_model(
        ReactantFunctionSpec(
            input_signature,
            output_signature,
            input_locations,
            serialized_module,
            Dict(k => Array(v) for (k, v) in state_dict),
        ),
        saved_model_path,
    )
end

function to_tf_saved_model(fn_spec::ReactantFunctionSpec, path::String)
    if !serialization_supported(Val(:SavedModel))
        error("Serialization to SavedModel is not supported. This might happen if \
               PythonCall hasn't been installed and loaded.")
    end
    return __to_tf_saved_model(fn_spec, path)
end

# Defined in the PythonCallExt module
function __to_tf_saved_model end

end
