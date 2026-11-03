module TFSavedModel

using ..Serialization: NUMPY_SIMPLE_TYPES
using ..Reactant: Reactant, Compiler, MLIR

# https://github.com/openxla/stablehlo/blob/955fa7e6e3b0a6411edc8ff6fcce1e644440acbd/stablehlo/integrations/python/stablehlo/savedmodel/stablehlo_to_tf_saved_model.py

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
    input_locations::Vector=[],
    state_dict::Dict=Dict(),
)
    isempty(thunk.module_string) && error(
        "To export a thunk, ensure that it has been compiled with `serializable=true`."
    )

    ensure_saved_model_supported()

    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        MLIR.IR.activate(ctx)
        try
            mlir_mod = parse(MLIR.IR.Module, thunk.module_string)
            ftype = MLIR.IR.FunctionType(first(MLIR.IR.body(mlir_mod)))

            input_signature = [
                VariableSignature(
                    collect(Int64, size(MLIR.IR.input(ftype, i))),
                    NUMPY_SIMPLE_TYPES[MLIR.IR.julia_type(eltype(MLIR.IR.input(ftype, i)))],
                ) for i in 1:MLIR.IR.ninputs(ftype)
            ]

            output_signature = [
                VariableSignature(
                    collect(Int64, size(MLIR.IR.result(ftype, i))),
                    NUMPY_SIMPLE_TYPES[MLIR.IR.julia_type(
                        eltype(MLIR.IR.result(ftype, i))
                    )],
                ) for i in 1:MLIR.IR.nresults(ftype)
            ]

            if isempty(input_locations)
                input_locations = [InputArgument(i) for i in 1:length(input_signature)]
            end

            @assert length(input_locations) == length(input_signature) "The number of input \
                                                                        locations must match the \
                                                                        number of input signatures."

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
        finally
            MLIR.IR.deactivate(ctx)
        end
    end
end

function to_tf_saved_model(fn_spec::ReactantFunctionSpec, path::String)
    ensure_saved_model_supported()
    return __to_tf_saved_model(fn_spec, path)
end

pythoncall_ext_loaded(::Val) = false

function ensure_saved_model_supported()
    if !pythoncall_ext_loaded(Val(:PythonCall))
        error(
            "Serialization to SavedModel is not supported because `PythonCall` hasn't " *
            "been installed and loaded. Load it with `using PythonCall` and ensure the " *
            "`tensorflow` Python package is installed (e.g. with " *
            "`using CondaPkg; CondaPkg.add_pip(\"tensorflow\")`).",
        )
    end
    return __ensure_tf_loaded!()
end

# Defined in the PythonCallExt module
function __ensure_tf_loaded! end
function __to_tf_saved_model end

end
