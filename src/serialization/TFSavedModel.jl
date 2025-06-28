module TFSavedModel

using ..Reactant: AbstractConcreteArray, AbstractConcreteNumber, Compiler, MLIR

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

function export_as_saved_model(
    thunk::Compiler.Thunk,
    saved_model_path::String,
    target_version::VersionNumber,
    input_locations,
    state_dict::Dict{String, <:Union{<:AbstractConcreteArray,<:AbstractConcreteNumber}},
)
    isempty(thunk.module_string) && error(
        "To export a thunk, ensure that it has been compiled with `serializable=true`."
    )

    mlir_mod = parse(MLIR.IR.Module, thunk.module_string)
    display(mlir_mod)

    return nothing
end

end
