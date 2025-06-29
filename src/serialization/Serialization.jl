"""
Implements serialization of Reactant compiled functions. Currently supported formats are:

- [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model)
"""
module Serialization

using ..Reactant: Reactant, MLIR

serialization_supported(::Val) = false

include("TFSavedModel.jl")

end
