"""
Implements serialization of Reactant compiled functions. Currently supported formats are:

- [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model)
- [EnzymeJAX](https://github.com/EnzymeAD/Enzyme-JAX) export for JAX integration
"""
module Serialization

using ..Reactant: Reactant, Compiler

serialization_supported(::Val) = false

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

include("TFSavedModel.jl")
include("EnzymeJAX.jl")

"""
    export_as_tf_saved_model(
        thunk::Compiler.Thunk,
        saved_model_path::String,
        target_version::VersionNumber,
        input_locations::Vector=[],
        state_dict::Dict=Dict(),
    )

Serializes a compiled reactant function (aka `Reactant.Compiler.Thunk`) to a
[Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model) which can be used
for deployemnt.

## Arguments

  - `thunk`: The compiled function to serialize (output of `@compile`). For this to work,
    the thunk must be compiled with `serializable=true`.
  - `saved_model_path`: The path where the SavedModel will be saved.
  - `target_version`: The version for serialization.
  - `input_locations`: A vector of input locations. This must be an empty vector or a
    vector of size equal to the number of inputs of the function. Each element can be one
    of:

    - `TFSavedModel.VariableType`: This indicates whether the variable is an input variable
      or a parameter. A parameter is serialized as a constant in the SavedModel, while
      an input variable is required at runtime.
    - `String`: The name of a parameter. This requires a corresponding entry in the
      `state_dict` to be present.
    - `Integer`: The position of an input argument. This is used to indicate that the
      input is an input argument.

  - `state_dict`: A dictionary mapping parameter names to their values. This is used to
    serialize the parameters of the function.

## Example

```julia
using Reactant, PythonCall

function fn(x, y)
    return sin.(x) .+ cos.(y.x[1:2, :])
end

x = Reactant.to_rarray(rand(Float32, 2, 10))
y = (; x=Reactant.to_rarray(rand(Float32, 4, 10)))

compiled_fn = @compile serializable = true fn(x, y)

Reactant.Serialization.export_as_tf_saved_model(
    compiled_fn,
    "/tmp/test_saved_model",
    v"1.8.5",
    [
        Reactant.Serialization.TFSavedModel.InputArgument(1),
        Reactant.Serialization.TFSavedModel.Parameter("y.x"),
    ],
    Dict("y.x" => y.x),
)
```

```python
import tensorflow as tf
import numpy as np

restored_model = tf.saved_model.load("/tmp/test_saved_model")

# Note that size of the input in python is reversed compared to Julia.
x = tf.constant(np.random.rand(10, 2))
restored_model.f(x)
```
"""
function export_as_tf_saved_model(
    thunk::Compiler.Thunk,
    saved_model_path::String,
    target_version::VersionNumber,
    input_locations::Vector=[],
    state_dict::Dict=Dict(),
)
    _input_locations = TFSavedModel.VariableType[]
    for loc in input_locations
        if loc isa TFSavedModel.VariableType
            push!(_input_locations, loc)
        elseif loc isa String
            push!(_input_locations, TFSavedModel.Parameter(loc))
        elseif loc isa Integer
            push!(_input_locations, TFSavedModel.InputArgument(Int(loc)))
        else
            error("Unsupported input location type: $(typeof(loc))")
        end
    end

    return TFSavedModel.export_as_saved_model(
        thunk, saved_model_path, target_version, _input_locations, state_dict
    )
end

const export_to_enzymejax = EnzymeJAX.export_to_enzymejax

end
