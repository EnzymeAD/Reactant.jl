```@meta
CollapsedDocStrings = true
```

# Serialization

```@docs
Reactant.Serialization
```

## Exporting to TensorFlow SavedModel

!!! note "Load PythonCall"

    Serialization to TensorFlow SavedModel requires PythonCall to be loaded. Loading
    PythonCall will automatically install tensorflow. If tensorflow installation fails,
    we won't be able to export to SavedModel.

A SavedModel contains a complete TensorFlow program, including trained parameters (i.e,
tf.Variables) and computation. It does not require the original model building code to run,
which makes it useful for sharing or deploying with [TFLite](https://tensorflow.org/lite),
[TensorFlow.js](https://js.tensorflow.org/),
[TensorFlow Serving](https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple),
or [TensorFlow Hub](https://tensorflow.org/hub). Refer to the
[official documentation](https://www.tensorflow.org/guide/saved_model) for more details.

```@docs
Reactant.Serialization.export_as_tf_saved_model
```

## Exporting to JAX via EnzymeAD

!!! note "Load NPZ"

    This export functionality requires the `NPZ` package to be loaded.

This export functionality generates:

1. A `.mlir` file containing the StableHLO representation of your Julia function
2. Input `.npz` files containing the input arrays for the function
3. A Python script that wraps the function for use with `enzyme_ad.jax.hlo_call`

The generated Python script can be immediately used with JAX and EnzymeAD without any
additional Julia dependencies.

```@docs
Reactant.Serialization.EnzymeJAX.export_to_enzymejax
```
