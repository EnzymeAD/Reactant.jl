# [Debugging compilation errors](@id debugging-compilation)

It may happen that compilation of some functions via Reactant may fail, or [raising](@ref Raising) may fail inside the Enzyme-JAX backend.
Not to worry, we're here to help!
But we need some help from you in order to look into the issue.
Below are some information about how to gather information that can be useful to the Reactant developers to solve your problems.

## Enabling debugging messages in Reactant

One option is to enable debugging message inside the `Reactant.jl` package.
This can be achieved by setting the environment variable [`JULIA_DEBUG`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_DEBUG) to `Reactant`.
This can be achieved by either exporting the variable before starting the Julia process

```sh
export JULIA_DEBUG=Reactant
julia ...

# or...

JULIA_DEBUG=Reactant julia ...
```

or inside your Julia code/REPL with

```julia
ENV["JULIA_DEBUG"] = "Reactant"
```

before loading the Reactant module.
Some of the debugging messages can provide information

## Dumping all MLIR modules

During its compilation pipeline, Reactant generates several MLIR modules, which go through a few stages of optimisation before generating the native code for the device you want to run the code on.
By default these modules are simply kept in memory, but for debugging purposes (especially for raising failures) it may be useful to save to disk all MLIR modules, which you can then share with the Reactant developers to help them fix bugs.
To do this, inside your Julia code/REPL set

```julia
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
```

before the offending compilation.

!!! tip "Changing directory where the MLIR modules are saved"
	The MLIR modules are saved by default in a subdirectory whose name starts with `reactant_` under [`tempdir()`](https://docs.julialang.org/en/v1/base/file/#Base.Filesystem.tempdir).
	Regularly you don't need to do this, but in the remote case you wanted to point `tempdir()` somewhere else, see the docstring of that function for how to change the location of the temporary directory.
	Alternatively, if you want to change only the directory where the MLIR modules are saved to (without changing the location of `tempdir()`), in your Julia code set the variable
	```julia
	Reactant.MLIR.IR.DUMP_MLIR_DIR[] = /path/to/directory
	```
	together with setting `Reactant.MLIR.IR.DUMP_MLIR_ALWAYS` as suggested above.


When the stacktrace of the error thrown during compilation mentions the pass manager (e.g. functions like `mlirPassManagerRunOnOp`, `run_pass_pipeline!`, etc.), the file we want to look at is the last one saved to disk (they're numbered, using an increasing counter) whose name looks like `module_.*_pre_all_pm.mlir`.
