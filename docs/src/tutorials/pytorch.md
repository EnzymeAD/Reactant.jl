# Importing PyTorch models

Reactant can trace an existing PyTorch model and splice it into a Reactant
computation, in the same way it traces JAX functions invoked through
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). This lets you reuse a
`torch.nn.Module` (including its trained weights) from Julia without rewriting it.

This feature lives in the `ReactantPythonCallExt` extension and is enabled when
`PythonCall`, `torch`, and `torchax` are all available. Under the hood the model is
traced with [`torch.export`](https://pytorch.org/docs/stable/export.html), lowered
to StableHLO with [`torchax`](https://github.com/pytorch/xla/tree/master/torchax),
and inlined into the current trace. The module's parameters and buffers are carried
across as inlined constants, mirroring how JAX closure parameters are captured. As
with closure-captured values in the JAX path, this embeds the weights directly in
the compiled module, so very large models produce correspondingly large modules.

## Usage

Construct (or load) the PyTorch module, then call it inside `@compile`/`@jit` with
Reactant arrays:

```julia
using Reactant, PythonCall

torch = pyimport("torch")
nn = pyimport("torch.nn")

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

x = Reactant.to_rarray(rand(Float32, 4, 8))   # (batch, features)
y = @jit model(x)                              # traced, compiled, executed
```

You provide only the model object and the Reactant input arrays; the weights are
read from the module automatically.

!!! note "The module is mutated in place"
    Importing a module switches it into eval mode (`model.eval()`), and for
    TorchScript inputs it also rewraps leaf parameters as `torch.nn.Parameter` on
    the module object you pass in. If you need the original module untouched, pass a
    copy.

!!! note "Array layout"
    The Reactant array's shape is the PyTorch tensor shape directly, with the batch
    dimension first (for example `(batch, channels, height, width)` for a 2D
    convolution). There is no axis reversal. Provide inputs in the layout the model
    expects, exactly as you would when calling the model in Python.

## TorchScript models

TorchScript artifacts produced by `torch.jit.script` or `torch.jit.trace` are
supported as well. Load the module and call it the same way:

```julia
scripted = torch.jit.load("model.pt")
y = @jit scripted(x)
```

## Matmul precision

The float32 matmul precision of an imported model is fixed by jax when it lowers the
model to StableHLO, and it is baked into the exported program. jax chooses it from
the platform it detects in the current process at trace time, not from the backend
Reactant ultimately runs on, so the two can disagree:

- With a GPU visible to jax (the default when `CUDA_VISIBLE_DEVICES` is unset, or any
  non-empty value), jax lowers float32 matmuls at reduced, TF32-style precision. That
  reduced precision is frozen in, so it applies even if Reactant then executes on CPU.
  On a CPU-only host this shows up as a roughly `5e-4` relative difference from eager
  PyTorch rather than the roughly `1e-7` of full float32.
- With only CPU detected (`CUDA_VISIBLE_DEVICES=""` hides all GPUs), jax lowers at
  full float32, and that too is frozen in, so you will not get the TF32 speedup even
  if you later run the compiled module on a GPU. TF32 also requires an Ampere or newer
  GPU in the first place.

The import path deliberately follows jax's default so callers can trade accuracy for
speed, and it emits a one-time warning reporting the precision in effect. For a
deterministic, platform-independent result, set the precision explicitly before
tracing:

```julia
pyimport("jax").config.update("jax_default_matmul_precision", "highest")
```

`"highest"` gives full float32 on any platform; the integration tests use it so their
comparisons against eager PyTorch are exact.

## Float64 (double precision)

Double precision models work directly. Build the module and the Reactant input in
`Float64` and call it as usual:

```julia
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
model.double()

x = Reactant.to_rarray(rand(Float64, 4, 8))
y = @jit model(x)            # y is a Float64 Reactant array
```

jax disables 64-bit support by default and would otherwise canonicalize float64 to
float32 during lowering. The import path enables jax's x64 mode only for the duration
of the export, so float64 models keep double precision without changing jax's global
configuration.

## Limitations

- The model must be exportable with `torch.export`. Data-dependent control flow or
  shapes that `torch.export` cannot capture are not supported.
- Inputs and outputs must be tensors. Nested or non-tensor arguments are not
  supported.
- Element types are limited to those Reactant maps to StableHLO (the floating point,
  integer, and boolean types in `Reactant.Serialization.NUMPY_SIMPLE_TYPES`).
- `torchax` depends on JAX, so a JAX installation compatible with your `torchax`
  version is required.
- Use a CPU build of PyTorch (`torch==...+cpu`). A CUDA PyTorch wheel ships a
  `libtorch_cuda.so` that clashes with Reactant's runtime when imported after
  Reactant, and the import fails. The extension already prevents the related
  Triton/LLVM clash by blocking Triton during export, so no special import order is
  needed.
