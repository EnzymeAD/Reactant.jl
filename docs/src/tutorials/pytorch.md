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
across as inlined constants, mirroring how JAX closure parameters are captured.

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
