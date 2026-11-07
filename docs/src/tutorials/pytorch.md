# [Importing PyTorch models](@id pytorch-import)

Reactant can trace an existing PyTorch model and splice it into a Reactant
computation, in the same way it traces JAX functions invoked through
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). This lets you reuse a
`torch.nn.Module` (including its trained weights) from Julia without rewriting it.

This feature lives in the `ReactantPythonCallExt` extension and is enabled when
`PythonCall`, `torch`, and `torchax` are all available. Under the hood the model is
traced with [`torch.export`](https://pytorch.org/docs/stable/export.html), lowered
to StableHLO with [`torchax`](https://github.com/pytorch/xla/tree/master/torchax),
and inlined into the current trace.

There are two ways to call a module. `Reactant.import_torch` passes the parameters
and buffers as arguments of the compiled function and returns updated state, which
is what you want for large models, for changing weights without recompiling, and for
training mode. Calling the module directly is a shortcut that embeds the parameters
and buffers in the compiled module as constants, mirroring how JAX closure parameters
are captured; very large models then produce correspondingly large modules.

## Usage

Construct (or load) the PyTorch module and import it:

```julia
using Reactant, PythonCall

torch = pyimport("torch")
nn = pyimport("torch.nn")

model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4))
model.eval()

tm, ps, st = Reactant.import_torch(model)

x = Reactant.to_rarray(rand(Float32, 4, 8))   # (batch, features)
y, st = @jit tm(x, ps, st)
```

`ps` holds the module's parameters and `st` its buffers, as flat `NamedTuple`s of
Reactant arrays keyed by the module's fully qualified names (FQNs), the same keys as
`model.state_dict()`:

```julia
keys(ps)                         # (Symbol("0.weight"), Symbol("0.bias"), ...)
ps.var"0.weight"                 # 16×8 ConcreteRArray{Float32, 2}
st.var"1.running_mean"           # buffers are inputs in eval mode too
st[Symbol("1.running_mean")]     # the same leaf, from an FQN string
```

### What goes where

Every leaf of `ps` and `st` is exactly one tensor of the module, named by its fully
qualified name (FQN), the same name it has in `model.state_dict()`. Parameters go in
`ps` and buffers in `st`, including non-persistent buffers, which are not part of
`state_dict()`. Displaying `tm` lists every tensor (for a large model the REPL shows
the first and last rows of each table; `Reactant.torch_layout(tm)` returns all of
them):

```julia-repl
julia> tm
TorchModule(Sequential) with 0 exports
  parameters (ps), 6 tensors, 244 elements:
     FQN       eltype   shape    owner
    -----------------------------------------
     0.weight  Float32  (16, 8)  Linear
     0.bias    Float32  (16,)    Linear
     1.weight  Float32  (16,)    BatchNorm1d
     1.bias    Float32  (16,)    BatchNorm1d
     3.weight  Float32  (4, 16)  Linear
     3.bias    Float32  (4,)     Linear
  buffers (st), 3 tensors, 33 elements:
     FQN                    eltype   shape  owner        persistent
    ----------------------------------------------------------------
     1.running_mean         Float32  (16,)  BatchNorm1d  true
     1.running_var          Float32  (16,)  BatchNorm1d  true
     1.num_batches_tracked  Int64    ()     BatchNorm1d  true
```

The same information is available as data:

```julia
Reactant.torch_layout(tm)                    # one TorchStateEntry per tensor
e = Reactant.torch_layout(tm, "1.running_mean")
e.kind, e.owner, e.eltype, e.shape           # (:buffer, "BatchNorm1d", Float32, (16,))
```

Because `ps` and `st` are plain `NamedTuple`s, the usual functions work on them:
`pairs(ps)` lists `fqn => leaf`, and `merge` returns a new tree with some leaves
replaced, for example new weights:

```julia
ps = merge(ps, (; var"3.weight"=Reactant.to_rarray(w)))   # w must be Float32 4×16
```

A replacement must keep the leaf's element type and size; a call with a mismatched
leaf raises an error naming the parameter or buffer.

### Saving and loading weights

Since the keys are the FQNs, `ps` and `st` save to any name-keyed format with the same
names PyTorch uses in `state_dict()`, with no separate mapping:

```julia
using Serialization

host(v) = v isa AbstractArray ? Array(v) : fill(Reactant.to_number(v))
weights = Dict(string(k) => host(v) for (k, v) in pairs(merge(ps, st)))
serialize("weights.jls", weights)
```

A `NamedTuple`'s key order is part of its type, and a compiled function only accepts
the order it was compiled for, so rebuild the trees in the module's order, given by
`keys(ps)` and `keys(st)` (or [`Reactant.torch_layout`](@ref)):

```julia
w = deserialize("weights.jls")
leaf(a) = ndims(a) == 0 ? ConcreteRNumber(a[]) : Reactant.to_rarray(a)
ps = NamedTuple{keys(ps)}(Tuple(leaf(w[string(k)]) for k in keys(ps)))
st = NamedTuple{keys(st)}(Tuple(leaf(w[string(k)]) for k in keys(st)))
```

After the first call, `display(tm)` also shows each cached export: which leaf of `ps`,
`st` or which input feeds each argument of the compiled program, and whether each
result becomes part of `y` or replaces a leaf of `st`.

### Calling the module

The call is `tm(x..., ps, st)`, with one argument per tensor input of `forward`, and
it always returns `(y, st)`, following the convention of Lux, Haiku and Equinox. When
`forward` returns several tensors, `y` is a tuple. Because `ps` and `st` are
arguments, you can swap in new values (for example weights updated by an optimizer)
and call `tm` again; each leaf must keep its element type and size, and a mismatch
raises an error naming the parameter or buffer.

Inputs are positional. A `forward` that takes keyword arguments, or nested tuples or
dictionaries of tensors, needs a small wrapper module that accepts each tensor as a
positional argument:

```julia
pyexec("""
import torch.nn as nn

class Positional(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
""", Main)

tm, ps, st = Reactant.import_torch(pyeval("Positional", Main)(model))
y, st = @jit tm(input_ids, attention_mask, ps, st)
```

The wrapper holds the original module as a submodule, so every key in `ps` and `st`
gains a `model.` prefix (`model.encoder.layer.0.attention.self.query.weight` rather
than `encoder.layer.0...`). Strip or add the prefix when matching keys against a
`state_dict()` saved from the unwrapped module.

`tm` exports the module once per mode and input signature (element types and sizes)
and caches the result, so calling it again with inputs of the same shape does not
re-export. `display(tm)` lists each cached export with its arguments and results.

### Training mode

The mode is read from `model.training` when the call is traced. The returned `st`
holds the new value of every buffer the module updates in place and the passed-in
value of every other buffer. For standard layers such as BatchNorm, updates happen
only in training mode, so feed the returned state into the next call:

```julia
model.train()
y, st = @jit tm(x, ps, st)   # BatchNorm uses batch statistics
st.var"1.num_batches_tracked"   # 1
y, st = @jit tm(x, ps, st)
st.var"1.num_batches_tracked"   # 2
```

A module that updates buffers in evaluation mode (a custom step counter, for example)
has those updates applied in the returned `st` too, matching eager PyTorch, and Reactant
warns once per export, since such updates are often unintended. Pass the original `st`
to the next call to discard them.

A function compiled with `@compile` keeps the mode it was traced with; compile it
again after calling `model.train()` or `model.eval()`. Exporting never changes the
module's own mode.

Modules that modify an input tensor or a parameter in place are rejected with an
error, since neither can be expressed by the `(y, st)` return value.

### Calling the module directly

For quick inference you can call the module itself:

```julia
y = @jit model(x)
```

This runs the module in evaluation mode, whatever `model.training` says, with its
parameters and buffers embedded as constants, and returns only `y`.

!!! note "TorchScript modules are modified"
    For TorchScript inputs the import rewraps leaf parameters as `torch.nn.Parameter`
    on the module object you pass in. If you need the original module untouched, pass
    a copy.

!!! note "Array layout"
    The Reactant array's shape is the PyTorch tensor shape directly, with the batch
    dimension first (for example `(batch, channels, height, width)` for a 2D
    convolution). There is no axis reversal. Provide inputs in the layout the model
    expects, exactly as you would when calling the model in Python.

## TorchScript models

TorchScript artifacts produced by `torch.jit.script` or `torch.jit.trace` are
supported as well. Load the module and use it the same way:

```julia
scripted = torch.jit.load("model.pt")
tm, ps, st = Reactant.import_torch(scripted)
y, st = @jit tm(x, ps, st)

y = @jit scripted(x)   # shortcut
```

A traced TorchScript module records the control flow of the mode it was traced in,
so its training behavior is fixed when the artifact is created. Modules whose mode
cannot change at all, frozen modules (`torch.jit.freeze`) and scripted modules that
list `training` in `__constants__`, import and run in evaluation mode; exporting them
in training mode raises an error naming the module. TorchScript does not record
whether a buffer is persistent, so every buffer of a TorchScript module is listed as
persistent in `torch_layout`.

## Lowering backends

The import has two stages. Capture uses only torch: `torch.export` and the graph
signature that names every tensor. A lowering backend then produces the StableHLO.
`:torchax` (torchax and jax) is currently the only backend, selected with
`import_torch(model; backend=:torchax)`; everything else described here is the same
whichever backend is used.

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
tm, ps, st = Reactant.import_torch(model)   # ps holds Float64 arrays
y, st = @jit tm(x, ps, st)                  # y is a Float64 Reactant array
```

jax disables 64-bit support by default and would otherwise canonicalize float64 to
float32 during lowering. The import path enables jax's x64 mode only for the duration
of the export, so float64 models keep double precision without changing jax's global
configuration.

## Limitations

- The model must be exportable with `torch.export`. Data-dependent control flow or
  shapes that `torch.export` cannot capture are not supported.
- Inputs and outputs must be tensors. Inputs are positional; keyword, nested or
  non-tensor arguments need a wrapper module (see "Calling the module" above).
- Element types are limited to those Reactant maps to StableHLO (the floating point,
  integer, and boolean types in `Reactant.Serialization.NUMPY_SIMPLE_TYPES`).
- `torchax` depends on JAX, so a JAX installation compatible with your `torchax`
  version is required.
- Use a CPU build of PyTorch (`torch==...+cpu`). A CUDA PyTorch wheel ships a
  `libtorch_cuda.so` that clashes with Reactant's runtime when imported after
  Reactant, and the import fails. The extension already prevents the related
  Triton/LLVM clash by blocking Triton during export, so no special import order is
  needed.
