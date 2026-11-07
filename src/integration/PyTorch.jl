# Functions of the PyTorch import that live in the PythonCall extension
# (ReactantPythonCallExt), declared here so that they are part of Reactant's API.

"""
    import_torch(model; backend=:torchax) -> (tm, ps, st)

Import a PyTorch `torch.nn.Module` (or a TorchScript `ScriptModule`) given as a
PythonCall `Py`. Returns a callable `tm`, the module's parameters `ps` and its buffers
`st`. `ps` and `st` are flat `NamedTuple`s of Reactant arrays keyed by the module's fully
qualified names, the same names `state_dict()` uses: `ps.var"0.weight"`,
`st.var"bn.running_mean"` or `st[Symbol("bn.running_mean")]`. Parameters
(`named_parameters()`) are in `ps` and buffers (`named_buffers()`) in `st`. A 0-dim
tensor becomes a `ConcreteRNumber`.

Call `tm` inside [`@jit`](@ref) or [`@compile`](@ref) as `tm(x..., ps, st)`. The call
always returns `(y, st_new)`. `ps` and `st` are arguments of the compiled function, not
embedded constants. Buffers the module updates in place, such as BatchNorm running
statistics in training mode (`model.train()`), are replaced in `st_new`; every other leaf
is the corresponding leaf of `st`. Updates in evaluation mode are applied as well, with a
warning, since they are usually unintended. The mode is read from `model.training` when
the call is traced.

Calling the module directly, `@jit model(x)`, is a shortcut that runs it in evaluation
mode with its state embedded as constants and returns only `y`.

Being plain `NamedTuple`s, `ps` and `st` work with `keys`, `pairs`, `merge` and
`Base.setindex`, for example `merge(ps, (; var"0.weight"=w))` to swap in new weights.
`display(tm)` lists every tensor with its element type, size and owning submodule;
[`torch_layout`](@ref) returns the same information as data.

`backend` names the lowering that turns the captured program into StableHLO.
`:torchax` (torchax and jax) is currently the only one.

Requires PythonCall with `torch` and `torchax` installed.
"""
function import_torch end

"""
    torch_layout(tm) -> Vector{TorchStateEntry}
    torch_layout(tm, fqn) -> TorchStateEntry

Describe where every parameter and buffer of a module imported with
[`import_torch`](@ref) lives. Each entry records the tensor's FQN, whether it is a
`:parameter` (a leaf of `ps`) or a `:buffer` (a leaf of `st`), whether it is
persistent (part of the module's `state_dict()`), the class of the submodule that owns
it, and its element type and size. Entries are in the order of `keys(ps)` followed by
`keys(st)`.

```julia
julia> torch_layout(tm, "1.running_mean")
TorchStateEntry("1.running_mean" => st.var"1.running_mean", Float32 (16,), BatchNorm1d)
```
"""
function torch_layout end
