# Import a PyTorch model into a Reactant trace.
#
# `Reactant.import_torch(model)` returns `(tm, ps, st)`: a `TorchModule` wrapping the
# `torch.nn.Module`, its parameters `ps` and its buffers `st` as Reactant arrays in
# flat NamedTuples keyed by the module's fully qualified names (FQNs). Calling
# `tm(x..., ps, st)` inside `@jit`/`@compile` captures the module with `torch.export`,
# lowers it to StableHLO with a lowering backend (torchax and jax, currently the only
# one), and splices that StableHLO into the current trace via `@opcall hlo_call`.
# Parameters and buffers are operands of that call, so they are arguments of the
# compiled function rather than embedded constants. The call always returns
# `(y, st_new)`: in training mode the buffers that the module updates (BatchNorm running
# statistics, for example) are replaced in `st_new`; every other leaf is passed through
# unchanged.
#
# Calling a module directly (`@jit model(x)`) routes through the `PythonCall.pycall`
# overlay (see overlays.jl) to `pycall_with_torch_export`, which uses the same path
# with the parameters and buffers embedded as constants and returns only `y`.

# Python side of the export, in two stages.
#
# Capture (`TORCH_CAPTURE_PY`) uses only torch: it runs `torch.export` in the requested
# mode, applies PyTorch's core ATen decompositions, and reads the graph signature to
# label every placeholder (user input, parameter, buffer or lifted constant) and every
# output (user output or buffer update) by FQN. Nothing in it depends on how the graph
# is lowered.
#
# Lowering turns a captured program into StableHLO. Each lowering is a Python function
# `lower(ep, prefix)` registered in `_LOWERINGS` under a backend name, returning
# `(mlir_text, arguments, results, n_user)` where
#   - the entry function takes the user inputs, then the parameters, then the buffers,
#     in PyTorch axis order; parameters and buffers it does not read may be pruned
#   - `arguments` labels each entry argument in order as (kind, key, dtype, shape), with
#     `key` the 0-based input position or the FQN without the wrapper prefix
#   - the results are the user outputs followed by the updated buffers, and `results`
#     labels each as (kind, key)
# `TORCHAX_LOWERING_PY` provides the "torchax" lowering. A lowering may decompose the
# program further and must label it with `_signature` after doing so.
#
# `torch.export` opportunistically imports `triton`. triton ships its own copy of
# LLVM whose static initializers clash with the LLVM already loaded by Reactant when
# triton is loaded second, segfaulting the process. triton is not needed here, so its
# import is blocked for the duration of the export; this makes the conversion work
# regardless of whether torch or Reactant was imported first.
const TORCH_CAPTURE_PY = """
import sys
import torch as _torch
from torch.export.graph_signature import InputKind as _InputKind, OutputKind as _OutputKind

_LOWERINGS = {}

class _TritonBlocker:
    def find_spec(self, name, path, target=None):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton import blocked by ReactantPythonCallExt to avoid an LLVM clash with Reactant")
        return None

def _state_layout(model):
    # One (kind, fqn, persistent, owner class, tensor) row per parameter and buffer,
    # parameters first, each in the module's own registration order. A buffer is
    # persistent when it is part of `state_dict()`.
    in_state_dict = set(model.state_dict().keys())
    def owner(fqn):
        path = fqn.rpartition(".")[0]
        try:
            m = model.get_submodule(path) if path else model
        except Exception:
            return "?"
        # A ScriptModule reports its Python class through `original_name`.
        return getattr(m, "original_name", None) or type(m).__name__
    rows = [("parameter", n, True, owner(n), t) for n, t in model.named_parameters()]
    rows += [("buffer", n, n in in_state_dict, owner(n), t) for n, t in model.named_buffers()]
    return rows

def _strip(fqn, prefix):
    return fqn[len(prefix):] if prefix and fqn.startswith(prefix) else fqn

def _set_training(model, train, prefix):
    # Put every submodule in the requested mode. Two kinds of TorchScript module cannot
    # change mode: a frozen module (`torch.jit.freeze`) has no `training` attribute at
    # all, its mode having been folded into the graph (freezing requires eval mode),
    # and a scripted module that lists `training` in `__constants__` raises on any
    # assignment. Both are left alone when they already match the request; otherwise
    # the export is refused with the module's name rather than a TorchScript error.
    for name, m in model.named_modules():
        # The wrapper's own module and its `jit_mod` child are both the user's root.
        rel = name[len(prefix):] if prefix and (name + ".").startswith(prefix) else name
        name = repr(rel) if rel else "the root module"
        if not hasattr(m, "training"):
            if train:
                raise ValueError(
                    f"{name} is a frozen TorchScript module, whose evaluation mode is "
                    "fixed in its graph; it cannot be exported in training mode"
                )
            continue
        if bool(m.training) == train:
            continue
        try:
            m.training = train
        except RuntimeError as e:
            if "constant" not in str(e):
                raise
            raise ValueError(
                f"{name} has `training` fixed to {bool(m.training)} as a TorchScript "
                f"constant; it cannot be exported in "
                f"{'training' if train else 'evaluation'} mode"
            ) from None

# TorchScript modules (`torch.jit.script` / `torch.jit.trace`) need two adjustments
# before `torch.export` accepts them. JIT modules store parameters as plain tensors,
# while the exporter's verifier requires real `torch.nn.Parameter`, so each leaf
# parameter is re-wrapped. And `torch.export` expects an `nn.Module`, so the
# ScriptModule is wrapped in a minimal forwarding module; its state FQNs gain a
# `jit_mod.` prefix, which is stripped again so `ps` and `st` use the ScriptModule's own
# names.
def _fix_jit_parameters(mod):
    for name in list(mod._parameters.keys()):
        p = mod._parameters[name]
        if p is not None and not isinstance(p, _torch.nn.Parameter):
            mod._parameters[name] = _torch.nn.Parameter(p.detach().clone(), requires_grad=False)
    for child in mod._modules.values():
        if child is not None:
            _fix_jit_parameters(child)

class _JitWrapper(_torch.nn.Module):
    def __init__(self, jit_mod):
        super().__init__()
        self.jit_mod = jit_mod
    def forward(self, *args):
        # Call the compiled method directly. Going through the module's `__call__`
        # fails under torch.export for a module scripted in this process (rather than
        # loaded from a file): the export's module-call hook resolves `forward` to the
        # original Python method and calls it with the wrong arguments.
        return self.jit_mod._c.forward(*args)

def _capture(model, example_inputs, train, strict, prefix):
    # Export in the requested mode without leaving the caller's module changed.
    saved_modes = [(m, bool(m.training)) for m in model.modules() if hasattr(m, "training")]
    try:
        _set_training(model, train, prefix)
        exported = _torch.export.export(model, example_inputs, strict=strict)
        return exported.run_decompositions()
    finally:
        for m, mode in saved_modes:
            if bool(m.training) != mode:
                m.training = mode

def _signature(ep, prefix):
    # Label the program's placeholders and outputs. `bind` has one (kind, key) per
    # placeholder, in order: ("user_input", 0-based position) or (kind, FQN as the
    # graph names it) for "parameter", "buffer" and "constant". `outs` has one
    # (kind, FQN or None) per graph output. In-place updates of inputs or parameters
    # cannot be returned through the (y, st) convention and are refused here.
    gs = ep.graph_signature
    bind = []
    n_user = 0
    for s in gs.input_specs:
        if s.kind == _InputKind.USER_INPUT:
            bind.append(("user_input", n_user))
            n_user += 1
        elif s.kind == _InputKind.PARAMETER:
            bind.append(("parameter", s.target))
        elif s.kind == _InputKind.BUFFER:
            bind.append(("buffer", s.target))
        elif s.kind == _InputKind.CONSTANT_TENSOR:
            bind.append(("constant", s.target))
        else:
            raise NotImplementedError(
                f"torch.export input kind {s.kind} ({s.target}) is not supported"
            )
    outs = []
    for s in gs.output_specs:
        if s.kind == _OutputKind.USER_OUTPUT:
            outs.append(("user_output", None))
        elif s.kind == _OutputKind.BUFFER_MUTATION:
            outs.append(("buffer_mutation", s.target))
        elif s.kind == _OutputKind.USER_INPUT_MUTATION:
            raise NotImplementedError(
                f"the module mutates its input {s.target!r} in place; in-place "
                "mutation of an input cannot be expressed by the (y, st) return "
                "convention of Reactant.import_torch"
            )
        elif s.kind == _OutputKind.PARAMETER_MUTATION:
            raise NotImplementedError(
                f"the module mutates its parameter {_strip(s.target, prefix)!r} in "
                "place; parameter mutation is not supported"
            )
        else:
            raise NotImplementedError(
                f"torch.export output kind {s.kind} ({s.target}) is not supported"
            )
    return bind, outs, n_user

def _torch_export_with_state(model, example_inputs, train, strict, prefix, backend):
    # Export `model` in training (`train=True`) or evaluation mode and lower it with
    # the named backend; see the contract above.
    if backend not in _LOWERINGS:
        raise ValueError(
            f"unknown PyTorch lowering backend {backend!r}; available: "
            + ", ".join(sorted(_LOWERINGS))
        )
    blocker = _TritonBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        ep = _capture(model, example_inputs, train, strict, prefix)
        return _LOWERINGS[backend](ep, prefix)
    finally:
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
"""

# The "torchax" lowering: torchax interprets the program's ATen ops as jax functions,
# and `jax.export` produces the StableHLO. Everything specific to torchax and jax lives
# here: the extra decompositions, the operator handlers torchax lacks, x64 handling and
# weak-scalar demotion.
const TORCHAX_LOWERING_PY = """
import jax as _jax
import jax.numpy as _jnp
import torch as _torch
import torchax as _torchax
import torchax.export as _txe
from torchax import decompositions as _decomps
from torchax.ops import mappings as _mappings
from torchax.ops import ops_registry as _ops_registry, jaten as _jaten

# JIT graphs emit the deprecated `aten._convolution.default` overload, which torchax
# does not register. Register a handler that drops the cuDNN flags and delegates to
# torchax's `aten.convolution`.
def _register_aten_convolution_default_handler():
    aten = _torch.ops.aten
    # Fail loudly if the overloads or helpers we rely on have moved, rather than
    # registering against a stale symbol or calling a missing function later.
    if not hasattr(aten, "_convolution") or not hasattr(aten._convolution, "default"):
        raise AttributeError(
            "torch.ops.aten._convolution.default is missing; "
            "the installed torch version is incompatible with ReactantPythonCallExt"
        )
    if not hasattr(_jaten, "_aten_convolution"):
        raise AttributeError(
            "torchax.ops.jaten._aten_convolution is missing; "
            "the installed torchax version is incompatible with ReactantPythonCallExt"
        )
    def _conv_handler(input, weight, bias, stride, padding, dilation, transposed,
                      output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32):
        return _jaten._aten_convolution(input, weight, bias, stride, padding, dilation,
                                         transposed, output_padding, groups)
    _ops_registry.register_torch_dispatch_op(aten._convolution.default, _conv_handler)

_register_aten_convolution_default_handler()

def _demote_weak_scalar_constant(w):
    # Reproduce PyTorch's weak-scalar promotion: a float64 *scalar* constant binds to
    # the dtype of the tensor it operates on instead of upcasting it. With x64 enabled
    # (see below) jax materializes it as a strong float64 array that upcasts and then
    # collides with float32 operands (e.g. a 0.001 constant reaching a float32 conv).
    # Cast 0-dim float64 constants to float32 so the constant's precision follows its
    # operands; genuine multi-dim float64 data is left alone, and jax promotes a
    # demoted scalar back to float64 wherever it meets a real float64 tensor. This is
    # applied only to lifted constants, which stay embedded in the exported program,
    # never to parameters or buffers, whose dtype must match what the caller holds.
    if getattr(w, "ndim", None) == 0 and getattr(w, "dtype", None) == _jnp.float64:
        return w.astype(_jnp.float32)
    return w

def _aval(t):
    return _jax.ShapeDtypeStruct(tuple(t.shape), _mappings.t2j_dtype(t.dtype))

def _lower_torchax(ep, prefix):
    # torchax's `exported_program_to_jax` is not used: the `func` it returns drops the
    # buffer mutation outputs (`res[num_mutations:]`), losing training-mode buffer
    # updates, and binds the states by position in an order that can differ from the
    # graph's placeholders. Instead apply the same decompositions and interpret the
    # graph directly, binding every state input by its FQN from the graph signature of
    # the program the interpreter runs.
    #
    # Enable jax's 64-bit mode for the lowering so genuinely double-precision models
    # keep their dtype. jax disables x64 by default and would otherwise canonicalize
    # float64 weights, inputs, and avals to float32, producing an f64/f32 mismatch when
    # Reactant feeds float64 operands to hlo_call. Save and restore the flag so this
    # stays scoped to the export and does not change global jax behavior (for example
    # the jax tracing path in pycall.jl). The constants and the exported program are
    # materialized inside this window, so they retain their dtype afterwards.
    prev_x64 = _jax.config.jax_enable_x64
    _jax.config.update("jax_enable_x64", True)
    try:
        ep = ep.run_decompositions(_decomps.DECOMPOSITIONS)
        bind, outs, n_user = _signature(ep, prefix)
        env = _torchax.default_env()

        def _state_value(fqn):
            if fqn in ep.state_dict:
                return ep.state_dict[fqn]
            return ep.constants[fqn]  # non-persistent buffers and lifted constants

        ps_avals, st_avals, consts = {}, {}, {}
        for kind, key in bind:
            if kind == "parameter":
                ps_avals[key] = _aval(_state_value(key))
            elif kind == "buffer":
                st_avals[key] = _aval(_state_value(key))
            elif kind == "constant":
                consts[key] = _demote_weak_scalar_constant(env.t2j_copy(_state_value(key)))

        gm = ep.graph_module
        # Lifted constants are closed over so jax embeds them in the program. Only the
        # mutated buffers are returned; the caller passes every other buffer through,
        # which keeps unread buffers eligible for dead-code elimination.
        def f(inputs, ps, st):
            args = []
            for kind, key in bind:
                if kind == "user_input":
                    args.append(inputs[key])
                elif kind == "parameter":
                    args.append(ps[key])
                elif kind == "buffer":
                    args.append(st[key])
                else:
                    args.append(consts[key])
            res = _txe.JaxInterpreter(gm).run(*args, enable_io_processing=False)
            ys = tuple(r for r, (kind, _) in zip(res, outs) if kind == "user_output")
            muts = {t: r for r, (kind, t) in zip(res, outs) if kind == "buffer_mutation"}
            return ys, muts

        user_avals = tuple(_txe.extract_avals(ep))
        # Matmul precision follows jax's default for the active platform, so float32
        # dot_general may lower at reduced (TF32-style) precision when jax initializes
        # for a GPU. This mirrors normal jax/torchax behavior and lets callers trade
        # accuracy for speed. Callers that need full precision can set
        # jax_default_matmul_precision="highest" before tracing (the tests do this for
        # deterministic comparisons against eager torch).
        je = _jax.export.export(_jax.jit(f))(user_avals, ps_avals, st_avals)

        # Map flat argument and result positions back to (kind, key). Dict leaves
        # flatten in sorted key order, so read the order from the trees rather than
        # assuming it.
        tu = _jax.tree_util
        (in_inputs, in_ps, in_st), _ = tu.tree_unflatten(je.in_tree, range(len(je.in_avals)))
        in_label = [None] * len(je.in_avals)
        for i, pos in enumerate(in_inputs):
            in_label[pos] = ("user_input", i)
        for fqn, pos in in_ps.items():
            in_label[pos] = ("parameter", _strip(fqn, prefix))
        for fqn, pos in in_st.items():
            in_label[pos] = ("buffer", _strip(fqn, prefix))
        out_ys, out_muts = tu.tree_unflatten(je.out_tree, range(len(je.out_avals)))
        out_label = [None] * len(je.out_avals)
        for pos in out_ys:
            out_label[pos] = ("user_output", None)
        for fqn, pos in out_muts.items():
            out_label[pos] = ("buffer_mutation", _strip(fqn, prefix))

        # `module_kept_var_idx` lists, in entry signature order, the flat argument
        # positions that survived dead-code elimination.
        inputs = []
        for pos in je.module_kept_var_idx:
            kind, key = in_label[pos]
            a = je.in_avals[pos]
            inputs.append((kind, key, str(a.dtype), tuple(int(d) for d in a.shape)))
        outputs = [(kind, key) for kind, key in out_label]
        return je.mlir_module(), inputs, outputs, n_user
    finally:
        _jax.config.update("jax_enable_x64", prev_x64)

_LOWERINGS["torchax"] = _lower_torchax
"""

is_torch_module(f::Py) = TORCH_EXPORT_SUPPORTED[] && pyisinstance(f, torchptr[].nn.Module)

function is_torchscript_module(f::Py)
    return TORCH_EXPORT_SUPPORTED[] && pyisinstance(f, torchptr[].jit.ScriptModule)
end

const NUMPY_DTYPE_TO_JULIA = Dict(string(v) => k for (k, v) in NUMPY_SIMPLE_TYPES)

function _julia_eltype(dtype::AbstractString)
    haskey(NUMPY_DTYPE_TO_JULIA, dtype) ||
        error("import_torch: no Julia element type for numpy dtype $dtype")
    return NUMPY_DTYPE_TO_JULIA[dtype]
end

## Types

"""
One argument of the StableHLO entry function of a [`TorchExport`](@ref), in argument
order.

- `kind`: `:parameter`, `:buffer` or `:user_input`
- `target`: for `:parameter` and `:buffer`, the FQN (`"1.running_mean"`) that is the key
  into `ps` or `st`; for `:user_input`, the 1-based position of the input in the call
  (torch.export gives no target for a user input, so the index is ours)
- `eltype`, `shape`: the expected element type and size, in PyTorch order

Lifted constants are embedded in the program and never appear here. Inputs the program
does not read (pruned by jax dead-code elimination) are also absent.
"""
struct TorchArgument
    kind::Symbol
    target::Union{String,Int}
    eltype::DataType
    shape::Dims
end

"""
One result of the StableHLO entry function of a [`TorchExport`](@ref), in result order.

- `kind`: `:user_output` (part of `y`) or `:buffer_mutation`
- `target`: for `:buffer_mutation`, the FQN of the buffer it replaces in `st`;
  `nothing` for `:user_output`
"""
struct TorchResult
    kind::Symbol
    target::Union{String,Nothing}
end

"""
One `torch.export` of a module for one mode and input signature: the StableHLO text
passed to `hlo_call` and the meaning of each of its arguments and results, and the
lowering `backend` that produced it (`:torchax`).
"""
struct TorchExport
    text::String
    arguments::Vector{TorchArgument}
    results::Vector{TorchResult}
    backend::Symbol
end

"""
One parameter or buffer of an imported module, as recorded by
[`Reactant.import_torch`](@ref). See [`Reactant.torch_layout`](@ref).

- `fqn`: the fully qualified name PyTorch uses (`"1.running_mean"`), the key of the
  leaf in `ps` (parameters) or `st` (buffers)
- `kind`: `:parameter` (in `ps`) or `:buffer` (in `st`)
- `persistent`: whether the tensor is part of the module's `state_dict()`; `false`
  for buffers registered with `persistent=False`. TorchScript does not record this, so
  every buffer of a TorchScript module is reported as persistent.
- `owner`: class name of the submodule that owns the tensor (`"BatchNorm1d"`)
- `eltype`, `shape`: element type and size of the leaf, in PyTorch order
"""
struct TorchStateEntry
    fqn::String
    kind::Symbol
    persistent::Bool
    owner::String
    eltype::DataType
    shape::Dims
end

# The Julia expression that reaches an entry's leaf, e.g. `st.var"1.running_mean"`.
function _julia_path(fqn::AbstractString, root::AbstractString)
    return Base.isidentifier(fqn) ? "$root.$fqn" : "$root.var\"$fqn\""
end
_julia_path(e::TorchStateEntry) = _julia_path(e.fqn, e.kind === :parameter ? "ps" : "st")

# (backend, mode, [(eltype, size) per user input]); mode is :eval or :train
const TorchExportKey = Tuple{Symbol,Symbol,Vector{Tuple{DataType,Dims}}}

"""
    TorchModule

A PyTorch module imported with [`Reactant.import_torch`](@ref). Call it as
`tm(x..., ps, st)` inside `@jit` or `@compile`; it returns `(y, st_new)`.

Each call looks up the export for the module's current mode (`model.training`) and the
element types and sizes of the inputs, exporting on first use with the module's lowering
backend. The Python module is never traced: the tracer treats a `TorchModule` as an
opaque constant.
"""
struct TorchModule
    model::Py                       # the module the user passed (mode is read here)
    export_model::Py                # what torch.export sees: the module or a wrapper
    prefix::String                  # FQN prefix a wrapper adds ("" if none), stripped
    backend::Symbol                 # lowering used by calls (see _torch_backends)
    layout::Vector{TorchStateEntry} # every parameter and buffer, in ps/st order
    cache::Dict{TorchExportKey,TorchExport}
end

# The lowering backends registered on the Python side, sorted by name.
function _torch_backends()
    names = pyconvert(Vector{String}, pylist(pyeval("_LOWERINGS", @__MODULE__)))
    return Symbol.(sort!(names))
end

function _check_backend(backend::Symbol, caller::String)
    available = _torch_backends()
    backend in available ||
        throw(ArgumentError("$caller: unknown backend :$backend; available: \
                             $(join(repr.(available), ", "))"))
    return backend
end

# Prepare `model` for export and read its parameters and buffers. `convert` maps each
# host array to the leaf stored in `ps`/`st`.
function _import(model::Py, convert; backend::Symbol=:torchax)
    TORCH_EXPORT_SUPPORTED[] || throw("torch/torchax could not be loaded.")
    _check_backend(backend, "import_torch")
    export_model, prefix = if is_torchscript_module(model)
        pyeval("_fix_jit_parameters", @__MODULE__)(model)
        pyeval("_JitWrapper", @__MODULE__)(model), "jit_mod."
    elseif is_torch_module(model)
        model, ""
    else
        throw(ArgumentError("import_torch: expected a torch.nn.Module, got $(model)"))
    end

    layout = TorchStateEntry[]
    ps = Pair{String,Any}[]
    st = Pair{String,Any}[]
    for row in pyeval("_state_layout", @__MODULE__)(model)
        kind = Symbol(pyconvert(String, row[0]))
        fqn = pyconvert(String, row[1])
        x = _torch_to_host(row[4])
        push!(
            layout,
            TorchStateEntry(
                fqn,
                kind,
                pyconvert(Bool, row[2]),
                pyconvert(String, row[3]),
                eltype(x),
                size(x),
            ),
        )
        push!(kind === :parameter ? ps : st, fqn => convert(x))
    end
    tm = TorchModule(
        model, export_model, prefix, backend, layout, Dict{TorchExportKey,TorchExport}()
    )
    return tm, _flat(ps), _flat(st)
end

# Keep the module out of tracing: it holds a Python handle and a cache that must be
# shared with the caller's object, not copied into a traced counterpart.
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:TorchModule}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    return T
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen, @nospecialize(prev::TorchModule), @nospecialize(path), mode; kwargs...
)
    if mode == Reactant.TracedToTypes
        push!(path, prev)
        return nothing
    end
    return prev
end

## Parameters and buffers as flat NamedTuples keyed by FQN

# Convert a torch tensor to a host array with the same logical shape (no reversal:
# Reactant maps a value's logical shape directly onto its MLIR tensor shape).
function _torch_to_host(t::Py)
    return pyconvert(Array, npptr[].asarray(t.detach().cpu()))
end

# A 0-dim tensor becomes a ConcreteRNumber, matching the TracedRNumber that hlo_call
# returns for a 0-dim result, so a returned `st` has the same types as the one passed in.
function _host_to_reactant(x::AbstractArray)
    return ndims(x) == 0 ? Reactant.ConcreteRNumber(x[]) : Reactant.to_rarray(x)
end
_host_to_reactant(x::Number) = Reactant.ConcreteRNumber(x)

# `ps` and `st` are flat NamedTuples whose keys are the FQNs, as in `state_dict()`:
# `st.var"1.running_mean"` or `st[Symbol("1.running_mean")]`. Being plain NamedTuples,
# they work with `keys`, `pairs`, `merge` and `Base.setindex` and are traced as is.
function _flat(pairs)
    return NamedTuple{Tuple(Symbol(first(p)) for p in pairs)}(Tuple(last(p) for p in pairs))
end

function _lookup(tree::NamedTuple, fqn::AbstractString, name::String)
    k = Symbol(fqn)
    haskey(tree, k) || throw(KeyError("$name has no leaf \"$fqn\""))
    return tree[k]
end

# `st` with the leaves named in `updates` replaced, keeping its key order.
function _replace_leaves(st::NamedTuple, updates::Dict{String})
    return merge(st, _flat(collect(updates)))
end

## Public FQN access

Reactant.torch_layout(tm::TorchModule) = tm.layout

function Reactant.torch_layout(tm::TorchModule, fqn::AbstractString)
    i = findfirst(e -> e.fqn == fqn, tm.layout)
    i === nothing && throw(KeyError("the module has no parameter or buffer \"$fqn\""))
    return tm.layout[i]
end

## Export

# Backend-specific notices issued before a lowering runs.
_before_lowering(::Val) = nothing

function _before_lowering(::Val{:torchax})
    # Report (once per session) the matmul precision jax will use to lower the model.
    # When this is left at jax's default, float32 matmuls may lower at reduced
    # (TF32-style) precision on a GPU-initialized jax, which changes results relative
    # to a CPU run.
    matmul_precision = pyconvert(
        String, pyimport("builtins").str(jaxptr[].config.jax_default_matmul_precision)
    )
    @warn "Tracing a PyTorch model: jax is lowering float32 matmuls with \
           jax_default_matmul_precision=$(matmul_precision), and that precision is frozen \
           into the exported StableHLO. jax picks it from the platform it detects at \
           trace time, not the backend Reactant executes on, so the two can disagree: \
           with GPUs visible jax can bake reduced (TF32-style) precision even if \
           execution falls back to CPU, and with only CPU detected it bakes full \
           float32 even if you later run on a GPU (and TF32 itself needs an Ampere or \
           newer GPU). Set jax_default_matmul_precision=\"highest\" before tracing for \
           deterministic full float32." maxlog = 1
    return nothing
end

function _torch_export(tm::TorchModule, backend::Symbol, mode::Symbol, inputs)
    np = npptr[]
    torch = torchptr[]

    # torch example inputs with the same shape and dtype as the traced arguments.
    example = Py[]
    for arg in inputs
        T = Reactant.unwrapped_eltype(arg)
        haskey(NUMPY_SIMPLE_TYPES, T) ||
            error("import_torch: no numpy dtype mapping for input element type $T")
        np_zeros = np.zeros(
            pytuple(collect(Int, size(arg))); dtype=string(NUMPY_SIMPLE_TYPES[T])
        )
        push!(example, torch.from_numpy(np_zeros))
    end

    _before_lowering(Val(backend))

    # Export with strict=false (the non-Dynamo tracer). Besides being required for
    # TorchScript, strict=true routes through Dynamo, whose accelerator/stream
    # handling conflicts with the torchax "jax" device and fails on current
    # torch+torchax. strict=false is also the direction torch.export is moving.
    result = torch_export_with_state[](
        tm.export_model,
        pytuple(Tuple(example)),
        mode === :train,
        false,
        tm.prefix,
        string(backend),
    )
    n_user = pyconvert(Int, result[3])
    n_user == length(inputs) ||
        error("import_torch: the module expects $n_user flattened tensor inputs but was \
               called with $(length(inputs)) (nested or non-tensor inputs are unsupported)")

    text = pyconvert(String, result[0])
    arguments = map(result[1]) do spec
        kind = Symbol(pyconvert(String, spec[0]))
        target = if kind === :user_input
            pyconvert(Int, spec[1]) + 1
        else
            pyconvert(String, spec[1])
        end
        eltype = _julia_eltype(pyconvert(String, spec[2]))
        shape = Tuple(pyconvert(Vector{Int}, spec[3]))
        TorchArgument(kind, target, eltype, shape)
    end
    results = map(result[2]) do spec
        kind = Symbol(pyconvert(String, spec[0]))
        target = kind === :buffer_mutation ? pyconvert(String, spec[1]) : nothing
        TorchResult(kind, target)
    end
    results = collect(results)

    # Buffer updates are expected in training mode (BatchNorm running statistics, for
    # example) but usually unintended in evaluation mode. They are still applied to the
    # returned `st`, matching eager PyTorch, so flag them. Exports are cached, so this
    # fires once per mode and input signature.
    if mode === :eval
        mutated = [s.target for s in results if s.kind === :buffer_mutation]
        if !isempty(mutated)
            name = pyconvert(String, tm.model.__class__.__name__)
            @warn "import_torch: $name updates buffers in evaluation mode: \
                   $(join(repr.(mutated), ", ")). The updated values are returned in \
                   `st`, as eager PyTorch would apply them; pass the returned `st` to \
                   the next call to keep them, or the original `st` to discard them."
        end
    end

    return TorchExport(text, collect(arguments), results, backend)
end

## Call

# A traced value is used as is; a host value (Array or Number) is embedded as a
# constant, which is how the `pycall` shortcut inlines the module's state.
_operand(x::Union{TracedRArray,TracedRNumber}) = x
_operand(x::AbstractArray) = @opcall constant(collect(x))
_operand(x::Number) = @opcall constant(fill(x))
function _operand(x)
    return error("import_torch: expected a Reactant array or number, got $(typeof(x))")
end

function _check_operand(x, spec::TorchArgument)
    what = if spec.kind === :user_input
        "input $(spec.target)"
    else
        "$(spec.kind) \"$(spec.target)\""
    end
    T = Reactant.unwrapped_eltype(x)
    sz = size(x)
    if T != spec.eltype || sz != spec.shape
        error("import_torch: $what has element type $T and size $sz, but the exported \
               module expects $(spec.eltype) and size $(spec.shape)")
    end
    return x
end

# The export of `tm` for a backend, mode and input signature, exporting on first use.
function _cached_export(tm::TorchModule, backend::Symbol, mode::Symbol, inputs)
    sig = Tuple{DataType,Dims}[(Reactant.unwrapped_eltype(x), size(x)) for x in inputs]
    return get!(
        () -> _torch_export(tm, backend, mode, inputs), tm.cache, (backend, mode, sig)
    )
end

function _torch_call(tm::TorchModule, inputs::Tuple, ps, st; mode::Symbol)
    for x in inputs
        x isa Union{TracedRArray,TracedRNumber} ||
            error("import_torch: inputs must be Reactant arrays or numbers traced by \
                   `@jit` or `@compile`; got $(typeof(x))")
    end
    exp = _cached_export(tm, tm.backend, mode, inputs)

    operands = Union{TracedRArray,TracedRNumber}[]
    for spec in exp.arguments
        v = if spec.kind === :user_input
            inputs[spec.target]
        elseif spec.kind === :parameter
            _operand(_lookup(ps, spec.target, "ps"))
        else
            _operand(_lookup(st, spec.target, "st"))
        end
        push!(operands, _check_operand(v, spec))
    end

    res = @opcall hlo_call(exp.text, operands...)

    ys = Any[]
    updates = Dict{String,Any}()
    for (r, spec) in zip(res, exp.results)
        if spec.kind === :user_output
            push!(ys, r)
        else
            _lookup(st, spec.target, "st")  # the updated buffer must exist in st
            updates[spec.target] = r
        end
    end
    y = isempty(ys) ? nothing : (length(ys) == 1 ? only(ys) : Tuple(ys))
    return y, (isempty(updates) ? st : _replace_leaves(st, updates))
end

# A frozen TorchScript module has no `training` attribute; its graph is fixed in
# evaluation mode.
function _torch_mode(tm::TorchModule)
    training = pygetattr(tm.model, "training", false)
    return pyconvert(Bool, training) ? :train : :eval
end

function (tm::TorchModule)(args...)
    length(args) >= 2 || throw(ArgumentError("a TorchModule is called as tm(x..., ps, st)"))
    return _torch_call(
        tm, args[1:(end - 2)], args[end - 1], args[end]; mode=_torch_mode(tm)
    )
end

function Reactant.import_torch(model::Py; backend::Symbol=:torchax)
    return _import(model, _host_to_reactant; backend)
end

# `@jit model(x)` shortcut: export in evaluation mode with the parameters and buffers
# embedded as constants, and return only `y`.
function pycall_with_torch_export(model::Py, args...)
    TORCH_EXPORT_SUPPORTED[] || throw("torch/torchax could not be loaded.")
    tm, ps, st = _import(model, identity)
    y, _ = _torch_call(tm, args, ps, st; mode=:eval)
    return y
end

## Display

function Base.show(io::IO, e::TorchStateEntry)
    return print(
        io,
        "TorchStateEntry(",
        repr(e.fqn),
        " => ",
        _julia_path(e),
        ", ",
        e.eltype,
        " ",
        e.shape,
        ", ",
        e.owner,
        e.persistent ? "" : ", non-persistent",
        ")",
    )
end

function Base.show(io::IO, tm::TorchModule)
    name = pyconvert(String, tm.model.__class__.__name__)
    n = length(tm.cache)
    return print(io, "TorchModule($name) with $n export", n == 1 ? "" : "s")
end

# The tables use PrettyTables (a dependency of Reactant) with a compact style: column
# labels, one rule under them, no other lines. When `io` is limited (the REPL sets
# `:limit => true`), a long table keeps its first and last rows around a `⋮` row and an
# "N rows omitted" note, sized to a third of the display height, so a model with
# hundreds of tensors stays readable. Each line is indented under its heading.
const _TABLE_FORMAT = PrettyTables.TextTableFormat(;
    borders=PrettyTables.text_table_borders__compact,
    horizontal_line_at_beginning=false,
    horizontal_line_after_column_labels=true,
    horizontal_line_after_data_rows=false,
    vertical_line_at_beginning=false,
    vertical_lines_at_data_columns=:none,
    vertical_line_after_data_columns=false,
)

function _print_table(io, labels::Vector{String}, rows::Vector{Vector{String}})
    isempty(rows) && return print(io, "\n     (none)")
    data = permutedims(reduce(hcat, rows))
    limit = get(io, :limit, false)::Bool
    height, width = displaysize(io)
    maxrows = max(8, height ÷ 3)
    table = sprint(; context=(:limit => limit)) do buf
        PrettyTables.pretty_table(
            buf,
            data;
            column_labels=labels,
            alignment=:l,
            table_format=_TABLE_FORMAT,
            vertical_crop_mode=:middle,
            fit_table_in_display_vertically=limit,
            fit_table_in_display_horizontally=false,
            display_size=(maxrows + 3, width),
        )
    end
    for line in split(rstrip(table, '\n'), '\n')
        print(io, "\n    ", rstrip(line))
    end
    return nothing
end

function _count_summary(entries)
    nelem = sum(e -> prod(e.shape; init=1), entries; init=0)
    nt = length(entries)
    return "$nt tensor$(nt == 1 ? "" : "s"), $(_group_digits(nelem)) element\
            $(nelem == 1 ? "" : "s")"
end
_group_digits(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

# The argument and result tables of one export.
function _print_export_tables(io, exp::TorchExport)
    print(io, "\n    arguments (hlo_call operands):")
    _print_table(
        io,
        ["kind", "source", "eltype", "shape"],
        [
            [
                repr(s.kind),
                if s.kind === :user_input
                    "x[$(s.target)]"
                else
                    _julia_path(s.target, s.kind === :parameter ? "ps" : "st")
                end,
                string(s.eltype),
                string(s.shape),
            ] for s in exp.arguments
        ],
    )
    print(io, "\n    results (hlo_call results):")
    ny = count(s -> s.kind === :user_output, exp.results)
    iy = 0
    _print_table(
        io,
        ["#", "kind", "destination"],
        map(enumerate(exp.results)) do (i, s)
            dest = if s.kind === :user_output
                iy += 1
                ny == 1 ? "y" : "y[$iy]"
            else
                _julia_path(s.target, "st")
            end
            [string(i), repr(s.kind), dest]
        end,
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tm::TorchModule)
    show(io, tm)
    params = filter(e -> e.kind === :parameter, tm.layout)
    print(io, "\n  parameters (ps), ", _count_summary(params), ":")
    _print_table(
        io,
        ["FQN", "eltype", "shape", "owner"],
        [[e.fqn, string(e.eltype), string(e.shape), e.owner] for e in params],
    )
    buffers = filter(e -> e.kind === :buffer, tm.layout)
    print(io, "\n  buffers (st), ", _count_summary(buffers), ":")
    _print_table(
        io,
        ["FQN", "eltype", "shape", "owner", "persistent"],
        [
            [e.fqn, string(e.eltype), string(e.shape), e.owner, string(e.persistent)] for
            e in buffers
        ],
    )
    for (key, exp) in sort!(collect(tm.cache); by=kv -> string(kv[1]))
        backend, mode, sig = key
        print(
            io,
            "\n  export ",
            repr(mode),
            " via ",
            backend,
            " for inputs ",
            join(("$T$(shape)" for (T, shape) in sig), ", "),
            ":",
        )
        _print_export_tables(io, exp)
    end
    return nothing
end
