# Import a PyTorch model into a Reactant trace.
#
# A `torch.nn.Module` is Python-callable, so invoking it inside `@compile`/`@jit`
# with Reactant arrays routes through the `PythonCall.pycall` overlay just like a
# JAX function does (see overlays.jl and pycall.jl). This file provides the torch
# analog of `pycall_with_jax_tracing`: it traces the module with `torch.export`,
# lowers it to StableHLO with torchax, and splices that StableHLO into the current
# trace via `@opcall hlo_call`. The module's parameters and buffers are carried
# across as inlined constants, mirroring how JAX closure parameters are captured.

# Trace `model` and lower it to StableHLO. torchax emits StableHLO whose entry
# signature is `(weights..., inputs...)`; this re-exports the lowered program
# through jax so the signature becomes `(inputs..., weights...)`, the order we build
# the hlo_call operand list in below. Returns `(weights, jax_export, n_inputs)` where
# `weights` is an indexable sequence of the materialized weight arrays, `jax_export`
# exposes `mlir_module()` and `module_kept_var_idx`, and `n_inputs` is the number of
# input arguments.
#
# `torch.export` opportunistically imports `triton`. triton ships its own copy of
# LLVM whose static initializers clash with the LLVM already loaded by Reactant when
# triton is loaded second, segfaulting the process. triton is not needed for this
# lowering path, so block its import for the duration of the export; this makes the
# conversion work regardless of whether torch or Reactant was imported first.
const TORCH_TO_STABLEHLO_INPUTS_FIRST_PY = """
import sys
import jax as _jax
import torch as _torch
import torchax.export as _txe

class _TritonBlocker:
    def find_spec(self, name, path, target=None):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton import blocked by ReactantPythonCallExt to avoid an LLVM clash with Reactant")
        return None

# torchax's `_extract_states_from_exported_program` builds the positional state
# list as params + buffers + lifted-constants, but torch.export can order the graph
# placeholders differently (e.g. params, constant, buffers). The interpreter binds
# states to placeholders by position, so the default order can feed the wrong tensor
# to each placeholder (a scalar constant where a conv buffer is expected, etc.),
# silently corrupting any model whose placeholder order differs. Override it to emit
# states in the graph's own `input_specs` order. This patches the torchax.export
# module global that `exported_program_to_jax` looks up at call time.
def _install_state_order_fix():
    _InputKind = _torch.export.graph_signature.InputKind
    def _extract_states_in_input_spec_order(exported_program):
        gs = exported_program.graph_signature
        state_dict = dict(exported_program.state_dict)
        constants = dict(getattr(exported_program, "constants", None) or {})
        names, values = [], []
        for spec in gs.input_specs:
            if spec.kind == _InputKind.USER_INPUT:
                continue
            target = spec.target
            if spec.kind in (_InputKind.PARAMETER, _InputKind.BUFFER):
                v = state_dict[target]
            elif spec.kind == _InputKind.CONSTANT_TENSOR:
                v = constants[target]
            else:
                v = constants.get(target, state_dict.get(target))
            names.append(target)
            values.append(v)
        return names, values
    # Fail loudly if the symbol we override has moved or been renamed: a plain
    # attribute assignment would otherwise succeed silently, leave torchax calling
    # its original function, and reintroduce the state-corruption bug this fixes.
    if not hasattr(_txe, "_extract_states_from_exported_program"):
        raise AttributeError(
            "torchax.export._extract_states_from_exported_program is missing; "
            "the installed torchax version is incompatible with ReactantPythonCallExt"
        )
    _txe._extract_states_from_exported_program = _extract_states_in_input_spec_order

_install_state_order_fix()

def _torch_to_stablehlo_inputs_first(model, example_inputs, strict):
    blocker = _TritonBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        exported = _torch.export.export(model, example_inputs, strict=strict)
        weights, func = _txe.exported_program_to_jax(exported)
        jax_avals = _txe.extract_avals(exported)
        def reordered(inputs, weights):
            return func(weights, inputs)
        jax_export = _jax.export.export(_jax.jit(reordered))((jax_avals,), weights)
        return weights, jax_export, len(jax_avals)
    finally:
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
"""

# TorchScript artifacts (`torch.jit.script` / `torch.jit.trace`) need three
# adjustments before `torch.export` accepts them:
#   1. JIT modules store parameters as plain tensors; the exporter's verifier
#      requires real `torch.nn.Parameter`, so re-wrap each leaf parameter.
#   2. JIT graphs emit the deprecated `aten._convolution.default` overload, which
#      torchax does not register. Install a handler that drops the cuDNN flags and
#      delegates to the supported `aten.convolution`. This mutates the global
#      torchax dispatch registry, so it runs once per process.
#   3. `torch.export` expects an `nn.Module`; wrap the `ScriptModule` in a minimal
#      forwarding module.
const TORCHSCRIPT_PATCHES_PY = """
import torch as _torch
from torchax.ops import ops_registry as _ops_registry, jaten as _jaten

def _fix_jit_parameters(mod):
    for name in list(mod._parameters.keys()):
        p = mod._parameters[name]
        if p is not None and not isinstance(p, _torch.nn.Parameter):
            mod._parameters[name] = _torch.nn.Parameter(p.detach().clone(), requires_grad=False)
    for child in mod._modules.values():
        if child is not None:
            _fix_jit_parameters(child)

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

class _JitWrapper(_torch.nn.Module):
    def __init__(self, jit_mod):
        super().__init__()
        self.jit_mod = jit_mod
    def forward(self, *args):
        return self.jit_mod(*args)
"""

const _torchscript_patched = Ref{Bool}(false)

function _ensure_torchscript_patches()
    _torchscript_patched[] && return nothing
    pyexec(TORCHSCRIPT_PATCHES_PY, @__MODULE__)
    pyeval("_register_aten_convolution_default_handler", @__MODULE__)()
    _torchscript_patched[] = true
    return nothing
end

is_torch_module(f::Py) =
    TORCH_EXPORT_SUPPORTED[] && pyisinstance(f, torchptr[].nn.Module)

is_torchscript_module(f::Py) =
    TORCH_EXPORT_SUPPORTED[] && pyisinstance(f, torchptr[].jit.ScriptModule)

function pycall_with_torch_export(model::Py, args...)
    TORCH_EXPORT_SUPPORTED[] || throw("torch/torchax could not be loaded.")
    @assert all(Base.Fix2(isa, Union{TracedRArray,TracedRNumber}), args) "pycall_with_torch_export: all inputs must be Reactant traced arrays or numbers"

    torch = torchptr[]
    np = npptr[]

    is_script = is_torchscript_module(model)
    if is_script
        _ensure_torchscript_patches()
        model.eval()
        pyeval("_fix_jit_parameters", @__MODULE__)(model)
        model = pyeval("_JitWrapper", @__MODULE__)(model)
    else
        model.eval()
    end

    # Build torch example inputs with the same shape and dtype as the traced
    # Reactant arguments. Reactant maps a value's logical shape directly onto its
    # MLIR tensor shape (see `Ops.mlir_type`), so the torch example shape equals
    # `size(arg)` with no reversal, and the resulting StableHLO input types match
    # the operands we feed to `hlo_call`.
    example = Py[]
    for arg in args
        T = Reactant.unwrapped_eltype(arg)
        haskey(NUMPY_SIMPLE_TYPES, T) ||
            error("pycall_with_torch_export: no numpy dtype mapping for $T")
        np_zeros = np.zeros(pytuple(collect(Int, size(arg))); dtype=string(NUMPY_SIMPLE_TYPES[T]))
        push!(example, torch.from_numpy(np_zeros))
    end
    py_args = pytuple(Tuple(example))

    # Export with strict=false (the non-Dynamo tracer). Besides being required for
    # TorchScript, strict=true routes through Dynamo, whose accelerator/stream
    # handling conflicts with the torchax "jax" device and fails on current
    # torch+torchax. strict=false is also the direction torch.export is moving.
    result = torch_to_stablehlo_inputs_first[](model, py_args, false)
    weights_py = result[0]
    jax_export = result[1]
    n_inputs = pyconvert(Int, result[2])
    n_inputs == length(args) || error(
        "pycall_with_torch_export: model expects $n_inputs flattened tensor inputs " *
        "but was called with $(length(args)) arguments (nested/non-tensor inputs are unsupported)",
    )

    text = pyconvert(String, jax_export.mlir_module())

    # `module_kept_var_idx` lists, in signature order, the flattened positions that
    # survived dead-code elimination. Positions `< n_inputs` are request inputs
    # (the traced args); positions `>= n_inputs` are weights at offset
    # `idx - n_inputs`. Pruned parameters/buffers (e.g. BatchNorm running stats that
    # fold away) simply never appear here, keeping operands aligned with the entry.
    kept_idx = Int[pyconvert(Int, i) for i in jax_export.module_kept_var_idx]
    operands = Union{TracedRArray,TracedRNumber}[]
    for g in kept_idx
        if g < n_inputs
            push!(operands, args[g + 1])
        else
            w = weights_py[g - n_inputs]
            jl = pyconvert(Array, np.asarray(w))
            push!(operands, @opcall constant(jl))
        end
    end

    res = @opcall hlo_call(text, operands...)
    return length(res) == 0 ? nothing : (length(res) == 1 ? res[1] : res)
end
