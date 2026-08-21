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
import jax.numpy as _jnp
import torch as _torch
import torchax.export as _txe

class _TritonBlocker:
    def find_spec(self, name, path, target=None):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton import blocked by ReactantPythonCallExt to avoid an LLVM clash with Reactant")
        return None

# torchax binds the model's states (parameters, buffers, lifted constants) to the
# decomposed graph's non-user placeholders by position, in graph order. It extracts
# them in `params + buffers + lifted-constants` order, but torch.export can order the
# placeholders differently (e.g. params, constant, buffers), so the default order can
# feed the wrong tensor to each placeholder (a scalar constant where a conv buffer is
# expected, etc.), silently corrupting any model whose placeholder order differs.
#
# `_reorder_weights_to_placeholder_order` below fixes this without monkey patching
# torchax: it recovers the decomposed `ExportedProgram` that the returned `func` binds
# to (it is captured in `func`'s closure) and permutes the already-converted weight
# list into the graph's own `input_specs` order before we feed it to `func`. Reading
# the order from the exact program `func` uses avoids any assumption about torchax's
# decomposition sequence; an incompatible torchax fails loudly here rather than
# producing silently wrong results.
def _reorder_weights_to_placeholder_order(weights, func):
    # `func` is the closure returned by `exported_program_to_jax`; it runs the
    # interpreter over a decomposed `ExportedProgram` it captured. Recover that program
    # by type (independent of closure cell order).
    decomposed = next(
        (cell.cell_contents for cell in (func.__closure__ or [])
         if isinstance(cell.cell_contents, _torch.export.ExportedProgram)),
        None,
    )
    if decomposed is None:
        raise RuntimeError(
            "could not recover the decomposed ExportedProgram from torchax's func "
            "closure; the installed torchax version is incompatible with ReactantPythonCallExt"
        )
    _InputKind = _torch.export.graph_signature.InputKind
    gs = decomposed.graph_signature
    # The order torchax extracted `weights` in (mirrors its
    # `_extract_states_from_exported_program`).
    extracted = (list(gs.parameters) + list(gs.buffers)
                 + list(getattr(gs, "lifted_tensor_constants", [])))
    # The order the interpreter consumes the states in: the graph's own placeholders.
    placeholder_order = [s.target for s in gs.input_specs if s.kind != _InputKind.USER_INPUT]
    if sorted(placeholder_order) != sorted(extracted):
        raise RuntimeError(
            "torchax state/placeholder set mismatch; the installed torchax version is "
            "incompatible with ReactantPythonCallExt"
        )
    pos = {name: i for i, name in enumerate(extracted)}
    return [weights[pos[target]] for target in placeholder_order]

def _demote_weak_scalar_constants(weights):
    # Reproduce PyTorch's weak-scalar promotion: a float64 *scalar* constant (a Python
    # float literal) binds to the dtype of the tensor it operates on instead of upcasting
    # it. jax materializes it as a strong float64 array that (with x64 on) upcasts and
    # then collides with float32 operands (e.g. a 0.001 constant reaching a float32 conv).
    # Cast 0-dim float64 constants to float32 so the constant's precision follows its
    # operands; genuine multi-dim float64 data is left alone, and jax promotes a demoted
    # scalar back to float64 wherever it meets a real float64 tensor.
    return [w.astype(_jnp.float32)
            if (getattr(w, "ndim", None) == 0 and getattr(w, "dtype", None) == _jnp.float64)
            else w
            for w in weights]

def _torch_to_stablehlo_inputs_first(model, example_inputs, strict):
    blocker = _TritonBlocker()
    sys.meta_path.insert(0, blocker)
    # Enable jax's 64-bit mode for the lowering so genuinely double-precision models keep
    # their dtype. jax disables x64 by default and would otherwise canonicalize float64
    # weights, inputs, and avals to float32, producing an f64/f32 mismatch when Reactant
    # feeds the original float64 operands to hlo_call. PyTorch's weak-scalar promotion (a
    # float64 Python-float constant binds to its operand's dtype rather than upcasting it)
    # is reproduced by `_demote_weak_scalar_constants`, so a float32 model's incidental
    # float64 scalars do not collide with float32 operands in strict ops (dot_general,
    # conv). Save and restore the flag so this stays scoped to the export and does not
    # change global jax behavior (for example the jax tracing path in pycall.jl): jax is
    # shared process-wide here, unlike a dedicated export process, so we do not set x64
    # globally. The weights and the exported program are materialized inside this window,
    # so they retain their dtype after the flag is restored.
    prev_x64 = _jax.config.jax_enable_x64
    _jax.config.update("jax_enable_x64", True)
    try:
        exported = _torch.export.export(model, example_inputs, strict=strict)
        weights, func = _txe.exported_program_to_jax(exported)
        # torchax extracts `weights` in params+buffers+constants order, which can
        # diverge from the order `func` binds them to the graph placeholders. Reorder
        # into placeholder order so each weight reaches the correct placeholder.
        weights = _reorder_weights_to_placeholder_order(weights, func)
        weights = _demote_weak_scalar_constants(weights)
        jax_avals = _txe.extract_avals(exported)
        def reordered(inputs, weights):
            return func(weights, inputs)
        # Matmul precision follows jax's default for the active platform, so float32
        # dot_general may lower at reduced (TF32-style) precision when jax initializes
        # for a GPU. This mirrors normal jax/torchax behavior and lets callers trade
        # accuracy for speed. Callers that need full precision can set
        # jax_default_matmul_precision="highest" before invoking the model (the tests
        # do this for deterministic comparisons against eager torch).
        jax_export = _jax.export.export(_jax.jit(reordered))((jax_avals,), weights)
        return weights, jax_export, len(jax_avals)
    finally:
        _jax.config.update("jax_enable_x64", prev_x64)
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

is_torch_module(f::Py) = TORCH_EXPORT_SUPPORTED[] && pyisinstance(f, torchptr[].nn.Module)

function is_torchscript_module(f::Py)
    return TORCH_EXPORT_SUPPORTED[] && pyisinstance(f, torchptr[].jit.ScriptModule)
end

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
        np_zeros = np.zeros(
            pytuple(collect(Int, size(arg))); dtype=string(NUMPY_SIMPLE_TYPES[T])
        )
        push!(example, torch.from_numpy(np_zeros))
    end
    py_args = pytuple(Tuple(example))

    # Report (once per session) the matmul precision jax will use to lower the model.
    # When this is left at jax's default, float32 matmuls may lower at reduced
    # (TF32-style) precision on a GPU-initialized jax (see the note in
    # `_torch_to_stablehlo_inputs_first`), which changes results relative to a CPU run.
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
