import ..Reactant
using ..Reactant: TracedRArray, ReactantRNG, Ops

# Must match mlir::impulse::{GuideKind,OptimizerKind} in ImpulseEnums.td.
const GUIDE_KIND_AUTO_NORMAL = Int32(0)
const GUIDE_KIND_AUTO_MVN = Int32(1)
const GUIDE_KIND_CUSTOM = Int32(2)
const GUIDE_KIND_AUTO_DELTA = Int32(3)

const OPTIMIZER_KIND_ADAM = Int32(0)
const OPTIMIZER_KIND_SGD = Int32(1)
const OPTIMIZER_KIND_CLIPPED_ADAM = Int32(2)
const OPTIMIZER_KIND_MOMENTUM = Int32(3)
const OPTIMIZER_KIND_RMSPROP = Int32(4)
const OPTIMIZER_KIND_ADAGRAD = Int32(5)
const OPTIMIZER_KIND_RMSPROP_MOMENTUM = Int32(6)

"""
    VIState(loc, scale, elbo, step_size, inv_mass_matrix, rng, slots)

Post-optimization variational state returned by `svi`.
  * `loc`             — converged guide location (unconstrained, `1×d`).
  * `scale`           — CONSTRAINED guide scale, reconstructed frontend-side from
                        the unconstrained state slot: `softplus` for AutoNormal
                        (`1×d`), `ScaledUnitLowerCholesky` for AutoMVN (`d×d`),
                        zeros for AutoDelta (`1×d`).
  * `elbo`            — final ELBO value (scalar; last entry of the trajectory).
  * `step_size`       — the learning rate used.
  * `inv_mass_matrix` — alias for `scale`, kept for compatibility with
                        `MCMCState` so VI output plugs into an HMC warm-start.
  * `rng`             — updated RNG state after `num_steps` SVI updates.
  * `slots`           — NamedTuple of the raw UNCONSTRAINED `state_out` slots
                        keyed by schema name. Feed a whole `VIState` (or this
                        NamedTuple) back via `svi(...; initial_state=state)` to
                        warm-restart: the slots are threaded as `state_in`, which
                        skips the SVI.init rng split and continues Adam's bias
                        correction from the previous `opt_step_count`.
"""
struct VIState{LocT,ScaleT,ElboT,StepT,RngT,SlotsT}
    loc::LocT
    scale::ScaleT
    elbo::ElboT
    step_size::StepT
    inv_mass_matrix::ScaleT
    rng::RngT
    slots::SlotsT
end

# Mirrors mlir::impulse::buildVIStateSchema; the op verifier rejects any other order.
function _vi_state_schema_names(guide::Symbol)
    if guide === :AutoDelta
        return String["loc", "opt_m_loc", "opt_v_loc", "opt_step_count"]
    elseif guide === :AutoMultivariateNormal
        return String[
            "loc",
            "scale_tril_unconstrained",
            "opt_m_loc",
            "opt_v_loc",
            "opt_m_scale_tril",
            "opt_v_scale_tril",
            "opt_step_count",
        ]
    else
        return String[
            "loc",
            "scale_unconstrained",
            "opt_m_loc",
            "opt_v_loc",
            "opt_m_scale",
            "opt_v_scale",
            "opt_step_count",
        ]
    end
end

function _vi_state_out_types(guide::Symbol, d::Int)
    f64 = MLIR.IR.Type(Float64)
    i64 = MLIR.IR.Type(Int64)
    vec_t = MLIR.IR.TensorType([1, d], f64)
    step_t = MLIR.IR.TensorType(Int64[], i64)
    if guide === :AutoDelta
        return MLIR.IR.Type[vec_t, vec_t, vec_t, step_t]
    elseif guide === :AutoMultivariateNormal
        flat = d * (d + 1) ÷ 2
        tril_t = MLIR.IR.TensorType([flat], f64)
        return MLIR.IR.Type[vec_t, tril_t, vec_t, vec_t, tril_t, tril_t, step_t]
    else
        return MLIR.IR.Type[vec_t, vec_t, vec_t, vec_t, vec_t, vec_t, step_t]
    end
end

function _vi_wrap_slot(name::String, value, d::Int)
    if name == "opt_step_count"
        return TracedRArray{Int64,0}((), value, ())
    elseif name == "scale_tril_unconstrained" ||
        name == "opt_m_scale_tril" ||
        name == "opt_v_scale_tril"
        flat = d * (d + 1) ÷ 2
        return TracedRArray{Float64,1}((), value, (flat,))
    else
        return TracedRArray{Float64,2}((), value, (1, d))
    end
end

# NumPyro's ScaledUnitLowerCholeskyTransform; must match VI::applyScaledUnitLowerCholesky.
# x_flat is off_diag (strict lower, row-major) followed by diag_pre.
function _scaled_unit_lower_cholesky(x_flat, d::Int)
    nOff = d * (d - 1) ÷ 2
    diag_pre = x_flat[(nOff + 1):(nOff + d)]
    diag = log1p.(exp.(diag_pre))
    off_diag = nOff > 0 ? x_flat[1:nOff] : x_flat[1:0]

    rows = Vector{Any}(undef, d)
    for i in 1:d
        diag_i = diag[i:i]
        nlow = i - 1
        kstart = (i - 1) * (i - 2) ÷ 2
        parts = Any[]
        if nlow > 0
            lower = off_diag[(kstart + 1):(kstart + nlow)] .* diag_i
            push!(parts, reshape(lower, 1, nlow))
        end
        push!(parts, reshape(diag_i, 1, 1))
        nup = d - i
        if nup > 0
            push!(parts, Ops.fill(0.0, (1, nup)))
        end
        rows[i] = length(parts) == 1 ? parts[1] : hcat(parts...)
    end
    return vcat(rows...)
end

"""
    svi(rng, original_trace, model, args...; selection, guide, optimizer,
        step_size, num_steps, num_particles, init_scale, adam_b1, adam_b2,
        adam_eps, ...; initial_position, initial_scale, initial_state)

Stochastic variational inference on a probabilistic `model`. Mirrors NumPyro's
`SVI(model, guide, optim, loss=Trace_ELBO(num_particles=K))` with pointwise-
identical RNG splitting.

Returns `(trace, elbo_trajectory, traced_result, state::VIState)`.

Cold start reads the initial location from `initial_position` (else the guide's
`init_to_uniform`) and the initial scale from the `initial_scale` operand or the
`init_scale` config. Warm restart: pass a previous `VIState` (or its `.slots`
NamedTuple) as `initial_state`; the state slots are threaded as `state_in`.
"""
function svi(
    rng::AbstractRNG,
    original_trace,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    guide::Symbol=:AutoNormal,
    optimizer::Symbol=:Adam,
    step_size,
    num_steps::Int,
    num_particles::Int=1,
    init_scale::Float64=0.1,
    adam_b1::Float64=0.9,
    adam_b2::Float64=0.999,
    adam_eps::Float64=1.0e-8,
    clip_norm::Float64=10.0,
    init_radius::Float64=2.0,
    momentum_mass::Float64=0.9,
    rmsprop_gamma::Float64=0.9,
    rmsprop_eps::Float64=1.0e-8,
    stable_update::Bool=false,
    initial_position=nothing,
    initial_scale=nothing,
    initial_state=nothing,
) where {Nargs}
    args = (rng, args...)

    tt = TracedTrace()
    ppf = ScopedValues.with(TRACING_TRACE => tt) do
        return process_probprog_function(f, args, "svi")
    end

    (;
        f_name,
        mlir_caller_args,
        mlir_result_types,
        traced_result,
        linear_results,
        fnwrapped,
        argprefix,
        resprefix,
    ) = ppf

    fn_attr = MLIR.IR.FlatSymbolRefAttribute(f_name)
    all_addresses_attr = build_selection_attr(tt)

    selection_pos_size = 0
    for address in selection
        for entry in tt.entries
            entry_path = [entry.parent_path..., entry.symbol]
            if entry_path == collect(address.path)
                selection_pos_size += entry.num_elements
                break
            end
        end
    end
    d = selection_pos_size

    selection_attr = MLIR.IR.Attribute[]
    for address in selection
        address_attr = MLIR.IR.Attribute[]
        for sym in address.path
            sym_addr = reinterpret(UInt64, pointer_from_objref(sym))
            push!(
                address_attr,
                MLIR.IR.Attribute(
                    MLIR.API.enzymeSymbolAttrGet(MLIR.IR.current_context(), sym_addr)
                ),
            )
        end
        push!(selection_attr, MLIR.IR.Attribute(address_attr))
    end

    trace_type = MLIR.IR.TensorType([Int64(num_steps), d], MLIR.IR.Type(Float64))
    elbo_type = if num_steps == 1
        MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))
    else
        MLIR.IR.TensorType([Int64(num_steps)], MLIR.IR.Type(Float64))
    end

    guide_int = if guide === :AutoNormal
        GUIDE_KIND_AUTO_NORMAL
    elseif guide === :AutoMultivariateNormal
        GUIDE_KIND_AUTO_MVN
    elseif guide === :AutoDelta
        GUIDE_KIND_AUTO_DELTA
    elseif guide === :Custom
        GUIDE_KIND_CUSTOM
    else
        error(
            "Unknown guide: $guide. Supported: :AutoNormal, " *
            ":AutoMultivariateNormal, :AutoDelta, :Custom",
        )
    end
    optim_int = if optimizer === :Adam
        OPTIMIZER_KIND_ADAM
    elseif optimizer === :SGD
        OPTIMIZER_KIND_SGD
    elseif optimizer === :ClippedAdam
        OPTIMIZER_KIND_CLIPPED_ADAM
    elseif optimizer === :Momentum
        OPTIMIZER_KIND_MOMENTUM
    elseif optimizer === :RMSProp
        OPTIMIZER_KIND_RMSPROP
    elseif optimizer === :Adagrad
        OPTIMIZER_KIND_ADAGRAD
    elseif optimizer === :RMSPropMomentum
        OPTIMIZER_KIND_RMSPROP_MOMENTUM
    else
        error(
            "Unknown optimizer: $optimizer. Supported: :Adam, :SGD, " *
            ":ClippedAdam, :Momentum, :RMSProp, :Adagrad, :RMSPropMomentum",
        )
    end

    vi_config_attr = MLIR.IR.Attribute(
        MLIR.API.enzymeVIConfigAttrGet(
            MLIR.IR.current_context(),
            guide_int,
            optim_int,
            Int64(num_particles),
            true,
            init_scale,
            true,
            adam_b1,
            true,
            adam_b2,
            true,
            adam_eps,
            true,
            clip_norm,
            true,
            init_radius,
            true,
            momentum_mass,
            true,
            rmsprop_gamma,
            true,
            rmsprop_eps,
            stable_update,
        ),
    )

    step_size_val = TracedUtils.get_mlir_data(step_size)

    trace_val = TracedUtils.get_mlir_data(original_trace)

    schema_names = _vi_state_schema_names(guide)

    resume_slots = if isnothing(initial_state)
        nothing
    elseif initial_state isa VIState
        initial_state.slots
    else
        initial_state
    end
    is_resume = !isnothing(resume_slots)

    state_in_vals = if is_resume
        MLIR.IR.Value[
            TracedUtils.get_mlir_data(resume_slots[Symbol(name)]) for name in schema_names
        ]
    else
        MLIR.IR.Value[]
    end

    initial_position_val = if is_resume || isnothing(initial_position)
        nothing
    else
        TracedUtils.get_mlir_data(initial_position)
    end
    initial_scale_val = if is_resume || isnothing(initial_scale)
        nothing
    else
        TracedUtils.get_mlir_data(initial_scale)
    end

    vi_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        state_in_vals,
        trace_val;
        init_step_size=step_size_val,
        init_position=initial_position_val,
        init_scale=initial_scale_val,
        trace=trace_type,
        diagnostics=elbo_type,
        # Zero-filled by the lowering; VI keeps its ELBO in diagnostics (Enzyme#2958).
        log_densities=elbo_type,
        output_rng_state=mlir_result_types[1],
        state_out=_vi_state_out_types(guide, d),
        fn=fn_attr,
        selection=MLIR.IR.Attribute(selection_attr),
        all_addresses=all_addresses_attr,
        vi_config=vi_config_attr,
        state_schema=_state_schema_attr(schema_names),
        num_warmup=Int64(0),
        num_samples=Int64(num_steps),
        thinning=Int64(1),
    )

    traced_result = process_probprog_outputs(
        vi_op,
        linear_results,
        traced_result,
        f,
        args,
        fnwrapped,
        resprefix,
        argprefix,
        3,
        true,
    )

    trace_out = TracedRArray{Float64,2}((), MLIR.IR.result(vi_op, 1), (num_steps, d))
    elbo_out = if num_steps == 1
        TracedRArray{Float64,0}((), MLIR.IR.result(vi_op, 2), ())
    else
        TracedRArray{Float64,1}((), MLIR.IR.result(vi_op, 2), (num_steps,))
    end
    rng_out = TracedRArray{UInt64,1}((), MLIR.IR.result(vi_op, 4), size(rng.seed))

    # state_out follows trace, diagnostics, log_densities and rng.
    slot_result(name) = MLIR.IR.result(vi_op, 4 + findfirst(==(name), schema_names))
    slots = NamedTuple(
        Symbol(name) => _vi_wrap_slot(name, slot_result(name), d) for name in schema_names
    )

    loc_out = slots[:loc]

    scale_out = if guide === :AutoDelta
        Ops.fill(0.0, (1, d))
    elseif guide === :AutoMultivariateNormal
        _scaled_unit_lower_cholesky(slots[:scale_tril_unconstrained], d)
    else
        log1p.(exp.(slots[:scale_unconstrained]))
    end

    elbo_final = if num_steps == 1
        elbo_out
    else
        Reactant.Ops.reshape(elbo_out[num_steps:num_steps], Int64[])
    end
    step_size_out = TracedRArray{Float64,0}((), step_size_val, ())

    state = VIState(
        loc_out, scale_out, elbo_final, step_size_out, scale_out, rng_out, slots
    )
    return trace_out, elbo_out, traced_result, state
end
