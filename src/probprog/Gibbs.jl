using ..Reactant: TracedRArray

"""
    gibbs(rng, trace, weight, f, args...; selection, constraint=nothing,
          constrained_addresses=nothing, num_sweeps=1)

Enumerative Gibbs: each selected site, in `selection` order, is redrawn from its
exact full conditional, computed by scoring every value of its finite support
(scalar Bernoulli sites). Only the density terms that depend on the site are
scored. Emits an `impulse.infer` with `gibbs_config`, lowered by the region
pipeline only (`Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = false`).

`constraint` / `constrained_addresses` supply observed values. A constrained site
inside a `ProbProg.if_` arm is scored from the constraint, so a flag that makes
the arm live scores its observation (never a stale trace slot), and the sweep
writes the observations into the trace for later moves.

Returns `(new_trace, new_weight, result)`. The weight advances by the score
difference of each redrawn site.
"""
function gibbs(
    rng::AbstractRNG,
    original_trace,
    original_weight,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    constraint=nothing,
    constrained_addresses=nothing,
    num_sweeps::Int=1,
) where {Nargs}
    args = (rng, args...)

    tt = TracedTrace()
    ppf = ScopedValues.with(TRACING_TRACE => tt) do
        process_probprog_function(f, args, "gibbs")
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

    function address_attr(address)
        return MLIR.IR.Attribute(
            MLIR.IR.Attribute[
                MLIR.IR.Attribute(
                    MLIR.API.enzymeSymbolAttrGet(
                        MLIR.IR.current_context(),
                        reinterpret(UInt64, pointer_from_objref(sym)),
                    ),
                ) for sym in address.path
            ],
        )
    end
    selected_attr = MLIR.IR.Attribute(MLIR.IR.Attribute[address_attr(a) for a in selection])

    width = size(original_trace, 2)
    trace_type = MLIR.IR.TensorType([1, width], MLIR.IR.Type(Float64))
    weight_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))
    diagnostics_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Bool))
    log_densities_type = MLIR.IR.TensorType(Int64[], MLIR.IR.Type(Float64))

    trace_mlir = TracedUtils.get_mlir_data(original_trace)
    weight_mlir = TracedUtils.get_mlir_data(original_weight)

    constraint_mlir = nothing
    constrained_attr = nothing
    if constraint !== nothing
        @assert constrained_addresses !== nothing "gibbs: `constrained_addresses` \
            required when `constraint` is provided"
        constraint_mlir = TracedUtils.get_mlir_data(constraint)
        constrained_attr = MLIR.IR.Attribute(
            MLIR.IR.Attribute[address_attr(a) for a in constrained_addresses]
        )
    end

    gibbs_config_attr = MLIR.IR.Attribute(
        MLIR.API.enzymeGibbsConfigAttrGet(MLIR.IR.current_context())
    )
    schema_attr = MLIR.IR.Attribute(MLIR.IR.Attribute[MLIR.IR.Attribute("weight")])

    infer_op = MLIR.Dialects.impulse.infer(
        mlir_caller_args,
        MLIR.IR.Value[weight_mlir],
        trace_mlir;
        constraint=constraint_mlir,
        trace=trace_type,
        diagnostics=diagnostics_type,
        log_densities=log_densities_type,
        output_rng_state=mlir_result_types[1],
        state_out=MLIR.IR.Type[weight_type],
        fn=fn_attr,
        selection=selected_attr,
        all_addresses=all_addresses_attr,
        gibbs_config=gibbs_config_attr,
        constrained_addresses=constrained_attr,
        state_schema=schema_attr,
        num_samples=Int64(num_sweeps),
    )

    traced_result = process_probprog_outputs(
        infer_op,
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

    new_trace = TracedRArray{Float64,2}((), MLIR.IR.result(infer_op, 1), (1, width))
    new_weight = TracedRArray{Float64,0}((), MLIR.IR.result(infer_op, 5), ())
    return new_trace, new_weight, traced_result
end

"""
    hmc(rng, trace, weight, f, args...; selection, step_size, num_steps,
        inverse_mass_matrix=nothing)

One HMC transition on the selected sites of `trace` (Gen's `hmc(trace, selection; L, eps)`):
`num_steps` leapfrog steps of exactly `step_size` (no adjustment), identity or the given
inverse mass, a Metropolis accept, no adaptation. Each selected site moves in the space its
support declares (support `:real`: the raw value). Built on `mcmc` (one `impulse.infer` with
`hmc_config`); the kept positions are written back into the selected sites' trace columns.

Returns `(new_trace, new_weight, accepted)`. `new_weight` is the log joint at the kept state
(minus the potential), which is the trace's joint when every selected site has support `:real`.
"""
function hmc(
    rng::AbstractRNG,
    trace,
    weight,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Selection,
    step_size,
    num_steps::Int,
    inverse_mass_matrix=nothing,
) where {Nargs}
    layout = Ref{Any}(nothing)
    samples, diagnostics, log_densities, _, _ = mcmc(
        rng,
        trace,
        f,
        args...;
        selection=selection,
        algorithm=:HMC,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=step_size,
        trajectory_length=1.0,
        num_steps=num_steps,
        num_warmup=0,
        num_samples=1,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        _layout=layout,
    )
    tt = layout[]
    new_trace = trace
    col = 0
    for address in selection
        entry = nothing
        for e in tt.entries
            if [e.parent_path..., e.symbol] == collect(address.path)
                entry = e
                break
            end
        end
        entry === nothing &&
            error("hmc: selected address $(address) is not a site of the model")
        n = entry.num_elements
        new_trace = Reactant.Ops.dynamic_update_slice(
            new_trace, samples[1:1, (col + 1):(col + n)], [1, entry.offset + 1]
        )
        col += n
    end
    # 0-dim like generate's weight: a loop-carried value must keep its type.
    new_weight = reshape(log_densities[1:1], ())
    accepted = Reactant.@allowscalar diagnostics[1, 1]
    return new_trace, new_weight, accepted
end
