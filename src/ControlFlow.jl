function ReactantCore.traced_if(cond::TracedRNumber{Bool}, true_fn, false_fn, args)
    # NOTE: This behavior is different from how we compile other functions, i.e., we keep
    #       things as constants if possible, but from a block we do need to return a
    #       traced value, so we force a conversion to a TracedType.
    seen_args = OrderedIdDict()
    traced_args = ntuple(length(args)) do i
        return make_tracer(
            seen_args, args[i], (:args, i), ArrayToTracedUnsafe; track_numbers=(Number,)
        )
    end

    true_block, true_res = generate_mlir_block(true_fn, [], MLIR.IR.Type[], traced_args)
    true_res = make_tracer(
        OrderedIdDict(), true_res, (), ArrayToTracedUnsafe; track_numbers=(Number,)
    )

    false_block, false_res = generate_mlir_block(false_fn, [], MLIR.IR.Type[], traced_args)
    false_res = make_tracer(
        OrderedIdDict(), false_res, (), ArrayToTracedUnsafe; track_numbers=(Number,)
    )

    @assert length(true_res) == length(false_res) "true branch returned $(length(true_res)) results, false branch returned $(length(false_res)). This shouldn't happen."

    seen_true_results = OrderedIdDict()
    traced_true_result = make_tracer(seen_true_results, true_res, (:result,), TracedSetPath)
    # for i in 1:length(traced_args)
    #     make_tracer(seen_true_results, traced_args[i], (:resargs, i), TracedTrack)
    # end
    true_linear_results = [v for v in values(seen_true_results) if v isa TracedType]

    seen_false_results = OrderedIdDict()
    traced_false_result = make_tracer(
        seen_false_results, false_res, (:result,), TracedSetPath
    )
    # for i in 1:length(traced_args)
    #     make_tracer(seen_false_results, traced_args[i], (:resargs, i), TracedTrack)
    # end
    false_linear_results = [v for v in values(seen_false_results) if v isa TracedType]

    final_true_linear_results = Vector{Any}(undef, length(true_res))
    final_false_linear_results = Vector{Any}(undef, length(false_res))

    @show true_linear_results
    @show false_linear_results

    for i in 1:length(true_res)
        true_idxs = findall(true_linear_results) do v
            v.paths[1][1] == :result && v.paths[1][2] == i
        end
        tr = true_linear_results[true_idxs]
        isempty(tr) && (tr = Any[MissingTracedValue()])

        false_idxs = findall(false_linear_results) do v
            v.paths[1][1] == :result && v.paths[1][2] == i
        end
        fr = false_linear_results[false_idxs]
        isempty(fr) && (fr = Any[MissingTracedValue()])

        resolved = false

        if length(tr) == length(fr)
            for (i, (trᵢ, frᵢ)) in enumerate(zip(tr, fr))
                if typeof(trᵢ) != typeof(frᵢ)
                    if !(trᵢ isa MissingTracedValue) && !(frᵢ isa MissingTracedValue)
                        error(
                            "Result #$(i) for the branches have different types: true branch \
                             returned `$(typeof(trᵢ))`, false branch returned `$(typeof(frᵢ))`.",
                        )
                    elseif trᵢ isa MissingTracedValue
                        tr[i] = new_traced_value(frᵢ)
                    else # frᵢ isa MissingTracedValue
                        fr[i] = new_traced_value(trᵢ)
                    end
                end
            end
            resolved = true
        elseif length(tr) == 1
            @assert tr[1] isa MissingTracedValue
            error("Not implemented yet")
        elseif length(fr) == 1
            @assert fr[1] isa MissingTracedValue
            error("Not implemented yet")
        else
            error("length(tr) = $(length(tr)), length(fr) = $(length(fr)). This shouldn't \
                   happen.")
        end

        if resolved
            final_true_linear_results[i] = tr
            final_false_linear_results[i] = fr
            continue
        else
            error("Could not resolve the return types of the branches")
        end
    end

    @show final_true_linear_results
    @show final_false_linear_results

    final_true_linear_results_flat = []
    for v in final_true_linear_results
        append!(final_true_linear_results_flat, v)
    end
    true_mlir_data = MLIR.IR.Value[x.mlir_data for x in final_true_linear_results_flat]
    true_region = MLIR.IR.Region()
    op = MLIR.Dialects.stablehlo.return_(true_mlir_data)
    MLIR.IR.rmfromparent!(op)
    push!(true_block, op)
    push!(true_region, true_block)

    final_false_linear_results_flat = []
    for v in final_false_linear_results
        append!(final_false_linear_results_flat, v)
    end
    false_mlir_data = MLIR.IR.Value[x.mlir_data for x in final_false_linear_results_flat]
    false_region = MLIR.IR.Region()
    op = MLIR.Dialects.stablehlo.return_(false_mlir_data)
    @show op
    MLIR.IR.rmfromparent!(op)
    push!(false_block, op)
    push!(false_region, false_block)

    result_types = MLIR.IR.Type[mlir_type(x) for x in final_true_linear_results_flat]

    # @show result_types
    # @show MLIR.IR.Type[mlir_type(x) for x in final_true_linear_results_flat]

    # if_compiled = MLIR.Dialects.stablehlo.if_(
    #     cond.mlir_data;
    #     true_branch=true_region,
    #     false_branch=false_region,
    #     result_0=result_types,
    # )

    # display(if_compiled)

    return error(1)

    # return map(1:MLIR.IR.nresults(if_compiled)) do i
    #     res = MLIR.IR.result(if_compiled, i)
    #     sz = size(MLIR.IR.type(res))
    #     T = MLIR.IR.julia_type(eltype(MLIR.IR.type(res)))
    #     isempty(sz) && return TracedRNumber{T}((), res)
    #     return TracedRArray{T,length(sz)}((), res, sz)
    # end
end
