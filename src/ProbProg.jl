module ProbProg

using ..Reactant: Reactant, XLA, MLIR, TracedUtils
using ReactantCore: ReactantCore

using Enzyme

@noinline function generate(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    argprefix::Symbol = gensym("generatearg")
    resprefix::Symbol = gensym("generateresult")
    resargprefix::Symbol = gensym("generateresarg")

    mlir_fn_res = TracedUtils.make_mlir_fn(
        f,
        args,
        (),
        string(f),
        false;
        args_in_result=:result_and_mutated,
        argprefix,
        resprefix,
        resargprefix,
    )
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]
    fname = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(batch_inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            TracedUtils.push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    gen_op = MLIR.Dialects.enzyme.generate(batch_inputs; outputs=out_tys, fn=fname)

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(gen_op, i)
        for path in res.paths
            isempty(path) && continue
            if path[1] == resprefix
                TracedUtils.set!(result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if idx == 1 && fnwrap
                    TracedUtils.set!(f, path[3:end], resv)
                else
                    if fnwrap
                        idx -= 1
                    end
                    TracedUtils.set!(args[idx], path[3:end], resv)
                end
            end
        end
    end

    return result
end

function sample!(f::Function, args::Vararg{Any,Nargs}) where {Nargs}
    argprefix::Symbol = gensym("samplearg")
    resprefix::Symbol = gensym("sampleresult")
    resargprefix::Symbol = gensym("sampleresarg")

    mlir_fn_res = TracedUtils.make_mlir_fn(
        f,
        args,
        (),
        string(f),
        false;
        args_in_result=:result_and_mutated,
        argprefix,
        resprefix,
        resargprefix,
    )
    (; result, linear_args, in_tys, linear_results) = mlir_fn_res
    fnwrap = mlir_fn_res.fnwrapped
    func2 = mlir_fn_res.f

    batch_inputs = MLIR.IR.Value[]
    for a in linear_args
        idx, path = TracedUtils.get_argidx(a, argprefix)
        if idx == 1 && fnwrap
            TracedUtils.push_val!(batch_inputs, f, path[3:end])
        else
            idx -= fnwrap ? 1 : 0
            TracedUtils.push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    out_tys = [MLIR.IR.type(TracedUtils.get_mlir_data(res)) for res in linear_results]

    sym = TracedUtils.get_attribute_by_name(func2, "sym_name")
    fn_attr = MLIR.IR.FlatSymbolRefAttribute(Base.String(sym))

    sample_op = MLIR.Dialects.enzyme.sample(batch_inputs; outputs=out_tys, fn=fn_attr)

    for (i, res) in enumerate(linear_results)
        resv = MLIR.IR.result(sample_op, i)

        for path in res.paths
            isempty(path) && continue
            if path[1] == resprefix
                TracedUtils.set!(result, path[2:end], resv)
            elseif path[1] == argprefix
                idx = path[2]::Int
                if idx == 1 && fnwrap
                    TracedUtils.set!(f, path[3:end], resv)
                else
                    if fnwrap
                        idx -= 1
                    end
                    TracedUtils.set!(args[idx], path[3:end], resv)
                end
            end
        end
    end

    return result
end

end
