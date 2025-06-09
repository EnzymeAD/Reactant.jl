# import Pkg
# 
# Pkg.activate("../KA")

using Reactant
using Reactant: MLIR, MLIR.IR, TracedRArray, Compiler
using Test, CUDA


function test_raising_pipeline(unopt_ir, nth_jitcall=1)
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    unopt_ir, inputs, name = IR.context!(ctx) do
        mod = parse(IR.Module, unopt_ir)
        main = MLIR.IR.lookup(MLIR.IR.SymbolTable(MLIR.IR.Operation(mod)), "main")

        # Get all jit_calls
        # and generate wrapper func
        body = IR.first_block(IR.region(main, 1))
        ops = IR.OperationIterator(body) |> collect
        jitcalls = filter(op -> IR.name(op) == "enzymexla.kernel_call", ops)

        jit_call = jitcalls[nth_jitcall]

        # names = map(jitcalls) do jit_call
        #     String(IR.API.mlirSymbolRefAttrGetRootReference(IR.attr(jit_call, "fn")))
        # end |> unique
        # for n in names
        #     println(n)
        # end
        # @show length(names)
        name = String(IR.API.mlirSymbolRefAttrGetRootReference(IR.attr(jit_call, "fn")))

        toskip = 8
        operands = IR.operands(jit_call)[toskip:end]
        nargs = IR.nargs(body)
        new_arg_types = IR.Type[IR.type(o) for o in operands]
        returns = IR.results(jit_call)

        deleteat!(ops, findfirst(==(jit_call), ops))
        for operand in Iterators.take(IR.operands(jit_call), 7)
            deleteat!(ops, findfirst(==(IR.op_owner(operand)), ops))
        end

        @info "here"

        for (i, ty) in enumerate(new_arg_types)
            new_arg = IR.push_argument!(body, ty)
            IR.operand!(jit_call, 7 + i, new_arg)
        end

        for op in ops
            IR.rmfromparent!(op)
        end
        for i in 1:nargs
            IR.erase_argument!(body, 1)
        end

        push!(body, MLIR.Dialects.func.return_(returns))

        @info "here!"

        ftype_attr = MLIR.IR.attr(main, "function_type")
        ftype = MLIR.IR.Type(ftype_attr)

        new_ftype = IR.FunctionType(
            new_arg_types,
            IR.Type[IR.type(r) for r in returns]
        )
        MLIR.IR.attr!(main, "function_type", IR.Attribute(new_ftype))

        inputs = AbstractArray[]
        for ty in new_arg_types
            et = IR.julia_type(eltype(ty))
            sz = size(ty)

            push!(inputs, rand(et, sz))
        end

        @info "here!!"

        pm = IR.PassManager()
        opm = IR.OpPassManager(pm)
        parse(opm, "builtin.module(symbol-dce)")
        IR.run!(pm, mod)

        ir = repr(mod)
        empty!(ops)
        GC.gc(true)
        return ir, inputs, name
    end
    GC.gc(true)

    @info "got ir"

    write(joinpath("kernels", "jitcall" * string(nth_jitcall) * ".mlir"), unopt_ir)

    # concrete_inputs = Tuple(map(Reactant.to_rarray, inputs))
    # 
    # concrete_results_raised = @jit raise=true Reactant.Ops.hlo_call(unopt_ir, concrete_inputs...)
    # concrete_results_unopt = @jit Reactant.Ops.hlo_call(unopt_ir, concrete_inputs...)
    # 
    # results_raised = map(Array, concrete_results_raised)
    # results_unopt = map(Array, concrete_results_unopt)
    # 
    # return results_raised, results_unopt, name
end


function do_test()
    test_raising_pipeline(read("/Users/bep7/Downloads/ocean-simulation-mlir-1/unopt_ocean_climate_simulation.mlir", String), parse(Int, ARGS[end]))
end

do_test()
