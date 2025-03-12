using Reactant
using Reactant: MLIR, MLIR.IR, TracedRArray, Compiler
using Test, CUDA


function test_raising_pipeline(unopt_ir)
    ctx = MLIR.IR.Context(Reactant.registry[], false)
    Reactant.Compiler.context_gc_vector[ctx] = Vector{TracedRArray}(undef, 0)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    IR.context!(ctx) do
        mod = parse(IR.Module, unopt_ir)
        main = MLIR.IR.lookup(MLIR.IR.SymbolTable(MLIR.IR.Operation(mod)), "main")

        ftype_attr = MLIR.IR.attr(main, "function_type")
        ftype = MLIR.IR.Type(ftype_attr)

        inputs = AbstractArray[]
        for i = 1:IR.ninputs(ftype)
            ty = IR.input(ftype, i)
            et = IR.julia_type(eltype(ty))
            sz = size(ty)

            push!(inputs, rand(et, sz))
        end

        opt_raised =
            Reactant.Compiler.run_reactant_pipeline!(
                mod;
                optimize = true,
                raise = true,
                no_nan = false,
                backend = "cpu",
            ) |> repr
        @debug "raised ir" ir = Text(opt_raised)

        function raised(inputs...)
            Reactant.Ops.hlo_call(opt_raised, inputs...)
        end

        function normal(inputs...)
            Reactant.Ops.hlo_call(unopt_ir, inputs...)
        end

        concrete_inputs = map(Reactant.to_rarray, inputs)

        concrete_results_raised = @jit raised(concrete_inputs...)
        concrete_results_unopt = @jit normal(concrete_inputs...)

        results_raised = map(Array, concrete_results_raised)
        results_unopt = map(Array, concrete_results_unopt)

        return results_raised, results_unopt
    end
end

function square_kernel!(x, y)
    i = threadIdx().x
    x[i] *= y[i]
    # We don't yet auto lower this via polygeist
    # sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x, y)
    @cuda blocks = 1 threads = length(x) square_kernel!(x, y)
    return nothing
end

@testset "square!" begin
    oA = collect(1:1:64)
    A = Reactant.to_rarray(oA)
    B = Reactant.to_rarray(100 .* oA)
    unopt_ir = repr(@code_hlo optimize = false square!(A, B))
    results_raised, results_unopt = test_raising_pipeline(unopt_ir)
    for (x, y) in zip(results_raised, results_unopt)
        @test x ≈ y
    end
end
