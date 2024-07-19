
using Reactant
using Enzyme, NNlib
using Reactant.MLIR

@noinline function no(@nospecialize(x))
    x = @ccall $(Base.@cfunction(identity, Any, (Any,)))(x::Any)::Any
    return x[]::Any
end

mutable struct Data
    v::Reactant.TracedRArray{Float64,1}
end
@noinline function tmp(a, b, d)
    @show d
    @show typeof(d)
    c = d.v
    @show typeof(c)

    return reshape(a, (4,)) ./ sqrt.(b .+ a)
end

function test()
    ctx = MLIR.IR.Context()
    Base.append!(Reactant.registry[]; context=ctx)
    @ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid

    MLIR.IR.context!(ctx) do
        mod = MLIR.IR.Module(MLIR.IR.Location())
        modbody = MLIR.IR.body(mod)

        in_tys = [MLIR.IR.TensorType([4], MLIR.IR.Type(Float64))]

        func = MLIR.Dialects.func.func_(;
            sym_name="main_tmp",
            function_type=MLIR.IR.FunctionType(in_tys, []),
            body=MLIR.IR.Region(),
        )

        fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for _ in in_tys])
        push!(MLIR.IR.region(func, 1), fnbody)

        GC.@preserve mod func fnbody begin
            MLIR.IR.block!(fnbody) do
                a = ones(4)
                b = ones(4)
                d = Data(
                    Reactant.TracedRArray{Float64,(4,),1}((), MLIR.IR.argument(fnbody, 1))
                )

                return tmp(a, b, d)
            end
        end

        return println(string(mod))
    end
end
test()

@testset "Activation Functions" begin
    sumabs2(f, x) = sum(abs2, f.(x))

    function ∇sumabs2(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    x_act = randn(Float32, 10, 10)
    x_act_ca = Reactant.ConcreteRArray(x_act)

    @testset "Activation: $act" for act in (
        identity, relu, sigmoid, tanh, tanh_fast, sigmoid_fast, gelu, abs2
    )
        f_compile = Reactant.compile(sumabs2, (act, x_act))

        y_simple = sumabs2(act, x_act)
        y_compile = f_compile(act, x_act_ca)

        ∂x_enz = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(act), Duplicated(x_act, ∂x_enz))

        ∇sumabs2_compiled = Reactant.compile(∇sumabs2, (act, x_act_ca))

        ∂x_compile = ∇sumabs2_compiled(act, x_act_ca)

        @test y_simple ≈ y_compile
    end
end
