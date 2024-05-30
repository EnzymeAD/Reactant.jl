
using Reactant

using Reactant.MLIR

@noinline function no(@nospecialize(x))
    x = @ccall $(Base.@cfunction(identity, Any, (Any,)))(x::Any)::Any
    return x[]::Any
end

mutable struct Data
    v::(Reactant.TracedRArray{Float64,S,1} where {S})
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
