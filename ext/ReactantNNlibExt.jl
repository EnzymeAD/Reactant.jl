module ReactantNNlibExt

using NNlib
using Reactant

function __init__()
for (jlop, hloop) in ((:(NNlib.tanh), :tanh),(:(NNlib.tanh_fast), :tanh),)
    @eval begin
        function Reactant.elem_apply(::typeof($jlop), lhs::Reactant.TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return Reactant.TracedRArray{ElType,Shape,N}((), Reactant.MLIR.IR.result(Reactant.MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1))
        end
    end
end
end

end