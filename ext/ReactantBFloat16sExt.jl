module ReactantBFloat16sExt

using BFloat16s: BFloat16
using Reactant: Reactant

@static if !isdefined(Core, :BFloat16) || Core.BFloat16 !== BFloat16
    function Reactant.MLIR.IR.Type(
        ::Type{BFloat16};
        context::Reactant.MLIR.IR.Context=Reactant.MLIR.IR.current_context(),
    )
        return Reactant.MLIR.IR.BFloat16Type(; context)
    end
end

end
