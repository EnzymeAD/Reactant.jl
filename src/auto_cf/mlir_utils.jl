function change_value!(from::Reactant.MLIR.IR.Value, to::Reactant.MLIR.IR.Value, op::Reactant.MLIR.IR.Operation)
    for i in 1:Reactant.MLIR.IR.noperands(op)
        Reactant.MLIR.IR.operand(op, i) == from || continue
        Reactant.MLIR.IR.operand!(op, i, to)
    end
    
    for i in 1:Reactant.MLIR.IR.nregions(op)
        r = Reactant.MLIR.IR.region(op, i)
        change_value!(from, to, r)
    end
end

function change_value!(from::Reactant.MLIR.IR.Value, to::Reactant.MLIR.IR.Value, region::Reactant.MLIR.IR.Region)
    for block in Reactant.MLIR.IR.BlockIterator(region)
        change_value!(from, to, block)
    end
end

function change_value!(from::Reactant.MLIR.IR.Value, to::Reactant.MLIR.IR.Value, block::Reactant.MLIR.IR.Block)
    for op in Reactant.MLIR.IR.OperationIterator(block)
        change_value!(from, to, op)
    end
end