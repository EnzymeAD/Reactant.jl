module IR

using ..Reactant
using ..API

import LLVM: activate, deactivate, dispose, @dispose

# WARN do not export `Type` nor `Module` as they are already defined in Core
# also, use `Core.Type` and `Core.Module` inside this module to avoid clash with
# MLIR `Type` and `Module`
export Attribute, Block, Context, Dialect, Location, Operation, Region, Value
export activate, deactivate, dispose, enable_multithreading!
export context, current_context, has_context, with_context
export block, current_block, has_block, with_block
export current_module, has_module, with_module
export type, settype!, location, typeid, dialect
export nattrs, getattr, setattr!, rmattr!
export nregions, region
export nresults, result, noperands, operand, setoperand!
export nsuccessors, successor
export @affinemap

using Random: randstring

include("Utils.jl")

include("LogicalResult.jl")
include("Context.jl")
include("Dialect.jl")
include("Location.jl")
include("Type.jl")
include("TypeID.jl")
include("Operation.jl")
include("Module.jl")
include("Block.jl")
include("Region.jl")
include("Value.jl")
include("OpOperand.jl")
include("Identifier.jl")
include("SymbolTable.jl")
include("AffineExpr.jl")
include("AffineMap.jl")
include("Attribute.jl")
include("IntegerSet.jl")

include("ExecutionEngine.jl")
include("Pass.jl")

end # module IR
