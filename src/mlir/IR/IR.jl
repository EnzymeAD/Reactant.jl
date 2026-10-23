module IR

using ..Reactant
using ..API

using BFloat16s: BFloat16s, BFloat16
using LLVM: LLVM, @dispose, mark_alloc, mark_use, mark_dispose, mark_untracked
import LLVM: activate, deactivate, dispose

# add an inner constructor to a wrapper type definition, which checks that the handle in
# its `ref` field isn't null
macro checked(typedef)
    Meta.isexpr(typedef, :struct) || error("expected a struct definition")
    name = typedef.args[2]
    name = Meta.isexpr(name, :<:) ? name.args[1] : name
    fields = filter(arg -> !(arg isa LineNumberNode), typedef.args[3].args)
    names = [Meta.isexpr(f, :(::)) ? f.args[1] : f for f in fields]
    push!(
        typedef.args[3].args,
        :(function $name($(fields...))
            ref.ptr == C_NULL && throw(UndefRefError())
            return new($(names...))
        end),
    )
    return esc(typedef)
end

# WARN do not export `Type` nor `Module` as they are already defined in Core
# also, use `Core.Type` and `Core.Module` inside this module to avoid clash with
# MLIR `Type` and `Module`
export Attribute,
    Block,
    Context,
    Dialect,
    Diagnostic,
    DiagnosticHandler,
    Location,
    Operation,
    Region,
    Value
export activate, deactivate, dispose, enable_multithreading!
export context, current_context, has_context, @with_context
export severity,
    nnotes, note, attach_diagnostic_handler!, detach_diagnostic_handler!, emit_error
export block, current_block, has_block, @with_block
export current_module, has_module, @with_module
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
include("DiagnosticHandler.jl")
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
