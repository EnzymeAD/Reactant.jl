module IR

using ..Reactant
using ..API

# do not export `Type`, as it is already defined in Core
# also, use `Core.Type` inside this module to avoid clash with MLIR `Type`
export Attribute, Block, Context, Dialect, Location, Operation, Region, Value
export activate!, deactivate!, dispose!, enable_multithreading!
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

# TODO try to fuse it with Ops.hlo_call?
function tryinject!(sym_name, code; verify=false, mod=current_module(), location=Location())
    fn = lookup(SymbolTable(Operation(mod)), sym_name)

    if fn === nothing
        ctx = current_context()
        block = body(mod)
        return @ccall API.mlir_c.mlirOperationInject(
            ctx::API.MlirContext,
            block::API.MlirBlock,
            code::API.MlirStringRef,
            location::API.MlirLocation,
            verify::Bool,
        )::Bool
    else
        return false
    end
end

function inject!(sym_name, code; kwargs...)
    success = tryinject!(sym_name, code; kwargs...)
    @assert success "Failed injecting MLIR to top-level block"
end

function tryinjectop!(sym_name, code; mod=current_module(), location=Location())
    fn = lookup(SymbolTable(Operation(mod)), sym_name)

    if isnothing(fn)
        top_level_block = body(mod)
        return parse(Operation, code; block=top_level_block, location)
    else
        return nothing
    end
end

end # module IR
