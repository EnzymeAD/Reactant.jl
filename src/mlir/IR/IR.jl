module IR

using ..Reactant
using ..API

using LLVM: LLVM, @checked, mark_alloc, mark_use, mark_dispose
import LLVM: activate, deactivate, dispose, refcheck
const activate! = activate
const deactivate! = deactivate
const dispose! = dispose

# fix for `@checked` on MLIR.API types
for AT in [
    :MlirDialect,
    :MlirDialectHandle,
    :MlirDialectRegistry,
    :MlirContext,
    :MlirLocation,
    :MlirType,
    :MlirTypeID,
    :MlirTypeIDAllocator,
    :MlirModule,
    :MlirOperation,
    :MlirOpOperand,
    :MlirBlock,
    :MlirRegion,
    :MlirValue,
    # :MlirLogicalResult,
    :MlirAffineExpr,
    :MlirAffineMap,
    # :MlirAttribute,
    # :MlirNamedAttribute,
    :MlirIntegerSet,
    :MlirIdentifier,
    :MlirSymbolTable,
    :MlirExecutionEngine,
    :MlirPassManager,
    :MlirOpPassManager,
]
    @eval refcheck(T::Core.Type, ref::API.$AT) = refcheck(T, ref.ptr)
end

# WARN do not export `Type` nor `Module` as they are already defined in Core
# also, use `Core.Type` and `Core.Module` inside this module to avoid clash with
# MLIR `Type` and `Module`
export Attribute, Block, Context, Dialect, Location, Operation, Region, Value
export dispose!, @scope, current_block, current_module, current_region
export context, type, settype!, location, typeid, block, dialect
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

# MLIR extra from ReactantExtra
const default_registry = Ref{Union{Nothing,DialectRegistry}}(nothing)

function register_enzymexla_dialects(ctx::Context)
    @ccall API.mlir_c.RegisterDialects(ctx::API.MlirContext)::Cvoid
end

# pass initialization state is required to avoid multiple initialization on precompilation
const passes_initialized = Ref(false)

function initialize_dialect()
    default_registry[] = DialectRegistry()
    @ccall API.mlir_c.InitializeRegistry(
        default_registry[]::API.MlirDialectRegistry
    )::Cvoid
    if !passes_initialized[]
        @ccall API.mlir_c.InitializePasses(
            default_registry[]::API.MlirDialectRegistry
        )::Cvoid
        passes_initialized[] = true
    end
    return nothing
end

function deinitialize_dialect()
    passes_initialized[] = false
    return default_registry[] = nothing
end

function __init__()
    initialize_dialect()

    # create a persistent context to hold data along the program lifetime
    # if any op/module is used after calling `compile`, MLIR attrs/types/... will segfault
    # this Context is freed at exit
    ctx = Context(default_registry[]; threading=false)
    register_enzymexla_dialects(ctx)
    activate!(ctx)
end

# TODO try to fuse it with Ops.hlo_call?
function tryinject!(sym_name, code; verify=false, mod=current_module(), location=Location())
    fn = lookup(SymbolTable(Operation(mod)), sym_name)

    if fn === nothing
        ctx = context()
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
