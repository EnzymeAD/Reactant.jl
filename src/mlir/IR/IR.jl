module IR

using ..API

# do not export `Type`, as it is already defined in Core
# also, use `Core.Type` inside this module to avoid clash with MLIR `Type`
export Attribute, Block, Context, Dialect, Location, Operation, Region, Value
export activate!, deactivate!, dispose!, enable_multithreading!, context!
export context, type, type!, location, typeid, block, dialect
export nattrs,
    attr,
    attr!,
    rmattr!,
    nregions,
    region,
    nresults,
    result,
    noperands,
    operand,
    operand!,
    nsuccessors,
    successor
export BlockIterator, RegionIterator, OperationIterator
export @affinemap

function mlirIsNull(val)
    return val.ptr == C_NULL
end

function print_callback(str::API.MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

macro llvmversioned(pred, expr)
    @assert Meta.isexpr(pred, :(=)) "Expected an expression as the first argument"

    predname, version = pred.args
    @assert predname in (:min, :max) "Expected 'min' or 'max' as the first argument"

    @assert Meta.isexpr(version, :macrocall) && version.args[1] == Symbol("@v_str") "Expected a VersionNumber"
    version = eval(version)

    if predname == :min && VersionNumber(19) >= version ||
        predname == :max && VersionNumber(19) <= version
        esc(expr)
    else
        esc(:(nothing))
    end
end

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
include("Iterators.jl")

include("ExecutionEngine.jl")
include("Pass.jl")

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Core.Type{API.MlirStringRef}, s::Union{Symbol,String}) = s
function Base.cconvert(::Core.Type{API.MlirStringRef}, s::AbstractString)
    return Base.cconvert(API.MlirStringRef, String(s)::String)
end

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(
    ::Core.Type{API.MlirStringRef}, s::Union{Symbol,String,AbstractVector{UInt8}}
)
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return API.MlirStringRef(p, sizeof(s))
end

function Base.String(str::API.MlirStringRef)
    return Base.unsafe_string(pointer(str.data), str.length)
end

Base.String(str::API.MlirIdentifier) = String(API.mlirIdentifierStr(str))

### Utils

function visit(f, op)
    all_ok = true
    for region in RegionIterator(op)
        for block in BlockIterator(region)
            for op in OperationIterator(block)
                all_ok &= f(op)
            end
        end
    end
    return all_ok
end

"""
    verifyall(operation; debug=false)

Prints the operations which could not be verified.
"""
function verifyall(operation::Operation; debug=false)
    io = IOBuffer()
    visit(operation) do op
        ok = verifyall(op; debug)
        if !ok || !verify(op)
            if ok
                show(IOContext(io, :debug => debug), op)
                error(String(take!(io)))
            end
            false
        else
            true
        end
    end
end
verifyall(module_::IR.Module; debug=false) = verifyall(Operation(module_); debug)

end # module IR
