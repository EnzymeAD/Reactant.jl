@checked struct SymbolTable
    ref::API.MlirSymbolTable
end

"""
    SymbolTable(operation)

Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.
"""
SymbolTable(op::Operation) = SymbolTable(mark_alloc(API.mlirSymbolTableCreate(op)))
SymbolTable(mod::Module) = SymbolTable(Operation(mod))

dispose(st::SymbolTable) = mark_dispose(API.mlirSymbolTableDestroy(st))

Base.cconvert(::Core.Type{API.MlirSymbolTable}, st::SymbolTable) = st
Base.unsafe_convert(::Core.Type{API.MlirSymbolTable}, st::SymbolTable) = mark_use(st).ref

# TODO(#2246) mlirSymbolTableGetSymbolAttributeName
# TODO(#2246) mlirSymbolTableGetVisibilityAttributeName

"""
    lookup(symboltable, name)

Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol.
If the symbol cannot be found, returns a null operation.
"""
function lookup(st::SymbolTable, name::AbstractString)
    raw_op = API.mlirSymbolTableLookup(st, name)
    if raw_op.ptr == C_NULL
        nothing
    else
        Operation(raw_op)
    end
end
function Base.getindex(st::SymbolTable, name::AbstractString)
    @something(lookup(st, name), throw(KeyError(name)))
end

"""
    Base.push!(symboltable, operation)

Inserts the given operation into the given symbol table. The operation must have the symbol trait.
If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness.
Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately.
Returns the name of the symbol after insertion.
"""
Base.push!(st::SymbolTable, op::Operation) = Attribute(API.mlirSymbolTableInsert(st, op))

"""
    Base.delete!(symboltable, operation)

Removes the given operation from the symbol table and erases it.
"""
Base.delete!(st::SymbolTable, op::Operation) = API.mlirSymbolTableErase(st, op)

# TODO(#2246) mlirSymbolTableReplaceAllSymbolUses
# TODO(#2246) mlirSymbolTableWalkSymbolTables
