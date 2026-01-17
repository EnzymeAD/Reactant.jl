@checked struct Identifier
    ref::API.MlirIdentifier
end

"""
    Identifier(context, str)

Gets an identifier with the given string value.
"""
function Identifier(str::String; context::Context=context())
    return Identifier(API.mlirIdentifierGet(context, str))
end

Base.cconvert(::Core.Type{API.MlirIdentifier}, id::Identifier) = id.ref

"""
    ==(ident, other)

Checks whether two identifiers are the same.
"""
Base.:(==)(a::Identifier, b::Identifier) = API.mlirIdentifierEqual(a, b)

"""
    context(ident)

Returns the context associated with this identifier
"""
context(id::Identifier) = Context(API.mlirIdentifierGetContext(id.identifier))

"""
    String(ident)

Gets the string value of the identifier.
"""
Base.String(id::Identifier) = String(API.mlirIdentifierStr(id.identifier))
