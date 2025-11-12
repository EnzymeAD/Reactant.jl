struct Attribute
    attribute::API.MlirAttribute
end

"""
    Attribute()

Returns an empty attribute.
"""
Attribute() = Attribute(API.mlirAttributeGetNull())

Base.convert(::Core.Type{API.MlirAttribute}, attribute::Attribute) = attribute.attribute

"""
    parse(::Core.Type{Attribute}, str; context=context())

Parses an attribute. The attribute is owned by the context.
"""
Base.parse(::Core.Type{Attribute}, str; context::Context=context()) =
    Attribute(API.mlirAttributeParseGet(context, str))

"""
    ==(a1, a2)

Checks if two attributes are equal.
"""
Base.:(==)(a::Attribute, b::Attribute) = API.mlirAttributeEqual(a, b)

"""
    context(attribute)

Gets the context that an attribute was created with.
"""
context(attr::Attribute) = Context(API.mlirAttributeGetContext(attr))

"""
    type(attribute)

Gets the type of this attribute.
"""
type(attr::Attribute) = Type(API.mlirAttributeGetType(attr))

"""
    typeid(attribute)

Gets the type id of the attribute.
"""
typeid(attr::Attribute) = TypeID(API.mlirAttributeGetTypeID(attr))

"""
    isaffinemap(attr)

Checks whether the given attribute is an affine map attribute.
"""
isaffinemap(attr::Attribute) = API.mlirAttributeIsAAffineMap(attr)

"""
    Attribute(affineMap)

Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.
"""
Attribute(map::AffineMap) = Attribute(API.mlirAffineMapAttrGet(map))

"""
    AffineMap(attr)

Returns the affine map wrapped in the given affine map attribute.
"""
AffineMap(attr::Attribute) = AffineMap(API.mlirAffineMapAttrGetValue(attr))

"""
    isarray(attr)

Checks whether the given attribute is an array attribute.
"""
isarray(attr::Attribute) = API.mlirAttributeIsAArray(attr)

"""
    Attribute(elements; context=context())

Creates an array element containing the given list of elements in the given context.
"""
Attribute(attrs::Vector{Attribute}; context::Context=context()) =
    Attribute(API.mlirArrayAttrGet(context, length(attrs), attrs))

"""
    isdict(attr)

Checks whether the given attribute is a dictionary attribute.
"""
isdict(attr::Attribute) = API.mlirAttributeIsADictionary(attr)

"""
    Attribute(elements; context=context())

Creates a dictionary attribute containing the given list of elements in the provided context.
"""
function Attribute(attrs::Dict; context::Context=context())
    attrs = [NamedAttribute(k, Attribute(v); context) for (k, v) in attrs]
    return Attribute(API.mlirDictionaryAttrGet(context, length(attrs), attrs))
end

"""
    isfloat(attr)

Checks whether the given attribute is a floating point attribute.
"""
isfloat(attr::Attribute) = API.mlirAttributeIsAFloat(attr)

"""
    Attribute(float; context=context(), location=Location(), check=false)

Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.
If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function Attribute(
    f::T; context::Context=context(), location::Location=Location(), check::Bool=false
) where {T<:AbstractFloat}
    if check
        Attribute(API.mlirFloatAttrDoubleGetChecked(location, Type(T), Float64(f)))
    else
        Attribute(API.mlirFloatAttrDoubleGet(context, Type(T), Float64(f)))
    end
end

"""
    Float64(attr)

Returns the value stored in the given floating point attribute, interpreting the value as double.
"""
function Base.Float64(attr::Attribute)
    @assert isfloat(attr) "attribute $(attr) is not a floating point attribute"
    return API.mlirFloatAttrGetValueDouble(attr)
end

"""
    Attribute(complex; context=context(), location=Location(), check=false)

Creates a complex attribute in the given context with the given complex value and double-precision FP semantics.
"""
function Attribute(
    c::T; context::Context=context(), location::Location=Location(), check::Bool=false
) where {T<:Complex}
    if check
        Attribute(
            API.mlirComplexAttrDoubleGetChecked(
                location, Type(T), Float64(real(c)), Float64(imag(c))
            ),
        )
    else
        Attribute(
            API.mlirComplexAttrDoubleGet(
                context, Type(T), Float64(real(c)), Float64(imag(c))
            ),
        )
    end
end

"""
    isinteger(attr)

Checks whether the given attribute is an integer attribute.
"""
isinteger(attr::Attribute) = API.mlirAttributeIsAInteger(attr)

"""
    Attribute(int)

Creates an integer attribute of the given type with the given integer value.
"""
Attribute(i::T, type=Type(T)) where {T<:Integer} =
    Attribute(API.mlirIntegerAttrGet(type, Int64(i)))

"""
    Int64(attr)

Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.
"""
function Base.Int64(attr::Attribute)
    @assert isinteger(attr) "attribute $(attr) is not an integer attribute"
    return API.mlirIntegerAttrGetValueInt(attr)
end

# TODO mlirIntegerAttrGetValueSInt

"""
    UInt64(attr)

Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.
"""
function Base.UInt64(attr::Attribute)
    @assert isinteger(attr) "attribute $(attr) is not an integer attribute"
    return API.mlirIntegerAttrGetValueUInt(attr)
end

"""
    isbool(attr)

Checks whether the given attribute is a bool attribute.
"""
isbool(attr::Attribute) = API.mlirAttributeIsABool(attr)

"""
    Attribute(value; context=context())

Creates a bool attribute in the given context with the given value.
"""
Attribute(b::Bool; context::Context=context()) = Attribute(API.mlirBoolAttrGet(context, b))

"""
    Bool(attr)

Returns the value stored in the given bool attribute.
"""
function Base.Bool(attr::Attribute)
    @assert isbool(attr) "attribute $(attr) is not a boolean attribute"
    return API.mlirBoolAttrGetValue(attr)
end

"""
    isintegerset(attr)

Checks whether the given attribute is an integer set attribute.
"""
isintegerset(attr::Attribute) = API.mlirAttributeIsAIntegerSet(attr)

"""
    isopaque(attr)

Checks whether the given attribute is an opaque attribute.
"""
isopaque(attr::Attribute) = API.mlirAttributeIsAOpaque(attr)

"""
    OpaqueAttribute(dialectNamespace, dataLength, data, type; context=context())

Creates an opaque attribute in the given context associated with the dialect identified by its namespace.
The attribute contains opaque byte data of the specified length (data need not be null-terminated).
"""
OpaqueAttribute(namespace, data, type; context::Context=context) =
    Attribute(API.mlirOpaqueAttrGet(context, namespace, length(data), data, type))

"""
    mlirOpaqueAttrGetDialectNamespace(attr)

Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.
"""
function namespace(attr::Attribute)
    @assert isopaque(attr) "attribute $(attr) is not an opaque attribute"
    return String(API.mlirOpaqueAttrGetDialectNamespace(attr))
end

"""
    data(attr)

Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function data(attr::Attribute)
    @assert isopaque(attr) "attribute $(attr) is not an opaque attribute"
    return String(API.mlirOpaqueAttrGetData(attr)) # TODO return as Base.CodeUnits{Int8,String}? or as a Vector{Int8}? or Pointer?
end

"""
    isstring(attr)

Checks whether the given attribute is a string attribute.
"""
isstring(attr::Attribute) = API.mlirAttributeIsAString(attr)

"""
    Attribute(str; context=context())

Creates a string attribute in the given context containing the given string.
"""
Attribute(str::AbstractString; context::Context=context()) =
    Attribute(API.mlirStringAttrGet(context, str))

"""
    Attribute(type, str)

Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.
"""
function Attribute(type::Type, str::AbstractString)
    return Attribute(API.mlirStringAttrTypedGet(type, str))
end

"""
    String(attr)

Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function Base.String(attr::Attribute)
    @assert isstring(attr) "attribute $(attr) is not a string attribute"
    return String(API.mlirStringAttrGetValue(attr))
end

"""
    issymbolref(attr)

Checks whether the given attribute is a symbol reference attribute.
"""
issymbolref(attr::Attribute) = API.mlirAttributeIsASymbolRef(attr)

"""
    SymbolRefAttribute(symbol, references; context=context())

Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references.
Each of the references in the list must not be nested.
"""
SymbolRefAttribute(
    symbol::String, references::Vector{Attribute}; context::Context=context()
) = Attribute(API.mlirSymbolRefAttrGet(context, symbol, length(references), references))

"""
    rootref(attr)

Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function rootref(attr::Attribute)
    @assert issymbolref(attr) "attribute $(attr) is not a symbol reference attribute"
    return String(API.mlirSymbolRefAttrGetRootReference(attr))
end

"""
    leafref(attr)

Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function leafref(attr::Attribute)
    @assert issymbolref(attr) "attribute $(attr) is not a symbol reference attribute"
    return String(API.mlirSymbolRefAttrGetLeafReference(attr))
end

"""
    nnestedrefs(attr)

Returns the number of references nested in the given symbol reference attribute.
"""
function nnestedrefs(attr::Attribute)
    @assert issymbolref(attr) "attribute $(attr) is not a symbol reference attribute"
    return API.mlirSymbolRefAttrGetNumNestedReferences(attr)
end

"""
    isflatsymbolref(attr)

Checks whether the given attribute is a flat symbol reference attribute.
"""
isflatsymbolref(attr::Attribute) = API.mlirAttributeIsAFlatSymbolRef(attr)

"""
    FlatSymbolRefAttribute(ctx, symbol)

Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.
"""
FlatSymbolRefAttribute(symbol::String; context::Context=context()) =
    Attribute(API.mlirFlatSymbolRefAttrGet(context, symbol))

"""
    flatsymbol(attr)

Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function flatsymbol(attr::Attribute)
    @assert isflatsymbolref(attr) "attribute $(attr) is not a flat symbol reference attribute"
    return String(API.mlirFlatSymbolRefAttrGetValue(attr))
end

"""
    istype(attr)

Checks whether the given attribute is a type attribute.
"""
istype(attr::Attribute) = API.mlirAttributeIsAType(attr)

"""
    Attribute(type)

Creates a type attribute wrapping the given type in the same context as the type.
"""
Attribute(type::Type) = Attribute(API.mlirTypeAttrGet(type))

"""
    Type(attr)

Returns the type stored in the given type attribute.
"""
Type(attr::Attribute) = Type(API.mlirTypeAttrGetValue(attr))

"""
    isunit(attr)

Checks whether the given attribute is a unit attribute.
"""
isunit(attr::Attribute) = API.mlirAttributeIsAUnit(attr)

"""
    UnitAttribute(; context=context())

Creates a unit attribute in the given context.
"""
UnitAttribute(; context::Context=context()) = Attribute(API.mlirUnitAttrGet(context))

"""
    iselements(attr)

Checks whether the given attribute is an elements attribute.
"""
iselements(attr::Attribute) = API.mlirAttributeIsAElements(attr)

# TODO mlirElementsAttrGetValue
# TODO mlirElementsAttrIsValidIndex

"""
    isdenseelements(attr)

Checks whether the given attribute is a dense elements attribute.
"""
isdenseelements(attr::Attribute) = API.mlirAttributeIsADenseElements(attr)
isdenseintelements(attr::Attribute) = API.mlirAttributeIsADenseIntElements(attr)
isdensefloatelements(attr::Attribute) = API.mlirAttributeIsADenseFPElements(attr)

"""
    DenseElementsAttribute(shapedType, elements)

Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.
"""
function DenseElementsAttribute(shaped_type::Type, elements::AbstractArray)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return Attribute(API.mlirDenseElementsAttrGet(shaped_type, length(elements), elements))
end

# TODO mlirDenseElementsAttrRawBufferGet

"""
    fill(attr, shapedType)

Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).
"""
function Base.fill(attr::Attribute, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return Attribute(API.mlirDenseElementsAttrSplatGet(shaped_type, attr))
end

function Base.fill(value::Bool, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrBoolSplatGet(shaped_type, value)
end

function Base.fill(value::UInt8, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrUInt8SplatGet(shaped_type, value)
end

function Base.fill(value::Int8, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrInt8SplatGet(shaped_type, value)
end

function Base.fill(value::UInt32, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrUInt32SplatGet(shaped_type, value)
end

function Base.fill(value::Int32, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrInt32SplatGet(shaped_type, value)
end

function Base.fill(value::UInt64, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrUInt64SplatGet(shaped_type, value)
end

function Base.fill(value::Int64, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrInt64SplatGet(shaped_type, value)
end

function Base.fill(value::Float32, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrFloatSplatGet(shaped_type, value)
end

function Base.fill(value::Float64, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    return API.mlirDenseElementsAttrDoubleSplatGet(shaped_type, value)
end

function Base.fill(::Core.Type{Attribute}, value, shape::Vector{Int})
    shaped_type = TensorType(shape, Type(typeof(value)))
    return Base.fill(value, shaped_type)
end

to_row_major(x) = permutedims(x, ndims(x):-1:1)
to_row_major(x::AbstractVector) = x
to_row_major(x::AbstractArray{T,0}) where {T} = x

"""
    DenseElementsAttribute(array::AbstractArray)

Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.
"""
function DenseElementsAttribute(values::AbstractArray{Bool})
    shaped_type = TensorType(collect(Int, size(values)), Type(Bool))
    return Attribute(
        API.mlirDenseElementsAttrBoolGet(
            shaped_type, length(values), AbstractArray{Cint}(to_row_major(values))
        ),
    )
end

function DenseElementsAttribute(values::AbstractArray{UInt8})
    shaped_type = TensorType(collect(Int, size(values)), Type(UInt8))
    return Attribute(
        API.mlirDenseElementsAttrUInt8Get(shaped_type, length(values), to_row_major(values))
    )
end

function DenseElementsAttribute(values::AbstractArray{Int8})
    shaped_type = TensorType(collect(Int, size(values)), Type(Int8))
    return Attribute(
        API.mlirDenseElementsAttrInt8Get(shaped_type, length(values), to_row_major(values))
    )
end

function DenseElementsAttribute(values::AbstractArray{UInt16})
    shaped_type = TensorType(collect(Int, size(values)), Type(UInt16))
    return Attribute(
        API.mlirDenseElementsAttrUInt16Get(
            shaped_type, length(values), to_row_major(values)
        ),
    )
end

function DenseElementsAttribute(values::AbstractArray{Int16})
    shaped_type = TensorType(collect(Int, size(values)), Type(Int16))
    return Attribute(
        API.mlirDenseElementsAttrInt16Get(shaped_type, length(values), to_row_major(values))
    )
end

function DenseElementsAttribute(values::AbstractArray{UInt32})
    shaped_type = TensorType(collect(Int, size(values)), Type(UInt32))
    return Attribute(
        API.mlirDenseElementsAttrUInt32Get(
            shaped_type, length(values), to_row_major(values)
        ),
    )
end

function DenseElementsAttribute(values::AbstractArray{Int32})
    shaped_type = TensorType(collect(Int, size(values)), Type(Int32))
    return Attribute(
        API.mlirDenseElementsAttrInt32Get(shaped_type, length(values), to_row_major(values))
    )
end

function DenseElementsAttribute(values::AbstractArray{UInt64})
    shaped_type = TensorType(collect(Int, size(values)), Type(UInt64))
    return Attribute(
        API.mlirDenseElementsAttrUInt64Get(
            shaped_type, length(values), to_row_major(values)
        ),
    )
end

function DenseElementsAttribute(values::AbstractArray{Int64})
    shaped_type = TensorType(collect(Int, size(values)), Type(Int64))
    return Attribute(
        API.mlirDenseElementsAttrInt64Get(shaped_type, length(values), to_row_major(values))
    )
end

function DenseElementsAttribute(values::AbstractArray{Float32})
    shaped_type = TensorType(collect(Int, size(values)), Type(Float32))
    return Attribute(
        API.mlirDenseElementsAttrFloatGet(shaped_type, length(values), to_row_major(values))
    )
end

function DenseElementsAttribute(values::AbstractArray{Float64})
    shaped_type = TensorType(collect(Int, size(values)), Type(Float64))
    return Attribute(
        API.mlirDenseElementsAttrDoubleGet(
            shaped_type, length(values), to_row_major(values)
        ),
    )
end

if isdefined(Core, :BFloat16)
    function DenseElementsAttribute(values::AbstractArray{Core.BFloat16})
        shaped_type = TensorType(collect(Int, size(values)), Type(Core.BFloat16))
        return Attribute(
            API.mlirDenseElementsAttrBFloat16Get(
                shaped_type, length(values), to_row_major(values)
            ),
        )
    end
end

function DenseElementsAttribute(values::AbstractArray{Float16})
    shaped_type = TensorType(collect(Int, size(values)), Type(Float16))
    return Attribute(
        API.mlirDenseElementsAttrFloat16Get(
            shaped_type, length(values), to_row_major(values)
        ),
    )
end

function DenseElementsAttribute(values::AbstractArray)
    shaped_type = TensorType(collect(Int, size(values)), Type(eltype(values)))
    return Attribute(
        API.mlirDenseElementsAttrRawBufferGet(
            shaped_type, length(values) * Base.elsize(values), to_row_major(values)
        ),
    )
end

"""
    DenseElementsAttribute(array::AbstractArray{String})

Creates a dense elements attribute with the given shaped type from string elements.
"""
function DenseElementsAttribute(values::AbstractArray{String})
    # TODO may fail because `Type(String)` is not defined
    shaped_type = TensorType(collect(Int, size(values)), Type(String))
    return Attribute(
        API.mlirDenseElementsAttrStringGet(
            shaped_type, length(values), to_row_major(values)
        ),
    )
end

"""
    Base.reshape(attr, shapedType)

Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.
"""
function Base.reshape(attr::Attribute, shape::Vector{Int})
    @assert isdenseelements(attr) "attribute $(attr) is not a dense elements attribute"
    @assert length(attr) == prod(shape) "new shape $(shape) has a different number of elements than the original attribute"
    element_type = eltype(type(attr))
    shaped_type = TensorType(shape, element_type)
    return Attribute(API.mlirDenseElementsAttrReshape(attr, shaped_type))
end

"""
    issplat(attr)

Checks whether the given dense elements attribute contains a single replicated value (splat).
"""
function issplat(attr::Attribute)
    @assert isdenseelements(attr) "attribute $(attr) is not a dense elements attribute"
    return API.mlirDenseElementsAttrIsSplat(attr) # TODO Base.allequal?
end

# TODO mlirDenseElementsAttrGetRawData

"""
    issparseelements(attr)

Checks whether the given attribute is a sparse elements attribute.
"""
issparseelements(attr::Attribute) = API.mlirAttributeIsASparseElements(attr)

# TODO mlirSparseElementsAttribute
# TODO mlirSparseElementsAttrGetIndices
# TODO mlirSparseElementsAttrGetValues

@llvmversioned min = v"16" """
      isdensearray(attr, ::Core.Type{T})

  Checks whether the given attribute is a dense array attribute.
  """
function isdensearray end

@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Bool}) =
    API.mlirAttributeIsADenseBoolArray(attr)
@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Int8}) =
    API.mlirAttributeIsADenseI8Array(attr)
@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Int16}) =
    API.mlirAttributeIsADenseI16Array(attr)
@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Int32}) =
    API.mlirAttributeIsADenseI32Array(attr)
@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Int64}) =
    API.mlirAttributeIsADenseI64Array(attr)
@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Float32}) =
    API.mlirAttributeIsADenseF32Array(attr)
@llvmversioned min = v"16" isdensearray(attr::Attribute, ::Core.Type{Float64}) =
    API.mlirAttributeIsADenseF64Array(attr)

@llvmversioned min = v"16" """
      DenseArrayAttribute(array; context=context())

  Create a dense array attribute with the given elements.
  """
function DenseArrayAttribute end

@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Bool}; context::Context=context()
) = Attribute(
    API.mlirDenseBoolArrayGet(
        context, length(values), AbstractArray{Cint}(to_row_major(values))
    ),
)
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Int8}; context::Context=context()
) = Attribute(API.mlirDenseI8ArrayGet(context, length(values), to_row_major(values)))
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{UInt8}; context::Context=context()
) = Attribute(API.mlirDenseI8ArrayGet(context, length(values), to_row_major(values)))
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Int16}; context::Context=context()
) = Attribute(API.mlirDenseI16ArrayGet(context, length(values), to_row_major(values)))
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Int32}; context::Context=context()
) = Attribute(API.mlirDenseI32ArrayGet(context, length(values), to_row_major(values)))
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Int64}; context::Context=context()
) = Attribute(API.mlirDenseI64ArrayGet(context, length(values), to_row_major(values)))
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Float32}; context::Context=context()
) = Attribute(API.mlirDenseF32ArrayGet(context, length(values), to_row_major(values)))
@llvmversioned min = v"16" DenseArrayAttribute(
    values::AbstractArray{Float64}; context::Context=context()
) = Attribute(API.mlirDenseF64ArrayGet(context, length(values), to_row_major(values)))

@llvmversioned min = v"16" Attribute(values::AbstractArray) = DenseArrayAttribute(values)

function Base.length(attr::Attribute)
    if isarray(attr)
        API.mlirArrayAttrGetNumElements(attr)
    elseif isdict(attr)
        API.mlirDictionaryAttrGetNumElements(attr)
    elseif iselements(attr)
        API.mlirElementsAttrGetNumElements(attr)
    else
        _isdensearray = any(
            T -> isdensearray(attr, T), [Bool, Int8, Int16, Int32, Int64, Float32, Float64]
        )
        if _isdensearray
            API.mlirDenseBoolArrayGetNumElements(attr)
        end
    end
end

function Base.getindex(attr::Attribute, i)
    if isarray(attr)
        Attribute(API.mlirArrayAttrGetElement(attr, i))
    elseif isdict(attr)
        if i isa String
            Attribute(API.mlirDictionaryAttrGetElementByName(attr, i))
        else
            NamedAttribute(API.mlirDictionaryAttrGetElement(attr, i))
        end
    elseif isdenseelements(attr)
        elem_type = julia_type(eltype(type(attr)))
        if elem_type isa Bool
            API.mlirDenseElementsAttrGetBoolValue(attr, i)
        elseif elem_type isa Int8
            API.mlirDenseElementsAttrGetInt8Value(attr, i)
        elseif elem_type isa UInt8
            API.mlirDenseElementsAttrGetUInt8Value(attr, i)
        elseif elem_type isa Int16
            API.mlirDenseElementsAttrGetInt16Value(attr, i)
        elseif elem_type isa UInt16
            API.mlirDenseElementsAttrGetUInt16Value(attr, i)
        elseif elem_type isa Int32
            API.mlirDenseElementsAttrGetInt32Value(attr, i)
        elseif elem_type isa UInt32
            API.mlirDenseElementsAttrGetUInt32Value(attr, i)
        elseif elem_type isa Int64
            API.mlirDenseElementsAttrGetInt64Value(attr, i)
        elseif elem_type isa UInt64
            API.mlirDenseElementsAttrGetUInt64Value(attr, i)
        elseif elem_type isa Float32
            API.mlirDenseElementsAttrGetFloatValue(attr, i)
        elseif elem_type isa Float64
            API.mlirDenseElementsAttrGetDoubleValue(attr, i)
        elseif elem_type isa String # TODO does this case work?
            String(API.mlirDenseElementsAttrGetStringValue(attr, i))
        else
            throw("unsupported element type $(elem_type)")
        end
    else
        if isdensearray(attr, Bool)
            API.mlirDenseBoolArrayGetElement(attr, i)
        elseif isdensearray(attr, Int8)
            API.mlirDenseI8ArrayGetElement(attr, i)
        elseif isdensearray(attr, Int16)
            API.mlirDenseI16ArrayGetElement(attr, i)
        elseif isdensearray(attr, Int32)
            API.mlirDenseI32ArrayGetElement(attr, i)
        elseif isdensearray(attr, Int64)
            API.mlirDenseI64ArrayGetElement(attr, i)
        elseif isdensearray(attr, Float32)
            API.mlirDenseF32ArrayGetElement(attr, i)
        elseif isdensearray(attr, Float64)
            API.mlirDenseF64ArrayGetElement(attr, i)
        end
    end
end

function Base.iterate(attr::Attribute, state=1)
    if state > length(attr)
        nothing
    else
        (attr[state], state + 1)
    end
end

function Base.getindex(attr::Attribute)
    @assert isdenseelements(attr) "attribute $(attr) is not a dense elements attribute"
    @assert issplat(attr) "attribute $(attr) is not splatted (more than one different elements)"
    elem_type = julia_type(eltype(type(attr)))
    if elem_type isa Bool
        API.mlirDenseElementsAttrGetBoolSplatValue(attr)
    elseif elem_type isa Int8
        API.mlirDenseElementsAttrGetInt8SplatValue(attr)
    elseif elem_type isa UInt8
        API.mlirDenseElementsAttrGetUInt8SplatValue(attr)
    elseif elem_type isa Int16
        API.mlirDenseElementsAttrGetInt16SplatValue(attr)
    elseif elem_type isa UInt16
        API.mlirDenseElementsAttrGetUInt16SplatValue(attr)
    elseif elem_type isa Int32
        API.mlirDenseElementsAttrGetInt32SplatValue(attr)
    elseif elem_type isa UInt32
        API.mlirDenseElementsAttrGetUInt32SplatValue(attr)
    elseif elem_type isa Int64
        API.mlirDenseElementsAttrGetInt64SplatValue(attr)
    elseif elem_type isa UInt64
        API.mlirDenseElementsAttrGetUInt64SplatValue(attr)
    elseif elem_type isa Float32
        API.mlirDenseElementsAttrGetFloatSplatValue(attr)
    elseif elem_type isa Float64
        API.mlirDenseElementsAttrGetDoubleSplatValue(attr)
    elseif elem_type isa String # TODO does this case work?
        String(API.mlirDenseElementsAttrGetStringSplatValue(attr))
    else
        throw("unsupported element type $(elem_type)")
    end
end

function Base.show(io::IO, attribute::Attribute)
    print(io, "Attribute(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirAttributePrint(attribute, c_print_callback, ref)
    return print(io, " =#)")
end

struct NamedAttribute
    named_attribute::API.MlirNamedAttribute
end

"""
    NamedAttribute(name, attr)

Associates an attribute with the name. Takes ownership of neither.
"""
function NamedAttribute(name, attribute; context=context(attribute))
    @assert !mlirIsNull(attribute.attribute)
    name = API.mlirIdentifierGet(context, name)
    return NamedAttribute(API.mlirNamedAttributeGet(name, attribute))
end

function Base.convert(::Core.Type{API.MlirAttribute}, named_attribute::NamedAttribute)
    return named_attribute.named_attribute
end
