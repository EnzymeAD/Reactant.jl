struct Value
    value::API.MlirValue

    function Value(value)
        @assert !mlirIsNull(value) "cannot create Value with null MlirValue"
        return new(value)
    end
end

Base.convert(::Core.Type{API.MlirValue}, value::Value) = value.value
Base.size(value::Value) = Base.size(type(value))
Base.ndims(value::Value) = Base.ndims(type(value))

"""
    ==(value1, value2)

Returns 1 if two values are equal, 0 otherwise.
"""
Base.:(==)(a::Value, b::Value) = API.mlirValueEqual(a, b)

"""
    is_block_arg(value)

Returns 1 if the value is a block argument, 0 otherwise.
"""
is_block_arg(value::Value) = API.mlirValueIsABlockArgument(value)

"""
    is_op_res(value)

Returns 1 if the value is an operation result, 0 otherwise.
"""
is_op_res(value::Value) = API.mlirValueIsAOpResult(value)

"""
    block_owner(value)

Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.
"""
function block_owner(value::Value)
    @assert is_block_arg(value) "could not get owner, value is not a block argument"
    return Block(API.mlirBlockArgumentGetOwner(value), false)
end

"""
    op_owner(value)

Returns an operation that produced this value as its result. Asserts if the value is not an op result.
"""
function op_owner(value::Value)
    @assert is_op_res(value) "could not get owner, value is not an op result"
    return Operation(API.mlirOpResultGetOwner(value), false)
end

function owner(value::Value)
    if is_block_arg(value)
        raw_block = API.mlirBlockArgumentGetOwner(value)
        mlirIsNull(raw_block) && return nothing
        return Block(raw_block, false)
    elseif is_op_res(value)
        raw_op = API.mlirOpResultGetOwner(value)
        mlirIsNull(raw_op) && return nothing
        return Operation(raw_op, false)
    else
        error("Value is neither a block argument nor an op result")
    end
end

"""
    block_arg_num(value)

Returns the position of the value in the argument list of its block.
"""
function block_arg_num(value::Value)
    @assert is_block_arg(value) "could not get arg number, value is not a block argument"
    return API.mlirBlockArgumentGetArgNumber(value)
end

"""
    op_res_num(value)

Returns the position of the value in the list of results of the operation that produced it.
"""
function op_res_num(value::Value)
    @assert is_op_res(value) "could not get result number, value is not an op result"
    return API.mlirOpResultGetResultNumber(value)
end

function position(value::Value)
    if is_block_arg(value)
        return block_arg_num(value)
    elseif is_op_res(value)
        return op_res_num(value)
    else
        error("Value is neither a block argument nor an op result")
    end
end

"""
    type(value)

Returns the type of the value.
"""
type(value::Value) = Type(API.mlirValueGetType(value))

"""
    set_type!(value, type)

Sets the type of the block argument to the given type.
"""
function type!(value, type)
    @assert is_a_block_argument(value) "could not set type, value is not a block argument"
    API.mlirBlockArgumentSetType(value, type)
    return value
end

function Base.show(io::IO, value::Value)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    GC.@preserve value ref begin
        API.mlirValuePrint(value, c_print_callback, ref)
    end
end
