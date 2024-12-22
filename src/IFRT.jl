module IFRT

using CxxWrap
using Reactant_jll

@wrapmodule(() -> Reactant_jll.libReactantExtra, :reactant_module_ifrt)

function __init__()
    @initcxx
end

# NOTE some DType kinds lack a corresponding Julia type
function Base.convert(::Type{DType}, type::Type)
    kind = if type === Bool
        DTypeKindPred
    elseif type === Int8
        DTypeKindS8
    elseif type === Int16
        DTypeKindS16
    elseif type === Int32
        DTypeKindS32
    elseif type === Int64
        DTypeKindS64
    elseif type === UInt8
        DTypeKindU8
    elseif type === UInt16
        DTypeKindU16
    elseif type === UInt32
        DTypeKindU32
    elseif type === UInt64
        DTypeKindU64
    elseif type === Float16
        DTypeKindF16
    elseif type === Float32
        DTypeKindF32
    elseif type === Float64
        DTypeKindF64
    elseif type === ComplexF32
        DTypeKindC64
    elseif type === ComplexF64
        DTypeKindC128
    elseif type === String
        DTypeKindString
    else
        @warn "`$type` can not be converted to DType"
        DTypeKindInvalid
    end

    return DType(kind)
end

end
