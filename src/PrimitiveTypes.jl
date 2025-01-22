# These only exist for the purpose of lowering. Since `ReactantPrimitive` is a fixed set of
# types, users can use these to convert their types to the primitive types supported by
# Reactant.
struct F8E5M2{T} <: AbstractFloat
    val::T
end

struct F8E4M3FN{T} <: AbstractFloat
    val::T
end

struct F8E4M3B11FNUZ{T} <: AbstractFloat
    val::T
end

struct F8E5M2FNUZ{T} <: AbstractFloat
    val::T
end

struct F8E4M3FNUZ{T} <: AbstractFloat
    val::T
end

# TODO: Quantized types

const ReactantFloat8 = Union{F8E5M2,F8E4M3FN,F8E4M3B11FNUZ,F8E5M2FNUZ,F8E4M3FNUZ}

@static if isdefined(Core, :BFloat16)
    const ReactantFloat = Union{
        Float16,Core.BFloat16,Float32,Float64,Base.uniontypes(ReactantFloat8)...
    }
else
    const ReactantFloat = Union{Float16,Float32,Float64,Base.uniontypes(ReactantFloat8)...}
end

const ReactantComplexFloat = Union{[Complex{T} for T in Base.uniontypes(ReactantFloat)]...}

const ReactantInt = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128}

const ReactantComplexInt = Union{
    Complex{Int8},
    Complex{UInt8},
    Complex{Int16},
    Complex{UInt16},
    Complex{Int32},
    Complex{UInt32},
    Complex{Int64},
    Complex{UInt64},
    Complex{Int128},
    Complex{UInt128},
}

const ReactantFloatInt = Union{
    Base.uniontypes(ReactantInt)...,Base.uniontypes(ReactantFloat)...
}

const ReactantPrimitive = Union{
    Bool,
    Base.uniontypes(ReactantFloatInt)...,
    Base.uniontypes(ReactantComplexInt)...,
    Base.uniontypes(ReactantComplexFloat)...,
}

"""
    to_reactant_primitive(val)

Constructs a Reactant primitive from the given value. Returns the Reactant primitive and a
function that can be used to convert the value back to the original type.
"""
to_reactant_primitive(::T) where {T} = nothing, nothing

for T in Base.uniontypes(ReactantPrimitive)
    @eval to_reactant_primitive(val::$T) = val, identity
end
