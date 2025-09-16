# The types listed in this file are the ones present in StableHLO specification.

# These only exist for the purpose of lowering. Since `ReactantPrimitive` is a fixed set of
# types, users can use these to convert their types to the primitive types supported by
# Reactant.
for T in (:F8E5M2, :F8E4M3FN, :F8E4M3B11FNUZ, :F8E5M2FNUZ, :F8E4M3FNUZ)
    @eval begin
        primitive type $(T) <: AbstractFloat 8 end

        Base.promote_rule(::Type{$(T)}, ::Type{Float16}) = Float16
        Base.promote_rule(::Type{Float16}, ::Type{$(T)}) = Float16

        Base.promote_rule(::Type{$(T)}, ::Type{Float32}) = Float32
        Base.promote_rule(::Type{Float32}, ::Type{$(T)}) = Float32

        Base.promote_rule(::Type{$(T)}, ::Type{Float64}) = Float64
        Base.promote_rule(::Type{Float64}, ::Type{$(T)}) = Float64

        Base.promote_rule(::Type{$(T)}, ::Type{<:Integer}) = $(T)
        Base.promote_rule(::Type{<:Integer}, ::Type{$(T)}) = $(T)

        @static if isdefined(Core, :BFloat16)
            Base.promote_rule(::Type{$(T)}, ::Type{Core.BFloat16}) = Core.BFloat16
            Base.promote_rule(::Type{Core.BFloat16}, ::Type{$(T)}) = Core.BFloat16
        end

        # For type conversion we simply rely on XLA
        (::Type{inT})(x::$(T)) where {inT<:Number} = convert(inT, x)
        (::Type{$(T)})(x::inT) where {inT<:Number} = convert($(T), x)
        Base.Complex(x::$(T)) = complex(x, zero(x))

        function Base.convert(::Type{inT}, x::$(T)) where {inT<:Number}
            @assert MLIR.IR._has_context() "currently only supported inside compiled functions"
            x isa TracedRNumber || (x = Ops.constant(x))
            return Ops.convert(TracedRNumber{inT}, x)
        end

        function Base.convert(::Type{$(T)}, x::inT) where {inT<:Number}
            @assert MLIR.IR._has_context() "currently only supported inside compiled functions"
            x isa TracedRNumber || (x = Ops.constant(x))
            return Ops.convert(TracedRNumber{unwrapped_eltype($(T))}, x)
        end
    end
end

primitive type TF32 <: AbstractFloat 32 end # currently only used to set precision

const ReactantFloat8 = Union{F8E5M2,F8E4M3FN,F8E4M3B11FNUZ,F8E5M2FNUZ,F8E4M3FNUZ}

# TODO: Quantized types

@static if isdefined(Core, :BFloat16)
    const ReactantFloat = Union{
        Float16,Core.BFloat16,Float32,Float64,Base.uniontypes(ReactantFloat8)...
    }
else
    const ReactantFloat = Union{Float16,Float32,Float64,Base.uniontypes(ReactantFloat8)...}
end

const ReactantComplexFloat = Union{Complex{Float32},Complex{Float64}}

const ReactantInt = Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64}

const ReactantComplexInt = Union{[Complex{T} for T in Base.uniontypes(ReactantInt)]...}

const ReactantFloatInt = Union{
    Base.uniontypes(ReactantInt)...,Base.uniontypes(ReactantFloat)...
}

const ReactantPrimitive = Union{
    Bool,
    Base.uniontypes(ReactantFloatInt)...,
    Base.uniontypes(ReactantComplexInt)...,
    Base.uniontypes(ReactantComplexFloat)...,
}

@inline to_reactant_primitive(v::T) where {T} = reinterpret(reactant_primitive(T), v)
@inline reactant_primitive(::Type{T}) where {T} = nothing

for T in Base.uniontypes(ReactantPrimitive)
    @eval @inline to_reactant_primitive(val::$(T)) = val
    @eval @inline reactant_primitive(::Type{$(T)}) = $(T)
end
