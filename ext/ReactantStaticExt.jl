module ReactantStaticExt

using Static: StaticBool, StaticInt, StaticFloat64, dynamic
using Reactant: Reactant, TracedRNumber

for (ST, T) in ((StaticBool, Bool), (StaticInt, Int), (StaticFloat64, Float64))
    @eval begin
        Reactant.unwrapped_eltype(::Type{<:$ST}) = $T
        Reactant.unwrapped_eltype(::$ST) = $T

        function Base.promote_rule(::Type{<:$ST}, ::Type{TracedRNumber{S}}) where {S}
            return TracedRNumber{promote_type($T, S)}
        end

        function Base.promote_rule(::Type{TracedRNumber{S}}, ::Type{<:$ST}) where {S}
            return TracedRNumber{promote_type($T, S)}
        end

        function Reactant.promote_to(::Type{TracedRNumber{S}}, x::$ST) where {S}
            return Reactant.promote_to(TracedRNumber{S}, dynamic(x))
        end
    end
end

end
