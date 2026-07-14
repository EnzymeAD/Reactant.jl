module ReactantStaticExt

using Static: StaticBool, StaticInt, StaticFloat64, dynamic
using Reactant: Reactant, TracedRNumber

# The promotion rules are defined per traced numeric kind: a kind-generic
# `::Type{<:TracedRNumber{S}}` method would be ambiguous with Static's own
# `promote_rule(::Type{<:StaticNumber}, ::Type{<:Union{Rational,AbstractFloat,Signed}})`,
# which matches the traced float kind through its `AbstractFloat` supertype.
for (ST, T) in ((StaticBool, Bool), (StaticInt, Int), (StaticFloat64, Float64))
    @eval begin
        Reactant.unwrapped_eltype(::Type{<:$ST}) = $T
        Reactant.unwrapped_eltype(::$ST) = $T

        function Reactant.promote_to(::Type{TN}, x::$ST) where {S,TN<:TracedRNumber{S}}
            return Reactant.promote_to(TN, dynamic(x))
        end
    end
    for K in (
        Reactant.TracedRInteger,
        Reactant.TracedRFloat,
        Reactant.TracedRComplex,
        Reactant.TracedRReal,
        TracedRNumber,
    )
        @eval begin
            function Base.promote_rule(::Type{<:$ST}, ::Type{$K{S}}) where {S}
                return Reactant.traced_number_type(promote_type($T, S))
            end

            function Base.promote_rule(::Type{$K{S}}, ::Type{<:$ST}) where {S}
                return Reactant.traced_number_type(promote_type($T, S))
            end
        end
    end
end

end
