module ReactantDLFP8TypesExt

using DLFP8Types: Float8_E4M3FN, Float8_E4M3FNUZ, Float8_E5M2, Float8_E5M2FNUZ
using Reactant: Reactant

Reactant.reactant_primitive(::Type{Float8_E4M3FN}) = Reactant.F8E4M3FN
Reactant.reactant_primitive(::Type{Float8_E4M3FNUZ}) = Reactant.F8E4M3FNUZ
Reactant.reactant_primitive(::Type{Float8_E5M2}) = Reactant.F8E5M2
Reactant.reactant_primitive(::Type{Float8_E5M2FNUZ}) = Reactant.F8E5M2FNUZ

end
