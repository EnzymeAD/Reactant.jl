module ReactantFloat8sExt

using Float8s: Float8_4
using Reactant: Reactant

Reactant.reactant_primitive(::Type{Float8_4}) = Reactant.F8E4M3FN

end
