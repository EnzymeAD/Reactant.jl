module Dialects

import ..IR: Attribute, AbstractAttribute, NamedAttribute, context
import ..API
import ....Reactant

using Reactant_jll
namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::API.MlirAttribute) = NamedAttribute(name, Attribute(val))
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

function operandsegmentsizes(segments)
    return namedattribute("operand_segment_sizes", Attribute(Int32.(segments)))
end

c(a::AbstractArray) = isempty(a) ? "[]" : a
c(x) = x

for file in readdir(joinpath(@__DIR__, "Dialects"))
    endswith(file, ".jl") || continue
    include(joinpath(@__DIR__, "Dialects", file))
end

end # module Dialects
