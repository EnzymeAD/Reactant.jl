module Dialects

import ..IR: Attribute, NamedAttribute, context
import ..API

using Reactant_jll

namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::Attribute) = NamedAttribute(name, val)
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

function operandsegmentsizes(segments)
    return namedattribute("operand_segment_sizes", Attribute(Int32.(segments)))
end

for file in readdir("Dialects")
    include(joinpath("Dialects", file))
end

end # module Dialects
