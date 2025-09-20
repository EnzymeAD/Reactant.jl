module Dialects

import ..IR: Attribute, NamedAttribute
import ..API

using Reactant_jll: Reactant_jll

namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::Attribute) = NamedAttribute(name, val)
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

function operandsegmentsizes(segments)
    return namedattribute("operand_segment_sizes", Attribute(Int32.(segments)))
end

for file in readdir(joinpath(@__DIR__, "Dialects"))
    endswith(file, ".jl") || continue
    include(joinpath(@__DIR__, "Dialects", file))
end

end # module Dialects
