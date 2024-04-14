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

operandsegmentsizes(segments) = namedattribute("operand_segment_sizes", Attribute(Int32.(segments)))

for path in readdir(Reactant_jll.artifact_dir; join=true)
    if endswith("inc.jl")(path)
        include(path)
    end
end

end # module Dialects
