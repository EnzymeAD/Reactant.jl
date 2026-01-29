module Dialects

using ..IR: NamedAttribute
using ..API

using Reactant_jll: Reactant_jll

operandsegmentsizes(segments) = NamedAttribute("operand_segment_sizes", Int32.(segments))

#! explicit-imports: off
for file in readdir(joinpath(@__DIR__, "Dialects"))
    endswith(file, ".jl") || continue
    include(joinpath(@__DIR__, "Dialects", file))
end
#! explicit-imports: on

end # module Dialects
