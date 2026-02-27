module Metal

using Reactant: Reactant

"""
    has_metal() -> Bool

Returns `true` when the Metal PJRT API has been initialized by `ReactantMetalExt`.
This works regardless of whether `Metal` was loaded in `Main` scope (e.g. Pluto
notebooks load packages in module scope, not `Main`).
"""
function has_metal()
    return Sys.isapple() && Reactant.XLA.PJRT._metal_pjrt_api_ptr[] != C_NULL
end

end # module Metal
