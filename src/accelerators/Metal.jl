module Metal

using Reactant: Reactant

"""
    has_metal() -> Bool

Returns `true` when all of the following are true:
  - running on Apple hardware (`Sys.isapple()`)
  - Metal.jl has been loaded into the session (`isdefined(Main, :Metal)`)
  - the Metal GPU stack is operational (`Metal.functional()`)

The Metal PJRT backend is registered automatically by `ReactantMetalExt`
when `using Metal` is evaluated, so callers typically only need this for
informational queries or conditional dispatch.
"""
function has_metal()
    return Sys.isapple() &&
           isdefined(Main, :Metal) &&
           Main.Metal.functional()
end

"""
    setup_metal!()

Placeholder hook for external callers.  The actual Metal PJRT client is
created inside `ReactantMetalExt.__init__()`, which Julia loads automatically
whenever `Metal` is brought into scope as a weak dependency.
"""
function setup_metal!()
    # Metal client registration is handled by ReactantMetalExt.__init__()
    # when Metal.jl is loaded as a weak dependency.
    return nothing
end

end # module Metal
