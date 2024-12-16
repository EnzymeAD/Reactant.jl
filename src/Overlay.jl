# NOTE: We are placing all the reactant_overrides here to avoid incompatibilities with
#       Revise.jl. Essentially files that contain reactant_overrides cannot be revised
#       correctly. Once that (https://github.com/timholy/Revise.jl/issues/646) is resolved
#       we should move all the reactant_overrides to relevant files.

# Helper Function to determine if we are inside the ReactantInterpreter
"""
    within_reactant_interpreter()

Returns `true` if we are currently inside the ReactantInterpreter.
"""
@noinline within_reactant_interpreter() = false
@reactant_overlay @noinline within_reactant_interpreter() = true

# Compiling within a compile should return simply the original function
@reactant_overlay function Compiler.compile(
    f, args; client=nothing, optimize=true, sync=false
)
    return f
end

# Enzyme.jl overlays
@reactant_overlay @noinline function Enzyme.autodiff_deferred(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    return overload_autodiff(rmode, f, rt, args...)
end

@reactant_overlay @noinline function Enzyme.autodiff(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    return overload_autodiff(rmode, f, rt, args...)
end

# Random.jl overlays
@reactant_overlay @noinline function Random.default_rng()
    return call_with_reactant(TracedRandom.default_rng)
end
