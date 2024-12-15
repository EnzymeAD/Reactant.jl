# NOTE: We are placing all the reactant_overrides here to avoid incompatibilities with
#       Revise.jl. Essentially files that contain reactant_overrides cannot be revised
#       correctly. Once that (https://github.com/timholy/Revise.jl/issues/646) is resolved
#       we should move all the reactant_overrides to relevant files.

# Compiling within a compile should return simply the original function
@reactant_override function Compiler.compile(
    f, args; client=nothing, optimize=true, sync=false
)
    return f
end

# Enzyme overrides
@reactant_override @noinline function Enzyme.autodiff_deferred(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    return overload_autodiff(rmode, f, rt, args...)
end

@reactant_override @noinline function Enzyme.autodiff(
    rmode::Enzyme.Mode, f::FA, rt::Type{A}, args::Vararg{Annotation,Nargs}
) where {FA<:Annotation,A<:Annotation,Nargs}
    return overload_autodiff(rmode, f, rt, args...)
end
