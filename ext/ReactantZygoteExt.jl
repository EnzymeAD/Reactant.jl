module ReactantZygoteExt

using Reactant:
    Reactant, CallWithReactant, @reactant_overlay, use_overlayed_version, call_with_reactant
using Zygote: Zygote
using Enzyme: Enzyme, Reverse, Active, Const, Duplicated

# TODO: overload the following as well
#       - Zygote.pullback
#       - Zygote.jacobian
#       - Zygote.hessian

@reactant_overlay function Zygote.gradient(f::F, args...) where {F}
    # TODO: check `f` as well once #1642 is merged
    if use_overlayed_version(args)
        @warn "Reactant doesn't support using Zygote for computing gradients. Replacing \
               `Zygote.gradient` with `Enzyme.autodiff` call. Please update your code to \
               not use `Zygote.gradient` and instead use `Enzyme.gradient` inside \
               `Reactant.@compile`."
        dargs = map(Enzyme.make_zero, args)
        duplicated = map(Duplicated, args, dargs)
        Reactant.overload_autodiff(Reverse, Const(f), Active, duplicated...)
        return dargs
    else
        return Base.inferencebarrier(Zygote.gradient)(CallWithReactant(f), args...)
    end
end

end
