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
    if Reactant.OVERLAY_ZYGOTE_CALLS[] && use_overlayed_version(args)
        @warn "Reactant doesn't support using Zygote for computing gradients. Replacing \
               `Zygote.gradient` with `Enzyme.autodiff` call. Please update your code to \
               not use `Zygote.gradient` and instead use `Enzyme.gradient` inside \
               `Reactant.@compile`. If this behavior is undesirable, set the \
               `overlay_zygote_calls` scoped value via `Reactant.with_config` to \
               `false`.\n\nReactant can remove this switching without any breaking change \
               and hence reliance on this behavior is strongly discouraged."
        return Enzyme.gradient(Reverse, Const(f), args...)
    else
        return Base.inferencebarrier(Zygote.gradient)(CallWithReactant(f), args...)
    end
end

end
