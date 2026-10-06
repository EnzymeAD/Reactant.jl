module ReactantCheckpointingExt

# Loops written with Checkpointing.jl's `@ad_checkpoint scheme for ...` become
# `Checkpointing.checkpoint_for(body, scheme, range)`. Traced by Reactant, that
# loop would be unrolled, and the EnzymeRules that do the checkpointing in
# Checkpointing.jl are never used, since Reactant differentiates the traced IR
# with Enzyme-MLIR. Here the loop is traced as `@trace for` instead, with the
# Enzyme-MLIR checkpointing schedule that matches the scheme.

using Reactant: Reactant, @reactant_overlay, @trace
using ReactantCore: ReactantCore
using Checkpointing: Checkpointing

# The schedule Enzyme-MLIR runs for a scheme, from its number of checkpoints.
# Enzyme-MLIR keeps the snapshots in its own buffers, so the scheme's storage
# is not used.
reactant_checkpointing(scheme::Checkpointing.Revolve) = ReactantCore.Binomial(scheme.acp)
reactant_checkpointing(scheme::Checkpointing.Periodic) = ReactantCore.Periodic(scheme.acp)
# Online_r2 is for loops whose length is only known at the end. Enzyme-MLIR
# reverses a while loop with a binomial schedule over the iterations it counted.
reactant_checkpointing(scheme::Checkpointing.Online_r2) = ReactantCore.Binomial(scheme.acp)

function reactant_checkpointing(scheme::Checkpointing.Scheme)
    return error(
        "Reactant does not support the Checkpointing.jl scheme $(typeof(scheme)). " *
        "Use Revolve, Periodic or Online_r2, or `@trace checkpointing=...`.",
    )
end

function check_storage(scheme::Checkpointing.Scheme)
    storage = scheme.storage
    if !(storage isa Checkpointing.ArrayStorage)
        error(
            "Reactant keeps checkpoints in device buffers and cannot use " *
            "$(typeof(storage)). Use the default ArrayStorage.",
        )
    end
    return nothing
end

@reactant_overlay function Checkpointing.checkpoint_for(
    body::Function, scheme::Checkpointing.Scheme, range
)
    check_storage(scheme)
    checkpointing = reactant_checkpointing(scheme)
    @trace checkpointing = checkpointing track_numbers = false for i in range
        body(i)
    end
    return nothing
end

@reactant_overlay function Checkpointing.checkpoint_while(
    body::Function, scheme::Checkpointing.Scheme
)
    check_storage(scheme)
    checkpointing = reactant_checkpointing(scheme)
    go = ReactantCore.promote_to_traced(true)
    @trace checkpointing = checkpointing track_numbers = false while go
        go = body()
    end
    return nothing
end

end
