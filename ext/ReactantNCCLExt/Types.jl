using Reactant
using Reactant: MLIR, Sharding
using ReactantCore: ReactantCore
using Reactant.TracedUtils: get_mlir_data, get_paths, set_paths!

# NOTE crossing the Reactant boundary is not yet supported for NCCL types
# i.e. there is no `ConcreteCommunicator` yet, so a `TracedCommunicator` can only be created inside a traced function

mutable struct TracedCommunicator
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
end

ReactantCore.is_traced(::TracedCommunicator) = true
ReactantCore.is_traced(::TracedCommunicator, _) = true

function Base.show(io::IOty, X::T) where {IOty<:Union{IO,IOContext},T<:TracedCommunicator}
    return print(io, "$T(paths=", X.paths, ")")
end

Reactant.TracedUtils.get_mlir_data(x::TracedCommunicator) = x.mlir_data
Reactant.TracedUtils.set_mlir_data!(x::TracedCommunicator, data) = (x.mlir_data = data; return x)

Reactant.TracedUtils.get_paths(x::TracedCommunicator) = x.paths
Reactant.TracedUtils.set_paths!(x::TracedCommunicator, paths) = (x.paths = paths; return x)

function Reactant.traced_type_inner(
    T::Type{TracedCommunicator},
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
)
    if mode == Reactant.TracedToConcrete
        error("Returning a NCCL communicator from a compiled function is not supported yet")
    end
    return T
end

function Reactant.make_tracer(
    seen,
    @nospecialize(prev::TracedCommunicator),
    @nospecialize(path),
    mode;
    @nospecialize(sharding = Sharding.NoSharding()),
    kwargs...,
)
    if Sharding.is_sharded(sharding)
        error("Simultaneous use of sharding and NCCL is not supported")
    end

    if mode == Reactant.ConcreteToTraced
        throw("Cannot trace existing trace type")

    elseif mode == Reactant.TracedToTypes
        push!(path, MLIR.IR.type(get_mlir_data(prev)))
        return nothing

    elseif mode == Reactant.TracedTrack || mode == Reactant.TracedSetPath
        set_paths!(prev, (get_paths(prev)..., path))
        if !haskey(seen, prev)
            return seen[prev] = prev
        end
        return prev

    elseif mode == Reactant.NoStopTracedTrack
        set_paths!(prev, (get_paths(prev)..., path))
        if !haskey(seen, prev)
            seen[prev] = prev # don't return!
        end
        return prev

    else
        throw("Trace mode $mode not implemented for TracedCommunicator")
    end
end

Reactant.use_overlayed_version(::TracedCommunicator) = true

function Reactant.Ops.mlir_type(::Type{TracedCommunicator})
    return MLIR.IR.Type(MLIR.API.enzymexlaCommNcclCommTypeGet(MLIR.IR.current_context()))
end
