@enumx TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
    TracedToTypes = 6
    NoStopTracedTrack = 7
end

struct VisitedObject
    id::Int
end

struct DefaultWalkWithInitialPath{IP} <: Functors.AbstractWalk
    initial_path::IP
end

function (walk::DefaultWalkWithInitialPath)(recurse, kp::Functors.KeyPath, x)
    x_children, re = Functors.functor(x)
    isempty(kp) && (kp = Functors.KeyPath(walk.initial_path))
    return re(
        Functors._map(
            recurse,
            Functors._map(Base.Fix1(Functors.KeyPath, kp), Functors._keys(x_children)),
            x_children,
        ),
    )
end

isleaf(::Functors.KeyPath, x) = isleaf(x)
isleaf(x) = Functors.isleaf(x)
function isleaf(x::AbstractArray{T}) where {T}
    parent(x) !== x && return Functors.isleaf(x)
    return T <: ReactantPrimitive
end

struct MakeTracerFn{TN,TB} <: Function
    mode::TraceMode.T
    toscalar::Bool
    tobatch::TB
end

function make_tracer(
    seen, x, path, mode; toscalar=false, tobatch=nothing, track_numbers::Type{TN}=Union{}
) where {TN}
    @assert mode != TraceMode.TracedToTypes # XXX: support TracedToTypes
    @assert mode != TraceMode.ConcreteToTraced # XXX: support ConcreteToTraced
    @assert mode != TraceMode.TracedTrack # XXX: support TracedTrack
    @assert mode != TraceMode.NoStopTracedTrack # XXX: support NoStopTracedTrack
    @assert mode != TraceMode.TracedSetPath # XXX: support TracedSetPath
    @assert mode != TraceMode.TracedToConcrete # XXX: support TracedToConcrete

    res = Functors.fmap_with_path(
        MakeTracerFn{TN,typeof(tobatch)}(mode, toscalar, tobatch),
        x;
        cache=seen,
        walk=DefaultWalkWithInitialPath(path),
        exclude=isleaf,
    )
    @show seen
    return res
end

function (f::MakeTracerFn{TN,TB})(
    path::Functors.KeyPath, x::ConcreteRArray{T,N}
) where {T,N,TN,TB}
    return error(1)
end

# function make_tracer(
#     seen, @nospecialize(prev::ConcreteRArray{T,N}), @nospecialize(path), mode; kwargs...
# ) where {T,N}
#     if mode == TracedToTypes
#         throw("Cannot have ConcreteRArray as function call argument.")
#     end
#     if mode == ArrayToConcrete
#         return prev
#     end
#     if mode != ConcreteToTraced
#         throw("Cannot trace concrete")
#     end
#     if haskey(seen, prev)
#         return seen[prev]::TracedRArray{T,N}
#     end
#     @assert N isa Int
#     res = TracedRArray{T,N}((path,), nothing, size(prev))
#     seen[prev] = res
#     return res
# end

# function make_tracer(
#     seen, prev::ConcreteRNumber{T}, @nospecialize(path), mode; kwargs...
# ) where {T}
#     if mode == TracedToTypes
#         throw("Cannot have ConcreteRNumber as function call argument.")
#     end
#     if mode == ArrayToConcrete
#         return prev
#     end
#     if mode != ConcreteToTraced
#         throw("Cannot trace existing trace type")
#     end
#     if haskey(seen, prev)
#         return seen[prev]::TracedRNumber{T}
#     end
#     res = TracedRNumber{T}((path,), nothing)
#     seen[prev] = res
#     return res
# end

# function make_tracer(
#     seen,
#     @nospecialize(prev::TracedRArray{T,N}),
#     @nospecialize(path),
#     mode;
#     toscalar=false,
#     tobatch=nothing,
#     kwargs...,
# ) where {T,N}
#     if mode == ConcreteToTraced
#         throw("Cannot trace existing trace type")
#     end
#     if mode == TracedToTypes
#         push!(path, MLIR.IR.type(prev.mlir_data))
#         return nothing
#     end
#     if mode == TracedTrack
#         TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             return seen[prev] = prev
#         end
#         return prev
#     end
#     if mode == NoStopTracedTrack
#         TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             seen[prev] = prev # don't return!
#         end
#         return prev
#     end
#     if mode == TracedSetPath
#         if haskey(seen, prev)
#             return seen[prev]
#         end
#         res = if toscalar
#             TracedRNumber{T}((path,), nothing)
#         elseif tobatch !== nothing
#             error("This should not happen...")
#         else
#             TracedRArray{T,N}((path,), prev.mlir_data, size(prev))
#         end
#         seen[prev] = res
#         return res
#     end

#     if mode == TracedToConcrete
#         if haskey(seen, prev)
#             return seen[prev]::ConcreteRArray{T,N}
#         end
#         res = ConcreteRArray{T,N}(XLA.AsyncEmptyBuffer, size(prev))
#         seen[prev] = res
#         return res
#     end

#     throw("Cannot Unknown trace mode $mode")
# end

# function make_tracer(
#     seen,
#     @nospecialize(prev::TracedRNumber{T}),
#     @nospecialize(path),
#     mode;
#     tobatch=nothing,
#     toscalar=false,
#     kwargs...,
# ) where {T}
#     if mode == ConcreteToTraced
#         throw("Cannot trace existing trace type")
#     end
#     if mode == TracedToTypes
#         push!(path, MLIR.IR.type(prev.mlir_data))
#         return nothing
#     end
#     if mode == TracedTrack
#         TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             return seen[prev] = prev
#         end
#         return prev
#     end
#     if mode == NoStopTracedTrack
#         TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             seen[prev] = prev # don't return!
#         end
#         return prev
#     end
#     if mode == TracedSetPath
#         if haskey(seen, prev)
#             return seen[prev]
#         end
#         res = if toscalar
#             TracedRNumber{T}((path,), nothing)
#         elseif tobatch !== nothing
#             TracedRArray{T,length(tobatch)}((path,), prev.mlir_data, tobatch)
#         else
#             TracedRNumber{T}((path,), prev.mlir_data)
#         end
#         seen[prev] = res
#         return res
#     end

#     if mode == TracedToConcrete
#         if haskey(seen, prev)
#             return seen[prev]::ConcreteRNumber{T}
#         end
#         res = ConcreteRNumber{T}(XLA.AsyncEmptyBuffer)
#         seen[prev] = res
#         return res
#     end

#     throw("Cannot Unknown trace mode $mode")
# end

# function make_tracer(
#     seen, @nospecialize(prev::MissingTracedValue), @nospecialize(path), mode; kwargs...
# )
#     if mode == ConcreteToTraced
#         throw("Cannot trace existing trace type")
#     end
#     if mode == TracedToTypes
#         throw("Cannot have MissingTracedValue as function call argument.")
#     end
#     if mode == TracedTrack
#         TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             return seen[prev] = prev
#         end
#         return prev
#     end
#     if mode == NoStopTracedTrack
#         TracedUtils.set_paths!(prev, (TracedUtils.get_paths(prev)..., path))
#         if !haskey(seen, prev)
#             seen[prev] = prev # don't return!
#         end
#         return prev
#     end
#     if mode == TracedSetPath
#         haskey(seen, prev) && return seen[prev]
#         res = MissingTracedValue((path,))
#         seen[res] = res
#         return res
#     end
#     if mode == TracedToConcrete
#         error("Cannot convert MissingTracedValue to Concrete. This is meant to be an \
#                internal implementation detail not exposed to the user.")
#     end
#     throw("Cannot Unknown trace mode $mode")
# end

# function make_tracer(
#     seen,
#     @nospecialize(prev::Number),
#     @nospecialize(path),
#     mode;
#     @nospecialize(track_numbers::Type = Union{}),
#     kwargs...,
# )
#     if mode == TracedToTypes
#         push!(path, prev)
#         return nothing
#     end
#     RT = Core.Typeof(prev)
#     if RT <: track_numbers
#         if mode == ArrayToConcrete
#             return ConcreteRNumber(prev)
#         else
#             if mode == TracedTrack || mode == NoStopTracedTrack
#                 res = TracedRNumber{RT}(
#                     (path,), TracedUtils.broadcast_to_size(prev, ()).mlir_data
#                 )
#                 if !haskey(seen, prev)
#                     return seen[prev] = res
#                 end
#                 return res
#             elseif mode == TracedSetPath
#                 haskey(seen, prev) && return seen[prev]
#                 res = TracedRNumber{RT}(
#                     (path,), TracedUtils.broadcast_to_size(prev, ()).mlir_data
#                 )
#                 seen[prev] = res
#                 return res
#             elseif mode == TracedToConcrete
#                 throw("Input is not a traced-type: $(RT)")
#             end
#         end
#     end
#     return prev
# end

function (f::MakeTracerFn{TN,TB})(path::Functors.KeyPath, x::Array{T,N}) where {T,N,TN,TB}
    if T <: ReactantPrimitive
        if f.mode == TraceMode.ArrayToConcrete
            return ConcreteRArray(x)
        end
    end
    error("TODO")
end

# function make_tracer(
#     seen,
#     @nospecialize(prev::Array),
#     @nospecialize(path),
#     mode;
#     track_numbers::Type=Union{},
#     kwargs...,
# )
#     RT = Core.Typeof(prev)
#     if mode != NoStopTracedTrack && haskey(seen, prev)
#         if mode == TracedToTypes
#             visited = seen[prev]
#             push!(path, visited)
#             return nothing
#         end
#         return seen[prev]
#     end
#     if eltype(RT) <: ReactantPrimitive
#         if mode == ArrayToConcrete && return seen[prev] = ConcreteRArray(prev)
#         elseif mode == TracedToTypes
#             # Original array can get mutated so we store a copy:
#             push!(path, copy(prev))
#             seen[prev] = VisitedObject(length(seen) + 1)
#             return nothing
#         end
#     elseif mode == TracedToTypes
#         push!(path, RT)
#         for I in eachindex(prev)
#             if isassigned(prev, I)
#                 pv = prev[I]
#                 make_tracer(seen, pv, path, mode; track_numbers, kwargs...)
#             end
#         end
#         return nothing
#     end
#     TT = traced_type(eltype(RT), Val(mode), track_numbers)
#     newa = Array{TT,ndims(RT)}(undef, size(prev))
#     seen[prev] = newa
#     same = true
#     for I in eachindex(prev)
#         if isassigned(prev, I)
#             pv = prev[I]
#             nv = make_tracer(seen, pv, append_path(path, I), mode; track_numbers, kwargs...)
#             if pv !== nv
#                 same = false
#             end
#             @inbounds newa[I] = nv
#         end
#     end
#     if same
#         seen[prev] = prev
#         return prev
#     end
#     return newa
# end

"""
    to_rarray(x; track_numbers::Union{Bool,Type}=false)

Recursively convert leaves of `x` to `TracedRArray`s.
"""
@inline function to_rarray(@nospecialize(x); track_numbers::Union{Bool,Type}=false)
    track_numbers isa Bool && (track_numbers = track_numbers ? Number : Union{})
    return make_tracer(OrderedIdDict(), x, (), TraceMode.ArrayToConcrete; track_numbers)
end
