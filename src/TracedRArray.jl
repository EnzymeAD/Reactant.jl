module TracedRArrayOverrides

using Adapt: WrappedArray
using Base.Broadcast
using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    AnyTracedRArray,
    AnyTracedRVector,
    Ops,
    MLIR,
    ancestor,
    allowscalar,
    aos_to_soa,
    unwrapped_eltype
using ..TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!, materialize_traced_array

using ReactantCore: ReactantCore
using GPUArraysCore: GPUArraysCore, @allowscalar

ReactantCore.is_traced(::TracedRArray, seen) = true
ReactantCore.is_traced(::TracedRArray) = true

Base.strides(x::TracedRArray) = Base.size_to_strides(1, size(x)...)

Base.IndexStyle(::Type{<:TracedRArray}) = Base.IndexLinear()

# This is required otherwise we will copy a tracedrarray each time
# we use it
function Base.convert(::Type{TracedRArray}, x::TracedRArray)
    return x
end

function Base.convert(::Type{TracedRArray}, x::AnyTracedRArray)
    return Base.convert(TracedRArray{unwrapped_eltype(x),ndims(x)}, x)
end

function Base.convert(::Type{TracedRArray}, x::AbstractArray)
    return Base.convert(TracedRArray{eltype(x),ndims(x)}, x)
end

function Base.convert(::Type{TracedRArray{T,N}}, x::AbstractArray) where {T,N}
    @assert ndims(x) == N
    if x isa TracedRArray
        eltype(x) == T && return x
        return Ops.convert(TracedRArray{T,N}, x)
    end
    if eltype(x) <: TracedRNumber
        return convert(TracedRArray{T,N}, aos_to_soa(materialize_traced_array(x)))
    end
    return convert(TracedRArray{T,N}, Ops.constant(collect(x)))
end

TracedRArray{T,N}(x::AbstractArray) where {T,N} = convert(TracedRArray{T,N}, x)

function Base.getindex(
    a::TracedRArray{T,N}, index::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    GPUArraysCore.assertscalar("getindex(::TracedRArray, ::Vararg{Int, N})")
    res = Ops.reshape(Ops.dynamic_slice(a, [index...], ones(Int32, N)), Int[])
    return TracedRNumber{T}((), res.mlir_data)
end

Base.getindex(a::TracedRArray{T,0}) where {T} = TracedRNumber{T}((), a.mlir_data)
function Base.getindex(a::TracedRArray{T,0}, ::CartesianIndex{0}) where {T}
    return TracedRNumber{T}((), a.mlir_data)
end

function generate_index_list(i1, is...)
    list = reshape(i1, :, 1)
    for i in is
        i = TracedUtils.broadcast_to_size(i, (length(i), 1))
        lorig = size(list, 1)
        list = repeat(list, size(i, 1), 1)
        i = repeat(i; inner=(lorig, 1))
        list = hcat(list, i)
    end
    return list
end

function scalar_index_to_cartesian(
    idx::AbstractVector{TracedRNumber{T}}, sz::NTuple{N,Int}
) where {T,N}
    idx = materialize_traced_array(idx)
    idx = Ops.subtract(idx, Ops.fill(T(1), size(idx)))
    idxs = materialize_traced_array(
        reshape(Ops.remainder(idx, Ops.fill(T(sz[1]), size(idx))), :, 1)
    )
    idx = Ops.divide(idx, Ops.fill(T(sz[1]), size(idx)))
    for i in 2:N
        idxs = hcat(idxs, Ops.remainder(idx, Ops.fill(T(sz[i]), size(idx))))
        idx = Ops.divide(idx, Ops.fill(T(sz[i]), size(idx)))
    end
    return Ops.add(idxs, Ops.fill(T(1), size(idxs)))
end

function scalar_index_to_cartesian(idx::T, sz::NTuple{N,Int}) where {T<:Number,N}
    idx = idx - 1
    idxs = (idx % T(sz[1]),)
    idx = idx ÷ T(sz[1])
    for i in 2:N
        idxs = (idxs..., idx % T(sz[i]))
        idx = idx ÷ T(sz[i])
    end
    return idxs .+ 1
end

function Base.getindex(
    a::TracedRArray{T,N}, index::Union{Int,TracedRNumber{Int}}
) where {T,N}
    GPUArraysCore.assertscalar("getindex(::TracedRArray, ::Union{Int, TracedRNumber{Int}})")
    return TracedRNumber{T}(
        (),
        Ops.reshape(
            Ops.dynamic_slice(
                a, collect(scalar_index_to_cartesian(index, size(a))), ones(Int32, N)
            ),
            Int64[],
        ).mlir_data,
    )
end

function _getindex_cartesian(a::TracedRArray{T,N}, index::CartesianIndex{N}) where {T,N}
    GPUArraysCore.assertscalar("getindex(::TracedRArray, ::CartesianIndex{N})")
    return TracedRNumber{T}(
        (),
        Ops.reshape(
            Ops.dynamic_slice(a, collect(Int64, index.I), ones(Int32, N)), Int64[]
        ).mlir_data,
    )
end

function Base.getindex(a::TracedRArray{T,N}, index::CartesianIndex{N}) where {T,N}
    return _getindex_cartesian(a, index)
end
function Base.getindex(a::TracedRArray{T,1}, indices::CartesianIndex{1}) where {T}
    return _getindex_cartesian(a, indices)
end

function _getindex_linear(a::TracedRArray{T,N}, indices::AbstractArray) where {T,N}
    if !(indices isa Reactant.TracedType) && __contiguous_indices(vec(indices))
        a_flat = materialize_traced_array(vec(a))
        indices_flat = vec(indices)
        return Ops.reshape(
            Ops.dynamic_slice(a_flat, [first(indices_flat)], [length(indices_flat)]),
            collect(size(indices)),
        )
    end

    if !(indices isa TracedRArray)
        indices = collect(indices)
        eltype(indices) <: CartesianIndex && (indices = LinearIndices(size(a))[indices])
        indices = TracedUtils.promote_to(TracedRArray{Int,ndims(indices)}, indices)
    end
    return Ops.reshape(
        Ops.gather_getindex(a, scalar_index_to_cartesian(vec(indices), size(a))),
        collect(size(indices)),
    )
end

function Base.getindex(a::TracedRArray{T,N}, indices::AbstractArray) where {T,N}
    return _getindex_linear(a, indices)
end
function Base.getindex(a::TracedRArray{T,1}, indices::AbstractArray) where {T}
    return _getindex_linear(a, indices)
end

Base.getindex(a::TracedRArray{T,N}, ::Colon) where {T,N} = materialize_traced_array(vec(a))

__contiguous_indices(::Base.LogicalIndex) = false
__contiguous_indices(x) = all(isone, diff(x))

function Base.getindex(a::TracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    indices = Base.to_indices(a, indices)

    use_gather_getindex = false
    for idxs in indices
        idxs isa Number && continue
        if idxs isa Reactant.TracedType
            use_gather_getindex = true
            break
        end
        contiguous = __contiguous_indices(vec(idxs))
        if typeof(contiguous) <: Bool && !contiguous
            use_gather_getindex = true
            break
        end
    end

    if use_gather_getindex
        # TODO: This will create a dynamically sized tensor and we need to implement
        #       `findall` for it.
        if any(i -> unwrapped_eltype(i) <: Bool, indices)
            error("Boolean indexing with TracedRArrays isn't fully supported yet.")
        end

        indices, integer_indices, result_size, preddim_result_size, _ = TracedUtils.traced_indices(
            indices...
        )
        res = Ops.reshape(
            Ops.gather_getindex(a, generate_index_list(indices...)), preddim_result_size
        )
        isempty(integer_indices) ||
            (res = materialize_traced_array(dropdims(res; dims=integer_indices)))
        return Ops.reshape(res, result_size)
    end

    x = Ops.dynamic_slice(a, [first.(indices)...], [length.(indices)...])
    ddims = findall(indices) do idx
        return idx isa Integer || idx isa TracedRNumber{<:Integer}
    end
    isempty(ddims) || return materialize_traced_array(dropdims(x; dims=Tuple(ddims)))
    return x
end

# Prevent ambiguity
# We only do it for specific arrays to avoid going down this path for most arrays
function Base.getindex(
    a::WrappedArray{TracedRNumber{T}}, index::Union{Int,TracedRNumber{Int}}...
) where {T}
    return getindex(ancestor(a), TracedUtils.get_ancestor_indices(a, index...)...)
end

function Base.getindex(a::WrappedArray{TracedRNumber{T}}, indices) where {T}
    return getindex(ancestor(a), TracedUtils.get_ancestor_indices(a, indices))
end

function Base.getindex(a::WrappedArray{TracedRNumber{T},1}, indices) where {T}
    return getindex(ancestor(a), TracedUtils.get_ancestor_indices(a, indices))
end
function Base.getindex(
    a::WrappedArray{TracedRNumber{T},N}, indices::Vararg{Any,N}
) where {T,N}
    return getindex(ancestor(a), TracedUtils.get_ancestor_indices(a, indices...)...)
end

## Specialize certain dispatches for better codegen
for aType in (
    Base.ReshapedArray{TracedRNumber{T}} where {T},
    PermutedDimsArray{TracedRNumber{T}} where {T},
)
    @eval begin
        function Base.getindex(a::$aType, indices::Union{Int,TracedRNumber{Int}}...)
            return getindex(materialize_traced_array(a), indices...)
        end

        function Base.getindex(a::$aType, indices...)
            return getindex(materialize_traced_array(a), indices...)
        end
    end
end

function maybe_assert_scalar_setindexing(
    ::TracedRArray{T,N}, ::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    GPUArraysCore.assertscalar("setindex!(::TracedRArray, v, ::Vararg{Int, N})")
    return nothing
end

maybe_assert_scalar_setindexing(args...) = nothing

function _setindex_scalar!(
    a::TracedRArray{T,N}, v, index::Union{Int,TracedRNumber{Int}}
) where {T,N}
    GPUArraysCore.assertscalar(
        "setindex!(::TracedRArray, v, ::Union{Int, TracedRNumber{Int}})"
    )

    res = Ops.reshape(
        Ops.dynamic_update_slice(
            a,
            TracedUtils.broadcast_to_size(
                TracedUtils.promote_to(TracedRNumber{T}, v), ntuple(Returns(1), N)
            ),
            collect(scalar_index_to_cartesian(index, size(a))),
        ),
        collect(size(a)),
    )
    set_mlir_data!(a, get_mlir_data(res))
    return a
end

function Base.setindex!(
    a::TracedRArray{T,N}, v, index::Union{Int,TracedRNumber{Int}}
) where {T,N}
    return _setindex_scalar!(a, v, index)
end
function Base.setindex!(
    a::TracedRArray{T,1}, v, index::Union{Int,TracedRNumber{Int}}
) where {T}
    return _setindex_scalar!(a, v, index)
end

function Base.setindex!(a::TracedRArray{T,N}, v, index::CartesianIndex{N}) where {T,N}
    GPUArraysCore.assertscalar("setindex!(::TracedRArray, v, ::CartesianIndex{N})")

    res = Ops.reshape(
        Ops.dynamic_update_slice(
            a,
            TracedUtils.broadcast_to_size(T(v), ntuple(Returns(1), N)),
            collect(Int64, index.I),
        ),
        collect(size(a)),
    )
    set_mlir_data!(a, get_mlir_data(res))
    return a
end

function _setindex_linear!(a::TracedRArray{T,N}, v, indices::AbstractArray) where {T,N}
    if !(indices isa Reactant.TracedType) && __contiguous_indices(vec(indices))
        res = Ops.reshape(
            Ops.dynamic_update_slice(
                materialize_traced_array(vec(a)),
                TracedUtils.broadcast_to_size(
                    TracedUtils.promote_to(TracedRArray{T,1}, vec(v)), (length(indices),)
                ),
                [first(indices)],
            ),
            collect(size(a)),
        )
        set_mlir_data!(a, get_mlir_data(res))
        return a
    end

    if !(indices isa TracedRArray)
        indices = collect(indices)
        eltype(indices) <: CartesianIndex && (indices = LinearIndices(size(a))[indices])
        indices = TracedUtils.promote_to(TracedRArray{Int,ndims(indices)}, indices)
    end
    res = Ops.scatter_setindex(
        a,
        scalar_index_to_cartesian(vec(indices), size(a)),
        materialize_traced_array(vec(v)),
    )
    set_mlir_data!(a, get_mlir_data(res))
    return a
end

function Base.setindex!(a::TracedRArray{T,N}, v, indices::AbstractArray) where {T,N}
    return _setindex_linear!(a, v, indices)
end
function Base.setindex!(a::TracedRArray{T,1}, v, indices::AbstractArray) where {T,N}
    return _setindex_linear!(a, v, indices)
end

function Base.setindex!(a::TracedRArray{T,N}, v, indices::Vararg{Any,N}) where {T,N}
    if (N == 1) && (indices isa Colon)
        # Remove ambiguity from the previous
        # ```julia
        # Base.setindex!(a::TracedRArray{T,N}, v, ::Colon) where {T,N}
        # ```
        # signature, which would be confused with this one for N=1.
        v = TracedUtils.broadcast_to_size(v, size(a))
        set_mlir_data!(a, get_mlir_data(v))
        return a
    end
    maybe_assert_scalar_setindexing(a, indices...)

    indices = Base.to_indices(a, indices)

    use_scatter_setindex = false
    for idxs in indices
        idxs isa Number && continue
        if idxs isa Reactant.TracedType
            use_scatter_setindex = true
            break
        end
        contiguous = __contiguous_indices(idxs)
        if typeof(contiguous) <: Bool && !contiguous
            use_scatter_setindex = true
            break
        end
    end

    if use_scatter_setindex
        # TODO: This will create a dynamically sized tensor and we need to implement
        #       `findall` for it.
        if any(i -> unwrapped_eltype(i) <: Bool, indices)
            error("Boolean indexing with TracedRArrays isn't fully supported yet.")
        end
        indices_list = map(Base.Fix1(TracedUtils.promote_to, TracedRArray{Int,1}), indices)
        indices_list = generate_index_list(indices_list...)
        res = Ops.scatter_setindex(a, indices_list, Ops.reshape(v, length(v)))
        set_mlir_data!(a, get_mlir_data(res))
        return v
    end

    if v isa Number
        v = TracedUtils.broadcast_to_size(v, length.(indices))
        v = TracedUtils.promote_to(TracedRArray{T,N}, v)
    else
        v = TracedUtils.promote_to(TracedRArray{T,ndims(v)}, v)
        non_integer_indices = [
            !(idx isa Union{Integer,TracedRNumber{<:Integer}}) for idx in indices
        ]
        broadcast_dims = findall(non_integer_indices)
        if length(broadcast_dims) == N
            v = TracedUtils.broadcast_to_size(v, length.(indices))
        else
            v = Ops.broadcast_in_dim(
                materialize_traced_array(v),
                broadcast_dims,
                collect(Int64, length.(indices)),
            )
        end
    end

    set_mlir_data!(
        a,
        Ops.dynamic_update_slice(
            a, v, [i isa Colon ? 1 : first(i) for i in indices]
        ).mlir_data,
    )
    return v
end

Base.Tuple(x::TracedRArray) = ntuple(Base.Fix1(Base.getindex, x), length(x))

Base.size(x::TracedRArray) = x.shape

Base.collect(x::TracedRArray) = copy(x) # XXX: Is this correct?

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), A.mlir_data, size(A))

function Base.similar(::TracedRArray, ::Type{T}, dims::Dims{N}) where {T,N}
    return Ops.fill(zero(unwrapped_eltype(T)), dims)
end

function Base.show(io::IOty, X::TracedRArray{T,N}) where {T,N,IOty<:Union{IO,IOContext}}
    return print(io, "TracedRArray{", T, ",", N, "N}(", X.paths, ", size=", size(X), ")")
    # TODO this line segfaults if MLIR IR has not correctly been generated
    # return print(io, X.mlir_data, ")")
end

function Base.permutedims(A::AnyTracedRArray{T,N}, perm) where {T,N}
    return Ops.transpose(materialize_traced_array(A), Int64[perm...])
end

Base.conj(A::AnyTracedRArray) = A
Base.conj(A::AnyTracedRArray{<:Complex}) = Ops.conj(materialize_traced_array(A))

Base.conj!(A::AnyTracedRArray) = A

function Base.conj!(A::AnyTracedRArray{<:Complex})
    TracedUtils.set_mlir_data!(A, Ops.conj(materialize_traced_array(A)).mlir_data)
    return A
end

Base.real(A::AnyTracedRArray) = A
Base.real(A::AnyTracedRArray{<:Complex}) = Ops.real(materialize_traced_array(A))

Base.imag(A::AnyTracedRArray) = zero(A)
Base.imag(A::AnyTracedRArray{<:Complex}) = Ops.imag(materialize_traced_array(A))

TracedUtils.promote_to(::Type{TracedRArray{T,N}}, rhs) where {T,N} = TracedRArray{T,N}(rhs)
function TracedUtils.promote_to(::TracedRArray{T,N}, rhs) where {T,N}
    return TracedUtils.promote_to(TracedRArray{T,N}, rhs)
end
function TracedUtils.promote_to(
    ::Type{TracedRArray{T,0}}, rhs::TracedRNumber{T2}
) where {T,T2}
    return TracedRArray{T,0}((), Ops.convert(TracedRNumber{T}, rhs).mlir_data, ())
end

for (jlop, hloop, hlocomp, merge) in
    ((:(Base.:(==)), :compare, "EQ", :all), (:(Base.:(!=)), :compare, "NE", :any))
    @eval function $jlop(
        @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs::TracedRArray{T,N})
    ) where {T,N}
        elems = $(jlop).(lhs, rhs)
        return N == 0 ? elems : $(merge)(elems)
    end
end

function overloaded_mapreduce(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(A::AnyTracedRArray{T,N});
    dims=:,
    init=nothing,
) where {T,N}
    A = materialize_traced_array(A)

    if dims isa Int
        dims = [dims]
    end

    op_in_T = Core.Compiler.return_type(f, Tuple{T})

    if init === nothing
        if op === min
            init = typemax(op_in_T)
        elseif op === max
            init = typemin(op_in_T)
        else
            init = Base.reduce_empty(Base.BottomRF(op), op_in_T)
        end

        if typeof(init) != op_in_T
            op_in_T = typeof(init)
            A = typeof(init).(A)
        end
    end

    init = [TracedUtils.broadcast_to_size(init, ()).mlir_data]

    inp = [broadcast(f, A).mlir_data]

    rdims = Int64[]

    if dims == (:)
        for i in 0:(N - 1)
            push!(rdims, i)
        end
    else
        for i in dims
            push!(rdims, i - 1)
        end
    end

    in_tys = [
        MLIR.IR.TensorType(Int64[], eltype(MLIR.IR.type(inp[1]))),
        MLIR.IR.TensorType(Int64[], eltype(MLIR.IR.type(init[1]))),
    ]

    fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location(), MLIR.IR.Location()])

    args = (
        TracedRNumber{Reactant.unwrapped_eltype(op_in_T)}((), MLIR.IR.argument(fnbody, 1)),
        TracedRNumber{Reactant.unwrapped_eltype(op_in_T)}((), MLIR.IR.argument(fnbody, 2)),
    )

    resty = MLIR.IR.block!(fnbody) do
        tmp = TracedUtils.broadcast_to_size(op(args...), ())
        Ops.return_(tmp)
        return eltype(MLIR.IR.type(tmp.mlir_data))
    end

    toonedims = Int[]
    outdims = Int[]
    for i in 1:N
        tmp = if in(i - 1, rdims)
            1
        else
            sz = size(A, i)
            push!(outdims, sz)
            sz
        end
        push!(toonedims, tmp)
    end

    TT = MLIR.IR.Type[MLIR.IR.TensorType(outdims, resty)]

    body = MLIR.IR.Region()
    push!(body, fnbody)
    red = MLIR.Dialects.stablehlo.reduce(
        inp, init; result_0=TT, dimensions=MLIR.IR.DenseArrayAttribute(rdims), body
    )

    red = MLIR.IR.result(red, 1)
    redT = eltype(MLIR.IR.julia_type(MLIR.IR.type(red)))

    if dims != (:)
        red = Ops.reshape(TracedRArray(red), toonedims...)
    else
        if length(outdims) == 0
            red = TracedRNumber{redT}((), red)
        else
            red = TracedRArray{redT,length(outdims)}((), red, (outdims...,))
        end
    end
    return red
end

function Base.mapreducedim!(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(R::AnyTracedRArray{T,N}),
    A::Base.AbstractArrayOrBroadcasted,
) where {T,N}
    @assert length(size(R)) == length(size(A))
    dims = map(enumerate(zip(size(R), size(A)))) do (i, (sR, sA))
        sR == sA && return nothing
        @assert sR == 1
        return i
    end
    tmp = mapreduce(f, op, A; dims=filter(!isnothing, dims))
    R .= op.(R, tmp) # match native Julia's behavior
    return R
end

function Base.fill!(A::AnyTracedRArray{T,N}, x) where {T,N}
    bcast = TracedUtils.broadcast_to_size(T(x), size(A))
    set_mlir_data!(A, get_mlir_data(bcast))
    return A
end

function Base.fill!(A::AnyTracedRArray{T,N}, x::TracedRNumber{T2}) where {T,N,T2}
    bcast = TracedUtils.broadcast_to_size(
        TracedUtils.promote_to(TracedRNumber{T}, x), size(A)
    )
    set_mlir_data!(A, get_mlir_data(bcast))
    return A
end

struct AbstractReactantArrayStyle{N} <: AbstractArrayStyle{N} end

AbstractReactantArrayStyle(::Val{N}) where {N} = AbstractReactantArrayStyle{N}()
AbstractReactantArrayStyle{M}(::Val{N}) where {N,M} = AbstractReactantArrayStyle{N}()

function BroadcastStyle(::Type{<:AnyTracedRArray{T,N}}) where {T,N}
    return AbstractReactantArrayStyle{N}()
end

function Base.similar(
    ::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{T}, dims
) where {T<:Reactant.ReactantPrimitive,N}
    @assert N isa Int
    return TracedRArray{T,length(dims)}((), nothing, map(length, dims))
end

function Base.similar(
    ::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{TracedRNumber{T}}, dims
) where {T<:Reactant.ReactantPrimitive,N}
    @assert N isa Int
    return TracedRArray{T,length(dims)}((), nothing, map(length, dims))
end

function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    dest = copyto!(similar(bc, ElType), bc)
    return dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

Base.eltype(::Broadcast.Extruded{T}) where {T} = eltype(T)

function first_scalar(x)
    Reactant.@allowscalar first(x)
end

# we need to override the outer copy method to make sure we never fall back to scalar
# iteration (see, e.g., CUDA.jl#145)
function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle})
    fn = if bc.f isa Type && bc.f <: Reactant.ReactantPrimitive
        TracedUtils.TypeCast{bc.f}()
    else
        bc.f
    end
    ElType = Broadcast.combine_eltypes(fn, bc.args)
    # Special case a union{} return so we can see the better error message
    if ElType === Union{}
        fn(map(first_scalar, bc.args)...)
    end
    @assert ElType != Any && ElType != Union{}
    sim = similar(bc, ElType)
    return copyto!(sim, bc)
end

function Base.materialize!(
    ::Style, dest, bc::Broadcasted
) where {Style<:AbstractReactantArrayStyle}
    return _copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end

Base.copyto!(dest::AnyTracedRArray, bc::Broadcasted{Nothing}) = _copyto!(dest, bc) # Keep it for ArrayConflict

function Base.copyto!(dest::TracedRArray{T,N}, src::TracedRArray{T,N}) where {T,N}
    dest.mlir_data = src.mlir_data
    return dest
end
function Base.copyto!(
    dest::Reactant.TracedRArray{T},
    dstart::Integer,
    src::Reactant.TracedRArray{T},
    sstart::Integer,
    n::Integer,
) where {T}
    setindex!(dest, src[sstart:(sstart + n - 1)], dstart:(dstart + n - 1))
    return dest
end

function Base.copyto!(dest::TracedRArray{T,N}, src::TracedRArray{T2,N}) where {T,T2,N}
    return copyto!(dest, Ops.convert(TracedRArray{T,N}, src))
end

function Base.copyto!(dest::TracedRArray{T,N}, src::Array{T2,N}) where {T,T2,N}
    return copyto!(dest, TracedUtils.promote_to(TracedRArray{T2,N}, src))
end

function _copyto!(dest::AnyTracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (TracedUtils.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = TracedUtils.promote_to(
        TracedRArray{unwrapped_eltype(dest),ndims(dest)},
        TracedUtils.elem_apply(bc.f, args...),
    )
    TracedUtils.set_mlir_data!(dest, res.mlir_data)
    return dest
end

function _copyto!(dest::Array{<:TracedRNumber}, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (TracedUtils.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = TracedUtils.elem_apply(bc.f, args...)
    for I in 1:length(dest)
        dest[I] = Reactant.@allowscalar res[I]
    end
    return dest
end

dispatch_val(x) = x
dispatch_val(::Val{D}) where {D} = D

@inline function Base._typed_vcat(
    ::Type{T}, X::Base.AbstractVecOrTuple{<:TracedRArray}
) where {T}
    return Base._cat_t(Val(1), T, X...)
end

@inline function Base._typed_hcat(
    ::Type{T}, X::Base.AbstractVecOrTuple{<:TracedRArray}
) where {T}
    return Base._cat_t(Val(2), T, X...)
end

# `Base.typed_hvcat` is overloaded for `AbstractVecOrMat` using `setindex!` that breaks Reactant
# generic implementation uses `typed_hcat` and `typed_vcat` which is alright
@inline function Base.typed_hvcat(
    ::Type{T}, rows::Tuple{Vararg{Int}}, as::TracedRArray...
) where {T}
    return invoke(
        Base.typed_hvcat, Tuple{Type{T},Tuple{Vararg{Int}},Vararg{Any}}, T, rows, as...
    )
end

function Base._typed_hvncat(
    T::Type, dims::NTuple{N,Int}, row_first::Bool, as::TracedRArray...
) where {N}
    As = if row_first
        perm = [2, 1, 3:N...]
        dims = [dims[2], dims[1], dims[3:end]...]
        permutedims(reshape(collect(as), dims...), perm)
    else
        reshape(collect(as), dims)
    end

    for d in 1:N
        Bs = Array{Any,N - d}(undef, size(As)[2:end]...)

        for (i, col) in
            zip(eachindex(Bs), eachslice(As; dims=Tuple(2:ndims(As)), drop=true))
            # TODO row_first affects the flattening?
            Bs[i] = Base._cat_t(d, T, col...)
        end

        As = Bs
    end

    return only(As)
end

function maybe_expand_dims(x::AbstractArray{T,N}, dims) where {T,N}
    dims = dispatch_val(dims)
    dims ≤ N && return x
    return reshape(x, ntuple(i -> i ≤ N ? size(x, i) : 1, dims))
end

function Base._cat_t(dims, ::Type{T}, X::TracedRArray...) where {T}
    dims = dispatch_val(dims)
    @assert dims isa Integer "Support for non-integer dimensions is not implemented yet."

    # MLIR expects the dimension `dims` to be ≤ the rank of the input tensors
    X = maybe_expand_dims.(X, (dims,))

    catdims = Base.dims2cat(dims)
    shape = Base.cat_size_shape(catdims, X...)
    RT = unwrapped_eltype(Base.promote_eltype(T, X...))

    # convert to the target eltype
    X = map(Base.Fix1(TracedUtils.promote_to, TracedRArray{RT,length(shape)}), X)

    return TracedRArray{RT,length(shape)}(
        (),
        MLIR.IR.result(
            # TODO maybe we should do some conversion?
            MLIR.Dialects.stablehlo.concatenate(
                collect(TracedUtils.get_mlir_data.(X));
                result_0=MLIR.IR.TensorType(collect(Int, shape), MLIR.IR.Type(RT)),
                dimension=dims - 1, # stablehlo expects this to be zero-indexed
            ),
            1,
        ),
        shape,
    )
end

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval function Base.clamp!(x::AnyTracedRArray, min::$(minT), max::$(maxT))
        y = Ops.clamp(min, materialize_traced_array(x), max)
        TracedUtils.set_mlir_data!(x, y.mlir_data)
        return x
    end
end

overloaded_all(f, x::AnyTracedRArray, dims) = mapreduce(f, &, x; dims)
overloaded_any(f, x::AnyTracedRArray, dims) = mapreduce(f, |, x; dims)

# outer repeat
function Base._RepeatInnerOuter.repeat_outer(
    x::AnyTracedRArray{T,N}, counts::NTuple{N,Any}
) where {T,N}
    # (d1, d2, ..., dP) -> (d1, 1, d2, 1, ..., dP, 1)
    interleaved_size = ones(Int, 2N)
    interleaved_size[1:2:(2N)] .= size(x)

    x_interleaved = reshape(materialize_traced_array(x), interleaved_size...)

    # (d1, 1, d2, 1, ..., dP, 1) -> (d1, r1, d2, r2, ..., dP, rP)
    broadcast_target_size = interleaved_size
    broadcast_target_size[2:2:(2N)] .= counts

    x_broadcasted = TracedUtils.broadcast_to_size(x_interleaved, broadcast_target_size)

    # (d1, r1, d2, r2, ..., dP, rP) -> (d1*r1, d2*r2, ..., dP*rP)
    final_size = vec(prod(reshape(broadcast_target_size, 2, :); dims=1))

    return materialize_traced_array(reshape(x_broadcasted, final_size...))
end

# inner repeat
function Base._RepeatInnerOuter.repeat_inner(
    x::AnyTracedRArray{T,N}, counts::NTuple{M,Any}
) where {T,N,M}
    P = max(N, M) # potentially padded

    # (d1, d2, ..., dP) -> (1, d1, 1, d2, 1, ..., 1, dP)
    interleaved_size = ones(Int, 2P)
    interleaved_size[2:2:(2N)] .= size(x)

    x_interleaved = reshape(materialize_traced_array(x), interleaved_size...)

    # (1, d1, 1, d2, 1, ..., 1, dP) -> (r1, d1, r2, d2, ..., rP, dP)
    broadcast_target_size = interleaved_size
    broadcast_target_size[1:2:(2N)] .= counts

    x_broadcasted = TracedUtils.broadcast_to_size(x_interleaved, broadcast_target_size)

    # (r1, d1, r2, d2, ..., rP, dP) -> (d1*r1, d2*r2, ..., dP*rP)
    final_size = vec(prod(reshape(broadcast_target_size, 2, :); dims=1))

    return materialize_traced_array(reshape(x_broadcasted, final_size...))
end

# stack
function overloaded_stack(dims::Union{Integer,Colon}, xs)
    @assert allequal(ndims.(xs)) "All arrays must have the same number of dimensions..."
    dims = dims isa Colon ? ndims(first(xs)) + 1 : dims
    res = map(xs) do x
        new_shape = ntuple(
            i -> i == dims ? 1 : (i < dims ? size(x, i) : size(x, i - 1)), ndims(x) + 1
        )
        return materialize_traced_array(reshape(x, new_shape))
    end
    return cat(res...; dims)
end

# sort
function Base.sort(x::AnyTracedRArray; alg=missing, order=missing, kwargs...)
    return sort!(copy(x); alg, order, kwargs...)
end
function Base.sort(x::AnyTracedRVector; alg=missing, order=missing, kwargs...)
    return sort!(copy(x); alg, order, kwargs...)
end

function Base.sort!(
    x::AnyTracedRVector; lt=isless, by=identity, rev::Bool=false, alg=missing, order=missing
)
    @assert alg === missing "Reactant doesn't support `alg` kwarg for `sort!`"
    @assert order === missing "Reactant doesn't support `order` kwarg for `sort!`"

    comparator = rev ? (a, b) -> !lt(by(a), by(b)) : (a, b) -> lt(by(a), by(b))
    res = only(Ops.sort(materialize_traced_array(x); comparator, dimension=1))
    set_mlir_data!(x, get_mlir_data(res))
    return x
end

function Base.sort!(
    x::AnyTracedRArray;
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Bool=false,
    alg=missing,
    order=missing,
)
    @assert alg === missing "Reactant doesn't support `alg` kwarg for `sort!`"
    @assert order === missing "Reactant doesn't support `order` kwarg for `sort!`"

    comparator = rev ? (a, b) -> !lt(by(a), by(b)) : (a, b) -> lt(by(a), by(b))
    res = only(Ops.sort(materialize_traced_array(x); dimension=dims, comparator))
    set_mlir_data!(x, get_mlir_data(res))
    return x
end

function Base.sortperm(x::AnyTracedRArray; alg=missing, order=missing, kwargs...)
    return sortperm!(similar(x, Int), x; alg, order, kwargs...)
end
function Base.sortperm(x::AnyTracedRVector; alg=missing, order=missing, kwargs...)
    return sortperm!(similar(x, Int), x; alg, order, dims=1, kwargs...)
end

function Base.sortperm!(
    ix::AnyTracedRArray{Int,N},
    x::AnyTracedRArray{<:Any,N};
    dims::Union{Integer,Nothing}=nothing,
    lt=isless,
    by=identity,
    rev::Bool=false,
    alg=missing,
    order=missing,
) where {N}
    if dims === nothing
        @assert ndims(x) == 1
        dims = 1
    end

    @assert alg === missing "Reactant doesn't support `alg` kwarg for `sortperm!`"
    @assert order === missing "Reactant doesn't support `order` kwarg for `sortperm!`"

    comparator =
        rev ? (a, b, i1, i2) -> !lt(by(a), by(b)) : (a, b, i1, i2) -> lt(by(a), by(b))
    idxs = Ops.constant(collect(LinearIndices(x)))
    _, res = Ops.sort(materialize_traced_array(x), idxs; dimension=dims, comparator)
    set_mlir_data!(ix, get_mlir_data(res))
    return ix
end

function Base.partialsort(
    x::AnyTracedRVector, k::Union{Integer,OrdinalRange}; rev=false, kwargs...
)
    if rev
        values, _ = overloaded_partialsort_descending(x, k; kwargs...)
    else
        values, _ = overloaded_partialsort_ascending(x, k; kwargs...)
    end
    k isa Integer && return @allowscalar(values[k])
    return view(values, k)
end

function Base.partialsort!(
    x::AnyTracedRVector, k::Union{Integer,OrdinalRange}; rev=false, kwargs...
)
    if rev
        values, _ = overloaded_partialsort_descending(x, k; kwargs...)
    else
        values, _ = overloaded_partialsort_ascending(x, k; kwargs...)
    end
    val = @allowscalar(values[k])
    @allowscalar setindex!(x, val, k)
    k isa Integer && return val
    return view(x, k)
end

function Base.partialsortperm(
    x::AnyTracedRVector, k::Union{Integer,OrdinalRange}; rev=false, kwargs...
)
    if rev
        _, idxs = overloaded_partialsort_descending(x, k; kwargs...)
    else
        _, idxs = overloaded_partialsort_ascending(x, k; kwargs...)
    end
    k isa Integer && return @allowscalar(idxs[k])
    return view(idxs, k)
end

function Base.partialsortperm!(
    ix::AnyTracedRVector{Int},
    x::AnyTracedRVector,
    k::Union{Integer,OrdinalRange};
    rev=false,
    kwargs...,
)
    if rev
        _, idxs = overloaded_partialsort_descending(x, k; kwargs...)
    else
        _, idxs = overloaded_partialsort_ascending(x, k; kwargs...)
    end
    val = @allowscalar(idxs[k])
    @allowscalar setindex!(ix, val, k)
    k isa Integer && return @allowscalar(ix[k])
    return val
end

function overloaded_partialsort_descending(
    x::AnyTracedRVector{T}, k::Union{Integer,OrdinalRange}; by=identity, lt=isless
) where {T}
    if lt !== isless || by !== identity
        sorted_x, sorted_idxs = Ops.sort(
            materialize_traced_array(x),
            Ops.constant(collect(LinearIndices(x)));
            dimension=1,
            comparator=(a, b, i1, i2) -> !lt(by(a), by(b)),
        )
        return sorted_x[1:maximum(k)], sorted_idxs[1:maximum(k)]
    end

    if Reactant.LOWER_PARTIALSORT_TO_APPROX_TOP_K[] && T <: Reactant.ReactantFloat
        result = Ops.approx_top_k(
            materialize_traced_array(x),
            maximum(k);
            comparator=(a, b, i1, i2) -> a > b,
            dimension=1,
            init_val=typemin(T),
        )
        return result.values[1:maximum(k)], result.indices[1:maximum(k)]
    end

    (; values, indices) = Ops.top_k(materialize_traced_array(x), maximum(k))
    return values, indices
end

function overloaded_partialsort_ascending(
    x::AnyTracedRVector{T}, k::Union{Integer,OrdinalRange}; by=identity, lt=isless
) where {T}
    if lt !== isless || by !== identity || T <: Unsigned
        sorted_x, sorted_idxs = Ops.sort(
            materialize_traced_array(x),
            Ops.constant(collect(LinearIndices(x)));
            dimension=1,
            comparator=(a, b, i1, i2) -> !lt(by(a), by(b)),
        )
        return sorted_x[1:maximum(k)], sorted_idxs[1:maximum(k)]
    end

    if Reactant.LOWER_PARTIALSORT_TO_APPROX_TOP_K[] && T <: Reactant.ReactantFloat
        result = Ops.approx_top_k(
            materialize_traced_array(x),
            maximum(k);
            comparator=(a, b, i1, i2) -> a < b,
            dimension=1,
            init_val=typemax(T),
        )
        return result.values[1:maximum(k)], result.indices[1:maximum(k)]
    end

    (; values, indices) = Ops.top_k(Ops.negate(materialize_traced_array(x)), maximum(k))
    return Ops.negate(values), indices
end

# arg* functions
function Base.argmin(f::F, x::AnyTracedRArray) where {F}
    idx = scalar_index_to_cartesian(argmin(f.(x)), size(x))
    return @allowscalar x[idx...]
end

function Base.argmax(f::F, x::AnyTracedRArray) where {F}
    idx = scalar_index_to_cartesian(argmax(f.(x)), size(x))
    return @allowscalar x[idx...]
end

Base.argmin(x::AnyTracedRArray; kwargs...) = findmin(identity, x; kwargs...)[2]
Base.argmax(x::AnyTracedRArray; kwargs...) = findmax(identity, x; kwargs...)[2]

# find* functions
Base.findfirst(x::AnyTracedRArray) = findfirst(identity, x)
Base.findlast(x::AnyTracedRArray) = findlast(identity, x)

function Base.findfirst(f::Function, x::AnyTracedRArray)
    fA = materialize_traced_array(vec(f.(x)))
    (; indices) = Ops.top_k(fA, 1)
    return @allowscalar indices[1]
end

function Base.findlast(f::Function, x::AnyTracedRArray)
    fA = Ops.reverse(materialize_traced_array(vec(f.(x))); dimensions=[1])
    (; indices) = Ops.top_k(fA, 1)
    return length(x) - @allowscalar(indices[1]) + 1
end

Base.findmin(x::AnyTracedRVector) = findmin(identity, x; dims=1)
function Base.findmin(x::AnyTracedRArray; dims::Union{Integer,Nothing}=nothing)
    return findmin(identity, x; dims)
end

Base.findmax(x::AnyTracedRVector) = findmax(identity, x; dims=1)
function Base.findmax(x::AnyTracedRArray; dims::Union{Integer,Nothing}=nothing)
    return findmax(identity, x; dims)
end

## To avoid scalar indexing and constructing an array of tuples, we return the linear index
## instead of the cartesian index
function Base.findmin(f, x::AnyTracedRArray; dims::Union{Integer,Nothing}=nothing)
    if dims === nothing
        if ndims(x) == 1
            dims = 1
        else
            return findmin(f, vec(x); dims=1)
        end
    end

    fx = Ops.negate(materialize_traced_array(f.(x)))
    (; values, indices) = Ops.top_k(fx, 1; dimension=dims)

    # Compute linear indices
    strds = strides(x)
    iotas = [Ops.iota(Int64, [size(indices)...]; iota_dimension=i) for i in 1:ndims(x)]
    iotas[dims] = Ops.subtract(indices, Ops.fill(Int64(1), size(indices)))
    linear_indices = Ops.fill(Int64(1), size(indices))
    for d in eachindex(iotas)
        linear_indices = Ops.add(
            linear_indices,
            Ops.multiply(iotas[d], Ops.fill(Int64(strds[d]), size(iotas[d]))),
        )
    end

    values = Ops.negate(values)
    ndims(x) == 1 && return @allowscalar (values[1], linear_indices[1])
    return (values, linear_indices)
end

function Base.findmax(f, x::AnyTracedRArray; dims::Union{Integer,Nothing}=nothing)
    if dims === nothing
        if ndims(x) == 1
            dims = 1
        else
            return findmax(f, vec(x); dims=1)
        end
    end

    fx = materialize_traced_array(f.(x))
    (; values, indices) = Ops.top_k(fx, 1; dimension=dims)

    # Compute linear indices
    strds = strides(x)
    iotas = [Ops.iota(Int64, [size(indices)...]; iota_dimension=i) for i in 1:ndims(x)]
    iotas[dims] = Ops.subtract(indices, Ops.fill(Int64(1), size(indices)))
    linear_indices = Ops.fill(Int64(1), size(indices))
    for d in eachindex(iotas)
        linear_indices = Ops.add(
            linear_indices,
            Ops.multiply(iotas[d], Ops.fill(Int64(strds[d]), size(iotas[d]))),
        )
    end

    ndims(x) == 1 && return @allowscalar (values[1], linear_indices[1])
    return (values, linear_indices)
end

Base.map(f, x::AnyTracedRArray) = f.(x)

function Base.map!(f, y::AnyTracedRArray, x::AbstractArray)
    y .= f.(x)
    return y
end

end
