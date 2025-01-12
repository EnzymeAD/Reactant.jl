module TracedRArrayOverrides

using Base.Broadcast
using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

using ..Reactant:
    Reactant,
    TracedRArray,
    TracedRNumber,
    ReactantPrimitive,
    WrappedTracedRArray,
    AnyTracedRArray,
    Ops,
    MLIR,
    ancestor,
    allowscalar,
    aos_to_soa,
    unwrapped_eltype
using ..TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!, materialize_traced_array

using ReactantCore: ReactantCore
using GPUArraysCore: GPUArraysCore

ReactantCore.is_traced(::TracedRArray) = true

function Base.convert(::Type{TracedRArray{T,N}}, x::AbstractArray) where {T,N}
    @assert ndims(x) == N
    if x isa TracedRArray
        eltype(x) == T && return x
        return Ops.convert(TracedRArray{T,N}, x)
    end
    x isa WrappedTracedRArray &&
        return convert(TracedRArray{T,N}, materialize_traced_array(x))
    if eltype(x) <: TracedRNumber
        return convert(TracedRArray{T,N}, aos_to_soa(x))
    end
    return convert(TracedRArray{T,N}, Ops.constant(collect(x)))
end

TracedRArray{T,N}(x::AbstractArray) where {T,N} = convert(TracedRArray{T,N}, x)

function Base.getindex(
    a::TracedRArray{T,N}, index::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    GPUArraysCore.assertscalar("getindex(::TracedRArray, ::Vararg{Int, N})")

    start_indices = [
        TracedUtils.promote_to(TracedRNumber{Int}, i - 1).mlir_data for i in index
    ]
    slice_sizes = [Int64(1) for _ in index]

    res1 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_slice(a.mlir_data, start_indices; slice_sizes), 1
    )
    res2 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.reshape(
            res1; result_0=MLIR.IR.TensorType(Int64[], eltype(MLIR.IR.type(res1)))
        ),
        1,
    )
    return TracedRNumber{T}((), res2)
end

Base.getindex(a::TracedRArray{T,0}) where {T} = TracedRNumber{T}((), a.mlir_data)

function generate_index_list(i1, is...)
    list = reshape(i1, :, 1) .- 1
    for i in is
        i = reshape(i, :, 1)
        lorig = size(list, 1)
        list = repeat(list, size(i, 1), 1)
        i = repeat(i; inner=(lorig, 1)) .- 1
        list = hcat(list, i)
    end
    return list
end

function scalar_index_to_cartesian(idx::AbstractVector{T}, sz::NTuple{N,Int}) where {T,N}
    idx = idx .- 1
    idxs = materialize_traced_array(reshape(idx .% T(sz[1]), :, 1))
    idx = idx .÷ T(sz[1])
    for i in 2:N
        idxs = hcat(idxs, idx .% T(sz[i]))
        idx = idx .÷ T(sz[i])
    end
    return idxs
end

function Base.getindex(
    a::TracedRArray{T,N}, indices::Union{Int,TracedRNumber{Int}}
) where {T,N}
    if indices isa Int
        indices = TracedUtils.promote_to(TracedRNumber{Int}, indices)
    end
    indices = TracedUtils.broadcast_to_size(indices, (1,))
    return Ops.gather_getindex(a, scalar_index_to_cartesian(indices, size(a)))[1]
end

function Base.getindex(a::TracedRArray{T,N}, indices) where {T,N}
    if !(indices isa TracedRArray)
        indices = TracedUtils.promote_to(TracedRArray{Int,1}, collect(indices))
    end
    return Ops.gather_getindex(a, scalar_index_to_cartesian(indices, size(a)))
end

Base.getindex(a::TracedRArray{T,N}, ::Colon) where {T,N} = materialize_traced_array(vec(a))

function Base.getindex(a::TracedRArray{T,N}, indices::CartesianIndex{N}) where {T,N}
    indices =
        materialize_traced_array(
            reshape(
                TracedUtils.promote_to(TracedRArray{Int,1}, vcat(Tuple(indices)...)), 1, N
            ),
        ) .- 1
    return Ops.gather_getindex(a, indices)[1]
end

function Base.getindex(a::TracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    indices = map(enumerate(indices)) do (idx, i)
        i isa Colon && return 1:size(a, idx)
        i isa CartesianIndex && return Tuple(i)
        i isa AbstractArray{<:Bool} && return findall(i)
        return i
    end

    use_gather_getindex = false
    for idxs in indices
        idxs isa Number && continue
        if idxs isa Reactant.TracedType
            use_gather_getindex = true
            break
        end
        contiguous = all(isone, diff(idxs))
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
        indices_list = map(Base.Fix1(TracedUtils.promote_to, TracedRArray{Int,1}), indices)
        indices_list = generate_index_list(indices_list...)
        res = Ops.gather_getindex(a, indices_list)
        return Ops.reshape(res, length.(indices)...)
    end

    start_indices = map(indices) do i
        return TracedUtils.promote_to(TracedRNumber{Int}, first(i) - 1).mlir_data
    end
    slice_sizes = [Int64(length(i)) for i in indices]
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_slice(a.mlir_data, start_indices; slice_sizes), 1
    )

    x = TracedRArray{T,N}((), res, Tuple(length.(indices)))
    ddims = findall(Base.Fix2(isa, Integer), indices)
    isempty(ddims) || return materialize_traced_array(dropdims(x; dims=Tuple(ddims)))
    return x
end

# Prevent ambiguity
function Base.getindex(a::WrappedTracedRArray, index::Union{Int,TracedRNumber{Int}}...)
    return getindex(ancestor(a), TracedUtils.get_ancestor_indices(a, index...)...)
end

function Base.getindex(a::WrappedTracedRArray, indices...)
    return getindex(ancestor(a), TracedUtils.get_ancestor_indices(a, indices...)...)
end

function maybe_assert_scalar_setindexing(
    ::TracedRArray{T,N}, ::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    GPUArraysCore.assertscalar("setindex!(::TracedRArray, v, ::Vararg{Int, N})")
    return nothing
end

maybe_assert_scalar_setindexing(args...) = nothing

function Base.setindex!(a::TracedRArray{T,N}, v, indices::Vararg{Any,N}) where {T,N}
    maybe_assert_scalar_setindexing(a, indices...)

    indices = map(enumerate(indices)) do (idx, i)
        i isa Colon && return 1:size(a, idx)
        i isa CartesianIndex && return Tuple(i)
        i isa AbstractArray{<:Bool} && return findall(i)
        return i
    end

    use_scatter_setindex = false
    for idxs in indices
        idxs isa Number && continue
        if idxs isa Reactant.TracedType
            use_scatter_setindex = true
            break
        end
        contiguous = all(isone, diff(idxs))
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
        a.mlir_data = res.mlir_data
        return v
    end

    if v isa Number
        v = TracedUtils.broadcast_to_size(v, length.(indices))
        v = TracedUtils.promote_to(TracedRArray{T,N}, v)
    else
        v = TracedUtils.promote_to(TracedRArray{T,ndims(v)}, v)
        non_integer_indices = [!(idx isa Integer) for idx in indices]
        broadcast_dims = findall(non_integer_indices)
        if length(broadcast_dims) == N
            v = TracedUtils.broadcast_to_size(v, length.(indices))
        else
            v = Ops.broadcast_in_dim(
                materialize_traced_array(v), broadcast_dims, Int64.(length.(indices))
            )
        end
    end

    indices = [
        (
            TracedUtils.promote_to(TracedRNumber{Int}, i isa Colon ? 1 : first(i)) - 1
        ).mlir_data for i in indices
    ]
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_update_slice(
            a.mlir_data, TracedUtils.get_mlir_data(v), indices
        ),
        1,
    )
    a.mlir_data = res
    return v
end

function Base.setindex!(a::AnyTracedRArray{T,N}, v, indices::Vararg{Any,N}) where {T,N}
    ancestor_indices = TracedUtils.get_ancestor_indices(a, indices...)
    setindex!(ancestor(a), v, ancestor_indices...)
    return a
end

Base.Tuple(x::TracedRArray) = ntuple(Base.Fix1(Base.getindex, x), length(x))

Base.size(x::TracedRArray) = x.shape

Base.collect(x::TracedRArray) = copy(x) # XXX: Is this correct?

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), A.mlir_data, size(A))

# TODO is there a way to create an unitialized `tensor`? does it show an advantage? maybe `fill`?
function Base.similar(::TracedRArray, ::Type{T}, dims::Dims{N}) where {T,N}
    return Ops.constant(zeros(unwrapped_eltype(T), dims))
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

for (jlop, hloop, hlocomp, merge) in
    ((:(Base.:(==)), :compare, "EQ", :all), (:(Base.:(!=)), :compare, "NE", :any))
    @eval function $jlop(
        @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs::TracedRArray{T,N})
    ) where {T,N}
        elems = $(jlop).(lhs, rhs)
        return N == 0 ? elems : $(merge)(elems)
    end
end

function Base.mapreduce(
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
        TracedRNumber{op_in_T}((), MLIR.IR.argument(fnbody, 1)),
        TracedRNumber{op_in_T}((), MLIR.IR.argument(fnbody, 2)),
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
    @nospecialize(R::TracedRArray),
    A::Base.AbstractArrayOrBroadcasted,
)
    tmp = TracedUtils.broadcast_to_size(
        Base.mapreduce(f, op, A; dims=1), (1, size(R)[2:end]...)
    )
    R.mlir_data = broadcast(op, R, tmp).mlir_data
    return R
end

function Base.fill!(A::TracedRArray{T,N}, x) where {T,N}
    bcast = TracedUtils.broadcast_to_size(T(x), size(A))
    A.mlir_data = bcast.mlir_data
    return A
end

function Base.fill!(A::TracedRArray{T,N}, x::TracedRNumber{T2}) where {T,N,T2}
    bcast = TracedUtils.broadcast_to_size(
        TracedUtils.promote_to(TracedRNumber{T}, x), size(A)
    )
    A.mlir_data = bcast.mlir_data
    return A
end

struct AbstractReactantArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

AbstractReactantArrayStyle(::Val{N}) where {N} = AbstractReactantArrayStyle{N}()
AbstractReactantArrayStyle{M}(::Val{N}) where {N,M} = AbstractReactantArrayStyle{N}()

function BroadcastStyle(::Type{<:AnyTracedRArray{T,N}}) where {T,N}
    return AbstractReactantArrayStyle{N}()
end

function Base.similar(
    ::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{T}, dims
) where {T<:ReactantPrimitive,N}
    @assert N isa Int
    return TracedRArray{T,length(dims)}((), nothing, map(length, dims))
end

function Base.similar(
    ::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{TracedRNumber{T}}, dims
) where {T<:ReactantPrimitive,N}
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
    fn = if bc.f isa Type && bc.f <: ReactantPrimitive
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

Base.copyto!(dest::TracedRArray, bc::Broadcasted{Nothing}) = _copyto!(dest, bc) # Keep it for ArrayConflict

function Base.copyto!(dest::TracedRArray{T,N}, src::TracedRArray{T,N}) where {T,N}
    dest.mlir_data = src.mlir_data
    return dest
end

function Base.copyto!(dest::TracedRArray{T,N}, src::TracedRArray{T2,N}) where {T,T2,N}
    return copyto!(dest, Ops.convert(TracedRArray{T,N}, src))
end

function _copyto!(dest::AnyTracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (TracedUtils.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = TracedUtils.elem_apply(bc.f, args...)
    TracedUtils.set_mlir_data!(dest, res.mlir_data)
    return dest
end

function _copyto!(dest::AbstractArray{<:TracedRNumber}, bc::Broadcasted)
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
                result_0=MLIR.IR.TensorType(shape, MLIR.IR.Type(RT)),
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

Base.all(f::Function, x::AnyTracedRArray) = mapreduce(f, &, x)
Base.any(f::Function, x::AnyTracedRArray) = mapreduce(f, |, x)

# outer repeat
function Base._RepeatInnerOuter.repeat_outer(
    x::AnyTracedRArray{T,N}, counts::NTuple{M,Int}
) where {T,N,M}
    P = max(N, M) # potentially padded

    # (d1, d2, ..., dP) -> (d1, 1, d2, 1, ..., dP, 1)
    interleaved_size = ones(Int, 2P)
    interleaved_size[1:2:(2N)] .= size(x)

    x_interleaved = reshape(materialize_traced_array(x), interleaved_size...)

    # (d1, 1, d2, 1, ..., dP, 1) -> (d1, r1, d2, r2, ..., dP, rP)
    broadcast_target_size = interleaved_size
    broadcast_target_size[2:2:(2M)] .= counts

    x_broadcasted = TracedUtils.broadcast_to_size(x_interleaved, broadcast_target_size)

    # (d1, r1, d2, r2, ..., dP, rP) -> (d1*r1, d2*r2, ..., dP*rP)
    final_size = vec(prod(reshape(broadcast_target_size, 2, :); dims=1))

    return materialize_traced_array(reshape(x_broadcasted, final_size...))
end

# inner repeat
function Base._RepeatInnerOuter.repeat_inner(
    x::AnyTracedRArray{T,N}, counts::NTuple{M,Int}
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

end
