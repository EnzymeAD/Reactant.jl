module TracedRArrayOverrides

using Base: Broadcast
using Base.Broadcast: Broadcasted, AbstractArrayStyle, instantiate

using ..Reactant: Reactant, TracedRArray, TracedRNumber, AnyTracedRArray, AnyTracedRVector
using ..Reactant: MLIR, unwrapped_eltype
using ..Ops: @opcall
using ..TracedUtils: TracedUtils, get_mlir_data, set_mlir_data!, materialize_traced_array

using ReactantCore: ReactantCore
using GPUArraysCore: GPUArraysCore, @allowscalar

__lt(::Base.Order.ForwardOrdering, a, b) = isless.(a, b)
__lt(o::Base.Order.ReverseOrdering, a, b) = __lt(o.fwd, b, a)
__lt(o::Base.Order.By, a, b) = __lt(o.order, o.by.(a), o.by.(b))
__lt(o::Base.Order.Lt, a, b) = o.lt.(a, b)

ReactantCore.is_traced(::TracedRArray, seen) = true
ReactantCore.is_traced(::TracedRArray) = true

Base.strides(x::TracedRArray) = Base.size_to_strides(1, size(x)...)

Base.IndexStyle(::Type{<:TracedRArray}) = Base.IndexLinear()

# This is required otherwise we will copy a tracedrarray each time
# we use it
Base.convert(T::Type{<:TracedRArray}, x::AbstractArray) = Reactant.promote_to(T, x)

# Base.complex
Base.complex(x::TracedRArray{<:Real}) = complex.(x)
Base.complex(x::TracedRArray{<:Complex}) = x

TracedRArray{T,N}(x::AbstractArray) where {T,N} = convert(TracedRArray{T,N}, x)

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

    res = @opcall(
        reshape(
            @opcall(
                dynamic_update_slice(
                    a,
                    Reactant.broadcast_to_size(
                        Reactant.promote_to(TracedRNumber{T}, v), ntuple(Returns(1), N)
                    ),
                    collect(
                        Reactant.TracedIndexing.scalar_index_to_cartesian(index, size(a))
                    ),
                )
            ),
            collect(size(a)),
        )
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

function _setindex_scalar_cartesian!(
    a::TracedRArray{T,N}, v, index::CartesianIndex{N}
) where {T,N}
    GPUArraysCore.assertscalar("setindex!(::TracedRArray, v, ::CartesianIndex{N})")
    res = @opcall(
        reshape(
            @opcall(
                dynamic_update_slice(
                    a,
                    Reactant.broadcast_to_size(T(v), ntuple(Returns(1), N)),
                    collect(Int64, index.I),
                )
            ),
            collect(size(a)),
        )
    )
    set_mlir_data!(a, get_mlir_data(res))
    return a
end

function Base.setindex!(a::TracedRArray{T,N}, v, index::CartesianIndex{N}) where {T,N}
    return _setindex_scalar_cartesian!(a, v, index)
end
function Base.setindex!(a::TracedRArray{T,1}, v, index::CartesianIndex{1}) where {T}
    return _setindex_scalar_cartesian!(a, v, index)
end

function _setindex_linear!(a::TracedRArray{T,N}, v, indices::AbstractArray) where {T,N}
    if !(indices isa Reactant.TracedType) && TracedUtils.__contiguous_indices(vec(indices))
        res = @opcall(
            reshape(
                @opcall(
                    dynamic_update_slice(
                        materialize_traced_array(vec(a)),
                        Reactant.broadcast_to_size(T.(vec(v)), (length(indices),)),
                        [first(indices)],
                    )
                ),
                collect(size(a)),
            )
        )
        set_mlir_data!(a, get_mlir_data(res))
        return a
    end

    if !(indices isa TracedRArray)
        indices = collect(indices)
        eltype(indices) <: CartesianIndex && (indices = LinearIndices(size(a))[indices])
        indices = Reactant.promote_to(TracedRArray{Int,ndims(indices)}, indices)
    end
    res = @opcall scatter_setindex(
        a,
        Reactant.TracedIndexing.scalar_index_to_cartesian(vec(indices), size(a)),
        Reactant.promote_to(TracedRArray{T,1}, materialize_traced_array(vec(v))),
    )
    set_mlir_data!(a, get_mlir_data(res))
    return a
end

function Base.setindex!(a::TracedRArray{T,N}, v, indices::AbstractArray) where {T,N}
    return _setindex_linear!(a, v, indices)
end
function Base.setindex!(a::TracedRArray{T,1}, v, indices::AbstractArray) where {T}
    return _setindex_linear!(a, v, indices)
end

function Base.setindex!(a::TracedRArray{T,N}, v, indices::Vararg{Any,N}) where {T,N}
    if (N == 1) && (indices isa Colon)
        # Remove ambiguity from the previous
        # ```julia
        # Base.setindex!(a::TracedRArray{T,N}, v, ::Colon) where {T,N}
        # ```
        # signature, which would be confused with this one for N=1.
        v = Reactant.broadcast_to_size(v, size(a))
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
        contiguous = TracedUtils.__contiguous_indices(idxs)
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

        gather_dims = TracedUtils.indices_to_gather_dims(indices...)

        v = @opcall convert(
            TracedRArray{T,ndims(v)},
            Reactant.promote_to(TracedRArray{unwrapped_eltype(v),ndims(v)}, v),
        )

        updates = @opcall transpose(v, invperm(gather_dims.permutation))
        n_collapsed = length(gather_dims.collapsed_slice_dims)
        updates_shape = Int64[
            prod(size(updates)[1:n_collapsed]), size(updates)[(n_collapsed + 1):end]...
        ]
        updates = @opcall reshape(updates, updates_shape)

        res = @opcall(
            scatter(
                (xᵢ, xⱼ) -> xⱼ,
                [a],
                gather_dims.start_indices,
                [updates];
                update_window_dims=gather_dims.offset_dims,
                inserted_window_dims=gather_dims.collapsed_slice_dims,
                input_batching_dims=Int64[],
                scatter_indices_batching_dims=Int64[],
                scatter_dims_to_operand_dims=gather_dims.start_index_map,
                index_vector_dim=gather_dims.index_vector_dim,
            )
        )[1]
        set_mlir_data!(a, get_mlir_data(res))
        return v
    end

    if v isa Number
        v = Reactant.broadcast_to_size(v, length.(indices))
        v = Reactant.promote_to(TracedRArray{T,N}, v)
    else
        v = Reactant.promote_to(TracedRArray{T,ndims(v)}, v)
        non_integer_indices = [
            !(idx isa Union{Integer,TracedRNumber{<:Integer}}) for idx in indices
        ]
        broadcast_dims = findall(non_integer_indices)
        if length(broadcast_dims) == N
            v = Reactant.broadcast_to_size(v, length.(indices))
        else
            v = @opcall broadcast_in_dim(
                materialize_traced_array(v),
                broadcast_dims,
                collect(Int64, length.(indices)),
            )
        end
    end

    set_mlir_data!(
        a,
        @opcall(
            dynamic_update_slice(a, v, [i isa Colon ? 1 : first(i) for i in indices])
        ).mlir_data,
    )
    return v
end

Base.Tuple(x::TracedRArray) = ntuple(Base.Fix1(getindex, x), length(x))

Base.size(x::TracedRArray) = x.shape

Base.collect(x::TracedRArray) = copy(x) # XXX: Is this correct?

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), A.mlir_data, size(A))

function Base.similar(::TracedRArray, ::Type{T}, dims::Dims{N}) where {T,N}
    return @opcall fill(zero(unwrapped_eltype(T)), dims)
end

function Base.show(io::IOty, X::AnyTracedRArray) where {IOty<:Union{IO,IOContext}}
    print(io, Core.Typeof(X), "(")
    parent(X) !== X && Base.show(io, parent(X))
    return print(io, ")")
end

function Base.show(io::IOty, X::TracedRArray{T,N}) where {T,N,IOty<:Union{IO,IOContext}}
    return print(io, "TracedRArray{", T, ",", N, "N}(", X.paths, ", size=", size(X), ")")
end

for ArrayType in (
    :(AnyTracedRArray{T,N}),
    :(TracedRArray{T,N}),
    :(SubArray{<:TracedRNumber{T},N,<:TracedRArray}),
    :(Base.ReshapedArray{<:TracedRNumber{T},N,<:TracedRArray})
    )
    @eval function Base.permutedims(A::$ArrayType, perm) where {T,N}
        return @opcall transpose(materialize_traced_array(A), Int64[perm...])
    end
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

function __default_init(
    ::Type{T}, ::Union{typeof(Base.min),typeof(Base.FastMath.min_fast)}
) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemax(T))
end
function __default_init(
    ::Type{T}, ::Union{typeof(Base.max),typeof(Base.FastMath.max_fast)}
) where {T}
    return Reactant.promote_to(TracedRNumber{T}, typemin(T))
end
function __default_init(::Type{T}, op::F) where {T,F}
    return Reactant.promote_to(TracedRNumber, Base.reduce_empty(Base.BottomRF(op), T))
end

function __default_init(
    T::Type{<:Reactant.ReactantFloat8},
    ::Union{typeof(Base.min),typeof(Base.FastMath.min_fast)},
)
    return Reactant.promote_to(TracedRNumber{T}, typemax(Float16))
end
function __default_init(
    T::Type{<:Reactant.ReactantFloat8},
    ::Union{typeof(Base.max),typeof(Base.FastMath.max_fast)},
)
    return Reactant.promote_to(TracedRNumber{T}, typemin(Float16))
end
function __default_init(T::Type{<:Reactant.ReactantFloat8}, op::F) where {F}
    return Reactant.promote_to(TracedRNumber{T}, __default_init(Float16, op))
end

function overloaded_mapreduce(
    @nospecialize(f), @nospecialize(op), @nospecialize(A); dims=:, init=Base._InitialValue()
)
    res = unwrapped_broadcast(f, A)
    # This means we are unable to use the optimized dispatches. For now we will
    # unroll the mapreduce.
    if typeof(res) == typeof(A)
        @assert dims == Colon() "dims not supported for mapreduce currently."
        return foldl(op, res; init)
    end
    return overloaded_mapreduce(identity, op, res; dims=:, init)
end

function overloaded_mapreduce(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(A::AnyTracedRArray{T,N});
    dims=:,
    init=Base._InitialValue(),
) where {T,N}
    A = materialize_traced_array(A)

    original_dims = dims
    dims isa Int && (dims = Int64[dims])
    dims isa Colon && (dims = collect(Int64, 1:N))
    dims isa Vector{Int64} || (dims = collect(Int64, dims))

    op_in_T = unwrapped_eltype(Core.Compiler.return_type(f, Tuple{T}))
    reduce_init = __default_init(op_in_T, op)
    riT = unwrapped_eltype(typeof(reduce_init))
    if riT != op_in_T
        op_in_T = riT
        A = riT.(A)
    end
    reduce_init = Reactant.promote_to(TracedRNumber{op_in_T}, reduce_init)

    reduce_input = materialize_traced_array(TracedUtils.elem_apply(f, A))

    res = @opcall reduce(reduce_input, reduce_init, dims, op)

    (init isa Base._InitialValue || init === nothing) || (res = op.(res, init))

    if original_dims isa Colon
        @assert size(res) == () "expected size of result to be (), got $(size(res))"
        return TracedRNumber{unwrapped_eltype(res)}((), res.mlir_data)
    end
    if res isa TracedRNumber
        res = TracedRArray{unwrapped_eltype(res),0}((), res.mlir_data, ())
    end
    return @opcall reshape(res, [ifelse(i in dims, 1, size(A, i)) for i in 1:N])
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
    copyto!(R, op.(R, tmp))
    return R
end

function Base.fill!(A::AnyTracedRArray{T,N}, x) where {T,N}
    bcast = Reactant.broadcast_to_size(T(x), size(A))
    set_mlir_data!(A, get_mlir_data(bcast))
    return A
end

function Base.fill!(A::AnyTracedRArray{T,N}, x::TracedRNumber{T2}) where {T,N,T2}
    bcast = Reactant.broadcast_to_size(Reactant.promote_to(TracedRNumber{T}, x), size(A))
    set_mlir_data!(A, get_mlir_data(bcast))
    return A
end

function Base.fill!(A::Array{T,N}, x::TracedRNumber{T2}) where {T,N,T2}
    throw(MethodError(fill!, (A, x)))
end

struct AbstractReactantArrayStyle{N} <: AbstractArrayStyle{N} end

AbstractReactantArrayStyle(::Val{N}) where {N} = AbstractReactantArrayStyle{N}()
AbstractReactantArrayStyle{M}(::Val{N}) where {N,M} = AbstractReactantArrayStyle{N}()

function Broadcast.BroadcastStyle(::Type{<:AnyTracedRArray{T,N}}) where {T,N}
    return AbstractReactantArrayStyle{N}()
end

function Base.similar(
    ::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{T}, dims
) where {T<:Reactant.ReactantPrimitive,N}
    @assert N isa Int
    return @opcall fill(zero(unwrapped_eltype(T)), dims)
end

function Base.similar(
    ::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{TracedRNumber{T}}, dims
) where {T<:Reactant.ReactantPrimitive,N}
    @assert N isa Int
    return @opcall fill(zero(T), dims)
end

function Base.copy(bc::Broadcasted{<:AbstractReactantArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    dest = copyto!(similar(bc, ElType), bc)
    return dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

Base.eltype(::Broadcast.Extruded{T}) where {T} = eltype(T)

first_scalar(x) = @allowscalar first(x)

# we need to override the outer copy method to make sure we never fall back to scalar
# iteration (see, e.g., CUDA.jl#145)
function Base.copy(bc::Broadcasted{<:AbstractReactantArrayStyle})
    fn = if bc.f isa Type && bc.f <: Reactant.ReactantPrimitive
        TracedUtils.TypeCast{bc.f}()
    else
        bc.f
    end
    ElType = Broadcast.combine_eltypes(fn, bc.args)
    # Special case a union{} return so we can see the better error message
    if ElType === Union{}
        fn(map(first_scalar, bc.args)...)
    elseif ElType == Any
        ElType = eltype(fn(map(first_scalar, bc.args)...))
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

function Base.copyto!(dest::AnyTracedRArray{T,N}, src::TracedRArray{T,N}) where {T,N}
    TracedUtils.set_mlir_data!(dest, src.mlir_data)
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
    src2 = if T != T2
        @opcall convert(TracedRArray{T,N}, src)
    else
        src
    end
    TracedUtils.set_mlir_data!(dest, src2.mlir_data)
    return dest
end

function Base.copyto!(
    dest::AnyTracedRArray{T1,N} where {T1}, src::AnyTracedRArray{T2,N} where {T2}
) where {N}
    return copyto!(dest, materialize_traced_array(src))
end

function Base.copyto!(dest::AnyTracedRArray{T,N}, src::Array{T2,N}) where {T,T2,N}
    return copyto!(dest, Reactant.promote_to(TracedRArray{T2,N}, src))
end

function _copyto!(dest::AnyTracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = Reactant.promote_to(
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

    args = (Reactant.broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = TracedUtils.elem_apply(bc.f, args...)
    for I in 1:length(dest)
        dest[I] = @allowscalar res[I]
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
@inline function Base.typed_hvcat(
    ::Type{T}, rows::Tuple{Int}, as::TracedRArray...
) where {T}
    return invoke(
        Base.typed_hvcat, Tuple{Type{T},Tuple{Vararg{Int}},Vararg{Any}}, T, rows, as...
    )
end

function Base._typed_hvncat(
    ::Type{T}, dims::NTuple{N,Int}, row_first::Bool, a::TracedRArray, as::TracedRArray...
) where {T,N}
    return _typed_hvncat_internal(T, dims, row_first, a, as...)
end
function Base._typed_hvncat(
    ::Type{T}, dims::Tuple{Int}, row_first::Bool, a::TracedRArray, as::TracedRArray...
) where {T}
    return _typed_hvncat_internal(T, dims, row_first, a, as...)
end
function Base._typed_hvncat(
    ::Type{T}, dims::Tuple{}, row_first::Bool, a::TracedRArray, as::TracedRArray...
) where {T}
    return _typed_hvncat_internal(T, dims, row_first, a, as...)
end

function _typed_hvncat_internal(
    ::Type{T}, dims::NTuple{N,Int}, row_first::Bool, as...
) where {T,N}
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
    X = map(Base.Fix1(Reactant.promote_to, TracedRArray{RT,length(shape)}), X)

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
        T = unwrapped_eltype(x)
        min = Reactant.promote_to(TracedRNumber{T}, min)
        max = Reactant.promote_to(TracedRNumber{T}, max)
        y = @opcall clamp(min, materialize_traced_array(x), max)
        TracedUtils.set_mlir_data!(x, y.mlir_data)
        return x
    end
end

# outer repeat
function repeat_outer_overloaded(x::AnyTracedRArray{T,N}, counts::Dims{N}) where {T,N}
    # (d1, d2, ..., dP) -> (d1, 1, d2, 1, ..., dP, 1)
    interleaved_size = ones(Int, 2N)
    interleaved_size[1:2:(2N)] .= size(x)

    x_interleaved = @opcall reshape(materialize_traced_array(x), interleaved_size...)

    # (d1, 1, d2, 1, ..., dP, 1) -> (d1, r1, d2, r2, ..., dP, rP)
    broadcast_target_size = interleaved_size
    broadcast_target_size[2:2:(2N)] .= counts

    x_broadcasted = Reactant.broadcast_to_size(x_interleaved, broadcast_target_size)

    # (d1, r1, d2, r2, ..., dP, rP) -> (d1*r1, d2*r2, ..., dP*rP)
    final_size = vec(prod(reshape(broadcast_target_size, 2, :); dims=1))

    return materialize_traced_array(reshape(x_broadcasted, final_size...))
end

function Base._RepeatInnerOuter.repeat_outer(
    x::AnyTracedRArray{T,1}, counts::Tuple{Any}
) where {T}
    return repeat_outer_overloaded(x, counts)
end
function Base._RepeatInnerOuter.repeat_outer(
    x::AnyTracedRArray{T,2}, counts::NTuple{2,Any}
) where {T}
    return repeat_outer_overloaded(x, counts)
end
function Base._RepeatInnerOuter.repeat_outer(
    x::AnyTracedRArray{T,N}, counts::NTuple{N,Any}
) where {T,N}
    return repeat_outer_overloaded(x, counts)
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

    x_broadcasted = Reactant.broadcast_to_size(x_interleaved, broadcast_target_size)

    # (r1, d1, r2, d2, ..., rP, dP) -> (d1*r1, d2*r2, ..., dP*rP)
    final_size = vec(prod(reshape(broadcast_target_size, 2, :); dims=1))

    return materialize_traced_array(reshape(x_broadcasted, final_size...))
end

# stack
function overloaded_stack(dims::Union{Integer,Colon}, xs)
    @assert allequal([ndims(x) for x in xs]) "All arrays must have the same number of \
                                              dimensions..."
    dims = dims isa Colon ? ndims(first(xs)) + 1 : dims
    res = []
    for x in xs
        new_shape = ntuple(
            i -> i == dims ? 1 : (i < dims ? size(x, i) : size(x, i - 1)), ndims(x) + 1
        )
        push!(res, materialize_traced_array(internal_stack_reshape(x, new_shape)))
    end
    return cat(res...; dims)
end

internal_stack_reshape(x, new_shape) = reshape(x, new_shape)
function internal_stack_reshape(x::TracedRNumber{T}, new_shape) where {T}
    return internal_stack_reshape(TracedRArray{T,0}((), x.mlir_data, ()), new_shape)
end

# sort
function Base.sort(x::AnyTracedRArray; alg=missing, kwargs...)
    return sort!(copy(x); alg, kwargs...)
end
function Base.sort(x::AnyTracedRVector; alg=missing, kwargs...)
    return sort!(copy(x); alg, kwargs...)
end

function Base.sort!(
    x::AnyTracedRVector;
    lt=isless,
    by=identity,
    rev::Bool=false,
    alg=missing,
    order=Base.Order.Forward,
)
    @assert alg === missing "Reactant doesn't support `alg` kwarg for `sort!`"

    ordering = Base.ord(lt, by, rev, order)
    comparator = (a, b) -> __lt(ordering, a, b)

    res = only(@opcall(sort(materialize_traced_array(x); comparator, dimension=1)))
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
    order=Base.Order.Forward,
)
    @assert alg === missing "Reactant doesn't support `alg` kwarg for `sort!`"

    ordering = Base.ord(lt, by, rev, order)
    comparator = (a, b) -> __lt(ordering, a, b)

    res = only(@opcall(sort(materialize_traced_array(x); dimension=dims, comparator)))
    set_mlir_data!(x, get_mlir_data(res))
    return x
end

function Base.sortperm(x::AnyTracedRArray; alg=missing, kwargs...)
    return sortperm!(similar(x, Int), x; alg, kwargs...)
end
function Base.sortperm(x::AnyTracedRVector; alg=missing, kwargs...)
    return sortperm!(similar(x, Int), x; alg, dims=1, kwargs...)
end

function Base.sortperm!(
    ix::AnyTracedRArray{Int,N},
    x::AnyTracedRArray{<:Any,N};
    dims::Union{Integer,Nothing}=nothing,
    lt=isless,
    by=identity,
    rev::Bool=false,
    alg=missing,
    order=Base.Order.Forward,
) where {N}
    if dims === nothing
        @assert ndims(x) == 1
        dims = 1
    end

    @assert alg === missing "Reactant doesn't support `alg` kwarg for `sortperm!`"

    ordering = Base.ord(lt, by, rev, order)
    comparator = (a, b, i1, i2) -> __lt(ordering, a, b)

    idxs = @opcall constant(collect(LinearIndices(x)))
    _, res = @opcall sort(materialize_traced_array(x), idxs; dimension=dims, comparator)
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
        sorted_x, sorted_idxs = @opcall sort(
            materialize_traced_array(x),
            @opcall(constant(collect(LinearIndices(x))));
            dimension=1,
            comparator=(a, b, i1, i2) -> !lt(by(a), by(b)),
        )
        return (getindex(sorted_x, 1:maximum(k)), getindex(sorted_idxs, 1:maximum(k)))
    end

    if Reactant.LOWER_PARTIALSORT_TO_APPROX_TOP_K[] && T <: Reactant.ReactantFloat
        result = @opcall approx_top_k(
            materialize_traced_array(x),
            maximum(k);
            comparator=(a, b, i1, i2) -> a > b,
            dimension=1,
            init_val=typemin(T),
        )
        return (
            getindex(result.values, 1:maximum(k)), getindex(result.indices, 1:maximum(k))
        )
    end

    (; values, indices) = @opcall top_k(materialize_traced_array(x), maximum(k))
    return values, indices
end

function overloaded_partialsort_ascending(
    x::AnyTracedRVector{T}, k::Union{Integer,OrdinalRange}; by=identity, lt=isless
) where {T}
    if lt !== isless || by !== identity || T <: Unsigned
        sorted_x, sorted_idxs = @opcall sort(
            materialize_traced_array(x),
            @opcall(constant(collect(LinearIndices(x))));
            dimension=1,
            comparator=(a, b, i1, i2) -> !lt(by(a), by(b)),
        )
        return (getindex(sorted_x, 1:maximum(k)), getindex(sorted_idxs, 1:maximum(k)))
    end

    if Reactant.LOWER_PARTIALSORT_TO_APPROX_TOP_K[] && T <: Reactant.ReactantFloat
        result = @opcall approx_top_k(
            materialize_traced_array(x),
            maximum(k);
            comparator=(a, b, i1, i2) -> a < b,
            dimension=1,
            init_val=typemax(T),
        )
        return (
            getindex(result.values, 1:maximum(k)), getindex(result.indices, 1:maximum(k))
        )
    end

    (; values, indices) = @opcall top_k(
        @opcall(negate(materialize_traced_array(x))), maximum(k)
    )
    return @opcall(negate(values)), indices
end

# arg* functions
function Base.argmin(f::F, x::AnyTracedRArray) where {F}
    idx = Reactant.TracedIndexing.scalar_index_to_cartesian(argmin(f.(x)), size(x))
    return @allowscalar x[idx...]
end

function Base.argmax(f::F, x::AnyTracedRArray) where {F}
    idx = Reactant.TracedIndexing.scalar_index_to_cartesian(argmax(f.(x)), size(x))
    return @allowscalar x[idx...]
end

Base.argmin(x::AnyTracedRArray; kwargs...) = findmin(identity, x; kwargs...)[2]
Base.argmax(x::AnyTracedRArray; kwargs...) = findmax(identity, x; kwargs...)[2]

# find* functions
Base.findfirst(x::AnyTracedRArray) = findfirst(identity, x)
Base.findlast(x::AnyTracedRArray) = findlast(identity, x)

# FIXME: we need to conditionally return `nothing` here if idx < 0
function Base.findfirst(f::Function, x::AnyTracedRArray)
    idx = @opcall findfirst(materialize_traced_array(vec(f.(x))))
    return TracedRNumber{Int}((), idx.mlir_data)
end

# FIXME: we need to conditionally return `nothing` here if idx < 0
function Base.findlast(f::Function, x::AnyTracedRArray)
    fA = @opcall reverse(materialize_traced_array(vec(f.(x))); dimensions=[1])
    idx = @opcall findfirst(fA)
    return length(x) + 1 - TracedRNumber{Int}((), idx.mlir_data)
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

    fx = @opcall negate(materialize_traced_array(f.(x)))
    (; values, indices) = @opcall top_k(fx, 1; dimension=dims)

    # Compute linear indices
    strds = strides(x)
    iotas = [@opcall(iota(Int64, [size(indices)...]; iota_dimension=i)) for i in 1:ndims(x)]
    iotas[dims] = @opcall subtract(indices, @opcall(fill(Int64(1), size(indices))))
    linear_indices = @opcall fill(Int64(1), size(indices))
    for d in eachindex(iotas)
        linear_indices = @opcall add(
            linear_indices,
            @opcall(multiply(iotas[d], @opcall(fill(Int64(strds[d]), size(iotas[d]))))),
        )
    end

    values = @opcall negate(values)
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
    (; values, indices) = @opcall top_k(fx, 1; dimension=dims)

    # Compute linear indices
    strds = strides(x)
    iotas = [@opcall(iota(Int64, [size(indices)...]; iota_dimension=i)) for i in 1:ndims(x)]
    iotas[dims] = @opcall subtract(indices, @opcall(fill(Int64(1), size(indices))))
    linear_indices = @opcall fill(Int64(1), size(indices))
    for d in eachindex(iotas)
        linear_indices = @opcall add(
            linear_indices,
            @opcall(multiply(iotas[d], @opcall(fill(Int64(strds[d]), size(iotas[d]))))),
        )
    end

    ndims(x) == 1 && return @allowscalar (values[1], linear_indices[1])
    return (values, linear_indices)
end

function overloaded_map(f, x::AbstractArray, xs::AbstractArray...)
    @assert allequal((axes(x), axes.(xs)...)) "Expected axes of all inputs to map to be \
                                               equal"

    needs_unrolling = falses(length(xs) + 1)
    inputs = ()
    for (i, input) in enumerate((x, xs...))
        if input isa AnyTracedRArray
            input = Reactant.materialize_traced_array(input)
        elseif eltype(input) <: Reactant.ReactantPrimitive
            input = Reactant.promote_to(TracedRArray{eltype(input),ndims(input)}, input)
        else
            needs_unrolling[i] = true
        end
        inputs = (inputs..., input)
    end

    @assert allequal(needs_unrolling) "All inputs to `overloaded_map` must be \
                                       unrolled or none of them. Open an issue."
    if needs_unrolling[1]
        length(inputs) == 1 && return unrolled_map(f, only(inputs))
        return unrolled_map(splat(f), zip(inputs...))
    end
    return TracedUtils.elem_apply(f, inputs...)
end

function overloaded_map!(f, y::AnyTracedRArray, x::AbstractArray, xs::AbstractArray...)
    copyto!(y, overloaded_map(f, x, xs...))
    return y
end

function Base.mapslices(f::F, A::AnyTracedRArray; dims) where {F}
    return mapslices(f, materialize_traced_array(A); dims)
end

function Base.mapslices(f::F, A::TracedRArray; dims) where {F}
    dims isa Integer && (dims = Int64[dims])
    dims isa AbstractVector || (dims = collect(Int64, dims))
    return @opcall batch(f, A, dims)
end

# accumulate interface
## Taken from https://github.com/JuliaGPU/CUDA.jl/blob/a4a7af45f54f0e57f5912bb52db48e2d27cf7b4f/src/accumulate.jl#L201
function Base.accumulate(
    op, A::AnyTracedRArray; dims::Union{Integer,Nothing}=nothing, kwargs...
)
    if dims === nothing && ndims(A) != 1
        return reshape(accumulate(op, A[:]), size(A)...)
    end

    nt = values(kwargs)
    # Base.promote_op was having issues
    if isempty(kwargs)
        zA = zero(unwrapped_eltype(A))
        out = similar(A, TracedRNumber{unwrapped_eltype(op(zA, zA))})
    elseif keys(nt) === (:init,)
        zA = zero(unwrapped_eltype(A))
        zI = zero(unwrapped_eltype(nt.init))
        out = similar(A, TracedRNumber{unwrapped_eltype(op(zA, zI))})
    else
        throw(
            ArgumentError(
                "accumulate does not support the keyword arguments $(setdiff(keys(nt), (:init,)))",
            ),
        )
    end

    return accumulate!(op, out, A; dims, kwargs...)
end

function Base.accumulate_pairwise!(op, A::AnyTracedRVector, B::AnyTracedRVector)
    return accumulate!(op, A, B; dims=1)
end

function Base._accumulate!(
    op, output::AnyTracedRArray, input::AnyTracedRVector, ::Nothing, ::Nothing
)
    return scan_impl!(op, output, input; dims=1)
end

function Base._accumulate!(
    op, output::AnyTracedRArray, input::AnyTracedRArray, dims::Integer, ::Nothing
)
    return scan_impl!(op, output, input; dims=dims)
end

function Base._accumulate!(
    op, output::AnyTracedRArray, input::AnyTracedRVector, ::Nothing, init::Some
)
    return scan_impl!(op, output, input; dims=1, init=init)
end

function Base._accumulate!(
    op, output::AnyTracedRArray, input::AnyTracedRArray, dims::Integer, init::Some
)
    return scan_impl!(op, output, input; dims=dims, init=init)
end

function scan_impl!(
    op,
    output::AnyTracedRArray{T,N},
    input::AnyTracedRArray{T,N};
    dims::Integer,
    init=Base._InitialValue(),
) where {T,N}
    @assert dims > 0 "dims must be a positive integer"
    @assert axes(output) == axes(input) "output and input must have the same shape"

    dims > ndims(input) && return copyto!(output, input)

    if init isa Base._InitialValue
        op_in_T = Core.Compiler.return_type(op, Tuple{T,T})
        op_in_T === Union{} && (op_in_T = T)
        init = __default_init(T, op)
        riT = unwrapped_eltype(typeof(init))
        if riT != op_in_T
            op_in_T = riT
            input = riT.(input)
        end
    else
        # TODO: fix this for TPUs
        if contains(string(first(Reactant.devices())), "TPU")
            initT = __default_init(T, op)
            if initT != init && initT != something(init)
                throw(
                    AssertionError(
                        "Currently, `init` is not supported on TPUs, provided value $init does not match identity $initT.",
                    ),
                )
            end
        end
    end

    init = something(init) # unwrap Some
    init = Reactant.promote_to(TracedRNumber{unwrapped_eltype(init)}, init)

    window_dimensions = ones(Int64, N)
    window_dimensions[dims] = size(input, dims)

    padding_low = zeros(Int64, N)
    padding_low[dims] = size(input, dims) - 1

    reduction_result = @opcall(
        reduce_window(
            op,
            [materialize_traced_array(input)],
            [init];
            window_dimensions=window_dimensions,
            window_strides=ones(Int64, N),
            base_dilations=ones(Int64, N),
            window_dilations=ones(Int64, N),
            padding_low=padding_low,
            padding_high=zeros(Int64, N),
            output_shape=collect(Int64, size(output)),
        )
    )[1]
    copyto!(output, reduction_result)

    return output
end

for (aType, xType) in (
    (AbstractRange{<:Real}, TracedRNumber{<:Real}),
    (AbstractRange{<:TracedRNumber}, Real),
    (AbstractRange{<:TracedRNumber}, TracedRNumber{<:Real}),
)
    @eval function Base.searchsortedfirst(
        a::$(aType), x::$(xType), o::Base.DirectOrdering
    )::TracedRNumber{keytype(a)}
        x = TracedUtils.promote_to(TracedRNumber{Reactant.unwrapped_eltype(a)}, x)

        f, h, l = first(a), step(a), last(a)
        n = round(Int, (x - f) / h + 1)

        return ifelse(
            !Base.Order.lt(o, f, x),
            1,
            ifelse(
                (h == 0) | Base.Order.lt(o, l, x),
                length(a) + 1,
                ifelse(Base.Order.lt(o, @allowscalar(a[n]), x), n + 1, n),
            ),
        )
    end
end

function overloaded_searchsortedfirst(v, x, lo::T, hi::T, o::Base.Ordering) where {T}
    v = TracedUtils.broadcast_to_size(v, size(v))
    x = TracedUtils.promote_to(TracedRNumber{Reactant.unwrapped_eltype(v)}, x)
    return sum(T.(__lt(o, v[lo:hi], x)); init=lo)
end

function overloaded_searchsortedlast(v, x, lo::T, hi::T, o::Base.Ordering) where {T}
    v = TracedUtils.broadcast_to_size(v, size(v))
    x = TracedUtils.promote_to(TracedRNumber{Reactant.unwrapped_eltype(v)}, x)
    return sum(T.(.!(__lt(o, x, v[lo:hi]))); init=lo - 1)
end

function overloaded_searchsorted(v, x, lo::T, hi::T, o::Base.Ordering) where {T}
    v = TracedUtils.broadcast_to_size(v, size(v))
    x = TracedUtils.promote_to(TracedRNumber{Reactant.unwrapped_eltype(v)}, x)
    firstidx = overloaded_searchsortedfirst(v, x, lo, hi, o)
    lastidx = overloaded_searchsortedlast(v, x, lo, hi, o)
    return Reactant.TracedUnitRange(firstidx, lastidx)
end

for op in (:searchsortedfirst, :searchsortedlast, :searchsorted)
    rop = Symbol(:overloaded_, op)

    @eval begin
        function Base.$(op)(
            x::AnyTracedRVector, v, lo::T, hi::T, o::Base.Ordering
        ) where {T<:Integer}
            return $(rop)(x, v, lo, hi, o)
        end
        function Base.$(op)(
            x::AbstractVector, v::TracedRNumber, lo::T, hi::T, o::Base.Ordering
        ) where {T<:Integer}
            return $(rop)(x, v, lo, hi, o)
        end
        function Base.$(op)(
            x::AnyTracedRVector, v::TracedRNumber, lo::T, hi::T, o::Base.Ordering
        ) where {T<:Integer}
            return $(rop)(x, v, lo, hi, o)
        end
    end
end

function Base.reverse(
    v::AnyTracedRVector{T}, start::Integer, stop::Integer=lastindex(v)
) where {T}
    v[start:stop] = reverse!(v[start:stop])
    return v
end

function Base.reverse!(
    v::AnyTracedRVector{T}, start::Integer, stop::Integer=lastindex(v)
) where {T}
    reverse!(view(v, start:stop))
    return v
end

function Base.reverse!(v::AnyTracedRVector{T}) where {T}
    v_mat = materialize_traced_array(v)
    copyto!(v, @opcall(reverse(v_mat; dimensions=1)))
    return v
end

function Base._reverse!(a::AnyTracedRArray{T,N}, dims::NTuple{M,Int}) where {T,N,M}
    a_mat = materialize_traced_array(a)
    copyto!(a, @opcall(reverse(a_mat; dimensions=dims)))
    return a
end

function Base.circshift!(
    dest::AnyTracedRArray{T,N}, src, shiftamt::Base.DimsInteger
) where {T,N}
    return circshift_internal!(dest, src, shiftamt)
end
function Base.circshift!(dest::AnyTracedRArray{T,N}, src, shiftamt::Tuple{}) where {T,N}
    return circshift_internal!(dest, src, shiftamt)
end

function circshift_internal!(
    dest::AnyTracedRArray{T,N}, src, shiftamt::Base.DimsInteger
) where {T,N}
    src = Reactant.promote_to(TracedRArray{T,N}, materialize_traced_array(src))
    shiftamt = Base.fill_to_length(shiftamt, 0, Val(N))

    for i in 1:N
        amt = shiftamt[i] % size(src, i)
        amt == 0 && continue
        if amt > 0
            src1 = selectdim(src, i, (size(src, i) - amt + 1):size(src, i))
            src2 = selectdim(src, i, 1:(size(src, i) - amt))
        else
            src1 = selectdim(src, i, (-amt + 1):size(src, i))
            src2 = selectdim(src, i, 1:(-amt))
        end
        src = cat(src1, src2; dims=i)
    end

    copyto!(dest, src)
    return dest
end

struct BroadcastIterator{F}
    f::F
end

(fn::BroadcastIterator)(args...) = fn.f((args...,))

function unwrapped_broadcast(f::F, x::Base.Iterators.Zip) where {F}
    min_length = Base.inferencebarrier(minimum)(length, x.is)
    itrs = [length(itr) > min_length ? itr[1:min_length] : itr for itr in x.is]
    any(Base.Fix2(isa, AnyTracedRArray), itrs) || return unrolled_map(f, x)
    return broadcast(BroadcastIterator(f), itrs...)
end

function unwrapped_broadcast(f::F, x::Base.Iterators.Enumerate) where {F}
    x.itr isa AnyTracedRArray || return unrolled_map(f, x)
    return broadcast(
        BroadcastIterator(f), Reactant.promote_to(TracedRArray, 1:length(x.itr)), x.itr
    )
end

unwrapped_broadcast(f::F, xs) where {F} = unrolled_map(f, xs)

# TODO: once traced_call supports internal mutations, we can use traced_call here
# TODO: we should overload this for Slices and use mapslices instead
function unrolled_map(f::F, itr) where {F}
    y = Reactant.call_with_reactant(iterate, itr)
    y === nothing && return []

    first, state = y
    res_first = Reactant.call_with_reactant(f, first)
    result = [res_first]

    while true
        y = Reactant.call_with_reactant(iterate, itr, state)
        y === nothing && break

        val, state = y
        res = Reactant.call_with_reactant(f, val)
        push!(result, res)
    end

    return result
end

end
