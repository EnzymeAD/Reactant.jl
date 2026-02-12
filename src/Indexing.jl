module TracedIndexing

using ..Reactant: Reactant, TracedRArray, TracedRNumber, TracedStepRangeLen, TracedUnitRange
using ..Reactant: AnyTracedRArray, ancestor, unwrapped_eltype
using ..Ops: @opcall
using ..TracedUtils: TracedUtils

using GPUArraysCore: @allowscalar, assertscalar
using ReactantCore: materialize_traced_array
using StructUtils: Selectors

using Base: TwicePrecision

function overloaded_unsafe_getindex end

## Number Indexing
Base.getindex(a::TracedRNumber) = a
Base.getindex(a::TracedRArray{T,0}) where {T} = TracedRNumber{T}((), a.mlir_data)
function Base.getindex(a::TracedRArray{T,0}, ::CartesianIndex{0}) where {T}
    return TracedRNumber{T}((), a.mlir_data)
end

function Base.getindex(r::Union{Base.StepRangeLen,Base.LinRange}, i::TracedRNumber{Int})
    @inline
    i isa TracedRNumber{Bool} && throw(ArgumentError("invalid index: $i of type Bool"))
    # @boundscheck checkbounds(r, i)
    return Base.unsafe_getindex(r, i)
end
function Base.getindex(r::Base.UnitRange, i::I) where {I<:TracedRNumber{Int}}
    val = convert(I, r.start + (i - oneunit(i)))
    # TODO(#2237): we should have error messages at some point.
    # @boundscheck Base._in_unit_range(v, val, i) || throw_boundserror(v, i)
    return val
end

## Array Indexing
### Scalar Indexing
function Base.getindex(
    a::TracedRArray{T,N}, index::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    assertscalar("getindex(::TracedRArray, ::Vararg{Int, N})")
    res = @opcall(reshape(@opcall(dynamic_slice(a, [index...], ones(Int32, N))), Int[]))
    return TracedRNumber{unwrapped_eltype(T)}((), res.mlir_data)
end

function Base.getindex(
    a::TracedRArray{T,N}, index::Union{Int,TracedRNumber{Int}}
) where {T,N}
    return getindex(a, scalar_index_to_cartesian(index, size(a))...)
end

function Base.getindex(a::TracedRArray{T,N}, index::CartesianIndex{N}) where {T,N}
    return getindex_cartesian(a, index)
end
function Base.getindex(a::TracedRArray{T,1}, index::CartesianIndex{1}) where {T}
    return getindex_cartesian(a, index)
end
function Base.getindex(a::TracedRArray, index::CartesianIndex{1})
    return getindex_cartesian(a, index)
end

function Base.getindex(a::Array{<:TracedRNumber,1}, index::CartesianIndex{1})
    return Base.unsafe_getindex(a, index)
end
function Base.getindex(a::Array{<:TracedRNumber,N}, index::CartesianIndex{N}) where {N}
    return Base.unsafe_getindex(a, index)
end

Base.getindex(a::TracedRArray{<:Any,1}, ::Colon) = vec(materialize_traced_array(a))
Base.getindex(a::TracedRArray, ::Colon) = vec(materialize_traced_array(a))

function Base.getindex(a::TracedRArray{T,N}, indices::AbstractArray) where {T,N}
    return getindex_linear(a, indices)
end
function Base.getindex(a::TracedRArray{T,1}, indices::AbstractArray) where {T}
    return getindex_linear(a, indices)
end

function Base.getindex(a::TracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    return getindex_general(a, indices...)
end

### Wrapped Array Types
function Base.getindex(
    a::AnyTracedRArray{T,N}, index::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    ancestor, idxs = TracedUtils.get_ancestor_and_indices(a, index...)
    return getindex(ancestor, idxs...)
end

# This method is needed exclusively to resolve an ambiguity
function Base.getindex(
    l::Selectors.List{TracedRNumber{T}}, index::Union{Int,TracedRNumber{Int}}
) where {T}
    return getindex(Reactant.promote_to(TracedRArray{T,1}, l), index)
end

function Base.getindex(
    a::AbstractRange{TracedRNumber{T}}, index::Union{Int,TracedRNumber{Int}}
) where {T}
    return getindex(Reactant.promote_to(TracedRArray{T,1}, a), index)
end

function Base.getindex(a::AnyTracedRArray{T,N}, linear_indices) where {T,N}
    ancestor, idxs = TracedUtils.get_ancestor_and_indices(a, linear_indices)
    return getindex(ancestor, idxs...)
end

function Base.getindex(a::AnyTracedRArray{T,1}, indices) where {T}
    ancestor, idxs = TracedUtils.get_ancestor_and_indices(a, indices)
    return getindex(ancestor, idxs...)
end

function Base.getindex(a::AnyTracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    ancestor, idxs = TracedUtils.get_ancestor_and_indices(a, indices...)
    return getindex(ancestor, idxs...)
end

### Specialize certain dispatches for better codegen
for aType in (
    Base.ReshapedArray{TracedRNumber{T}} where {T},
    PermutedDimsArray{TracedRNumber{T}} where {T},
)
    @eval begin
        function Base.getindex(a::$(aType), indices::Union{Int,TracedRNumber{Int}}...)
            return getindex(materialize_traced_array(a), indices...)
        end

        function Base.getindex(a::$(aType), indices...)
            return getindex(materialize_traced_array(a), indices...)
        end
    end
end

for aType in (
    Base.ReshapedArray{TracedRNumber{T},N,P,Tuple{}} where {T,N,P<:AbstractArray},
    Base.ReshapedArray{TracedRNumber{T},1,P,Tuple{}} where {T,P<:AbstractArray},
)
    @eval function Base.getindex(a::$(aType), indices::Int)
        return getindex(materialize_traced_array(a), indices)
    end
end

function Base.getindex(
    x::Base.ReshapedArray{TracedRNumber{T}}, index::Base.ReshapedIndex
) where {T}
    return getindex(parent(x), index.parentindex)
end

function Base.getindex(
    x::Base.Sort.WithoutMissingVector{TracedRNumber{T}}, i::Int
) where {T}
    out = getindex(x.data, i)
    @assert !(out isa Missing)
    return out
end

function Base.getindex(x::Base.OneTo{TracedRNumber{T}}, i::Int) where {T}
    return @allowscalar getindex(Reactant.promote_to(TracedRNumber{T}, x), i)
end

function Base.getindex(
    x::Union{LinRange{TracedRNumber{T}},StepRangeLen{TracedRNumber{T}}}, i::Int
) where {T}
    return @allowscalar getindex(Reactant.promote_to(TracedRNumber{T}, x), i)
end

function Base.getindex(x::Base.UnitRange{TracedRNumber{T}}, i::Int) where {T}
    return @allowscalar getindex(Reactant.promote_to(TracedRNumber{T}, x), i)
end

## StepRangeLen Indexing
function Base.getindex(r::TracedStepRangeLen{T}, s::OrdinalRange{S}) where {T,S}
    @inline
    @boundscheck checkbounds(r, s)

    len = length(s)
    sstep = Base.step_hp(s)
    rstep = Base.step_hp(r)
    L = typeof(len)
    if S === Bool
        rstep *= one(sstep)
        if len == 0
            return TracedStepRangeLen{T}(first(r), rstep, zero(L), oneunit(L))
        elseif len == 1
            if first(s)
                return TracedStepRangeLen{T}(first(r), rstep, oneunit(L), oneunit(L))
            else
                return TracedStepRangeLen{T}(first(r), rstep, zero(L), oneunit(L))
            end
        else # len == 2
            return TracedStepRangeLen{T}(last(r), rstep, oneunit(L), oneunit(L))
        end
    else
        # Find closest approach to offset by s
        ind = LinearIndices(s)
        offset = L(
            max(min(1 + round(L, (r.offset - first(s)) / sstep), last(ind)), first(ind))
        )
        ref = Base._getindex_hiprec(r, first(s) + (offset - oneunit(offset)) * sstep)
        return TracedStepRangeLen{T}(ref, rstep * sstep, len, offset)
    end
end

function overloaded_unsafe_getindex(
    v::TracedStepRangeLen{T}, i::Union{I,TracedRNumber{I}}
) where {T,I}
    finalT = T
    offsetT = typeof(v.offset)
    if i isa TracedRNumber
        if !(T <: TracedRNumber)
            finalT = TracedRNumber{T}
        end
        if !(v.offset isa TracedRNumber)
            offsetT = TracedRNumber{offsetT}
        end
    end
    return finalT(v.ref + (convert(offsetT, i) - v.offset) * v.step)
end

function overloaded_unsafe_getindex(
    r::Union{
        Base.StepRangeLen{T,<:TwicePrecision,<:TwicePrecision},
        TracedStepRangeLen{T,<:TwicePrecision,<:TwicePrecision,<:TwicePrecision},
    },
    i::TracedRNumber{Int},
) where {T}
    # Very similar to _getindex_hiprec, but optimized to avoid a 2nd call to add12
    @inline
    i isa TracedRNumber{Bool} && throw(ArgumentError("invalid index: $i of type Bool"))
    OT = TracedRNumber{unwrapped_eltype(r.offset)}
    u = Base.convert(OT, i)::OT - r.offset
    shift_hi, shift_lo = u * r.step.hi, u * r.step.lo
    x_hi, x_lo = Base.add12(r.ref.hi, shift_hi)
    T2 = TracedRNumber{unwrapped_eltype(T)}
    return T2(x_hi + (x_lo + (shift_lo + r.ref.lo)))
end

function Base.getindex(r::TracedStepRangeLen, i::TracedRNumber{Int})
    return Base.unsafe_getindex(r, i)
end

# This assumes that r.step has already been split so that (0:len-1)*r.step.hi is exact
function Base.unsafe_getindex(r::TracedStepRangeLen, i::Integer)
    return overloaded_unsafe_getindex(r, i)
end
function Base.unsafe_getindex(r::TracedStepRangeLen, i::TracedRNumber{Int})
    return overloaded_unsafe_getindex(r, i)
end

function Base.unsafe_getindex(
    r::Union{
        Base.StepRangeLen{T,<:TwicePrecision,<:TwicePrecision},
        TracedStepRangeLen{T,<:TwicePrecision,<:TwicePrecision,<:TwicePrecision},
    },
    i::TracedRNumber{Int},
) where {T}
    return overloaded_unsafe_getindex(r, i)
end

function Base._getindex_hiprec(r::TracedStepRangeLen, i::Integer)  # without rounding by T
    u = oftype(r.offset, i) - r.offset
    return r.ref + u * r.step
end

## UnitRange Indexing
function Base.getindex(v::TracedUnitRange{T}, i::CartesianIndex{1}) where {T}
    return getindex(v, i.I...)
end

Base.getindex(v::TracedUnitRange, ::Colon) = v

for iType in (Int, TracedRNumber{Int}, Integer)
    @eval function Base.getindex(v::TracedUnitRange{T}, i::$iType) where {T}
        return convert(T, v.start + (i - oneunit(i)))
        # TODO(#2237): we should have error messages at some point.
        # @boundscheck Base._in_unit_range(v, val, i) || throw_boundserror(v, i)
    end
end

# TODO(#2237): some of these dispatches can be optimized
for idxtype in (AbstractArray, AbstractUnitRange{<:Integer}, StepRange{<:Integer})
    @eval function Base.getindex(v::TracedUnitRange{T}, i::$idxtype) where {T}
        return getindex(Reactant.promote_to(TracedRArray{T,1}, v), i)
    end
end

# setindex!
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
    return _setindex_scalar_cartesian!(a, v, index)
end
function Base.setindex!(a::TracedRArray{T,1}, v, index::CartesianIndex{1}) where {T}
    return _setindex_scalar_cartesian!(a, v, index)
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
        TracedUtils.set_mlir_data!(a, TracedUtils.get_mlir_data(v))
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
        # TODO(#2237): This will create a dynamically sized tensor and we need to implement
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
                (_, xⱼ) -> xⱼ,
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
        TracedUtils.set_mlir_data!(a, TracedUtils.get_mlir_data(res))
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

    TracedUtils.set_mlir_data!(
        a,
        @opcall(
            dynamic_update_slice(a, v, [i isa Colon ? 1 : first(i) for i in indices])
        ).mlir_data,
    )
    return v
end

# common helper methods
function generate_index_list(i1, is...)
    list = reshape(i1, :, 1)
    for i in is
        i = Reactant.broadcast_to_size(i, (length(i), 1))
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
    idx = @opcall(subtract(idx, @opcall(fill(T(1), size(idx)))))
    idxs = materialize_traced_array(
        reshape(@opcall(remainder(idx, @opcall(fill(T(sz[1]), size(idx))))), :, 1)
    )
    idx = @opcall(divide(idx, @opcall(fill(T(sz[1]), size(idx)))))
    for i in 2:N
        idxs = hcat(idxs, @opcall(remainder(idx, @opcall(fill(T(sz[i]), size(idx))))))
        idx = @opcall(divide(idx, @opcall(fill(T(sz[i]), size(idx)))))
    end
    return @opcall(add(idxs, @opcall(fill(T(1), size(idxs)))))
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

function getindex_cartesian(a::TracedRArray{T,N}, index::CartesianIndex{N}) where {T,N}
    assertscalar("getindex(::TracedRArray, ::CartesianIndex{N})")
    return TracedRNumber{unwrapped_eltype(T)}(
        (),
        @opcall(
            reshape(
                @opcall(dynamic_slice(a, collect(Int64, index.I), ones(Int32, N))), Int64[]
            )
        ).mlir_data,
    )
end

get_slice_stride(::Base.LogicalIndex) = -1
get_slice_stride(x::CartesianIndex) = -1
function get_slice_stride(x)
    length(x) == 1 && return 1
    strides = diff(x)
    isempty(strides) && return -1
    allequal(strides) || return -1
    val = first(strides)
    val isa Number || return -1
    return val
end

function getindex_linear(a::TracedRArray{T,N}, indices::AbstractArray) where {T,N}
    if !(indices isa Reactant.TracedType)
        if length(indices) == 1 && first(indices) isa CartesianIndex
            # fast-path else we will end up with a gather
            return Reactant.broadcast_to_size(
                @allowscalar(getindex_cartesian(a, first(indices))), (1,)
            )
        end
        stride = get_slice_stride(vec(indices))
        if stride > 0
            a_flat = materialize_traced_array(vec(a))
            indices_flat = vec(indices)
            return @opcall(
                reshape(
                    @opcall(
                        slice(
                            a_flat,
                            Int64[first(indices_flat)],
                            Int64[last(indices_flat)];
                            strides=Int64[stride],
                        )
                    ),
                    collect(Int64, size(indices)),
                )
            )
        end
    end

    if !(indices isa TracedRArray)
        indices = collect(indices)
        eltype(indices) <: CartesianIndex && (indices = LinearIndices(size(a))[indices])
        indices = Reactant.promote_to(TracedRArray{Int}, indices)
    end
    return @opcall(
        reshape(
            @opcall(gather_getindex(a, scalar_index_to_cartesian(vec(indices), size(a)))),
            collect(size(indices)),
        )
    )
end

function getindex_general(a::TracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    indices = Base.to_indices(a, indices)

    use_gather_getindex = false
    use_dynamic_slice = false
    strides = Int64[]
    for idxs in indices
        if idxs isa Number
            idxs isa TracedRNumber && (use_dynamic_slice = true)
            push!(strides, 1)
            continue
        end
        if idxs isa Reactant.TracedType
            use_gather_getindex = true
            break
        end
        stride = get_slice_stride(vec(idxs))
        push!(strides, stride)
        if stride ≤ 0 || (use_dynamic_slice && stride != 1)
            use_gather_getindex = true
            break
        end
    end

    if use_gather_getindex
        # TODO(#2237): This will create a dynamically sized tensor and we need to implement
        #       `findall` for it.
        if any(i -> unwrapped_eltype(i) <: Bool, indices)
            error("Boolean indexing with TracedRArrays isn't fully supported yet.")
        end

        gather_dims = TracedUtils.indices_to_gather_dims(indices...)

        return @opcall(
            reshape(
                @opcall(
                    transpose(
                        @opcall(
                            reshape(
                                @opcall(
                                    gather(
                                        a,
                                        gather_dims.start_indices;
                                        gather_dims.offset_dims,
                                        gather_dims.collapsed_slice_dims,
                                        operand_batching_dims=Int64[],
                                        start_indices_batching_dims=Int64[],
                                        gather_dims.start_index_map,
                                        gather_dims.index_vector_dim,
                                        gather_dims.slice_sizes,
                                    )
                                ),
                                gather_dims.gather_reshape_shape,
                            )
                        ),
                        gather_dims.permutation,
                    )
                ),
                gather_dims.result_shape,
            )
        )
    end

    if use_dynamic_slice
        @assert all(isone, strides) "This should not happen, please report a bug"
        x = @opcall dynamic_slice(a, [first.(indices)...], [length.(indices)...])
    else
        x = @opcall slice(a, [first.(indices)...], [last.(indices)...]; strides)
    end

    ddims = findall(indices) do idx
        return idx isa Integer || idx isa TracedRNumber{Int}
    end
    isempty(ddims) || return materialize_traced_array(dropdims(x; dims=Tuple(ddims)))
    return x
end

function overloaded_unsafe_getindex(
    ::IndexLinear, a::Array{T,N}, idxs::Vararg{Any,N}
) where {T,N}
    return Base.unsafe_getindex(@opcall(constant(a)), idxs...)
end

function maybe_assert_scalar_setindexing(
    ::TracedRArray{T,N}, ::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    assertscalar("setindex!(::TracedRArray, v, ::Vararg{Int, N})")
    return nothing
end

maybe_assert_scalar_setindexing(_args...) = nothing

function _setindex_scalar!(
    a::TracedRArray{T,N}, v, index::Union{Int,TracedRNumber{Int}}
) where {T,N}
    assertscalar("setindex!(::TracedRArray, v, ::Union{Int, TracedRNumber{Int}})")

    res = @opcall(
        reshape(
            @opcall(
                dynamic_update_slice(
                    a,
                    Reactant.broadcast_to_size(
                        Reactant.promote_to(TracedRNumber{T}, v), ntuple(Returns(1), N)
                    ),
                    collect(scalar_index_to_cartesian(index, size(a))),
                )
            ),
            collect(size(a)),
        )
    )
    TracedUtils.set_mlir_data!(a, TracedUtils.get_mlir_data(res))
    return a
end

function _setindex_scalar_cartesian!(
    a::TracedRArray{T,N}, v, index::CartesianIndex{N}
) where {T,N}
    assertscalar("setindex!(::TracedRArray, v, ::CartesianIndex{N})")
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
    TracedUtils.set_mlir_data!(a, TracedUtils.get_mlir_data(res))
    return a
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
        TracedUtils.set_mlir_data!(a, TracedUtils.get_mlir_data(res))
        return a
    end

    if !(indices isa TracedRArray)
        indices = collect(indices)
        eltype(indices) <: CartesianIndex && (indices = LinearIndices(size(a))[indices])
        indices = Reactant.promote_to(TracedRArray{Int,ndims(indices)}, indices)
    end
    res = @opcall scatter_setindex(
        a,
        scalar_index_to_cartesian(vec(indices), size(a)),
        Reactant.promote_to(TracedRArray{T,1}, materialize_traced_array(vec(v))),
    )
    TracedUtils.set_mlir_data!(a, TracedUtils.get_mlir_data(res))
    return a
end

end
