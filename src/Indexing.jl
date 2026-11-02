module TracedIndexing

using ..Reactant: Reactant, TracedRArray, TracedRNumber, TracedStepRangeLen, TracedUnitRange
using ..Reactant: AnyTracedRArray, ancestor, unwrapped_eltype
using ..Ops: @opcall
using ..TracedUtils: TracedUtils

using GPUArraysCore: @allowscalar, assertscalar
using ReactantCore: ReactantCore, materialize_traced_array
using StructUtils: Selectors

using Base: TwicePrecision

function overloaded_unsafe_getindex end

## Number Indexing
Base.getindex(a::TracedRNumber) = a
Base.getindex(a::TracedRArray{T,0}) where {T} = TracedRNumber{T}((), a.mlir_data)
function Base.getindex(a::TracedRArray{T,0}, ::CartesianIndex{0}) where {T}
    return TracedRNumber{T}((), a.mlir_data)
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
    l::Selectors.List{<:TracedRNumber{T}}, index::Union{Int,TracedRNumber{Int}}
) where {T}
    return getindex(Reactant.promote_to(TracedRArray{T,1}, l), index)
end

for (aT, iT) in (
    (AbstractRange, TracedRNumber{<:Integer}),
    (AbstractRange{<:TracedRNumber}, Int),
    (AbstractRange{<:TracedRNumber}, TracedRNumber{<:Integer}),
    (AbstractRange{<:TracedRNumber}, TracedRNumber{Int}),
    # 1.10 specific ambiguity fixes
    (UnitRange{<:TracedRNumber}, Int),
)
    @eval function Base.getindex(a::$aT, index::$iT)
        return convert(
            TracedRNumber{Reactant.unwrapped_eltype(a)},
            first(a) + (index - oneunit(index)) * Base.step_hp(a),
        )
    end
end

for (aT, iT) in (
    (Base.OneTo, TracedRNumber{<:Integer}),
    (Base.OneTo{<:TracedRNumber}, Int),
    (Base.OneTo{<:TracedRNumber}, TracedRNumber{<:Integer}),
    (Base.OneTo{<:TracedRNumber}, TracedRNumber{Int}),
)
    @eval function Base.getindex(a::$aT, index::$iT)
        return convert(TracedRNumber{Reactant.unwrapped_eltype(a)}, index)
    end
end

for (aT, iT) in (
    (StepRangeLen, TracedRNumber{<:Integer}),
    (StepRangeLen{<:TracedRNumber}, Int),
    (StepRangeLen{<:TracedRNumber}, TracedRNumber{<:Integer}),
    (StepRangeLen{<:TracedRNumber}, TracedRNumber{Int}),
)
    @eval function Base.getindex(r::$aT, index::$iT)
        # FIXME: this crashes for some reason
        # u = convert(TracedRNumber{Reactant.unwrapped_eltype(r)}, index) - r.offset
        # return convert(TracedRNumber{Reactant.unwrapped_eltype(r)}, r.ref + u * r.step)
        return @allowscalar getindex(
            Reactant.promote_to(TracedRArray{Reactant.unwrapped_eltype(r),1}, r), index
        )
    end
end

for (aT, iT) in (
    (LinRange, TracedRNumber{<:Integer}),
    (LinRange{<:TracedRNumber}, Int),
    (LinRange{<:TracedRNumber}, TracedRNumber{<:Integer}),
    (LinRange{<:TracedRNumber}, TracedRNumber{Int}),
)
    @eval function Base.getindex(r::$aT, index::$iT)
        # `Base.lerpi`, which takes an `Integer` index
        t = (index - oneunit(index)) / r.lendiv
        return convert(
            TracedRNumber{Reactant.unwrapped_eltype(r)}, (1 - t) * r.start + t * r.stop
        )
    end
end

# v1.10 specific ambiguity fixes
function Base.getindex(
    a::Union{LinRange{TracedRNumber{T}},StepRangeLen{TracedRNumber{T}}}, index::Int
) where {T}
    return getindex(Reactant.promote_to(TracedRArray{T,1}, a), index)
end

function Base.getindex(
    a::Union{
        LinRange{T} where T<:Reactant.TracedRNumber,
        StepRangeLen{T} where T<:Reactant.TracedRNumber,
    },
    index::Int,
)
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
for aType in (Base.ReshapedArray{<:TracedRNumber}, PermutedDimsArray{<:TracedRNumber})
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
    Base.ReshapedArray{<:TracedRNumber,<:Any,<:AbstractArray,Tuple{}},
    Base.ReshapedArray{<:TracedRNumber,1,<:AbstractArray,Tuple{}},
)
    @eval function Base.getindex(a::$(aType), indices::Int)
        return getindex(materialize_traced_array(a), indices)
    end
end

function Base.getindex(x::Base.ReshapedArray{<:TracedRNumber}, index::Base.ReshapedIndex)
    return getindex(parent(x), index.parentindex)
end

function Base.getindex(
    x::Base.Sort.WithoutMissingVector{TracedRNumber{T}}, i::Int
) where {T}
    out = getindex(x.data, i)
    @assert !(out isa Missing)
    return out
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
    idx::AbstractVector{<:TracedRNumber{T}}, sz::NTuple{N,Int}
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
    if !(indices isa Reactant.TracedType || eltype(indices) <: TracedRNumber)
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

    if !(indices isa TracedRArray || eltype(indices) <: TracedRNumber)
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

_is_scalar_indexing(idxs::Tuple) = all(i -> i isa Union{Int,TracedRNumber{Int}}, idxs)

function _nonscalar_output_size(idxs::Tuple)
    dims = Int[]
    for idx in idxs
        if !(idx isa Union{Int,TracedRNumber{Int}})
            append!(dims, size(idx))
        end
    end
    return Tuple(dims)
end

function _normalize_jl_array_indices(idxs::Tuple)
    return map(idxs) do idx
        if idx isa TracedRNumber{Bool} || idx isa Bool
            throw(ArgumentError("invalid index: $idx of type Bool"))
        elseif idx isa TracedRNumber{<:Integer}
            return convert(TracedRNumber{Int}, idx)
        elseif idx isa Integer
            return Int(idx)
        else
            return idx
        end
    end
end

function _traced_numbers_to_rarray(a::Array, ::Type{S}) where {S}
    isempty(a) && return @opcall(fill(zero(S), collect(Int, size(a))))
    inputs = Vector{TracedRArray{S,1}}(undef, length(a))
    for (k, idx) in enumerate(eachindex(a))
        v = if isassigned(a, idx)
            Reactant.promote_to(TracedRNumber{S}, a[idx])
        else
            Reactant.promote_to(TracedRNumber{S}, zero(S))
        end
        inputs[k] = @opcall reshape(TracedRArray{S,0}((), v.mlir_data, ()), Int[1])
    end
    flat = length(inputs) == 1 ? inputs[1] : @opcall(concatenate(inputs, 1))
    return @opcall reshape(flat, collect(Int, size(a)))
end

function _index_array_of_traced_rarrays(a::Array, norm_idxs::Tuple, is_scalar::Bool)
    x1 = first(a)
    K = ndims(x1)
    inner_sz = size(x1)
    for x in a
        if ndims(x) != K || size(x) != inner_sz
            throw(
                ArgumentError(
                    "cannot index into an array of TracedRArrays with non-uniform shapes"
                ),
            )
        end
    end
    S = mapreduce(unwrapped_eltype, promote_type, a)
    if !(isconcretetype(S) && S <: Reactant.ReactantPrimitive)
        throw(
            ArgumentError(
                "cannot index into an array of TracedRArrays with element type $S"
            ),
        )
    end
    inputs = Vector{TracedRArray{S,K + 1}}(undef, length(a))
    reshaped_inner = Int[1, inner_sz...]
    for (k, idx) in enumerate(eachindex(a))
        xk = a[idx]
        xk_conv = unwrapped_eltype(xk) === S ? xk : @opcall(convert(TracedRArray{S,K}, xk))
        inputs[k] = @opcall reshape(xk_conv, reshaped_inner)
    end
    flat = length(inputs) == 1 ? inputs[1] : @opcall(concatenate(inputs, 1))
    if K == 0
        stacked0 = @opcall reshape(flat, collect(Int, size(a)))
        sliced0 = @allowscalar Base.unsafe_getindex(stacked0, norm_idxs...)
        is_scalar && return TracedRArray{S,0}((), sliced0.mlir_data, ())
        out_sz = _nonscalar_output_size(norm_idxs)
        res0 = Array{TracedRArray{S,0},length(out_sz)}(undef, out_sz)
        for I in CartesianIndices(out_sz)
            res0[I] = TracedRArray{S,0}((), (@allowscalar sliced0[I]).mlir_data, ())
        end
        return res0
    end
    stacked = @opcall reshape(flat, Int[size(a)..., inner_sz...])
    colons = ntuple(Returns(Colon()), K)
    sliced = @allowscalar Base.unsafe_getindex(stacked, norm_idxs..., colons...)
    is_scalar && return sliced
    out_sz = _nonscalar_output_size(norm_idxs)
    res = Array{TracedRArray{S,K},length(out_sz)}(undef, out_sz)
    for I in CartesianIndices(out_sz)
        res[I] = @allowscalar sliced[Tuple(I)..., colons...]
    end
    return res
end

function _has_typevar(@nospecialize(t))
    tvs = Base.IdSet{TypeVar}()
    Reactant.collect_tvars_in_type!(tvs, t)
    return !isempty(tvs)
end

function _resolve_indexed_struct_type(ET1::DataType, flds::Vector{Any}, nf_def::Int)
    all(f -> flds[f] isa fieldtype(ET1, f), 1:nf_def) && return ET1

    TT_traced = try
        Reactant.traced_type_inner(
            ET1, Dict{Type,Type}(), Reactant.NoStopTracedTrack, Number, Val(1), nothing
        )
    catch
        nothing
    end
    if TT_traced isa DataType &&
        isconcretetype(TT_traced) &&
        fieldcount(TT_traced) == fieldcount(ET1) &&
        all(f -> flds[f] isa fieldtype(TT_traced, f), 1:nf_def)
        return TT_traced
    end

    wrapper = ET1.name.wrapper
    wrapper isa UnionAll || return nothing
    body = Base.unwrap_unionall(wrapper)
    ft_tuple = Base.rewrap_unionall(Tuple{body.types[1:nf_def]...}, wrapper)
    actual = Tuple{map(Core.Typeof, flds)...}
    sz = @ccall jl_subtype_env_size(ft_tuple::Any)::Cint
    env = Array{Any}(undef, sz)
    ok =
        (@ccall jl_subtype_env(
            actual::Any, ft_tuple::Any, env::Ptr{Any}, sz::Cint
        )::Cint) == 1
    ok || return nothing

    tvar_vals = IdDict{TypeVar,Any}()
    cur_ft = ft_tuple
    for v in env
        if cur_ft isa UnionAll
            tvar_vals[cur_ft.var] = v
            cur_ft = cur_ft.body
        end
    end
    params = Any[]
    cur = wrapper
    idx = 1
    while cur isa UnionAll
        tv = cur.var
        val = get(tvar_vals, tv, tv)
        if val isa TypeVar
            val = ET1.parameters[idx]
        end
        push!(params, val)
        cur = cur.body
        idx += 1
    end
    TT = try
        first(Reactant.apply_type_with_promotion(wrapper, params))
    catch
        try
            Core.apply_type(wrapper, params...)
        catch
            return nothing
        end
    end
    if TT isa DataType &&
        isconcretetype(TT) &&
        fieldcount(TT) == fieldcount(ET1) &&
        all(f -> flds[f] isa fieldtype(TT, f), 1:nf_def)
        return TT
    end
    return nothing
end

function _construct_indexed_struct(ET1::DataType, flds::Vector{Any}, nf_def::Int)
    if ET1 <: Tuple
        return Tuple(flds)
    elseif ET1 <: NamedTuple
        return NamedTuple{ET1.parameters[1]}(Tuple(flds))
    end
    TT = _resolve_indexed_struct_type(ET1, flds, nf_def)
    if isnothing(TT)
        throw(
            ArgumentError(
                "cannot index into an array of $ET1 with traced indices: " *
                "indexed field types $(Tuple(map(Core.Typeof, flds))) are incompatible with $ET1",
            ),
        )
    end
    if ismutabletype(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        for f in 1:nf_def
            ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, f - 1, flds[f])
        end
        return y
    end
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), TT, flds, nf_def)
end

function _index_julia_array(a::Array{T,N}, norm_idxs...) where {T,N}
    M = length(norm_idxs)
    if M == 1 && N != 1
        return _index_julia_array(vec(a), norm_idxs[1])
    elseif 1 < M < N
        collapsed_sz = (size(a)[1:(M - 1)]..., prod(size(a)[M:N]))
        return _index_julia_array(reshape(a, collapsed_sz), norm_idxs...)
    elseif M > N
        padded_sz = (size(a)..., ntuple(Returns(1), M - N)...)
        return _index_julia_array(reshape(a, padded_sz), norm_idxs...)
    end

    is_scalar = _is_scalar_indexing(norm_idxs)
    if isempty(a)
        is_scalar && throw(BoundsError(a, norm_idxs))
        out_sz = _nonscalar_output_size(norm_idxs)
        prod(out_sz) > 0 && throw(BoundsError(a, norm_idxs))
        if T <: Reactant.ReactantPrimitive && !(T <: Complex{<:Integer})
            return @opcall fill(zero(T), collect(Int, out_sz))
        elseif T <: TracedRNumber && isconcretetype(T)
            S = unwrapped_eltype(T)
            if S <: Reactant.ReactantPrimitive && !(S <: Complex{<:Integer})
                return @opcall fill(zero(S), collect(Int, out_sz))
            end
        end
        TT = try
            Reactant.traced_type_inner(
                T,
                Dict{Type,Type}(),
                Reactant.NoStopTracedTrack,
                Number,
                Val(1),
                nothing,
            )
        catch
            T
        end
        return Array{TT,length(out_sz)}(undef, out_sz)
    end

    if T <: Reactant.ReactantPrimitive && !(T <: Complex{<:Integer})
        return @allowscalar Base.unsafe_getindex(@opcall(constant(a)), norm_idxs...)
    end

    if T <: TracedRNumber && isconcretetype(T)
        S = unwrapped_eltype(T)
        if S <: Reactant.ReactantPrimitive && !(S <: Complex{<:Integer})
            ra = _traced_numbers_to_rarray(a, S)
            return @allowscalar Base.unsafe_getindex(ra, norm_idxs...)
        end
    end

    for k in eachindex(a)
        isassigned(a, k) || throw(
            ArgumentError(
                "cannot index into an array with undefined elements with traced indices"
            ),
        )
    end

    if T <: Base.Enum && isconcretetype(T)
        BT = Reactant.enum_basetype(T)
        int_res = _index_julia_array(map(BT, a), norm_idxs...)
        is_scalar && return Reactant.TracedEnum{T}(int_res)
        out_sz = _nonscalar_output_size(norm_idxs)
        res_enum = Array{Reactant.TracedEnum{T},length(out_sz)}(undef, out_sz)
        for I in CartesianIndices(out_sz)
            res_enum[I] = Reactant.TracedEnum{T}(@allowscalar(int_res[I]))
        end
        return res_enum
    end

    if all(
        x ->
            x isa Union{Reactant.ReactantPrimitive,TracedRNumber} &&
                !(unwrapped_eltype(x) <: Complex{<:Integer}),
        a,
    )
        S = mapreduce(unwrapped_eltype, promote_type, a)
        if isconcretetype(S) && S <: Reactant.ReactantPrimitive
            if !any(x -> x isa TracedRNumber, a)
                return @allowscalar Base.unsafe_getindex(
                    @opcall(constant(map(S, a))), norm_idxs...
                )
            end
            ra = _traced_numbers_to_rarray(a, S)
            return @allowscalar Base.unsafe_getindex(ra, norm_idxs...)
        end
    end

    if all(x -> x isa TracedRArray, a)
        return _index_array_of_traced_rarrays(a, norm_idxs, is_scalar)
    end

    if all(x -> x isa Array, a)
        x1_arr = first(a)
        K = ndims(x1_arr)
        inner_sz = size(x1_arr)
        for x in a
            if ndims(x) != K || size(x) != inner_sz
                throw(
                    ArgumentError(
                        "cannot index into an array of Arrays with non-uniform shapes"
                    ),
                )
            end
        end
        if isempty(x1_arr)
            ET = eltype(x1_arr)
            if all(x -> eltype(x) === ET, a)
                TT = try
                    Reactant.traced_type_inner(
                        ET,
                        Dict{Type,Type}(),
                        Reactant.NoStopTracedTrack,
                        Number,
                        Val(1),
                        nothing,
                    )
                catch
                    ET
                end
                is_scalar && return Array{TT,K}(undef, inner_sz)
                out_sz = _nonscalar_output_size(norm_idxs)
                res_empty = Array{Array{TT,K},length(out_sz)}(undef, out_sz)
                for I in CartesianIndices(out_sz)
                    res_empty[I] = Array{TT,K}(undef, inner_sz)
                end
                return res_empty
            end
        end
        indexed_entries = [
            _index_julia_array([a[I][J] for I in CartesianIndices(a)], norm_idxs...) for
            J in CartesianIndices(inner_sz)
        ]
        if is_scalar
            first_entry = first(indexed_entries)
            if all(e -> Core.Typeof(e) === Core.Typeof(first_entry), indexed_entries)
                res_s = Array{Core.Typeof(first_entry),K}(undef, inner_sz)
                for J in CartesianIndices(inner_sz)
                    res_s[J] = indexed_entries[J]
                end
                return res_s
            end
            return indexed_entries
        end
        out_sz = _nonscalar_output_size(norm_idxs)
        if prod(out_sz) == 0
            return Array{Array,length(out_sz)}(undef, out_sz)
        end
        I1 = first(CartesianIndices(out_sz))
        sample0 = @allowscalar indexed_entries[first(CartesianIndices(inner_sz))][I1]
        same_inner_ty = all(
            J -> Core.Typeof(@allowscalar(indexed_entries[J][I1])) === Core.Typeof(sample0),
            CartesianIndices(inner_sz),
        )
        InnerArrT = same_inner_ty ? Array{Core.Typeof(sample0),K} : Array{Any,K}
        res_ns = Array{InnerArrT,length(out_sz)}(undef, out_sz)
        for I in CartesianIndices(out_sz)
            inner = InnerArrT(undef, inner_sz)
            for J in CartesianIndices(inner_sz)
                inner[J] = @allowscalar indexed_entries[J][I]
            end
            res_ns[I] = inner
        end
        return res_ns
    end

    x1 = first(a)
    if x1 isa Union{Symbol,AbstractString,AbstractChar,Type,Module,VersionNumber}
        if all(x -> x == x1, a)
            is_scalar && return x1
            out_sz = _nonscalar_output_size(norm_idxs)
            return fill(x1, out_sz)
        end
        throw(
            ArgumentError(
                "cannot index into an array with element type $T with traced indices: " *
                "non-uniform values of type $(typeof(x1)) are not supported",
            ),
        )
    end

    ET1 = Core.Typeof(x1)
    if !(ET1 isa DataType) || isprimitivetype(ET1)
        throw(
            ArgumentError(
                "cannot index into an array with element type $T with traced indices"
            ),
        )
    end
    for x in a
        ET = Core.Typeof(x)
        compatible = if ET === ET1
            true
        elseif ET1 <: Tuple && ET <: Tuple
            fieldcount(ET) == fieldcount(ET1)
        elseif ET1 <: NamedTuple && ET <: NamedTuple
            ET.parameters[1] === ET1.parameters[1]
        elseif ET isa DataType && !isprimitivetype(ET)
            ET.name.wrapper === ET1.name.wrapper && fieldcount(ET) == fieldcount(ET1)
        else
            false
        end
        compatible || throw(
            ArgumentError(
                "cannot index into an array with element type $T with traced indices: " *
                "incompatible element types $ET1 and $ET",
            ),
        )
    end

    nf = fieldcount(ET1)
    if nf == 0
        if is_scalar
            return ismutabletype(ET1) ? ccall(:jl_new_struct_uninit, Any, (Any,), ET1) : x1
        end
        out_sz = _nonscalar_output_size(norm_idxs)
        res_nf0 = Array{ET1,length(out_sz)}(undef, out_sz)
        for I in CartesianIndices(out_sz)
            res_nf0[I] =
                ismutabletype(ET1) ? ccall(:jl_new_struct_uninit, Any, (Any,), ET1) : x1
        end
        return res_nf0
    end

    nf_def = nf
    for f in 1:nf
        def1 = isdefined(x1, f)
        for x in a
            if isdefined(x, f) != def1
                throw(
                    ArgumentError(
                        "cannot index into an array of $ET1 with inconsistently defined field $f",
                    ),
                )
            end
        end
        if !def1
            for f2 in (f + 1):nf, x in a
                isdefined(x, f2) && throw(
                    ArgumentError(
                        "cannot index into an array of $ET1 with non-trailing undefined field $f",
                    ),
                )
            end
            nf_def = f - 1
            break
        end
    end

    generic_body =
        ET1 <: Union{Tuple,NamedTuple} ? nothing : Base.unwrap_unionall(ET1.name.wrapper)
    sliced_fields = Vector{Any}(undef, nf_def)
    for f in 1:nf_def
        f_arr = [getfield(a[I], f) for I in CartesianIndices(a)]
        fixed_const = false
        if !isnothing(generic_body)
            def_ft = fieldtype(generic_body, f)
            if !_has_typevar(def_ft) &&
                !any(ReactantCore.is_traced, f_arr) &&
                allequal(f_arr) &&
                first(f_arr) isa def_ft
                # Check if tracing this field would produce a type incompatible with def_ft
                f1 = first(f_arr)
                if f1 isa Reactant.ReactantPrimitive &&
                    !(TracedRNumber{typeof(f1)} <: def_ft)
                    fixed_const = true
                end
            end
        end
        if fixed_const
            sliced_fields[f] = if is_scalar
                first(f_arr)
            else
                fill(first(f_arr), _nonscalar_output_size(norm_idxs))
            end
        else
            sliced_fields[f] = _index_julia_array(f_arr, norm_idxs...)
        end
    end

    if is_scalar
        return _construct_indexed_struct(ET1, sliced_fields, nf_def)
    end

    out_sz = _nonscalar_output_size(norm_idxs)
    if prod(out_sz) == 0
        TT_empty = try
            Reactant.traced_type_inner(
                ET1,
                Dict{Type,Type}(),
                Reactant.NoStopTracedTrack,
                Number,
                Val(1),
                nothing,
            )
        catch
            ET1
        end
        return Array{TT_empty,length(out_sz)}(undef, out_sz)
    end

    cart = CartesianIndices(out_sz)
    I_first = first(cart)
    flds_first = Any[@allowscalar(sliced_fields[f][I_first]) for f in 1:nf_def]
    elem_first = _construct_indexed_struct(ET1, flds_first, nf_def)
    res_struct = Array{Core.Typeof(elem_first),length(out_sz)}(undef, out_sz)
    for I in cart
        if I == I_first
            res_struct[I] = elem_first
        else
            flds_I = Any[@allowscalar(sliced_fields[f][I]) for f in 1:nf_def]
            res_struct[I] = _construct_indexed_struct(ET1, flds_I, nf_def)
        end
    end
    return res_struct
end

function overloaded_unsafe_getindex(::IndexLinear, a::Array, idxs...)
    norm_idxs = _normalize_jl_array_indices(idxs)
    return _index_julia_array(a, norm_idxs...)
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
