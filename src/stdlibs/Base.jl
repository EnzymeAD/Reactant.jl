_to_traced_cat_arg(x::Number) = broadcast_to_size(x, (1,))
function _to_traced_cat_arg(x::AbstractArray)
    m = ReactantCore.materialize_traced_array(x)
    m isa TracedRArray && return m
    return promote_to(TracedRArray{unwrapped_eltype(eltype(m)),ndims(m)}, m)
end

struct CallCat{Dims,T}
    dims::Dims
end

function (c::CallCat{Dims,T})(xs::Tuple) where {Dims,T}
    return Base._cat_t(c.dims, T, map(_to_traced_cat_arg, xs)...)
end

struct CallCatInto{Dims}
    dims::Dims
end

function (c::CallCatInto{Dims})(dest, xs::Tuple) where {Dims}
    copyto!(dest, Base._cat_t(c.dims, eltype(dest), map(_to_traced_cat_arg, xs)...))
    return nothing
end

@inline function Base.vcat(a::Number, b::Union{UnionAnyConcreteRArray,AnyTracedRArray})
    return vcat(fill!(similar(b, typeof(a), (1, size(b)[2:end]...)), a), b)
end
@inline function Base.vcat(a::Union{UnionAnyConcreteRArray,AnyTracedRArray}, b::Number)
    return vcat(a, fill!(similar(a, typeof(b), (1, size(a)[2:end]...)), b))
end
@inline function Base.hcat(a::Number, b::Union{UnionAnyConcreteRArray,AnyTracedRArray})
    return hcat(fill!(similar(b, typeof(a), (size(b)[1:(end - 1)]..., 1)), a), b)
end
@inline function Base.hcat(a::Union{UnionAnyConcreteRArray,AnyTracedRArray}, b::Number)
    return hcat(a, fill!(similar(a, typeof(b), (size(a)[1:(end - 1)]..., 1)), b))
end

function Base._typed_vcat!(
    a::UnionAnyConcreteRArray{T,1}, V::Base.AbstractVecOrTuple{AbstractVector}
) where {T}
    isempty(V) && return a
    xs = Tuple(V)
    total_len = 0
    for k in 1:length(xs)
        total_len += length(xs[k])
    end
    if total_len != length(a)
        throw(
            DimensionMismatch(
                "destination length $(length(a)) does not match concatenated length $total_len",
            ),
        )
    end
    if a isa ConcretePJRTArray{T,1} || a isa ConcreteIFRTArray{T,1}
        fn = compile(CallCat{Val{1},T}(Val(1)), (xs,))
        res = fn(xs)
        if typeof(res) === typeof(a)
            a.data = res.data
            return a
        end
    end
    fn = compile(CallCatInto{Val{1}}(Val(1)), (a, xs))
    fn(a, xs)
    return a
end

function Base._typed_vcat(::Type{T}, ::Union{Tuple{},AbstractVector{Union{}}}) where {T}
    return Vector{T}()
end
function Base._typed_hcat(::Type{T}, ::Union{Tuple{},AbstractVector{Union{}}}) where {T}
    return Vector{T}()
end

function Base._typed_vcat(
    ::Type{T}, X::Base.AbstractVecOrTuple{UnionAnyConcreteRArray}
) where {T}
    isempty(X) && return Vector{T}()
    xs = Tuple(X)
    Base.cat_size_shape((true,), xs...)
    fn = compile(CallCat{Val{1},T}(Val(1)), (xs,))
    return fn(xs)
end

function Base._typed_hcat(
    ::Type{T}, X::Base.AbstractVecOrTuple{UnionAnyConcreteRArray}
) where {T}
    isempty(X) && return Vector{T}()
    xs = Tuple(X)
    Base.cat_size_shape((false, true), xs...)
    fn = compile(CallCat{Val{2},T}(Val(2)), (xs,))
    return fn(xs)
end

_unwrap_cat_dims(x) = x
_unwrap_cat_dims(::Val{D}) where {D} = D

function Base.similar(
    a::Base.ReshapedArray{T,M,<:ConcretePJRTArray}, ::Type{S}, dims::Dims
) where {T,M,S}
    anc = ancestor(a)
    if Sharding.is_sharded(anc) && length(dims) != ndims(anc)
        sh = Sharding.unwrap_shardinfo(anc.sharding)
        new_sharding =
            hasproperty(sh, :mesh) ? Sharding.Replicated(sh.mesh) : Sharding.NoSharding()
        return ConcretePJRTArray(
            zeros(S, dims);
            client=XLA.client(anc),
            device=XLA.device(anc),
            sharding=new_sharding,
        )
    end
    return similar(anc, S, dims)
end

function Base._cat_t(
    dims::Union{Integer,Val},
    ::Type{T},
    x1::UnionAnyConcreteRArray,
    X::UnionAnyConcreteRArray...,
) where {T}
    d = _unwrap_cat_dims(dims)
    all_xs = (x1, X...)
    if d isa Integer
        catdims = Base.dims2cat(d)
        Base.cat_size_shape(catdims, all_xs...)
        fn = compile(CallCat{Val{Int(d)},T}(Val(Int(d))), (all_xs,))
        return fn(all_xs)
    end
    return invoke(Base._cat_t, Tuple{Any,Type{T},Vararg{Any}}, dims, T, all_xs...)
end

function Base.setindex!(a::ConcretePJRTArray{T}, v, args::Vararg{Any,M}) where {T,M}
    T <: TracedRNumber && throw(MethodError(setindex!, (a, v, args...)))
    isempty(a) && throw("Cannot setindex! to empty buffer")
    @boundscheck checkbounds(a, args...)
    fn = compile(mysetindex!, (a, v, args...))
    fn(a, v, args...)
    return a
end

function Base.setindex!(a::ConcreteIFRTArray, v, args::Vararg{Any,M}) where {M}
    eltype(a) <: TracedRNumber && throw(MethodError(setindex!, (a, v, args...)))
    isempty(a) && throw("Cannot setindex! to empty buffer")
    @boundscheck checkbounds(a, args...)
    fn = compile(mysetindex!, (a, v, args...))
    fn(a, v, args...)
    return a
end
