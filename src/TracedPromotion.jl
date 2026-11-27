# convert julia types to traced types
## Promote to a Traced Type
promote_to(::TracedRArray{T,N}, rhs) where {T,N} = promote_to(TracedRArray{T,N}, rhs)
promote_to(::TracedRNumber{T}, rhs) where {T} = promote_to(TracedRNumber{T}, rhs)

struct _DecomposedLazyStruct{T,F<:Tuple}
    fields::F
end

@generated function _construct_from_fields(::Type{T}, fields::Tuple) where {T}
    nf = fieldcount(T)
    exprs = Vector{Any}(undef, nf)
    for i in 1:nf
        exprs[i] = :(fields[$i])
    end
    return Expr(:new, T, exprs...)
end

_recompose_lazy_struct(x) = x
function _recompose_lazy_struct(d::_DecomposedLazyStruct{T}) where {T}
    return _construct_from_fields(T, map(_recompose_lazy_struct, d.fields))
end
_recompose_lazy_struct(t::Tuple) = map(_recompose_lazy_struct, t)
function _recompose_lazy_struct(nt::NamedTuple{K}) where {K}
    return NamedTuple{K}(map(_recompose_lazy_struct, Tuple(nt)))
end

function _decompose_lazy_struct(@nospecialize(x))
    T = Core.Typeof(x)
    if !isstructtype(T) ||
        isprimitivetype(T) ||
        ismutabletype(T) ||
        x isa Union{
            TracedRArray,
            TracedRNumber,
            MissingTracedValue,
            Array,
            AbstractRange,
            Type,
            Symbol,
            Module,
        }
        return x
    end
    nf = nfields(x)
    nf == 0 && return x
    for i in 1:nf
        isdefined(x, i) || return x
    end
    fvals = Vector{Any}(undef, nf)
    changed = false
    for i in 1:nf
        orig_i = getfield(x, i)
        dec_i = _decompose_lazy_struct(orig_i)
        fvals[i] = dec_i
        changed |= (dec_i !== orig_i)
    end
    if x isa Tuple
        return changed ? Tuple(fvals) : x
    elseif x isa NamedTuple
        return changed ? NamedTuple{fieldnames(T)}(Tuple(fvals)) : x
    end
    if (x isa AbstractArray && !isbitstype(eltype(x)) && !(ancestor(x) isa RArray)) ||
        changed
        ftup = Tuple(fvals)
        return _DecomposedLazyStruct{T,typeof(ftup)}(ftup)
    end
    return x
end

struct _LazyTracedArrayIndexFn{A} <: Function
    arr::A
end

@inline function (fn::_LazyTracedArrayIndexFn)(idxs::Vararg{Any,N}) where {N}
    return @allowscalar getindex(_recompose_lazy_struct(fn.arr), idxs...)
end

_has_traced_leaves(::Union{TracedRArray,TracedRNumber,MissingTracedValue}, _) = true
function _has_traced_leaves(
    ::Union{Symbol,Module,DataType,Nothing,Missing,Core.SimpleVector,Exception,Ptr}, _
)
    return false
end
function _has_traced_leaves(x::Array, seen)
    ET = eltype(x)
    ET !== Union{} && ET <: TracedRNumber && return true
    isconcretetype(ET) && return false
    x in seen && return false
    push!(seen, x)
    for i in eachindex(x)
        isassigned(x, i) || continue
        _has_traced_leaves(@inbounds(x[i]), seen) && return true
    end
    return false
end
function _has_traced_leaves(@nospecialize(x), seen=Base.IdSet())
    T = typeof(x)
    (!isstructtype(T) || isprimitivetype(T)) && return false
    nf = nfields(x)
    nf == 0 && return false
    if ismutabletype(T)
        x in seen && return false
        push!(seen, x)
    end
    for i in 1:nf
        isdefined(x, i) || continue
        _has_traced_leaves(getfield(x, i), seen) && return true
    end
    return false
end

function _materialize_lazy_traced_array(x::PermutedDimsArray{<:Any,<:Any,perm}) where {perm}
    return permutedims(_materialize_lazy_traced_array(parent(x)), perm)
end

function _materialize_lazy_traced_array(x::Base.ReshapedArray)
    return @opcall reshape(_materialize_lazy_traced_array(parent(x)), size(x)...)
end

function _materialize_lazy_traced_array(x::LinearAlgebra.Transpose)
    p = _materialize_lazy_traced_array(parent(x))
    return ndims(p) == 1 ? @opcall(reshape(p, 1, length(p))) : permutedims(p, (2, 1))
end

function _materialize_lazy_traced_array(x::LinearAlgebra.Adjoint)
    return conj(_materialize_lazy_traced_array(transpose(parent(x))))
end

function _materialize_lazy_traced_array(x::SubArray{<:Any,N}) where {N}
    p = parent(x)
    pidxs = parentindices(x)
    while p isa SubArray && length(pidxs) == ndims(p)
        pidxs = Base.reindex(parentindices(p), pidxs)
        p = parent(p)
    end
    if p isa AnyTracedRArray
        return materialize_traced_array(view(p, pidxs...))
    end
    if N == 0
        scalar_pidxs = map(first, pidxs)
        val = @allowscalar call_with_reactant(getindex, p, scalar_pidxs...)
        ET = val isa TracedRNumber ? unwrapped_eltype(val) : unwrapped_eltype(typeof(val))
        return promote_to(TracedRArray{ET,0}, val)
    end
    if isempty(x)
        ET = unwrapped_eltype(x)
        T_empty = ET <: ReactantPrimitive ? ET : Float64
        return @opcall fill(zero(T_empty), collect(Int64, size(x)))
    end
    target_shape = collect(Int64, size(x))
    if length(pidxs) == ndims(p)
        args = Vector{Any}(undef, length(pidxs))
        out_dim = 1
        for (m, pidx) in enumerate(pidxs)
            if pidx isa Real
                args[m] = Ref(pidx)
            else
                k = ndims(pidx)
                t_idx = promote_to(TracedRArray{Int,k}, pidx)
                bdims = collect(Int64, out_dim:(out_dim + k - 1))
                args[m] = @opcall broadcast_in_dim(t_idx, bdims, target_shape)
                out_dim += k
            end
        end
        return TracedUtils.elem_apply_via_while_loop(
            _LazyTracedArrayIndexFn(_decompose_lazy_struct(p)), args...
        )
    elseif length(pidxs) == 1
        pidx = only(pidxs)
        k = ndims(pidx)
        t_idx = promote_to(TracedRArray{Int,k}, pidx)
        bdims = collect(Int64, 1:k)
        lin_arg = @opcall broadcast_in_dim(t_idx, bdims, target_shape)
        return TracedUtils.elem_apply_via_while_loop(
            _LazyTracedArrayIndexFn(_decompose_lazy_struct(p)), lin_arg
        )
    end
    return invoke(_materialize_lazy_traced_array, Tuple{AbstractArray{<:Any,N}}, x)
end

function _materialize_lazy_traced_array(rhs::AbstractArray{<:Any,N}) where {N}
    if N == 0
        val = @allowscalar call_with_reactant(getindex, rhs)
        ET = val isa TracedRNumber ? unwrapped_eltype(val) : unwrapped_eltype(typeof(val))
        return promote_to(TracedRArray{ET,0}, val)
    end
    if isempty(rhs)
        ET = unwrapped_eltype(rhs)
        T_empty = ET <: ReactantPrimitive ? ET : Float64
        return @opcall fill(zero(T_empty), collect(Int64, size(rhs)))
    end
    target_shape = collect(Int64, size(rhs))
    coords = Vector{TracedRArray{Int,N}}(undef, N)
    for d in 1:N
        ax = promote_to(TracedRArray{Int,1}, axes(rhs, d))
        coords[d] = @opcall broadcast_in_dim(ax, Int64[d], target_shape)
    end
    return TracedUtils.elem_apply_via_while_loop(
        _LazyTracedArrayIndexFn(_decompose_lazy_struct(rhs)), Tuple(coords)...
    )
end

## Array types
function promote_to(::Type{TracedRArray}, rhs)
    if rhs isa AbstractArray &&
        !(rhs isa TracedRArray) &&
        !(ancestor(rhs) isa AnyTracedRArray) &&
        _has_traced_leaves(rhs)
        return _materialize_lazy_traced_array(rhs)
    end
    return promote_to(TracedRArray{unwrapped_eltype(rhs),ndims(rhs)}, rhs)
end
function promote_to(::Type{TracedRArray{T}}, rhs) where {T}
    return promote_to(TracedRArray{T,ndims(rhs)}, rhs)
end

promote_to(::Type{<:TracedRArray{T,N}}, rhs::TracedRArray{T,N}) where {T,N} = rhs
function promote_to(::Type{<:TracedRArray{T,N}}, rhs::TracedRArray{T2,N}) where {T,T2,N}
    return @opcall convert(TracedRArray{T,N}, rhs)
end

function promote_to(
    ::Type{<:TracedRArray{T,N}}, rhs::AbstractArray{<:TracedRNumber,N}
) where {T,N}
    return @opcall convert(TracedRArray{T,N}, aos_to_soa(materialize_traced_array(rhs)))
end

function promote_to(::Type{<:TracedRArray{T,1}}, rhs::AbstractRange) where {T}
    return @opcall convert(TracedRArray{T,1}, materialize_traced_array(rhs))
end

function promote_to(::Type{<:TracedRArray{T,1}}, rhs::Base.OneTo) where {T}
    return promote_to(TracedRArray{T,1}, first(rhs):last(rhs))
end

function promote_to(::Type{<:TracedRArray{T,N}}, rhs::LinearAlgebra.Diagonal) where {T,N}
    return LinearAlgebra.diagm(promote_to(TracedRArray{T,1}, rhs.diag))
end

function promote_to(::Type{<:TracedRArray{T,N}}, rhs::AbstractArray{<:Any,N}) where {T,N}
    if ancestor(rhs) isa AnyTracedRArray
        return promote_to(TracedRArray{T,N}, materialize_traced_array(rhs))
    end
    if _has_traced_leaves(rhs)
        mat = _materialize_lazy_traced_array(rhs)
        return T === Any ? mat : promote_to(TracedRArray{T,N}, mat)
    end
    ## fallback option, almost certainly will emit a really bad IR
    return promote_to(TracedRArray{T,N}, @opcall(constant(rhs)))
end

function promote_to(::Type{<:TracedRArray{T,N}}, rhs::Number) where {T,N}
    return promote_to(TracedRArray{T,N}, [rhs])
end

## Number types
function promote_to(::Type{TracedRNumber}, rhs)
    T = rhs isa AbstractIrrational ? Float64 : unwrapped_eltype(rhs)
    return promote_to(TracedRNumber{T}, rhs)
end

function promote_to(
    ::Type{TracedRNumber}, rhs::Union{Rational{T},TracedRational{T}}
) where {T}
    return promote_to(TracedRNumber{float(unwrapped_eltype(T))}, rhs)
end

promote_to(::Type{<:TracedRNumber{T}}, rhs::TracedRNumber{T}) where {T} = rhs
function promote_to(::Type{<:TracedRNumber{T}}, rhs::TracedRNumber{T2}) where {T,T2}
    return @opcall convert(TracedRNumber{T}, rhs)
end
function promote_to(
    ::Type{<:TracedRNumber{T}}, rhs::TracedRNumber{T2}
) where {T<:ReactantFloat8,T2}
    return @opcall convert(TracedRNumber{T}, rhs)
end

function promote_to(::Type{<:TracedRArray{T,0}}, rhs::TracedRNumber{T2}) where {T,T2}
    return TracedRArray{T,0}((), @opcall(convert(TracedRNumber{T}, rhs)).mlir_data, ())
end
function promote_to(::Type{<:TracedRNumber{T}}, rhs::TracedRArray{T2,0}) where {T,T2}
    return TracedRNumber{T}((), @opcall(convert(TracedRArray{T,0}, rhs)).mlir_data)
end

function promote_to(::Type{<:TracedRNumber{T}}, rhs::Number) where {T<:ReactantFloat8}
    res = @opcall(fill(rhs))
    return @opcall convert(
        TracedRNumber{T}, TracedRNumber{unwrapped_eltype(res)}((), res.mlir_data)
    )
end

function promote_to(::Type{<:TracedRNumber{T}}, rhs::Number) where {T}
    return promote_to(TracedRNumber{T}, T(rhs))
end

function promote_to(::Type{<:TracedRNumber{T}}, rhs::T) where {T<:ReactantPrimitive}
    res = @opcall fill(rhs)
    return TracedRNumber{T}((), res.mlir_data)
end

function ReactantCore.promote_to_traced(x)
    return promote_to(TracedRNumber{unwrapped_eltype(typeof(x))}, x)
end

## Promote to a Traced Type and broadcast to a given size
function broadcast_to_size(arg::AbstractArray, rsize)
    return broadcast_to_size(promote_to(TracedRArray, arg), rsize)
end

function broadcast_to_size(arg::Number, rsize)
    return broadcast_to_size(promote_to(TracedRNumber, arg), rsize)
end

broadcast_to_size(arg::TracedRNumber, ::Tuple{}) = arg
broadcast_to_size(arg::TracedRNumber, rsize) = @opcall fill(arg, collect(Int64, rsize))

function broadcast_to_size(arg::TracedRArray, rsize)
    return @opcall broadcast_in_dim(
        arg, collect(Int64, 1:ndims(arg)), collect(Int64, rsize)
    )
end

function broadcast_to_size(arg::Broadcast.Extruded, rsize)
    rsize2 = (keep ? rsizev : 1 for (keep, rsizev) in zip(arg.keeps, rsize))
    return broadcast_to_size(broadcast_to_size(arg.x, rsize2), rsize)
end

broadcast_to_size(arg::Base.RefValue, rsize) = arg
