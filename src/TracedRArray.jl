using Base.Broadcast
using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

mutable struct TracedRArray{T,N} <: RArray{T,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N}
        shape = Tuple(shape)
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape
        end
        return new{T,N}(paths, mlir_data, shape)
    end
end

const WrappedTracedRArray{T,N} = WrappedArray{T,N,TracedRArray,TracedRArray{T,N}}
const AnyTracedRArray{T,N} = Union{TracedRArray{T,N},WrappedTracedRArray{T,N}}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = Union{
    AnyTracedRArray{T,2},LinearAlgebra.Diagonal{T,TracedRArray{T,1}}
}
const AnyTracedRVecOrMat{T} = Union{AnyTracedRVector{T},AnyTracedRMatrix{T}}

function TracedRArray(data::MLIR.IR.Value)
    data_type = MLIR.IR.type(data)
    return TracedRArray{eltype(MLIR.IR.julia_type(data_type)),ndims(data_type)}(
        (), data, size(data_type)
    )
end

ReactantCore.is_traced(::TracedRArray) = true

new_traced_value(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), nothing, size(A))

function Base.convert(::Type{TracedRArray{T,N}}, x::AbstractArray) where {T,N}
    @assert ndims(x) == N
    if x isa TracedRArray
        eltype(x) == T && return x
        return Ops.convert(TracedRArray{T,N}, x)
    end
    x isa WrappedTracedRArray &&
        return convert(TracedRArray{T,N}, materialize_traced_array(x))
    return convert(TracedRArray{T,N}, Ops.constant(collect(x)))
end

TracedRArray{T,N}(x::AbstractArray) where {T,N} = convert(TracedRArray{T,N}, x)

materialize_traced_array(x::TracedRArray) = x
materialize_traced_array(x::WrappedTracedRArray) = x[axes(x)...]
function materialize_traced_array(
    x::Adapt.WrappedReshapedArray{T,N,<:TracedRArray}
) where {T,N}
    return Ops.reshape(materialize_traced_array(parent(x)), size(x)...)
end
function materialize_traced_array(
    x::LinearAlgebra.Transpose{T,TracedRArray{T,N}}
) where {T,N}
    px = parent(x)
    A = ndims(px) == 1 ? reshape(px, :, 1) : px
    return permutedims(A, (2, 1))
end
function materialize_traced_array(x::LinearAlgebra.Adjoint{T,TracedRArray{T,N}}) where {T,N}
    return conj(materialize_traced_array(transpose(parent(x))))
end
function materialize_traced_array(
    x::PermutedDimsArray{T,N,perm,iperm,<:TracedRArray{T,N}}
) where {T,N,perm,iperm}
    return permutedims(parent(x), perm)
end
function materialize_traced_array(x::LinearAlgebra.Diagonal{T,TracedRArray{T,1}}) where {T}
    return LinearAlgebra.diagm(parent(x))
end

get_mlir_data(x::TracedRArray) = x.mlir_data
get_mlir_data(x::AnyTracedRArray) = get_mlir_data(materialize_traced_array(x))

function set_mlir_data!(x::TracedRArray, data)
    x.mlir_data = data
    return x
end
function set_mlir_data!(x::Adapt.WrappedReshapedArray{T,N,<:TracedRArray}, data) where {T,N}
    res_mlir_data = Ops.reshape(TracedRArray(data), size(parent(x))...).mlir_data
    set_mlir_data!(parent(x), res_mlir_data)
    return x
end
function set_mlir_data!(x::LinearAlgebra.Transpose{T,TracedRArray{T,N}}, data) where {T,N}
    tdata = TracedRArray(data)
    px = parent(x)
    px.mlir_data = (
        if ndims(px) == 1
            Ops.reshape(tdata, length(tdata))
        else
            Ops.transpose(tdata, [2, 1])
        end
    ).mlir_data
    return x
end
function set_mlir_data!(x::LinearAlgebra.Adjoint{T,TracedRArray{T,N}}, data) where {T,N}
    tdata = TracedRArray(data)
    px = parent(x)
    transposed_data =
        ndims(px) == 1 ? Ops.reshape(tdata, length(tdata)) : Ops.transpose(tdata, [2, 1])
    px.mlir_data = (T <: Real ? transposed_data : Ops.conj(transposed_data)).mlir_data
    return x
end
function set_mlir_data!(
    x::PermutedDimsArray{T,N,perm,iperm,TracedRArray{T,N}}, data
) where {T,N,perm,iperm}
    parent(x).mlir_data = permutedims(TracedRArray(data), iperm).mlir_data
    return x
end
function set_mlir_data!(x::LinearAlgebra.Diagonal{T,TracedRArray{T,1}}, data) where {T}
    parent(x).mlir_data = LinearAlgebra.diag(TracedRArray(data)).mlir_data
    return x
end
function set_mlir_data!(x::AnyTracedRArray, data)
    setindex!(x, TracedRArray(data), axes(x)...)
    return x
end

ancestor(x::TracedRArray) = x
ancestor(x::WrappedTracedRArray) = ancestor(parent(x))

get_ancestor_indices(::TracedRArray, indices...) = indices
function get_ancestor_indices(x::WrappedTracedRArray, indices...)
    return get_ancestor_indices(parent(x), Base.reindex(parentindices(x), indices)...)
end

function Base.getindex(
    a::TracedRArray{T,N}, index::Vararg{Union{Int,TracedRNumber{Int}},N}
) where {T,N}
    GPUArraysCore.assertscalar("getindex(::TracedRArray, ::Vararg{Int, N})")

    start_indices = [promote_to(TracedRNumber{Int}, i - 1).mlir_data for i in index]
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

function Base.getindex(a::TracedRArray{T,0}) where {T}
    return TracedRNumber{T}((), a.mlir_data)
end

# XXX: We want to support https://github.com/EnzymeAD/Reactant.jl/issues/242 eventually
function Base.getindex(a::TracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    indices = map(enumerate(indices)) do (idx, i)
        i isa Colon && return 1:size(a, idx)
        i isa CartesianIndex && return Tuple(i)
        return i
    end

    foreach(indices) do idxs
        idxs isa Number && return nothing
        contiguous = all(isone, diff(idxs))
        # XXX: We want to throw error even for dynamic indexing
        if typeof(a) <: Bool
            contiguous || error("non-contiguous indexing is not supported")
        end
    end

    start_indices = map(indices) do i
        return promote_to(TracedRNumber{Int}, first(i) - 1).mlir_data
    end
    slice_sizes = [Int64(length(i)) for i in indices]
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_slice(a.mlir_data, start_indices; slice_sizes), 1
    )

    x = TracedRArray{T,N}((), res, Tuple(length.(indices)))
    ddims = findall(Base.Fix2(isa, Integer), indices)
    isempty(ddims) || return dropdims(x; dims=Tuple(ddims))
    return x
end

# Prevent ambiguity
function Base.getindex(a::WrappedTracedRArray, index::Union{Int,TracedRNumber{Int}}...)
    return getindex(ancestor(a), get_ancestor_indices(a, index...)...)
end

function Base.getindex(a::WrappedTracedRArray, indices...)
    return getindex(ancestor(a), get_ancestor_indices(a, indices...)...)
end

function Base.setindex!(
    a::TracedRArray{T,N},
    v,
    indices::Vararg{Union{Base.AbstractUnitRange,Colon,Int,TracedRNumber{Int}},N},
) where {T,N}
    indices = map(enumerate(indices)) do (idx, i)
        i isa Int ? (i:i) : (i isa Colon ? (1:size(a, idx)) : i)
    end
    v = broadcast_to_size(v, length.(indices))
    v = promote_to(TracedRArray{T,N}, v)
    indices = [
        (promote_to(TracedRNumber{Int}, i isa Colon ? 1 : first(i)) - 1).mlir_data for
        i in indices
    ]
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_update_slice(
            a.mlir_data, get_mlir_data(v), indices
        ),
        1,
    )
    a.mlir_data = res
    return v
end

function Base.setindex!(
    a::AnyTracedRArray{T,N},
    v,
    indices::Vararg{Union{Base.AbstractUnitRange,Colon,Int,TracedRNumber{Int}},N},
) where {T,N}
    ancestor_indices = get_ancestor_indices(a, indices...)
    setindex!(ancestor(a), v, ancestor_indices...)
    return a
end

Base.Tuple(x::TracedRArray) = ntuple(Base.Fix1(Base.getindex, x), length(x))

Base.size(x::TracedRArray) = x.shape

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), A.mlir_data, size(A))

# TODO is there a way to create an unitialized `tensor`? does it show an advantage? maybe `fill`?
function Base.similar(::TracedRArray, ::Type{T}, dims::Dims{N}) where {T,N}
    return Ops.constant(zeros(T, dims))
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
    set_mlir_data!(A, Ops.conj(materialize_traced_array(A)).mlir_data)
    return A
end

Base.real(A::AnyTracedRArray) = A
Base.real(A::AnyTracedRArray{<:Complex}) = Ops.real(materialize_traced_array(A))

Base.imag(A::AnyTracedRArray) = zero(A)
Base.imag(A::AnyTracedRArray{<:Complex}) = Ops.imag(materialize_traced_array(A))

promote_to(::Type{TracedRArray{T,N}}, rhs) where {T,N} = TracedRArray{T,N}(rhs)

promote_to(::TracedRArray{T,N}, rhs) where {T,N} = promote_to(TracedRArray{T,N}, rhs)

elem_apply(::Type{T}, x::TracedRArray{T}) where {T<:ReactantPrimitive} = x
function elem_apply(
    ::Type{T}, x::TracedRArray{T2}
) where {T<:ReactantPrimitive,T2<:ReactantPrimitive}
    # Special Path to prevent going down a despecialized path
    return elem_apply(TypeCast{T}(), x)
end

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    if all(iszero ∘ ndims, args)
        scalar_args = map(args) do arg
            return promote_to(TracedRNumber{eltype(arg)}, arg)
        end
        return f(scalar_args...)
    end

    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        f, args, (), string(f) * "_broadcast_scalar", false; toscalar=true
    )

    invmap = IdDict()
    for (k, v) in seen_args
        invmap[v] = k
    end

    keys_seen = [k for k in keys(seen_args) if k isa TracedType]
    input_shapes = size.(keys_seen)
    # by the time we reach here all args must have same size
    @assert allequal(input_shapes) "input shapes are $(input_shapes)"
    OutShape = isempty(seen_args) ? nothing : first(input_shapes)
    @assert !isnothing(OutShape)

    in_tys2 = [mlir_type(invmap[arg]) for arg in linear_args]

    out_tys2 = [
        MLIR.IR.TensorType(OutShape, MLIR.IR.Type(eltype(arg))) for arg in linear_results
    ]

    fname = get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))

    batch_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = get_argidx(a)
        if idx == 1 && fnwrap
            push_val!(batch_inputs, f, path[3:end])
        else
            if fnwrap
                idx -= 1
            end
            push_val!(batch_inputs, args[idx], path[3:end])
        end
    end

    res = MLIR.Dialects.enzyme.batch(
        batch_inputs;
        outputs=out_tys2,
        fn=fname,
        batch_shape=MLIR.IR.DenseArrayAttribute([Int64(i) for i in OutShape]),
    )

    residx = 1

    for a in linear_results
        if has_residx(a)
            path = get_residx(a)
            set!(result, path[2:end], MLIR.IR.result(res, residx))
            residx += 1
        else
            idx, path = get_argidx(a)
            if idx == 1 && fnwrap
                set!(f, path[3:end], MLIR.IR.result(res, residx))
                residx += 1
            else
                if fnwrap
                    idx -= 1
                end
                set!(args[idx], path[3:end], MLIR.IR.result(res, residx))
                residx += 1
            end
        end
    end

    seen_results = OrderedIdDict()
    traced2_result = make_tracer(seen_results, result, (), TracedSetPath; tobatch=OutShape)

    func2.operation = MLIR.API.MlirOperation(C_NULL)

    return traced2_result
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

function Enzyme.Compiler.active_reg_inner(
    ::Type{TracedRArray{T,N}},
    seen::ST,
    world::Union{Nothing,UInt},
    ::Val{justActive}=Val(false),
    ::Val{UnionSret}=Val(false),
)::Enzyme.Compiler.ActivityState where {ST,T,N,justActive,UnionSret}
    if Enzyme.Compiler.active_reg_inner(T, seen, world, Val(justActive), Val(UnionSret)) ==
        Enzyme.Compiler.AnyState
        return Enzyme.Compiler.AnyState
    else
        return Enzyme.Compiler.DupState
    end
end

function Base.mapreduce(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(A::TracedRArray{T,N});
    dims=:,
    init=nothing,
) where {T,N}
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
    else
        init = init::T
    end

    init = [broadcast_to_size(init, ()).mlir_data]

    inp = [broadcast(f, A).mlir_data]

    rdims = if dims == (:)
        Int64[i for i in 0:(N - 1)]
    else
        Int64[i - 1 for i in dims]
    end

    in_tys = [
        MLIR.IR.TensorType(Int64[], eltype(MLIR.IR.type(arg))) for arg in (inp[1], init[1])
    ]

    fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in in_tys])

    args = (
        TracedRNumber{op_in_T}((), MLIR.IR.argument(fnbody, i)) for
        (i, ty) in enumerate(in_tys)
    )

    res = MLIR.IR.block!(fnbody) do
        tmp = broadcast_to_size(op(args...), ()).mlir_data
        MLIR.Dialects.stablehlo.return_(MLIR.IR.Value[tmp])
        return tmp
    end

    toonedims = [(in(i - 1, rdims) ? 1 : size(A, i)) for i in 1:N]
    outdims = [size(A, i) for i in 1:N if (i - 1) ∉ rdims]

    TT = [
        MLIR.IR.TensorType(outdims, eltype(MLIR.IR.type(inp0))) for
        (inp0, res0) in zip(inp, (res,))
    ]

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
    tmp = broadcast_to_size(Base.mapreduce(f, op, A; dims=1), (1, size(R)[2:end]...))
    R.mlir_data = broadcast(op, R, tmp).mlir_data
    return R
end

function Base.fill!(A::TracedRArray{T,N}, x) where {T,N}
    bcast = broadcast_to_size(T(x), size(A))
    A.mlir_data = bcast.mlir_data
    return A
end

function Base.fill!(A::TracedRArray{T,N}, x::TracedRNumber{T2}) where {T,N,T2}
    bcast = broadcast_to_size(promote_to(TracedRNumber{T}, x), size(A))
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
    bc::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{<:TracedRNumber{T}}, dims
) where {T<:ReactantPrimitive,N}
    @assert N isa Int
    return TracedRArray{T,N}((), nothing, map(length, dims))
end

function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    dest = copyto!(similar(bc, ElType), bc)
    return dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

Base.eltype(::Broadcast.Extruded{T}) where {T} = eltype(T)

# we need to override the outer copy method to make sure we never fall back to scalar
# iteration (see, e.g., CUDA.jl#145)
function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    if ElType == Any
        a1 = bc.args[1]
        @show a1
        b1 = a1.args[1]
        @show b1
        @show typeof(b1)
        @show eltype(b1)
        @show Broadcast._broadcast_getindex_eltype(a1.args[1])
        @show Broadcast.eltypes(a1.args)
        @show Broadcast._broadcast_getindex_eltype(a1)
        @show typeof(bc.args)
        argT = Broadcast.eltypes(bc.args)
        @show argT
        RT = Base._return_type(bc.f, argT)
        @show RT
        T = Base.promote_typejoin_union(RT)
        @show T
        @show bc.f, bc.args
    end
    @assert ElType != Any
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

broadcast_to_size(arg::AbstractArray, rsize) = broadcast_to_size(Ops.constant(arg), rsize)

function broadcast_to_size(arg::Base.RefValue, rsize)
    # XXX: don't we want to expand here to rsize?
    return arg
end

broadcast_to_size(arg::Number, rsize) = Ops.constant(Base.fill(arg, Tuple(rsize)))

function broadcast_to_size(arg::TracedRNumber, rsize)
    length(rsize) == 0 && return arg
    return broadcast_to_size_internal(
        TracedRArray{eltype(arg),0}((), arg.mlir_data, ()), rsize
    )
end

function broadcast_to_size(arg::AnyTracedRArray{T,0}, rsize) where {T}
    arg = materialize_traced_array(arg)
    return broadcast_to_size(TracedRNumber{T}((), arg.mlir_data), rsize)
end

function broadcast_to_size(arg::AnyTracedRArray, rsize)
    arg = materialize_traced_array(arg)
    size(arg) == Tuple(rsize) && return arg
    return broadcast_to_size_internal(arg, rsize)
end

function broadcast_to_size(arg::Broadcast.Extruded, rsize)
    rsize2 = (keep ? rsizev : 1 for (keep, rsizev) in zip(arg.keeps, rsize))
    x = broadcast_to_size(arg.x, rsize2)
    size(x) == rsize && return x
    return broadcast_to_size_internal(x, rsize)
end

function broadcast_to_size_internal(x::TracedRArray, rsize)
    dims = collect(Int64, 0:(length(size(x)) - 1))

    if length(size(MLIR.IR.type(x.mlir_data))) != length(dims)
        @show x
        @show arg
        @show rsize
        @show rsize2
        @show dims
    end
    @assert length(size(MLIR.IR.type(x.mlir_data))) == length(dims)
    mlirty = MLIR.IR.type(x.mlir_data)

    return TracedRArray{eltype(x),Int(length(rsize))}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.broadcast_in_dim(
                x.mlir_data;
                result_0=MLIR.IR.TensorType([t for t in rsize], eltype(mlirty)),
                broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims),
            ),
            1,
        ),
        collect(rsize),
    )
end

function _copyto!(dest::AnyTracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = elem_apply(bc.f, args...)
    set_mlir_data!(dest, res.mlir_data)
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

function Base._cat_t(dims, ::Type{T}, X::TracedRArray...) where {T}
    dims = dispatch_val(dims)
    @assert dims isa Integer "Support for non-integer dimensions is not implemented yet."

    # MLIR expects the dimension `dims` to be ≤ the rank of the input tensors
    X = maybe_expand_dims.(X, (dims,))

    catdims = Base.dims2cat(dims)
    shape = Base.cat_size_shape(catdims, X...)
    RT = Base.promote_eltype(T, X...)

    # convert to the target eltype
    X = map(Base.Fix1(promote_to, TracedRArray{RT,length(shape)}), X)

    return TracedRArray{RT,length(shape)}(
        (),
        MLIR.IR.result(
            # TODO maybe we should do some conversion?
            MLIR.Dialects.stablehlo.concatenate(
                collect(get_mlir_data.(X));
                result_0=MLIR.IR.TensorType(shape, MLIR.IR.Type(RT)),
                dimension=dims - 1, # stablehlo expects this to be zero-indexed
            ),
            1,
        ),
        shape,
    )
end

function maybe_expand_dims(x::AbstractArray{T,N}, dims) where {T,N}
    dims = dispatch_val(dims)
    dims ≤ N && return x
    return reshape(x, ntuple(i -> i ≤ N ? size(x, i) : 1, dims))
end

for (minT, maxT) in Iterators.product((Number, TracedRNumber), (Number, TracedRNumber))
    @eval function Base.clamp!(x::AnyTracedRArray, min::$(minT), max::$(maxT))
        y = Ops.clamp(min, materialize_traced_array(x), max)
        set_mlir_data!(x, y.mlir_data)
        return x
    end
end

# outer repeat
function Base.repeat(x::AnyTracedRArray{T,N}, counts::Vararg{Int,M}) where {T,N,M}
    P = max(N, M) # potentially padded

    # (d1, d2, ..., dP) -> (d1, 1, d2, 1, ..., dP, 1)
    interleaved_size = ones(Int, 2P)
    interleaved_size[1:2:2N] .= size(x)

    x_interleaved = reshape(x, interleaved_size...)

    # (d1, 1, d2, 1, ..., dP, 1) -> (d1, r1, d2, r2, ..., dP, rP)
    broadcast_target_size = interleaved_size
    broadcast_target_size[2:2:2M] .= counts

    x_broadcasted = broadcast_to_size(x_interleaved, broadcast_target_size)

    # (d1, r1, d2, r2, ..., dP, rP) -> (d1*r1, d2*r2, ..., dP*rP)
    final_size = vec(prod(reshape(broadcast_target_size, 2, :), dims=1))

    x_final = reshape(x_broadcasted, final_size...)

    return x_final
end
