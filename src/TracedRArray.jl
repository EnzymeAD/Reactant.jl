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

TracedRArray{T,N}(x::TracedRArray{T,N}) where {T,N} = x

const WrappedTracedRArray{T,N} = WrappedArray{T,N,TracedRArray,TracedRArray{T,N}}
const AnyTracedRArray{T,N} = Union{TracedRArray{T,N},WrappedTracedRArray{T,N}}
const AnyTracedRScalar{T} = AnyTracedRArray{T,0}
const AnyTracedRVector{T} = AnyTracedRArray{T,1}
const AnyTracedRMatrix{T} = AnyTracedRArray{T,2}
const AnyTracedRVecOrMat{T} = Union{AnyTracedRVector{T},AnyTracedRMatrix{T}}

materialize_traced_array(x::TracedRArray) = x
materialize_traced_array(x::WrappedTracedRArray) = x[axes(x)...]

get_mlir_data(x::TracedRArray) = x.mlir_data
get_mlir_data(x::AnyTracedRArray) = get_mlir_data(materialize_traced_array(x))

ancestor(x::TracedRArray) = x
ancestor(x::WrappedTracedRArray) = ancestor(parent(x))

get_ancestor_indices(::TracedRArray, indices...) = indices
function get_ancestor_indices(x::WrappedTracedRArray, indices...)
    return get_ancestor_indices(parent(x), Base.reindex(parentindices(x), indices)...)
end

Base.getindex(a::AnyTracedRScalar{T}) where {T} = a

Base.zero(::AnyTracedRScalar{T}) where {T} = promote_to(TracedRArray{T,0}, zero(T))
Base.one(::AnyTracedRScalar{T}) where {T} = promote_to(TracedRArray{T,0}, one(T))

function Base.convert(::Type{<:AnyTracedRScalar{T}}, x::Number) where {T}
    return promote_to(TracedRArray{T,0}, T(x))
end

function Base.getindex(a::TracedRArray{T,N}, index::Vararg{Int,N}) where {T,N}
    @warn(
        """Performing scalar indexing on task $(current_task()).
Invocation resulted in scalar indexing of a TracedRArray.
This is typically caused by calling an iterating implementation of a method.
Such implementations *do not* execute on device, but very slowly on the CPU,
and require expensive copies and synchronization each time and therefore should be avoided."""
    )

    res1 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.slice(
            a.mlir_data;
            start_indices=MLIR.IR.DenseArrayAttribute([Int64(i - 1) for i in index]),
            limit_indices=MLIR.IR.DenseArrayAttribute([Int64(i) for i in index]),
            strides=MLIR.IR.DenseArrayAttribute([Int64(1) for i in index]),
        ),
        1,
    )
    res2 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.reshape(
            res1; result_0=MLIR.IR.TensorType(Int64[], eltype(MLIR.IR.type(res1)))
        ),
        1,
    )
    return TracedRArray{T,0}((), res2, ())
end

function Base.getindex(a::TracedRArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    indices = [i isa Colon ? (1:size(a, idx)) : i for (idx, i) in enumerate(indices)]
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.slice(
            a.mlir_data;
            start_indices=MLIR.IR.DenseArrayAttribute([
                Int64(first(i) - 1) for i in indices
            ]),
            limit_indices=MLIR.IR.DenseArrayAttribute([Int64(last(i)) for i in indices]),
            strides=MLIR.IR.DenseArrayAttribute([Int64(1) for i in indices]),
        ),
        1,
    )
    x = TracedRArray{T,N}((), res, Tuple(length.(indices)))
    ddims = findall(x -> x isa Integer, indices)
    !isempty(ddims) && return dropdims(x; dims=Tuple(ddims))
    return x
end

# Prevent ambiguity
function Base.getindex(a::WrappedTracedRArray, index::Int...)
    return getindex(ancestor(a), get_ancestor_indices(a, index...)...)
end

function Base.getindex(a::WrappedTracedRArray, indices...)
    return getindex(ancestor(a), get_ancestor_indices(a, indices...)...)
end

function Base.setindex!(
    a::TracedRArray{T,N}, v, indices::Vararg{Union{Base.AbstractUnitRange,Colon},N}
) where {T,N}
    indices = [
        (promote_to(TracedRArray{Int,0}, i isa Colon ? 1 : first(i)) - 1).mlir_data for
        i in indices
    ]
    v = promote_to(TracedRArray{T,N}, v)
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_update_slice(a.mlir_data, v.mlir_data, indices), 1
    )
    a.mlir_data = res
    return v
end

Base.size(x::TracedRArray) = x.shape

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray{T,N}((), A.mlir_data, size(A))

function Base.similar(x::TracedRArray{T,N}, ::Type{T2}) where {T,N,T2}
    return TracedRArray{T2,N}((), nothing, size(x))
end

function Base.show(io::IOty, X::TracedRArray{T,N}) where {T,N,IOty<:Union{IO,IOContext}}
    return print(io, "TracedRArray{", T, ",", N, "N}(", X.paths, ", size=", size(X), ")")
    # TODO this line segfaults if MLIR IR has not correctly been generated
    # return print(io, X.mlir_data, ")")
end

Base.only(A::AnyTracedRScalar{T}) where {T} = A

function Base.reshape(A::AnyTracedRArray{T,N}, dims::NTuple{NT,Int}) where {T,N,NT}
    if prod(dims) != prod(size(A))
        throw(
            DimensionMismatch(
                "new shape $(dims) is incompatible with array size $(size(A))"
            ),
        )
    end

    # HLO reshape semantics collapse the opposite way
    res1 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.transpose(
            get_mlir_data(A);
            permutation=MLIR.IR.DenseArrayAttribute([Int64(N - 1 - i) for i in 0:(N - 1)]),
        ),
        1,
    )

    res2 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.reshape(
            res1;
            result_0=MLIR.IR.TensorType(
                [Int64(i) for i in reverse(dims)], eltype(MLIR.IR.type(res1))
            ),
        ),
    )

    res3 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.transpose(
            res2;
            permutation=MLIR.IR.DenseArrayAttribute([
                Int64(NT - 1 - i) for i in 0:(NT - 1)
            ]),
        ),
        1,
    )

    return TracedRArray{T,NT}((), res3, dims)
end

function Base.permutedims(A::AnyTracedRArray{T,N}, perm) where {T,N}
    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.transpose(
                get_mlir_data(A);
                permutation=MLIR.IR.DenseArrayAttribute([Int64(i - 1) for i in perm]),
            ),
            1,
        ),
        Tuple(size(A, i) for i in perm),
    )
end

function Base.transpose(A::AnyTracedRVecOrMat)
    A = ndims(A) == 1 ? reshape(A, :, 1) : A
    return permutedims(A, (2, 1))
end
Base.adjoint(A::AnyTracedRVecOrMat{<:Real}) = transpose(A)

function Base.promote_rule(
    ::Type{TracedRArray{T,N}}, ::Type{TracedRArray{S,N}}
) where {T,S,N}
    return TracedRArray{Base.promote_type(T, S),N}
end

function Base.promote_rule(::Type{T}, ::Type{TracedRArray{S,N}}) where {T,S,N}
    return TracedRArray{Base.promote_type(T, S),N}
end

function promote_to(::Type{TracedRArray{T,N}}, rhs) where {T,N}
    if isa(rhs, TracedRArray)
        if typeof(rhs) == TracedRArray{T,N}
            return rhs
        end
        return TracedRArray{T,N}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.convert(
                    rhs.mlir_data; result=mlir_type(TracedRArray{T,N}, size(rhs))
                ),
                1,
            ),
            size(rhs),
        )
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(T(rhs)), mlir_type(TracedRArray{T,N}, size(rhs)))
        ta = TracedRArray{T,N}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), size(rhs)
        )
        return ta
    end
    T0 = eltype(rhs)
    attr = MLIR.IR.DenseElementsAttribute(collect(rhs))
    return promote_to(
        TracedRArray{T,N},
        TracedRArray{T0,length(size(rhs))}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), size(rhs)
        ),
    )
end

function promote_to(::TracedRArray{T,N}, rhs) where {T,N}
    return promote_to(TracedRArray{T,N}, rhs)
end

for (jlop, hloop) in (
    (:(Base.min), :minimum),
    (:(Base.max), :maximum),
    (:(Base.:+), :add),
    (:(Base.:-), :subtract),
    (:(Base.:*), :multiply),
    (:(Base.:/), :divide),
    (:(Base.:^), :power),
)
    @eval begin
        function $(jlop)(
            @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs::TracedRArray{T,0})
        ) where {T}
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$(hloop)(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRArray{T1,0}), @nospecialize(rhs::TracedRArray{T2,0})
        ) where {T1,T2}
            commonTy = TracedRArray{Base.promote_type(T1, T2),0}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return $(jlop)(lhs, rhs)
        end
    end

    for otherType in (Number, Any)
        @eval begin
            function $(jlop)(
                @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs::$(otherType))
            ) where {T}
                rhs = promote_to(lhs, rhs)
                return $(jlop)(lhs, rhs)
            end

            function $(jlop)(
                @nospecialize(lhs::$(otherType)), @nospecialize(rhs::TracedRArray{T,0})
            ) where {T}
                lhs = promote_to(rhs, lhs)
                return $(jlop)(lhs, rhs)
            end
        end
    end
end

function Base.ifelse(
    @nospecialize(pred::TracedRArray{Bool,0}),
    @nospecialize(x::TracedRArray{T1,0}),
    @nospecialize(y::TracedRArray{T2,0})
) where {T1,T2}
    return TracedRArray{promote_type(T1, T2),0}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.select(pred.mlir_data, x.mlir_data, y.mlir_data), 1
        ),
        size(pred),
    )
end

Base.abs2(x::Reactant.TracedRArray{T,0}) where {T} = x * conj(x)

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::TracedRArray{T,0}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
    (:(Base.abs), :abs),
    (:(Base.:-), :negate),
    (:(Base.sin), :sine),
    (:(Base.cos), :cosine),
    (:(Base.tanh), :tanh),
    (:(Base.FastMath.tanh_fast), :tanh),
    (:(Base.exp), :exponential),
    (:(Base.FastMath.exp_fast), :exponential),
    (:(Base.log), :log),
    (:(Base.sqrt), :sqrt),
)
    @eval begin
        function $jlop(@nospecialize(lhs::TracedRArray{T,0})) where {T}
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1),
                size(lhs),
            )
        end
    end
end

struct TypeCast{T<:Number} <: Function end

function (::TypeCast{T})(x::TracedRArray{T2,0}) where {T,T2}
    return promote_to(TracedRArray{T,0}, x)
end

elem_apply(::Type{T}, x::TracedRArray{T}) where {T<:Number} = x
function elem_apply(::Type{T}, x::TracedRArray{T2}) where {T<:Number,T2<:Number}
    # Special Path to prevent going down a despecialized path
    return elem_apply(TypeCast{T}(), x)
end

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    all(iszero ∘ ndims, args) && return f(args...)

    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        f, args, (), string(f) * "_broadcast_scalar", false; toscalar=true
    )

    invmap = IdDict()
    for (k, v) in seen_args
        invmap[v] = k
    end

    input_shapes = size.(keys(seen_args))
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

for (jlop, hloop, hlocomp, merge) in (
    (:(Base.:(==)), :compare, "EQ", :all),
    (:(Base.:(!=)), :compare, "NE", :any),
    (:(Base.:(>=)), :compare, "GE", nothing),
    (:(Base.:(>)), :compare, "GT", nothing),
    (:(Base.:(<=)), :compare, "LE", nothing),
    (:(Base.:(<)), :compare, "LT", nothing),
)
    @eval begin
        function $(jlop)(
            @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs::TracedRArray{T,0})
        ) where {T}
            return TracedRArray{Bool,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(
                        lhs.mlir_data,
                        rhs.mlir_data;
                        comparison_direction=MLIR.API.stablehloComparisonDirectionAttrGet(
                            MLIR.IR.context(), $hlocomp
                        ),
                    ),
                    1,
                ),
                size(lhs),
            )
        end

        function $(jlop)(
            @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs)
        ) where {T}
            return $(jlop)(lhs, promote_to(lhs, rhs))
        end

        function $(jlop)(
            @nospecialize(lhs), @nospecialize(rhs::TracedRArray{T,0})
        ) where {T}
            return $(jlop)(promote_to(rhs, lhs), rhs)
        end
    end

    if merge !== nothing
        @eval begin
            function $jlop(
                @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs::TracedRArray{T,N})
            ) where {T,N}
                elems = $(jlop).(lhs, rhs)
                return N == 0 ? elems : $(merge)(elems)
            end
        end
    end
end

function Base.:*(
    @nospecialize(lhs::TracedRArray{T,2}), @nospecialize(rhs::TracedRArray{T,2})
) where {T}
    lhsty = MLIR.IR.type(lhs.mlir_data)
    rhsty = MLIR.IR.type(rhs.mlir_data)
    resty = MLIR.IR.TensorType((size(lhs, 1), size(rhs, 2)), eltype(lhsty))
    dot_dimension_numbers = MLIR.API.stablehloDotDimensionNumbersGet(
        MLIR.IR.context(), 0, [], 0, [], 1, [1], 1, [0]
    )
    prec = MLIR.IR.Attribute(
        MLIR.API.stablehloPrecisionAttrGet(MLIR.IR.context(), "DEFAULT")
    )
    precar = MLIR.IR.Attribute([prec, prec])
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dot_general(
            lhs.mlir_data,
            rhs.mlir_data;
            result_0=resty,
            dot_dimension_numbers=dot_dimension_numbers,
            precision_config=precar,
        ),
        1,
    )
    return TracedRArray{T,2}((), res, (size(lhs, 1), size(rhs, 2)))
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

    if isnothing(init)
        init = Base.reduce_empty(Base.BottomRF(op), T)
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
        TracedRArray{T,0}((), MLIR.IR.argument(fnbody, i), ()) for
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

    if dims != (:)
        red = MLIR.IR.result(
            MLIR.Dialects.stablehlo.reshape(
                red; result_0=MLIR.IR.TensorType(toonedims, eltype(MLIR.IR.type(red)))
            ),
            1,
        )
        red = TracedRArray{T,length(toonedims)}((), red, (toonedims...,))
    else
        red = TracedRArray{T,length(outdims)}((), red, (outdims...,))
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

struct AbstractReactantArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

AbstractReactantArrayStyle(::Val{N}) where {N} = AbstractReactantArrayStyle{N}()
AbstractReactantArrayStyle{M}(::Val{N}) where {N,M} = AbstractReactantArrayStyle{N}()

# function Broadcast.materialize(bc::Broadcasted) 
#    @show bc
#    inst = instantiate(bc)
#    @show inst
#    copy(inst)
# end

function BroadcastStyle(::Type{<:AnyTracedRArray{T,N}}) where {T,N}
    return AbstractReactantArrayStyle{N}()
end

function Base.similar(
    bc::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{T}, dims
) where {T,N}
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

function broadcast_to_size(arg::AbstractArray, rsize)
    attr = MLIR.IR.DenseElementsAttribute(arg)
    len = ndims(arg)
    @assert typeof(len) == Int
    arg = TracedRArray{eltype(arg),len}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), size(arg)
    )
    return broadcast_to_size(arg, rsize)
end

function broadcast_to_size(arg::Base.RefValue, rsize)
    # XXX: don't we want to expand here to rsize?
    return arg
end

function broadcast_to_size(arg::T, rsize) where {T<:Number}
    TT = MLIR.IR.TensorType([Int64(s) for s in rsize], MLIR.IR.Type(typeof(arg)))
    attr = Base.fill(arg, TT)
    return arg = TracedRArray{T,length(rsize)}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), rsize
    )
end

function broadcast_to_size(arg::AnyTracedRArray, rsize)
    arg = materialize_traced_array(arg)
    size(arg) == rsize && return arg
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

function _copyto!(dest::TracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = elem_apply(bc.f, args...)
    dest.mlir_data = res.mlir_data
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
