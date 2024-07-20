using Base.Broadcast
using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

mutable struct TracedRArray{T,N} <: RArray{T,N}
    paths::Tuple
    mlir_data::Union{Nothing,MLIR.IR.Value}
    shape::NTuple{N,Int}

    function TracedRArray{T,N}(
        paths::Tuple, mlir_data::Union{Nothing,MLIR.IR.Value}, shape
    ) where {T,N}
        if !isnothing(mlir_data)
            @assert size(MLIR.IR.type(mlir_data)) == shape
        end
        return new{T,N}(paths, mlir_data, shape)
    end
end

Base.size(x::TracedRArray) = x.shape

Base.copy(A::TracedRArray{T,N}) where {T,N} = TracedRArray((), A.mlir_data, size(A))

function Base.similar(x::TracedRArray{T,N}, ::Type{T2}) where {T,N,T2}
    return TracedRArray{T2,N}((), nothing, size(x))
end

function Base.show(io::IOty, X::TracedRArray{T,N}) where {T,N,IOty<:Union{IO,IOContext}}
    print(io, "TracedRArray{", T, ",", N, "N}(", X.paths, ", ")
    return print(io, X.mlir_data, ")")
end

function Base.reshape(A::TracedRArray{T,N}, dims::NTuple{NT,Int}) where {T,N,NT}
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))

    # HLO reshape semantics collapse the opposite way
    res1 = MLIR.IR.result(
        MLIR.Dialects.stablehlo.transpose(
            A.mlir_data;
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

function Base.permutedims(A::TracedRArray{T,N}, perm) where {T,N}
    return TracedArray{T,N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.transpose(
                A.mlir_data, MLIR.IR.DenseArrayAttribute([Int64(i - 1) for i in perm])
            ),
            1,
        ),
        tuple(size(A, i) for i in perm),
    )
end

function Base.promote_rule(
    ::Type{TracedRArray{T,N}}, ::Type{TracedRArray{S,N}}
) where {T,S,N}
    return TracedRArray{Base.promote_type(T, S),N}
end

function Base.promote_rule(A::Type{T}, B::Type{TracedRArray{S,N}}) where {T,S,N}
    return TracedRArray{Base.promote_type(T, S),N}
end

function promote_to(::Type{TracedRArray{T,N}}, rhs) where {T,N}
    if isa(rhs, TracedRArray)
        return TracedRArray{T,N}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.convert(
                    rhs.mlir_data; result=mlir_type(TracedRArray{T,N})
                ),
                1,
            ),
        )
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(T(rhs)), mlir_type(TracedRArray{T,N}))
        ta = TracedRArray{T,N}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
        )
        return ta
    end
    attr = MLIR.IR.DenseElementsAttribute(mlir_type(TracedRArray{T,N}), rhs)
    return TracedRArray{T,N}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), size(rhs)
    )
end

function promote_to(lhs::TracedRArray{T,N}, rhs) where {T,N}
    return promote_to(TracedRArray{T,N}, rhs)
end

for (jlop, hloop) in (
    (:(Base.min), :minimum),
    (:(Base.max), :maximum),
    (:(Base.:+), :add),
    (:(Base.:-), :subtract),
)
    @eval begin
        function $jlop(lhs::TracedRArray{T,N}, rhs::TracedRArray{T2,N}) where {T,T2,N}
            commonTy = TracedRArray{Base.promote_type(T, T2),N}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return commonTy(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                size(lhs),
            )
        end

        function $jlop(lhs::TracedRArray{T,N}, rhs::TracedRArray{T,N}) where {T,N}
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                size(lhs),
            )
        end

        function $jlop(lhs::TracedRArray{T,N}, rhs) where {T,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                size(lhs),
            )
        end

        function $jlop(lhs, rhs::TracedRArray{T,N}) where {T,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                size(lhs),
            )
        end
    end
end

for (jlop, hloop) in ((:(Base.:*), :multiply), (:(Base.:/), :divide), (:(Base.:^), :power))
    @eval begin
        function $jlop(lhs::TracedRArray{T,0}, rhs::TracedRArray{T2,0}) where {T,T2}
            commonTy = TracedRArray{Base.promote_type(T, T2),0}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return commonTy(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $jlop(lhs::TracedRArray{T,0}, rhs::TracedRArray{T,0}) where {T}
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $jlop(lhs::TracedRArray{T,0}, rhs) where {T}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $jlop(lhs, rhs::TracedRArray{T,0}) where {T}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{T,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end
    end
end

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::TracedRArray{T,0}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop) in (
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
        function $jlop(lhs::TracedRArray{T,N}) where {T,N}
            return TracedRArray{T,N}(
                (),
                MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1),
                size(lhs),
            )
        end
    end
end

function elem_apply(f, args::Vararg{Any,Nargs}) where {Nargs}
    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        f, args, (), string(f) * "_broadcast_scalar", false; toscalar=true
    )

    invmap = IdDict()
    OutShape = nothing
    for (k, v) in seen_args
        invmap[v] = k
        OutShape = size(k)
    end
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

    seen_results = IdDict()
    traced2_result = make_tracer(seen_results, result, (), TracedSetPath; tobatch=OutShape)

    func2.operation = MLIR.API.MlirOperation(C_NULL)

    return traced2_result
end

for (jlop, hloop, hlocomp) in (
    (:(Base.:(==)), :compare, "EQ"),
    (:(Base.:(!=)), :compare, "NE"),
    (:(Base.:(>=)), :compare, "GE"),
    (:(Base.:(>)), :compare, "GT"),
    (:(Base.:(<=)), :compare, "LE"),
    (:(Base.:(<)), :compare, "LT"),
)
    @eval begin
        function elem_apply(
            ::typeof($jlop), lhs::TracedRArray{T,N}, rhs::TracedRArray{T,N}
        ) where {T,N}
            return TracedRArray{T,N}(
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

        function elem_apply(::typeof($jlop), lhs::TracedRArray{T,N}, rhs) where {T,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{T,N}(
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

        function elem_apply(::typeof($jlop), lhs, rhs::TracedRArray{T,N}) where {T,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{T,N}(
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
    end
end

function Base.:*(lhs::TracedRArray{T,2}, rhs::TracedRArray{T,2}) where {T}
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

function Base.mapreduce(f, op, A::TracedRArray{T,N}; dims=:, init=nothing) where {T,N}
    if dims isa Int
        dims = [dims]
    end

    if isnothing(init)
        init = Base.reduce_empty(Base.BottomRF(op), T)
    else
        init = init::T
    end

    init = [broadcast_to_size(init, ()).mlir_data]

    inp = [elem_apply(f, A).mlir_data]

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

function Base.mapreducedim!(f, op, R::TracedRArray, A::Base.AbstractArrayOrBroadcasted)
    tmp = broadcast_to_size(Base.mapreduce(f, op, A; dims=1), (1, size(R)[2:end]...))
    R.mlir_data = elem_apply(op, R, tmp).mlir_data
    return R
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

BroadcastStyle(::Type{T}) where {T<:TracedRArray} = AbstractReactantArrayStyle{ndims(T)}()

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
    return arg
end

function broadcast_to_size(arg::TracedRArray, rsize)
    return arg
end

function broadcast_to_size(arg::Base.RefValue, rsize)
    return arg
end

function Base.fill!(A::TracedRArray{T,N}, x) where {T,N}
    bcast = broadcast_to_size(T(x), size(A))
    A.mlir_data = bcast.mlir_data
    return A
end

function broadcast_to_size(arg::T, rsize) where {T<:Number}
    TT = MLIR.IR.TensorType([Int64(s) for s in rsize], MLIR.IR.Type(typeof(arg)))
    attr = Base.fill(arg, TT)
    return arg = TracedRArray{T,length(rsize)}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), rsize
    )
end

function broadcast_to_size(arg::Broadcast.Extruded, rsize)
    rsize2 = (keep ? rsizev : 1 for (keep, rsizev) in zip(arg.keeps, rsize))

    x = broadcast_to_size(arg.x, rsize2)

    if size(x) == rsize
        return x
    end

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

    len = length(rsize)
    @assert typeof(len) == Int
    return TracedRArray{eltype(x),len}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.broadcast_in_dim(
                x.mlir_data;
                result_0=MLIR.IR.TensorType([t for t in rsize], eltype(mlirty)),
                broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims),
            ),
            1,
        ),
        rsize,
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

@enum TraceMode begin
    ConcreteToTraced = 1
    TracedTrack = 2
    TracedToConcrete = 3
    ArrayToConcrete = 4
    TracedSetPath = 5
end

for T in (
    DataType,
    Module,
    Nothing,
    Symbol,
    AbstractChar,
    AbstractFloat,
    Integer,
    AbstractString,
    RArray,
)
    @eval function traced_type(::Type{T}, seen, mode) where {T<:$T}
        return T
    end
end

function traced_type(::Type{C}, seen, mode) where {T,C<:Complex{T}}
    if !(C isa UnionAll)
        return Complex{traced_type(T, seen, mode)}
    else
        return @invoke traced_type(C::Type{Any}, seen, mode)
    end
end

function traced_type(::Type{T}, seen, mode) where {T<:Function}
    # functions are directly returned
    if sizeof(T) == 0
        return T
    end

    # in closures, enclosured variables need to be traced
    N = fieldcount(T)
    traced_fieldtypes = ntuple(Val(N)) do i
        return traced_type(fieldtype(T, i), seen, mode)
    end

    # closure are struct types with the types of enclosured vars as type parameters
    return Core.apply_type(T.name.wrapper, traced_fieldtypes...)
end

@inline is_concrete_tuple(x::T2) where {T2} =
    (x <: Tuple) && !(x === Tuple) && !(x isa UnionAll)

function traced_type(::Type{T}, seen, mode) where {T<:Tuple}
    if !Base.isconcretetype(T) || !is_concrete_tuple(T) || T isa UnionAll
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    elseif is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
        # Tuple{((T2 isa Core.TypeofVararg ? Any : T2) for T2 in T.parameters)...}
        throw(AssertionError("Type tuple of vararg $T is not supported"))
    end
    return Tuple{traced_type(T.parameters[i], seen, mode) for i in 1:length(T.parameters)}
end

function traced_type(::Type{T}, seen, mode) where {N,V,T<:NamedTuple{N,V}}
    return NamedTuple{N,traced_type(V, seen, mode)}
end

function traced_type(::Type{T}, seen, mode) where {K,V,T<:AbstractDict{K,V}}
    dictty = T.name.wrapper
    return dictty{K,traced_type(V, seen, mode)}
end

@inline getmap(::Val{T}) where {T} = nothing
@inline getmap(::Val{T}, a, b, args...) where {T} = getmap(Val(T), args...)
@inline getmap(::Val{T}, ::Val{T}, ::Val{T2}, args...) where {T,T2} = T2

function traced_type(::Type{T}, seen, mode) where {T}
    if T === Any
        return T
    end

    if T === Union{}
        return T
    end

    if Enzyme.Compiler.isghostty(T) || Core.Compiler.isconstType(T)
        return T
    end

    # unknown number of fields
    if T isa UnionAll
        aT = Base.argument_datatype(T)
        if isnothing(aT)
            throw("Unhandled type $T")
        end
        if isnothing(Base.datatype_fieldcount(aT))
            throw("Unhandled type $T")
        end
    end

    if T isa Union
        return Union{traced_type(T.a, seen, mode),traced_type(T.b, seen, mode)}
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        throw("Unhandled abstract type $T")
    end

    if !(Base.isconcretetype(T) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    nextTy = getmap(Val(T), seen...)
    if !isnothing(nextTy)
        return nextTy
    end

    seen2 = (Val(T), Val(T), seen...)

    changed = false
    subTys = Type[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type(subT, seen2, mode)
        changed |= subT != subTT
        push!(subTys, subTT)
    end

    if !changed
        return T
    end

    subParms = []
    for SST in T.parameters
        if SST isa Type
            TrT = traced_type(SST, seen, mode)
            push!(subParms, TrT)
        else
            push!(subParms, SST)
        end
    end

    TT2 = Core.apply_type(T.name.wrapper, subParms...)
    seen3 = (Val(T), Val(TT2), seen...)
    if fieldcount(T) == fieldcount(TT2)
        legal = true
        for f in 1:fieldcount(T)
            subT = fieldtype(T, f)
            subT2 = fieldtype(TT2, f)
            subTT = traced_type(subT, seen3, mode)
            if subT2 != subTT
                legal = false
                break
            end
        end
        if legal
            return TT2
        end
    end

    name = Symbol[]
    throw(error("Cannot convert type $T, best attempt $TT2 failed"))
end

function traced_type(::Type{T}, seen, ::Val{mode}) where {T<:ConcreteRArray,mode}
    if mode == ConcreteToTraced
        @inline base_typet(TV::TT) where {TT<:UnionAll} =
            UnionAll(TV.var, base_typet(TV.body))
        @inline base_typet(TV::TT) where {TT<:DataType} = TracedRArray{TV.parameters...}
        return base_typet(T)
    elseif mode == TracedToConcrete
        return T
    else
        throw("Abstract RArray cannot be made concrete")
    end
end

function traced_type(::Type{T}, seen::ST, ::Val{mode}) where {ST,T<:TracedRArray,mode}
    if mode == ConcreteToTraced
        throw("TracedRArray $T cannot be traced")
    elseif mode == TracedToConcrete
        @inline base_typec(TV::TT) where {TT<:UnionAll} =
            UnionAll(TV.var, base_typec(TV.body))
        @inline base_typec(TV::TT) where {TT<:DataType} = ConcreteRArray{TV.parameters...}
        return base_typec(T)
    elseif mode == TracedTrack || mode == TracedSetPath
        return T
    else
        throw("Abstract RArray $T cannot be made concrete in mode $mode")
    end
end

function traced_type(::Type{T}, seen, mode) where {T<:XLAArray}
    throw("XLA $T array cannot be traced")
end

function traced_type(::Type{A}, seen::ST, ::Val{mode}) where {T,N,A<:Array{T,N},ST,mode}
    if mode == ArrayToConcrete && T <: AbstractFloat
        return ConcreteRArray{T,N}
    else
        return Array{traced_type(T, seen, Val(mode)),N}
    end
end

for P in (Ptr, Core.LLVMPtr, Base.RefValue)
    @eval function traced_type(::Type{P}, seen, mode) where {T,P<:$P{T}}
        return $P{traced_type(T, seen, mode)}
    end
end

function traced_type(::Type{Val{T}}, seen, mode) where {T}
    if traced_type(typeof(T), seen, mode) == typeof(T)
        return T
    end
    throw("Val type $T cannot be traced")
end

append_path(path, i) = (path..., i)

function make_tracer(seen, prev::RT, path, mode; toscalar=false, tobatch=nothing) where {RT}
    if haskey(seen, prev)
        return seen[prev]
    end
    TT = traced_type(RT, (), Val(mode))
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    if TT <: NamedTuple
        changed = false
        subs = []
        for i in 1:nf
            xi = Base.getfield(prev, i)
            xi2 = make_tracer(seen, xi, append_path(path, i), mode; toscalar, tobatch)
            if xi !== xi2
                changed = true
            end
            push!(subs, xi2)
        end
        if !changed
            seen[prev] = prev
            return prev
        end
        tup = (subs...,)
        return NamedTuple{TT.parameters[1],typeof(tup)}(tup)
    end

    if ismutabletype(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        seen[prev] = y
        changed = false
        for i in 1:nf
            if isdefined(prev, i)
                xi = Base.getfield(prev, i)
                xi2 = make_tracer(seen, xi, append_path(path, i), mode; toscalar, tobatch)
                if xi !== xi2
                    changed = true
                end
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, xi2)
            end
        end
        if !changed
            seen[prev] = prev
            return prev
        end
        return y
    end

    if nf == 0
        return prev
    end

    flds = Vector{Any}(undef, nf)
    changed = false
    for i in 1:nf
        if isdefined(prev, i)
            xi = Base.getfield(prev, i)
            xi2 = make_tracer(seen, xi, append_path(path, i), mode; toscalar, tobatch)
            if xi !== xi2
                changed = true
            end
            flds[i] = xi2
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    if !changed
        seen[prev] = prev
        return prev
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), TT, flds, nf)
    seen[prev] = y
    return y
end

function make_tracer(seen, prev::ConcreteRArray{T,N}, path, mode; kwargs...) where {T,N}
    if mode == ArrayToConcrete
        return prev
    end
    if mode != ConcreteToTraced
        throw("Cannot trace concrete")
    end
    if haskey(seen, prev)
        return seen[prev]::TracedRArray{T,N}
    end
    @assert N isa Int
    res = TracedRArray{T,N}((path,), nothing, size(prev))
    seen[prev] = res
    return res
end

function make_tracer(
    seen, prev::TracedRArray{T,N}, path, mode; toscalar=false, tobatch=nothing
) where {T,N}
    if mode == ConcreteToTraced
        throw("Cannot trace existing trace type")
    end
    if mode == TracedTrack
        prev.paths = (prev.paths..., path)
        if !haskey(seen, prev)
            return seen[prev] = prev
        end
        return prev
    end
    if mode == TracedSetPath
        if haskey(seen, prev)
            return seen[prev]
        end
        res = if toscalar
            TracedRArray{T,0}((path,), nothing, ())
        elseif !isnothing(tobatch)
            TracedRArray{T,length(tobatch)}((path,), prev.mlir_data, tobatch)
        else
            TracedRArray{T,N}((path,), prev.mlir_data, size(prev))
        end
        seen[prev] = res
        return res
    end

    if mode == TracedToConcrete
        if haskey(seen, prev)
            return seen[prev]::ConcreteRArray{T,N}
        end
        res = ConcreteRArray{T,N}(XLA.AsyncEmptyBuffer, size(prev))
        seen[prev] = res
        return res
    end

    throw("Cannot Unknown trace mode $mode")
end

make_tracer(seen, prev::RT, path, mode; kwargs...) where {RT<:AbstractFloat} = prev

make_tracer(seen, prev::Symbol, path, mode; kwargs...) = prev

function make_tracer(
    seen, prev::Complex{RT}, path, mode; toscalar=false, tobatch=nothing
) where {RT}
    return Complex(
        make_tracer(seen, prev.re, append_path(path, :re), mode; toscalar, tobatch),
        make_tracer(seen, prev.im, append_path(path, :im), mode; toscalar, tobatch),
    )
end

function make_tracer(seen, prev::RT, path, mode; kwargs...) where {RT<:Array}
    if haskey(seen, prev)
        return seen[prev]
    end
    if mode == ArrayToConcrete && eltype(RT) <: AbstractFloat
        return seen[prev] = ConcreteRArray(prev)
    end
    TT = traced_type(eltype(RT), (), Val(mode))
    newa = Array{TT,ndims(RT)}(undef, size(prev))
    seen[prev] = newa
    same = true
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            nv = make_tracer(seen, pv, append_path(path, I), mode; kwargs...)
            if pv !== nv
                same = false
            end
            @inbounds newa[I] = nv
        end
    end
    if same
        seen[prev] = prev
        return prev
    end
    return newa
end

function make_tracer(seen, prev::RT, path, mode; kwargs...) where {RT<:Tuple}
    return (
        (
            make_tracer(seen, v, append_path(path, i), mode; kwargs...) for
            (i, v) in enumerate(prev)
        )...,
    )
end

function make_tracer(seen, prev::NamedTuple{A,RT}, path, mode; kwargs...) where {A,RT}
    return NamedTuple{A,traced_type(RT, (), Val(mode))}((
        (
            make_tracer(
                seen, Base.getfield(prev, i), append_path(path, i), mode; kwargs...
            ) for i in 1:length(A)
        )...,
    ))
end

function make_tracer(seen, prev::Core.Box, path, mode; kwargs...)
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    tr = make_tracer(seen, prev2, append_path(path, :contents), mode; kwargs...)
    if tr == prev2
        seen[prev] = prev
        return prev
    end
    res = Core.Box(tr)
    seen[prev] = res
    return res
end
