function promote_to(::Type{TracedRArray{ElType,Shape,N}}, rhs) where {ElType,Shape,N}
    if isa(rhs, TracedRArray)
        return TracedRArray{ElType,Shape,N}(
            (),
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.convert(
                    rhs.mlir_data; result=mlir_type(TracedRArray{ElType,Shape,N})
                ),
                1,
            ),
        )
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(ElType(rhs)), mlir_type(TracedRArray{ElType,Shape,N}))
        ta = TracedRArray{ElType,Shape,N}(
            (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
        )
        return ta
    end
    attr = MLIR.IR.DenseElementsAttribute(mlir_type(TracedRArray{ElType,Shape,N}), rhs)
    return TracedRArray{ElType,Shape,N}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
    )
end

function promote_to(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
    return promote_to(TracedRArray{ElType,Shape,N}, rhs)
end

for (jlop, hloop, RT) in (
    (:(Base.min), :minimum, :ElType),
    (:(Base.max), :maximum, :ElType),
    (:(Base.:+), :add, :ElType),
    (:(Base.:-), :subtract, :ElType),
)
    @eval begin
        function $jlop(
            lhs::TracedRArray{ElType,Shape,N}, rhs::TracedRArray{ElType2,Shape,N}
        ) where {ElType,ElType2,Shape,N}
            commonTy = TracedRArray{Base.promote_type(ElType, ElType2),Shape,N}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return commonTy(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(
            lhs::TracedRArray{ElType,Shape,N}, rhs::TracedRArray{ElType,Shape,N}
        ) where {ElType,Shape,N}
            return TracedRArray{$RT,Shape,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(
            lhs::TracedRArray{ElType,(),0}, rhs::TracedRArray{ElType,(),0}
        ) where {ElType}
            return TracedRArray{$RT,(),0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end
    end

    for otherType in (Number, Any, TracedRArray{S,(),0} where {S})
        @eval begin
            function $jlop(
                lhs::TracedRArray{ElType,Shape,N}, rhs::$otherType
            ) where {ElType,Shape,N}
                rhs = promote_to(lhs, rhs)
                return TracedRArray{$RT,Shape,N}(
                    (),
                    MLIR.IR.result(
                        MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                    ),
                )
            end

            function $jlop(
                lhs::$otherType, rhs::TracedRArray{ElType,Shape,N}
            ) where {ElType,Shape,N}
                lhs = promote_to(rhs, lhs)
                return TracedRArray{$RT,Shape,N}(
                    (),
                    MLIR.IR.result(
                        MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                    ),
                )
            end
        end
    end
end

Base.abs2(x::Reactant.TracedRArray{T,(),0}) where {T} = x * conj(x)

function Base.literal_pow(
    ::Base.RefValue{typeof(^)}, x::Reactant.TracedRArray{T,(),0}, ::Base.RefValue{Val{P}}
) where {T,P}
    return Base.literal_pow(^, x, Val(P))
end

for (jlop, hloop, RT) in (
    (:(Base.:*), :multiply, :ElType),
    (:(Base.:/), :divide, :ElType),
    (:(Base.:^), :power, :ElType),
)
    @eval begin
        function $jlop(
            lhs::TracedRArray{ElType,Shape,0}, rhs::TracedRArray{ElType2,Shape,0}
        ) where {ElType,ElType2,Shape}
            commonTy = TracedRArray{Base.promote_type(ElType, ElType2),Shape,0}
            lhs = promote_to(commonTy, lhs)
            rhs = promote_to(commonTy, rhs)
            return commonTy(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(
            lhs::TracedRArray{ElType,Shape,0}, rhs::TracedRArray{ElType,Shape,0}
        ) where {ElType,Shape}
            return TracedRArray{$RT,Shape,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs::TracedRArray{ElType,Shape,0}, rhs) where {ElType,Shape}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,Shape,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        # Base defines ::AbstractArray / ::Number, so we need this to avoid ambiguity
        function $jlop(lhs::TracedRArray{ElType,Shape,0}, rhs::Number) where {ElType,Shape}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,Shape,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs, rhs::TracedRArray{ElType,Shape,0}) where {ElType,Shape}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{$RT,Shape,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end
    end
end

function elem_apply(
    ::typeof(Base.ifelse),
    pred::TracedRArray{Bool,Shape,N},
    x::TracedRArray{ElType1,Shape,N},
    y::TracedRArray{ElType2,Shape,N},
) where {ElType1,ElType2,Shape,N}
    return TracedRArray{promote_type(ElType1, ElType2),Shape,N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.select(pred.mlir_data, x.mlir_data, y.mlir_data), 1
        ),
    )
end

function Base.:*(
    lhs::TracedRArray{ElType,Shape,2}, rhs::TracedRArray{ElType,Shape2,2}
) where {ElType,Shape,Shape2}
    lhsty = MLIR.IR.type(lhs.mlir_data)
    rhsty = MLIR.IR.type(rhs.mlir_data)
    resty = MLIR.IR.TensorType((Base.size(lhsty)[1], Base.size(rhsty)[2]), eltype(lhsty))
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
    return TracedRArray{ElType,(Shape[1], Shape2[2]),2}((), res)
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
        function $jlop(lhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}(
                (), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1)
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
    @assert OutShape !== nothing
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

for (jlop, hloop, hlocomp, RT) in (
    (:(Base.:(==)), :compare, "EQ", :ElType),
    (:(Base.:(!=)), :compare, "NE", :ElType),
    (:(Base.:(>=)), :compare, "GE", :ElType),
    (:(Base.:(>)), :compare, "GT", :ElType),
    (:(Base.:(<=)), :compare, "LE", :ElType),
    (:(Base.:(<)), :compare, "LT", :ElType),
)
    @eval begin
        function elem_apply(
            ::typeof($jlop),
            lhs::TracedRArray{ElType,Shape,N},
            rhs::TracedRArray{ElType,Shape,N},
        ) where {ElType,Shape,N}
            return TracedRArray{$RT,Shape,N}(
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
            )
        end

        function elem_apply(
            ::typeof($jlop), lhs::TracedRArray{ElType,Shape,N}, rhs
        ) where {ElType,Shape,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,Shape,N}(
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
            )
        end

        function elem_apply(
            ::typeof($jlop), lhs, rhs::TracedRArray{ElType,Shape,N}
        ) where {ElType,Shape,N}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{$RT,Shape,N}(
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
            )
        end
    end
end

@inline function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    return reshape(A, Base._reshape_uncolon(A, dims))
end

@inline function Base.reshape(
    A::ConcreteRArray{T,Shape,N}, dims::NTuple{NT,Int}
) where {T,Shape,N,NT}
    prod(dims) == prod(size(A)) || Base._throw_dmrsa(dims, prod(size(A)))
    host = convert(Array{T,N}, A)
    # HLO reshape semantics collapse the opposite so enforce on Julia Side
    # until we later make the transpose/reshape/transpose
    host = reshape(host, dims)
    client = XLA.client(A.data)
    device = XLA.device(A.data)
    return ConcreteRArray{T,dims,NT}(
        XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, host, device), nothing)
    )
    # ConcreteRArray{T, dims, NT}(XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, XLA.to_row_major(host), device), nothing))
end

Base.copy(A::TracedRArray{T,Shape,N}) where {T,Shape,N} = TracedRArray((), A.mlir_data)

@inline function Base.permutedims(A::TracedRArray{T,Shape,N}, perm) where {T,Shape,N}
    return TracedArray{T,tuple(Shape[i] for i in perm),N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.transpose(
                A.mlir_data, MLIR.IR.DenseArrayAttribute([Int64(i - 1) for i in perm])
            ),
            1,
        ),
    )
end

@inline function Base.reshape(
    A::TracedRArray{T,Shape,N}, dims::NTuple{NT,Int}
) where {T,Shape,N,NT}
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

    return TracedRArray{T,dims,NT}((), res3)
end

using Base.Broadcast

using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle, instantiate

struct AbstractReactantArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
AbstractReactantArrayStyle(::Val{N}) where {N} = AbstractReactantArrayStyle{N}()
AbstractReactantArrayStyle{M}(::Val{N}) where {N,M} = AbstractReactantArrayStyle{N}()

# @inline function Broadcast.materialize(bc::Broadcasted) 
#    @show bc
#    inst = instantiate(bc)
#    @show inst
#    copy(inst)
# end

BroadcastStyle(::Type{T}) where {T<:TracedRArray} = AbstractReactantArrayStyle{ndims(T)}()

function Base.similar(x::TracedRArray{T,Shape,N}, ::Type{T2}) where {T,Shape,N,T2}
    return TracedRArray{T2,Shape,N}((), nothing)
end

@inline function Base.similar(
    bc::Broadcasted{AbstractReactantArrayStyle{N}}, ::Type{T}, dims
) where {T,N}
    @assert N isa Int
    return TracedRArray{T,map(length, dims),N}((), nothing)
end

function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle{0}})
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    dest = copyto!(similar(bc, ElType), bc)
    return dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

@inline Base.eltype(b::Broadcast.Extruded{T}) where {T} = eltype(T)

# we need to override the outer copy method to make sure we never fall back to scalar
# iteration (see, e.g., CUDA.jl#145)
@inline function Broadcast.copy(bc::Broadcasted{<:AbstractReactantArrayStyle})
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

@inline function Base.materialize!(
    ::Style, dest, bc::Broadcasted
) where {Style<:AbstractReactantArrayStyle}
    return _copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end

@inline Base.copyto!(dest::TracedRArray, bc::Broadcasted{Nothing}) = _copyto!(dest, bc) # Keep it for ArrayConflict

@inline function Base.copyto!(
    dest::TracedRArray{ElType,Shape,N}, src::TracedRArray{ElType,Shape,N}
) where {ElType,Shape,N}
    dest.mlir_data = src.mlir_data
    return dest
end

@inline function broadcast_to_size(arg::AbstractArray, rsize)
    attr = MLIR.IR.DenseElementsAttribute(arg)
    len = ndims(arg)
    @assert typeof(len) == Int
    arg = TracedRArray{eltype(arg),size(arg),len}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
    )
    return arg
end

@inline function broadcast_to_size(arg::TracedRArray, rsize)
    return arg
end

@inline function broadcast_to_size(arg::Base.RefValue, rsize)
    return arg
end

function Base.fill!(A::TracedRArray{T,Shape,N}, x) where {T,Shape,N}
    bcast = broadcast_to_size(T(x), Shape)
    A.mlir_data = bcast.mlir_data
    return A
end

@inline function broadcast_to_size(arg::T, rsize) where {T<:Number}
    TT = MLIR.IR.TensorType([Int64(s) for s in rsize], MLIR.IR.Type(typeof(arg)))
    attr = Base.fill(arg, TT)
    return arg = TracedRArray{T,rsize,length(rsize)}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
    )
end

@inline function broadcast_to_size(arg::Broadcast.Extruded, rsize)
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
    return TracedRArray{eltype(x),rsize,len}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.broadcast_in_dim(
                x.mlir_data;
                result_0=MLIR.IR.TensorType([t for t in rsize], eltype(mlirty)),
                broadcast_dimensions=MLIR.IR.DenseArrayAttribute(dims),
            ),
            1,
        ),
    )
end

@inline function _copyto!(dest::TracedRArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest

    bc = Broadcast.preprocess(dest, bc)

    args = (broadcast_to_size(Base.materialize(a), size(bc)) for a in bc.args)

    res = elem_apply(bc.f, args...)
    dest.mlir_data = res.mlir_data
    return dest
end

function Base.mapreduce(
    f, op, A::TracedRArray{ElType,Shape,N}; dims=:, init=nothing
) where {ElType,Shape,N}
    if dims isa Int
        dims = [dims]
    end

    if init == nothing
        init = Base.reduce_empty(Base.BottomRF(op), ElType)
    else
        init = init::ElType
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
        TracedRArray{ElType,(),0}((), MLIR.IR.argument(fnbody, i)) for
        (i, ty) in enumerate(in_tys)
    )

    res = MLIR.IR.block!(fnbody) do
        tmp = broadcast_to_size(op(args...), ()).mlir_data
        MLIR.Dialects.stablehlo.return_(MLIR.IR.Value[tmp])
        return tmp
    end

    toonedims = [(in(i - 1, rdims) ? 1 : Shape[i]) for i in 1:N]
    outdims = [Shape[i] for i in 1:N if (i - 1) ∉ rdims]

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
        red = TracedRArray{ElType,(toonedims...,),length(toonedims)}((), red)
    else
        red = TracedRArray{ElType,(outdims...,),length(outdims)}((), red)
    end
    return red
end

function Base.mapreducedim!(f, op, R::TracedRArray, A::Base.AbstractArrayOrBroadcasted)
    tmp = broadcast_to_size(Base.mapreduce(f, op, A; dims=1), (1, size(R)[2:end]...))
    R.mlir_data = elem_apply(op, R, tmp).mlir_data
    return R
end

# Stdlib overloads
## Statistics
function Statistics.mean(A::TracedRArray{T,Shape,N}; dims=:) where {T,Shape,N}
    denom = dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)
    return mapreduce(identity, +, A; dims) / denom
end
function Statistics.var(
    A::TracedRArray{T,Shape,N}; dims=:, mean=nothing, corrected=true
) where {T,Shape,N}
    mean === nothing && (mean = Statistics.mean(A; dims))
    denom = (dims isa Colon ? length(A) : prod(Base.Fix1(size, A), dims)) - corrected
    return mapreduce(abs2, +, A .- mean; dims) / denom
end
