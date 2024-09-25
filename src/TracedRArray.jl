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

function Base.getindex(a::TracedRArray{T,N}, index::Vararg{Integer,N}) where {T,N}
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

function Base.getindex(
    a::TracedRArray{T,N}, indices::Vararg{Union{Base.AbstractUnitRange,Colon},N}
) where {T,N}
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
    return TracedRArray{T,N}((), res, Tuple(length.(indices)))
end

function Base.view(
    a::TracedRArray{T,N}, indices::Vararg{Union{Base.AbstractUnitRange,Colon},N}
) where {T,N}
    # TODO: Implement before merging the PR
    return error("view is not supported yet")
end

function Base.setindex!(
    a::TracedRArray{T,N}, v, indices::Vararg{Union{Base.AbstractUnitRange,Colon},N}
) where {T,N}
    indices = [promote_to(TracedRArray{Int, 0}, i isa Colon ? 1 : first(i))-1 for i in indices]
    v = promote_to(TracedRArray{T,N}, v)
    res = MLIR.IR.result(
        MLIR.Dialects.stablehlo.dynamic_update_slice(
           a.mlir_data, v, indices...
        ),
        1,
    )
    a.mlir_data = v.mlir_data
    return v
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

Base.only(A::TracedRArray{T,0}) where {T} = A

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
    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.transpose(
                A.mlir_data;
                permutation=MLIR.IR.DenseArrayAttribute([Int64(i - 1) for i in perm]),
            ),
            1,
        ),
        Tuple(size(A, i) for i in perm),
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
        if typeof(rhs) == TracedRArray{T, N}
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
    return promote_to(TracedRArray{T, N}, TracedRArray{T0,length(size(rhs))}(
        (), MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1), size(rhs)
    ))
end

function promote_to(lhs::TracedRArray{T,N}, rhs) where {T,N}
    return promote_to(TracedRArray{T,N}, rhs)
end

for (jlop, hloop, RT) in (
    (:(Base.min), :minimum, :T),
    (:(Base.max), :maximum, :T),
    (:(Base.:+), :add, :T),
    (:(Base.:-), :subtract, :T),
)
    @eval begin
        function $jlop(
            @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs::TracedRArray{T2,N})
        ) where {T,T2,N}
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

        function $jlop(
            @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs::TracedRArray{T,N})
        ) where {T,N}
            return TracedRArray{$RT,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                size(lhs),
            )
        end
    end

    for otherType in (Number, Any) #=TracedRArray{S,0} where {S}=#
        @eval begin
            function $jlop(
                @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs::$otherType)
            ) where {T,N}
                rhs = promote_to(lhs, rhs)
                return TracedRArray{$RT,N}(
                    (),
                    MLIR.IR.result(
                        MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                    ),
                    size(lhs),
                )
            end

            function $jlop(
                @nospecialize(lhs::$otherType), @nospecialize(rhs::TracedRArray{T,N})
            ) where {T,N}
                lhs = promote_to(rhs, lhs)
                return TracedRArray{$RT,N}(
                    (),
                    MLIR.IR.result(
                        MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                    ),
                    size(lhs),
                )
            end
        end
    end
end

for (jlop, hloop, RT) in
    ((:(Base.:*), :multiply, :T), (:(Base.:/), :divide, :T), (:(Base.:^), :power, :T))
    @eval begin
        function $jlop(
            @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs::TracedRArray{T2,0})
        ) where {T,T2}
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

        function $jlop(
            @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs::TracedRArray{T,0})
        ) where {T}
            return TracedRArray{$RT,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $jlop(@nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs)) where {T}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $jlop(@nospecialize(lhs), @nospecialize(rhs::TracedRArray{T,0})) where {T}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{$RT,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        # Base defines ::AbstractArray / ::Number, so we need this to avoid ambiguity
        function $jlop(
            @nospecialize(lhs::TracedRArray{T,0}), @nospecialize(rhs::Number)
        ) where {T}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
        end

        function $jlop(
            @nospecialize(lhs::Number), @nospecialize(rhs::TracedRArray{T,0})
        ) where {T}
            lhs = promote_to(rhs, lhs)
            return TracedRArray{$RT,0}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
                (),
            )
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
        function $jlop(@nospecialize(lhs::TracedRArray{T,N})) where {T,N}
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
            ::typeof($jlop),
            @nospecialize(lhs::TracedRArray{T,N}),
            @nospecialize(rhs::TracedRArray{T,N})
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

        function elem_apply(
            ::typeof($jlop), @nospecialize(lhs::TracedRArray{T,N}), @nospecialize(rhs)
        ) where {T,N}
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

        function elem_apply(
            ::typeof($jlop), @nospecialize(lhs), @nospecialize(rhs::TracedRArray{T,N})
        ) where {T,N}
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

function Base.mapreducedim!(
    @nospecialize(f),
    @nospecialize(op),
    @nospecialize(R::TracedRArray),
    A::Base.AbstractArrayOrBroadcasted,
)
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

function Base._cat(dims::Val{D}, A::TracedRArray{T,N}, Bs::TracedRArray...) where {T,N,D}
    @assert D isa Integer "Support for non-integer dimensions is not implemented yet."
    catdims = Base.dims2cat(dims)
    shape = Base.cat_size_shape(catdims, A, Bs...)
    RT = Base.promote_eltype(A, Bs...)
    Res = TracedRArray{RT,length(shape)}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.concatenate(
                [A.mlir_data, [B.mlir_data for B in Bs]...];
                result_0=MLIR.IR.TensorType(shape, MLIR.IR.Type(RT)),
                dimension=D - 1, # stablehlo expects this to be zero-indexed
            ),
            1,
        ),
        shape,
    )
    return Res
end
