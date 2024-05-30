
using Cassette

using Enzyme

Cassette.@context TraceCtx;

const enzyme_out = 0
const enzyme_dup = 1
const enzyme_const = 2
const enzyme_dupnoneed = 3
const enzyme_outnoneed = 4
const enzyme_constnoneed = 5

function get_argidx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == "args"
            return path[2]::Int, path
        end
    end
    throw(AssertionError("No path found"))
end
function get_residx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == "result"
            return path
        end
    end
    throw(AssertionError("No path found $x"))
end

function has_residx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == "result"
            return true
        end
    end
    return false
end

@inline act_from_type(x, reverse, needs_primal=true) =
    throw(AssertionError("Unhandled activity $(typeof(x))"))
@inline act_from_type(::Enzyme.Const, reverse, needs_primal=true) =
    act_from_type(Enzyme.Const, reverse, needs_primal)
@inline act_from_type(::Enzyme.Duplicated, reverse, needs_primal=true) =
    act_from_type(Enzyme.Duplicated, reverse, needs_primal)
@inline act_from_type(::Enzyme.DuplicatedNoNeed, reverse, needs_primal=true) =
    reverse ? enzyme_out : enzyme_dupnoneed
@inline act_from_type(::Enzyme.Active, reverse, needs_primal=true) =
    act_from_tuple(Enzyme.Active, reverse, needs_primal)

@inline act_from_type(::Type{<:Enzyme.Const}, reverse, needs_primal) =
    if needs_primal
        enzyme_const
    else
        enzyme_constnoneed
    end
@inline act_from_type(::Type{<:Enzyme.Duplicated}, reverse, needs_primal) =
    if reverse
        if needs_primal
            enzyme_out
        else
            enzyme_outnoneed
        end
    else
        if needs_primal
            enzyme_dup
        else
            enzyme_dupnoneed
        end
    end
@inline act_from_type(::Type{<:Enzyme.Active}, reverse, needs_primal) =
    if needs_primal
        enzyme_out
    else
        enzyme_outnoneed
    end

function push_val!(ad_inputs, x, path)
    for p in path
        x = getfield(x, p)
    end
    x = x.mlir_data
    return push!(ad_inputs, x)
end

function push_acts!(ad_inputs, x::Const, path, reverse)
    return push_val!(ad_inputs, x.val, path)
end

function push_acts!(ad_inputs, x::Active, path, reverse)
    return push_val!(ad_inputs, x.val, path)
end

function push_acts!(ad_inputs, x::Duplicated, path, reverse)
    push_val!(ad_inputs, x.val, path)
    if !reverse
        push_val!(ad_inputs, x.dval, path)
    end
end

function push_acts!(ad_inputs, x::DuplicatedNoNeed, path, reverse)
    push_val!(ad_inputs, x.val, path)
    if !reverse
        push_val!(ad_inputs, x.dval, path)
    end
end

function set_act!(inp, path, reverse, tostore; emptypath=false)
    x = if inp isa Enzyme.Active
        inp.val
    else
        inp.dval
    end

    for p in path
        x = getfield(x, p)
    end

    #if inp isa Enzyme.Active || !reverse
    x.mlir_data = tostore
    #else
    #    x.mlir_data = MLIR.IR.result(MLIR.Dialects.stablehlo.add(x.mlir_data, tostore), 1)
    #end

    if emptypath
        x.paths = ()
    end
end

function set!(x, path, tostore; emptypath=false)
    for p in path
        x = getfield(x, p)
    end

    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function Cassette.overdub(
    ::TraceCtx,
    ::typeof(Enzyme.autodiff),
    ::CMode,
    f::FA,
    ::Type{A},
    args::Vararg{Enzyme.Annotation,Nargs},
) where {CMode<:Enzyme.Mode,FA<:Enzyme.Annotation,A<:Enzyme.Annotation,Nargs}
    reverse = CMode <: Enzyme.ReverseMode

    primf = f.val
    primargs = ((v.val for v in args)...,)

    mod = MLIR.IR.mmodule()

    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        mod, primf, primargs, (), string(f) * "_autodiff", false
    )

    activity = Int32[]
    ad_inputs = MLIR.IR.Value[]

    for a in linear_args
        idx, path = get_argidx(a)
        if idx == 1 && fnwrap
            push!(activity, act_from_type(f, reverse))
            push_acts!(ad_inputs, f, path[3:end], reverse)
        else
            if fnwrap
                idx -= 1
            end
            push!(activity, act_from_type(args[idx], reverse))
            push_acts!(ad_inputs, args[idx], path[3:end], reverse)
        end
    end

    outtys = MLIR.IR.Type[]
    @inline needs_primal(::Type{<:Enzyme.ReverseMode{ReturnPrimal}}) where {ReturnPrimal} =
        ReturnPrimal
    for a in linear_results
        if has_residx(a)
            if needs_primal(CMode)
                push!(outtys, transpose_ty(MLIR.IR.type(a.mlir_data)))
            end
        else
            push!(outtys, transpose_ty(MLIR.IR.type(a.mlir_data)))
        end
    end
    for (i, act) in enumerate(activity)
        if act == enzyme_out || (reverse && (act == enzyme_dup || act == enzyme_dupnoneed))
            push!(outtys, in_tys[i])# transpose_ty(MLIR.IR.type(MLIR.IR.operand(ret, i))))
        end
    end

    ret_activity = Int32[]
    for a in linear_results
        if has_residx(a)
            act = act_from_type(A, reverse, needs_primal(CMode))
            push!(ret_activity, act)
            if act == enzyme_out || act == enzyme_outnoneed
                attr = fill(MLIR.IR.Attribute(eltype(a)(1)), mlir_type(a))
                cst = MLIR.IR.result(MLIR.Dialects.stablehlo.constant(; value=attr), 1)
                push!(ad_inputs, cst)
            end
        else
            idx, path = get_argidx(a)
            if idx == 1 && fnwrap
                act = act_from_type(f, reverse, true)
                push!(ret_activity, act)
                if act != enzyme_out && act != enzyme_outnoneed
                    continue
                end
                push_val!(ad_inputs, f.dval, path[3:end])
            else
                if fnwrap
                    idx -= 1
                end
                act = act_from_type(args[idx], reverse, true)
                push!(ret_activity, act)
                if act != enzyme_out && act != enzyme_outnoneed
                    continue
                end
                push_val!(ad_inputs, args[idx].dval, path[3:end])
            end
        end
    end

    function get_attribute_by_name(operation, name)
        return MLIR.IR.Attribute(MLIR.API.mlirOperationGetAttributeByName(operation, name))
    end

    function act_attr(val)
        val = @ccall MLIR.API.mlir_c.enzymeActivityAttrGet(
            MLIR.IR.context()::MLIR.API.MlirContext, val::Int32
        )::MLIR.API.MlirAttribute
        return MLIR.IR.Attribute(val)
    end
    fname = get_attribute_by_name(func2, "sym_name")
    fname = MLIR.IR.FlatSymbolRefAttribute(Base.String(fname))
    res = (reverse ? MLIR.Dialects.enzyme.autodiff : MLIR.Dialects.enzyme.fwddiff)(
        [transpose_val(v) for v in ad_inputs];
        outputs=outtys,
        fn=fname,
        activity=MLIR.IR.Attribute([act_attr(a) for a in activity]),
        ret_activity=MLIR.IR.Attribute([act_attr(a) for a in ret_activity]),
    )

    residx = 1

    for a in linear_results
        if has_residx(a)
            if needs_primal(CMode)
                path = get_residx(a)
                set!(result, path[2:end], transpose_val(MLIR.IR.result(res, residx)))
                residx += 1
            end
        else
            idx, path = get_argidx(a)
            if idx == 1 && fnwrap
                set!(f.val, path[3:end], transpose_val(MLIR.IR.result(res, residx)))
                residx += 1
            else
                if fnwrap
                    idx -= 1
                end
                set!(args[idx].val, path[3:end], transpose_val(MLIR.IR.result(res, residx)))
                residx += 1
            end
        end
    end

    restup = Any[(a isa Active) ? copy(a) : nothing for a in args]
    for a in linear_args
        idx, path = get_argidx(a)
        if idx == 1 && fnwrap
            if act_from_type(f, reverse) != enzyme_out
                continue
            end
            if f isa Enzyme.Active
                @assert false
                residx += 1
                continue
            end
            set_act!(f, path[3:end], reverse, transpose_val(MLIR.IR.result(res, residx)))
        else
            if fnwrap
                idx -= 1
            end
            if act_from_type(args[idx], reverse) != enzyme_out
                continue
            end
            if args[idx] isa Enzyme.Active
                set_act!(
                    args[idx],
                    path[3:end],
                    false,
                    transpose_val(MLIR.IR.result(res, residx));
                    emptypaths=true,
                ) #=reverse=#
                residx += 1
                continue
            end
            set_act!(
                args[idx], path[3:end], reverse, transpose_val(MLIR.IR.result(res, residx))
            )
        end
        residx += 1
    end

    if reverse
        resv = if needs_primal(CMode)
            result
        else
            nothing
        end
        return ((restup...,), resv)
    else
        if A <: Const
            return result
        else
            dres = copy(result)
            throw(AssertionError("TODO implement forward mode handler"))
            if A <: Duplicated
                return ()
            end
        end
    end
end

function promote_to(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
    if !(rhs <: Number)
        if ElType != eltype(rhs)
            throw(
                ArgumentError(
                    "Cannot promote $(typeof(rhs)) to $(TracedRArray{ElType,Shape,N}) with different element types",
                ),
            )
        end
        if Shape != size(rhs)
            throw(ArgumentError("Cannot promote to TracedRArray with different shapes"))
        end
    end

    if isa(rhs, TracedRArray)
        if isa(rhs, Number)
            throw(ArgumentError("TODO broadcast"))
        end
        return rhs
    end
    if isa(rhs, Number)
        attr = fill(MLIR.IR.Attribute(ElType(rhs)), mlir_type(lhs))
        return TracedRArray{ElType,Shape,N}(nothing, MLIR.Dialects.stablehlo.constant(attr))
    end
    attr = MLIR.IR.DenseElementsAttribute(mlir_type(lhs), rhs)
    return TracedRArray{ElType,Shape,N}(nothing, MLIR.Dialects.stablehlo.constant(attr))
end

for (jlop, hloop, RT) in (
    (:(Base.min), :minimum, :ElType),
    (:(Base.max), :maximum, :ElType),
    (:(Base.:+), :add, :ElType),
    (:(Base.:-), :subtract, :ElType),
)
    @eval begin
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

        function $jlop(lhs::TracedRArray{ElType,Shape,N}, rhs) where {ElType,Shape,N}
            rhs = promote_to(lhs, rhs)
            return TracedRArray{$RT,Shape,N}(
                (),
                MLIR.IR.result(
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end

        function $jlop(lhs, rhs::TracedRArray{ElType,Shape,N}) where {ElType,Shape,N}
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

Cassette.overdub(context::TraceCtx, f::typeof(Enzyme.make_zero), args...) = f(args...)

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
    return TracedRArray{ElType,(Base.size(lhsty)[1], Base.size(rhsty)[2]),2}((), res)
end

Cassette.overdub(context::TraceCtx, f::typeof(Base.:*), args...) = f(args...)

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
        Cassette.overdub(context::TraceCtx, f::typeof($jlop), args...) = f(args...)
    end
end

for (jlop, hloop, RT) in (
    (:(Base.min), :minimum, :ElType),
    (:(Base.max), :maximum, :ElType),
    (:(Base.:+), :add, :ElType),
    (:(Base.add_sum), :add, :ElType),
    (:(Base.:-), :subtract, :ElType),
    (:(Base.:*), :multiply, :ElType),
    (:(Base.:/), :divide, :ElType),
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
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
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
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
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
                    MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data, rhs.mlir_data), 1
                ),
            )
        end
    end
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

function elem_apply(::typeof(identity), lhs)
    return lhs
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
        function elem_apply(
            ::typeof($jlop), lhs::TracedRArray{ElType,Shape,N}
        ) where {ElType,Shape,N}
            return TracedRArray{ElType,Shape,N}(
                (), MLIR.IR.result(MLIR.Dialects.stablehlo.$hloop(lhs.mlir_data), 1)
            )
        end
    end
end

Cassette.overdub(context::TraceCtx, f::typeof(elem_apply), args...) = f(args...)

@inline function Base.reshape(A::RArray, dims::Tuple{Vararg{Union{Int,Colon}}})
    return reshape(A, Base._reshape_uncolon(A, dims))
end
Cassette.overdub(context::TraceCtx, f::typeof(Base.reshape), args...) = f(args...)

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
Cassette.overdub(context::TraceCtx, f::typeof(Base.copy), args...) = f(args...)

@inline function Base.permutedims(A::TracedRArray{T,Shape,N}, perm) where {T,Shape,N}
    return TracedArray{T,tuple(Shape[i] for i in perm),N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.stablehlo.transpose(
                A.mlir_data, DenseArrayAttribute([Int64(i - 1) for i in perm])
            ),
            1,
        ),
    )
end
Cassette.overdub(context::TraceCtx, f::typeof(Base.permutedims), args...) = f(args...)

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
Cassette.overdub(context::TraceCtx, f::typeof(Base.similar), args...) = f(args...)

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
Cassette.overdub(context::TraceCtx, f::typeof(Base.materialize!), args...) = f(args...)

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

function Base.fill!(A::TracedRArray{T,Shape,N}, x) where {T,Shape,N}
    bcast = broadcast_to_size(T(x), Shape)
    A.mlir_data = bcast.mlir_data
    return A
end
Cassette.overdub(context::TraceCtx, f::typeof(Base.fill!), args...) = f(args...)

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

function Cassette.overdub(
    context::Cassette.Context,
    ::Core.kwftype(typeof(Base.mapreduce)),
    kwargs::Any,
    ::typeof(Base.mapreduce),
    args...,
)
    return Base.mapreduce(args...; kwargs...)
end

Cassette.overdub(context::Cassette.Context, f::typeof(Base.mapreduce), args...) = f(args...)

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

    inp = [elem_apply(f, A.mlir_data)]

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
        tmp = broadcast_to_size(op(args...), (1,)).mlir_data
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
Cassette.overdub(context::TraceCtx, f::typeof(Base.mapreducedim!), args...) = f(args...)
