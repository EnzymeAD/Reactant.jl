# Taken from https://github.com/JuliaLang/julia/pull/52964/files#diff-936d33e524bcd097015043bd6410824119be5c210d43185c4d19634eb4912708
# Other references:
# - https://github.com/JuliaLang/julia/blob/0fd1f04dc7d4b905b0172b7130e9b1beab9bc4c9/test/compiler/AbstractInterpreter.jl#L228-L234
# - https://github.com/JuliaLang/julia/blob/v1.10.4/test/compiler/newinterp.jl#L9

const CC = Core.Compiler
using Enzyme

const HAS_INTEGRATED_CACHE = VERSION >= v"1.11.0-DEV.1552"

Base.Experimental.@MethodTable(ReactantMethodTable)

macro reactant_override(expr)
    return :(Base.Experimental.@overlay ReactantMethodTable $(expr))
end

@static if !HAS_INTEGRATED_CACHE
    struct ReactantCache
        dict::IdDict{Core.MethodInstance,Core.CodeInstance}
    end
    ReactantCache() = ReactantCache(IdDict{Core.MethodInstance,Core.CodeInstance}())

    function CC.get(wvc::CC.WorldView{ReactantCache}, mi::Core.MethodInstance, default)
        return get(wvc.cache.dict, mi, default)
    end
    function CC.getindex(wvc::CC.WorldView{ReactantCache}, mi::Core.MethodInstance)
        return getindex(wvc.cache.dict, mi)
    end
    function CC.haskey(wvc::CC.WorldView{ReactantCache}, mi::Core.MethodInstance)
        return haskey(wvc.cache.dict, mi)
    end
    function CC.setindex!(
        wvc::CC.WorldView{ReactantCache}, ci::Core.CodeInstance, mi::Core.MethodInstance
    )
        return setindex!(wvc.cache.dict, ci, mi)
    end
end

struct ReactantInterpreter <: CC.AbstractInterpreter
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    @static if !HAS_INTEGRATED_CACHE
        code_cache::ReactantCache
    end

    @static if HAS_INTEGRATED_CACHE
        function ReactantInterpreter(;
            world::UInt=Base.get_world_counter(),
            inf_params::CC.InferenceParams=CC.InferenceParams(),
            opt_params::CC.OptimizationParams=CC.OptimizationParams(),
            inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[],
        )
            return new(world, inf_params, opt_params, inf_cache)
        end
    else
        function ReactantInterpreter(;
            world::UInt=Base.get_world_counter(),
            inf_params::CC.InferenceParams=CC.InferenceParams(),
            opt_params::CC.OptimizationParams=CC.OptimizationParams(),
            inf_cache::Vector{CC.InferenceResult}=CC.InferenceResult[],
            code_cache=ReactantCache(),
        )
            return new(world, inf_params, opt_params, inf_cache, code_cache)
        end
    end
end

REACTANT_INTERPRETER::Union{Nothing,ReactantInterpreter} = nothing

@static if HAS_INTEGRATED_CACHE
    CC.get_inference_world(interp::ReactantInterpreter) = interp.world
else
    CC.get_world_counter(interp::ReactantInterpreter) = interp.world
end

CC.InferenceParams(interp::ReactantInterpreter) = interp.inf_params
CC.OptimizationParams(interp::ReactantInterpreter) = interp.opt_params
CC.get_inference_cache(interp::ReactantInterpreter) = interp.inf_cache

@static if HAS_INTEGRATED_CACHE
    # TODO what does this do? taken from https://github.com/JuliaLang/julia/blob/v1.11.0-rc1/test/compiler/newinterp.jl
    @eval CC.cache_owner(interp::ReactantInterpreter) =
        $(QuoteNode(gensym(:ReactantInterpreterCache)))
else
    function CC.code_cache(interp::ReactantInterpreter)
        return CC.WorldView(interp.code_cache, CC.WorldRange(interp.world))
    end
end

function CC.method_table(interp::ReactantInterpreter)
    return CC.OverlayMethodTable(interp.world, ReactantMethodTable)
end

const enzyme_out = 0
const enzyme_dup = 1
const enzyme_const = 2
const enzyme_dupnoneed = 3
const enzyme_outnoneed = 4
const enzyme_constnoneed = 5

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

function get_argidx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :args
            return path[2]::Int, path
        end
    end
    throw(AssertionError("No path found for $x"))
end
function get_residx(x)
    for path in x.paths
        if length(path) == 0
            continue
        end
        if path[1] == :result
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
        if path[1] == :result
            return true
        end
    end
    return false
end

function get_attribute_by_name(operation, name)
    return MLIR.IR.Attribute(MLIR.API.mlirOperationGetAttributeByName(operation, name))
end

@reactant_override function Enzyme.autodiff(
    ::CMode, f::FA, ::Type{A}, args::Vararg{Enzyme.Annotation,Nargs}
) where {CMode<:Enzyme.Mode,FA<:Enzyme.Annotation,A<:Enzyme.Annotation,Nargs}
    reverse = CMode <: Enzyme.ReverseMode

    primf = f.val
    primargs = ((v.val for v in args)...,)

    fnwrap, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results = make_mlir_fn(
        primf, primargs, (), string(f) * "_autodiff", false
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

    func2.operation = MLIR.API.MlirOperation(C_NULL)

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
