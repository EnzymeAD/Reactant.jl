# Taken from https://github.com/JuliaLang/julia/pull/52964/files#diff-936d33e524bcd097015043bd6410824119be5c210d43185c4d19634eb4912708
# Other references:
# - https://github.com/JuliaLang/julia/blob/0fd1f04dc7d4b905b0172b7130e9b1beab9bc4c9/test/compiler/AbstractInterpreter.jl#L228-L234
# - https://github.com/JuliaLang/julia/blob/v1.10.4/test/compiler/newinterp.jl#L9

const CC = Core.Compiler
using Enzyme

const HAS_INTEGRATED_CACHE = VERSION >= v"1.11.0-DEV.1552"

Base.Experimental.@MethodTable(ReactantMethodTable)

function var"@reactant_override"(__source__::LineNumberNode, __module__::Module, def)
    return Base.Experimental.var"@overlay"(
        __source__, __module__, :(Reactant.ReactantMethodTable), def
    )
end

@static if !HAS_INTEGRATED_CACHE
    struct ReactantCache
        dict::IdDict{Core.MethodInstance,Core.CodeInstance}
    end
    ReactantCache() = ReactantCache(IdDict{Core.MethodInstance,Core.CodeInstance}())

    const REACTANT_CACHE = ReactantCache()

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
        x = traced_getfield(x, p)
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
        x = traced_getfield(x, p)
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
