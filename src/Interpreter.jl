# Taken from https://github.com/JuliaLang/julia/pull/52964/files#diff-936d33e524bcd097015043bd6410824119be5c210d43185c4d19634eb4912708
# Other references:
# - https://github.com/JuliaLang/julia/blob/0fd1f04dc7d4b905b0172b7130e9b1beab9bc4c9/test/compiler/AbstractInterpreter.jl#L228-L234
# - https://github.com/JuliaLang/julia/blob/v1.10.4/test/compiler/newinterp.jl#L9

const CC = Core.Compiler
using Enzyme

import Core.Compiler:
    AbstractInterpreter,
    abstract_call,
    abstract_call_known,
    ArgInfo,
    StmtInfo,
    AbsIntState,
    get_max_methods,
    CallMeta,
    Effects,
    NoCallInfo,
    MethodResultPure

Base.Experimental.@MethodTable(REACTANT_METHOD_TABLE)

function var"@reactant_overlay"(__source__::LineNumberNode, __module__::Module, def)
    return Base.Experimental.var"@overlay"(
        __source__, __module__, :(Reactant.REACTANT_METHOD_TABLE), def
    )
end

function set_reactant_abi(
    interp,
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int=get_max_methods(interp, f, sv),
)
    (; fargs, argtypes) = arginfo

    if f === ReactantCore.within_compile
        if length(argtypes) != 1
            @static if VERSION < v"1.11.0-"
                return CallMeta(Union{}, Effects(), NoCallInfo())
            elseif VERSION < v"1.12.0-"
                return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
            else
                return Core.Compiler.Future{Core.Compiler.CallMeta}(
                    CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
                )
            end
        end
        @static if VERSION < v"1.11.0-"
            return CallMeta(
                Core.Const(true), Core.Compiler.EFFECTS_TOTAL, MethodResultPure()
            )
        elseif VERSION < v"1.12.0-"
            return CallMeta(
                Core.Const(true), Union{}, Core.Compiler.EFFECTS_TOTAL, MethodResultPure()
            )
        else
            return Core.Compiler.Future{Core.Compiler.CallMeta}(
                CallMeta(
                    Core.Const(true),
                    Union{},
                    Core.Compiler.EFFECTS_TOTAL,
                    MethodResultPure(),
                ),
            )
        end
    end

    # Improve inference by considering call_with_reactant as having the same results as
    # the original call
    if f === call_with_reactant
        arginfo2 = ArgInfo(fargs isa Nothing ? nothing : fargs[2:end], argtypes[2:end])
        return abstract_call(interp, arginfo2::ArgInfo, si, sv, max_methods)
    elseif !(interp.within_autodiff_rewrite) && f === overload_autodiff
        interp′ = Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            interp; within_autodiff_rewrite=true
        )
        return Base.@invoke abstract_call_known(
            interp′::Enzyme.Compiler.Interpreter.EnzymeInterpreter,
            f,
            arginfo,
            si,
            sv,
            max_methods,
        )
    end

    return Base.@invoke abstract_call_known(
        interp::AbstractInterpreter,
        f::Any,
        arginfo::ArgInfo,
        si::StmtInfo,
        sv::AbsIntState,
        max_methods::Int,
    )
end

@static if Enzyme.GPUCompiler.HAS_INTEGRATED_CACHE
    struct ReactantCacheToken end

    function ReactantInterpreter(;
        world::UInt=Base.get_world_counter(), within_autodiff=false
    )
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            ReactantCacheToken(),
            REACTANT_METHOD_TABLE,
            world,
            false,            #=forward_rules=#
            false,            #=reverse_rules=#
            false,            #=inactive_rules=#
            false,            #=broadcast_rewrite=#
            within_autodiff,  #=within_autodiff_rewrite=#
            set_reactant_abi,
        )
    end
else
    const REACTANT_CACHE = Enzyme.GPUCompiler.CodeCache()

    function ReactantInterpreter(;
        world::UInt=Base.get_world_counter(),
        code_cache=REACTANT_CACHE,
        within_autodiff=false,
    )
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            REACTANT_CACHE,
            REACTANT_METHOD_TABLE,
            world,
            false,            #=forward_rules=#
            false,            #=reverse_rules=#
            false,            #=inactive_rules=#
            false,            #=broadcast_rewrite=#
            within_autodiff,  #=within_autodiff_rewrite=#
            set_reactant_abi,
        )
    end
end
