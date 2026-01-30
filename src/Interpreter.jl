# Taken from https://github.com/JuliaLang/julia/pull/52964/files#diff-936d33e524bcd097015043bd6410824119be5c210d43185c4d19634eb4912708
# Other references:
# - https://github.com/JuliaLang/julia/blob/0fd1f04dc7d4b905b0172b7130e9b1beab9bc4c9/test/compiler/AbstractInterpreter.jl#L228-L234
# - https://github.com/JuliaLang/julia/blob/v1.10.4/test/compiler/newinterp.jl#L9

const CC = Core.Compiler

import Core.Compiler:
    AbstractInterpreter,
    abstract_call_known,
    ArgInfo,
    StmtInfo,
    AbsIntState,
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

@inline function set_reactant_abi(
    interp::Enzyme.Compiler.Interpreter.EnzymeInterpreter{typeof(set_reactant_abi)},
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int,
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
        arginfo2 =
            if length(argtypes) >= 2 &&
                Core.Compiler.widenconst(argtypes[2]) <: EnsureReturnType
                ArgInfo(fargs isa Nothing ? nothing : fargs[3:end], argtypes[3:end])
            else
                ArgInfo(fargs isa Nothing ? nothing : fargs[2:end], argtypes[2:end])
            end

        si2 = if VERSION < v"1.12"
            StmtInfo(true)
        else
            StmtInfo(true, false)
        end
        return Core.Compiler.abstract_call(interp, arginfo2::ArgInfo, si2, sv, max_methods)
    end

    if !should_rewrite_call(Core.Typeof(f))
        ninterp = Core.Compiler.NativeInterpreter(interp.world)
        # Note: mildly sus, but gabe said this was fine?
        @static if VERSION >= v"1.12"
            if hasproperty(sv, :interp)
                sv.interp = ninterp
                # sv2 = Compiler.OptimizationState(sv.result.linfo, ninterp)
            end
        end

        result = Base.@invoke abstract_call_known(
            ninterp::Core.Compiler.NativeInterpreter,
            f::Any,
            arginfo::ArgInfo,
            si::StmtInfo,
            sv::AbsIntState,
            max_methods::Int,
        )
        @static if VERSION >= v"1.12"
            if hasproperty(sv, :interp)
                sv.interp = interp
            end
        end
        return result
    else
        return Base.@invoke abstract_call_known(
            interp::AbstractInterpreter,
            f::Any,
            arginfo::ArgInfo,
            si::StmtInfo,
            sv::AbsIntState,
            max_methods::Int,
        )
    end
end

@static if Enzyme.GPUCompiler.HAS_INTEGRATED_CACHE
    struct ReactantCacheToken end

    function ReactantInterpreter(; world::UInt=Base.get_world_counter())
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            ReactantCacheToken(),
            REACTANT_METHOD_TABLE,
            world,
            false,            #=forward_rules=#
            false,            #=reverse_rules=#
            false,            #=inactive_rules=#
            false,            #=broadcast_rewrite=#
            false,            #=within_autodiff_rewrite=#
            set_reactant_abi, #=handler=#
        )
    end
else
    const REACTANT_CACHE = Enzyme.GPUCompiler.CodeCache()

    function ReactantInterpreter(;
        world::UInt=Base.get_world_counter(), code_cache=REACTANT_CACHE
    )
        return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
            REACTANT_CACHE,
            REACTANT_METHOD_TABLE,
            world,
            false,            #=forward_rules=#
            false,            #=reverse_rules=#
            false,            #=inactive_rules=#
            false,            #=broadcast_rewrite=#
            false,            #=within_autodiff_rewrite=#
            set_reactant_abi, #=handler=#
        )
    end
end

# Based on Enzyme.jl/src/typeutils/inference.jl
function return_type(interp::CC.AbstractInterpreter, mi::Core.MethodInstance)::Type
    @static if VERSION < v"1.11.0"
        #! explicit-imports: off
        code = CC.get(CC.code_cache(interp), mi, nothing)
        #! explicit-imports: on
        if code isa Core.CodeInstance
            return code.rettype
        end
        result = CC.InferenceResult(mi, CC.typeinf_lattice(interp))
        CC.typeinf(interp, result, :global)
        CC.is_inferred(result) || return Any
        CC.widenconst(CC.ignorelimited(result.result))
    else
        something(CC.typeinf_type(interp, mi), Any)
    end
end

function primal_interp_world(::Mode, world::UInt)
    return ReactantInterpreter(; world)
end

function primal_return_type_world(
    @nospecialize(mode::Mode), world::UInt, @nospecialize(TT::Type)
)
    return CC._return_type(primal_interp_world(mode, world), TT)
end

function primal_return_type_world(
    @nospecialize(mode::Mode), world::UInt, mi::Core.MethodInstance
)
    interp = primal_interp_world(mode, world)
    return return_type(interp, mi)
end

function primal_return_type_world(
    @nospecialize(mode::Mode), world::UInt, @nospecialize(FT::Type), @nospecialize(TT::Type)
)
    return primal_return_type_world(mode, world, Tuple{FT,TT.parameters...})
end

function primal_return_type end

function primal_return_type_generator(
    world::UInt,
    source,
    self,
    @nospecialize(mode::Type),
    @nospecialize(ft::Type),
    @nospecialize(tt::Type)
)
    @nospecialize
    @assert Base.isType(ft) && Base.isType(tt)
    @assert mode <: Mode
    mode = mode()
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    # validation
    ft <: Core.Builtin &&
        error("$(GPUCompiler.unsafe_function_from_type(ft)) is not a generic function")

    # look up the method    
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))

    mi = Enzyme.my_methodinstance(mode, ft, tt, world, min_world, max_world)

    slotnames = Core.svec(Symbol("#self#"), :mode, :ft, :tt)
    stub = Core.GeneratedFunctionStub(primal_return_type, slotnames, Core.svec())
    mi === nothing && return stub(world, source, :(throw(MethodError(ft, tt, $world))))

    result = primal_return_type_world(mode, world, mi)
    code = Any[Core.ReturnNode(result)]
    # create an empty CodeInfo to return the result
    ci = Enzyme.create_fresh_codeinfo(primal_return_type, source, world, slotnames, code)
    ci.max_world = max_world[]

    ci.edges = Any[]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.
    Enzyme.add_edge!(ci.edges, mi)

    return ci
end

@eval Base.@assume_effects :removable :foldable :nothrow @inline function primal_return_type(
    mode::Mode, ft::Type, tt::Type
)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, primal_return_type_generator))
end
