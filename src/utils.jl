
function apply(f, args...; kwargs...)
    return f(args...; kwargs...)
end

function call_with_reactant end

function maybe_argextype(@nospecialize(x), src)
    return try
        Core.Compiler.argextype(x, src)
    catch err
        !(err isa Core.Compiler.InvalidIRError) && rethrow()
        nothing
    end
end

"""
    Reactant.REDUB_ARGUMENTS_NAME

The variable name bound to `call_with_reactant`'s tuple of arguments in its
`@generated` method definition.

This binding can be used to manually reference/destructure `call_with_reactants` arguments

This is required because user arguments could have a name which clashes with whatever name we choose for
our argument. Thus we gensym to create it.

This originates from https://github.com/JuliaLabs/Cassette.jl/blob/c29b237c1ec0deda3a1037ec519eebe216952bfe/src/overdub.jl#L154
"""
const REDUB_ARGUMENTS_NAME = gensym("redub_arguments")

function throw_method_error(argtys)
    throw(MethodError(argtys[1], argtys[2:end]))
end



@inline function lookup_world(@nospecialize(sig::Type), world::UInt, mt::Union{Nothing,Core.MethodTable}, min_world::Ref{UInt}, max_world::Ref{UInt})
    res = ccall(:jl_gf_invoke_lookup_worlds, Any,
                  (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
                  sig, mt, world, min_world, max_world)
    return res
end

@inline function lookup_world(@nospecialize(sig::Type), world::UInt, mt::Core.Compiler.InternalMethodTable, min_world::Ref{UInt}, max_world::Ref{UInt})
    res = lookup_world(sig, mt.world, nothing, min_world, max_world)
    return res
end

@inline function lookup_world(@nospecialize(sig::Type), world::UInt, mt::Core.Compiler.OverlayMethodTable, min_world::Ref{UInt}, max_world::Ref{UInt})
    res = lookup_world(sig, mt.world, mt.mt, min_world, max_world)
    if res !== nothing
        return res
    else
        return lookup_world(sig, mt.world, nothing, min_world, max_world)
    end
end

function has_ancestor(query::Module, target::Module)
    query == target && return true
    while true
        next = parentmodule(query)
        next == target && return true
        next == query && return false
        query = next
    end
end

function should_rewrite_ft(@nospecialize(ft))
    # Don't rewrite builtin or intrinsics
    if ft <: Core.IntrinsicFunction || ft <: Core.Builtin
        return false
    end
    if ft <: Core.Function
        mod = ft.name.module
        # Don't rewrite primitive ops, tracing utilities, or any MLIR-based functions
        if has_ancestor(mod, Reactant.Ops) || has_ancestor(mod, Reactant.TracedUtils) || has_ancestor(mod, Reactant.MLIR)
            return false
        end
    end
    # Don't rewrite Val
    if ft === Type{Base.Val}
        return false
    end
    # Don't rewrite exception constructors
    if ft <: Type{<:Core.Exception}
        return false
    end

    # Default assume all functions need to be reactant-ified
    return true
end

# Avoid recursively interpreting into methods we define explicitly
# as overloads, which we assume should handle the entirety of the
# translation (and if not they can use call_in_reactant).
function is_reactant_method(mi::Core.MethodInstance)
    meth = mi.def
    if !isdefined(meth, :external_mt)
        return false
    end
    mt = meth.external_mt
    return mt === REACTANT_METHOD_TABLE
end

function rewrite_inst(inst, ir, interp)
    if Meta.isexpr(inst, :call)
        # Even if type unstable we do not want (or need) to replace intrinsic
        # calls or builtins with our version.
        ft = Core.Compiler.widenconst(maybe_argextype(inst.args[1], ir))
        if should_rewrite_ft(ft)
            rep = Expr(:call, call_with_reactant, inst.args...)
            return true, rep
        end
    end
    if Meta.isexpr(inst, :invoke)
        omi = inst.args[1]::Core.MethodInstance
        sig = omi.specTypes
        ft = sig.parameters[1]

        if should_rewrite_ft(ft) && !is_reactant_method(omi)

            min_world = Ref{UInt}(typemin(UInt))
            max_world = Ref{UInt}(typemax(UInt))

            lookup_result = lookup_world(Tuple{typeof(call_with_reactant), sig.parameters...}, interp.world, Core.Compiler.method_table(interp), min_world, max_world)
    
            match = lookup_result::Core.MethodMatch
            # look up the method and code instance
            mi = ccall(
                :jl_specializations_get_linfo,
                Ref{Core.MethodInstance},
                (Any, Any, Any),
                match.method,
                match.spec_types,
                match.sparams,
            )
            rep = Expr(:invoke, mi, call_with_reactant, inst.args[2:end]...)
            return true, rep
        end
    end
    return false, inst
end

function make_oc(sig, rt, src, nargs, isva, f)::Core.OpaqueClosure
    ccall(:jl_new_opaque_closure_from_code_info, Any, (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any, Cint),
        sig, rt, rt, @__MODULE__, src, 0, nothing, nargs, isva, f, true)::Core.OpaqueClosure
end

# Generator function which ensures that all calls to the function are executed within the ReactantInterpreter
# In particular this entails two pieces:
#   1) We enforce the use of the ReactantInterpreter method table when generating the original methodinstance
#   2) Post type inference (using of course the reactant interpreter), all type unstable call functions are
#      replaced with calls to `call_with_reactant`. This allows us to circumvent long standing issues in Julia
#      using a custom interpreter in type unstable code.
# `redub_arguments` is `(typeof(original_function), map(typeof, original_args_tuple)...)`
function call_with_reactant_generator(
    world::UInt, source::LineNumberNode, self, @nospecialize(redub_arguments)
)
    @nospecialize
    args = redub_arguments

    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(:call_with_reactant, REDUB_ARGUMENTS_NAME), Core.svec()
    )

    # look up the method match
    builtin_error = :(throw(
        AssertionError("Unsupported call_with_reactant of builtin $redub_arguments")
    ))

    if args[1] <: Core.Builtin
        return stub(world, source, builtin_error)
    end
    method_error = :(throw(MethodError($REDUB_ARGUMENTS_NAME[1], $REDUB_ARGUMENTS_NAME[2:end], $world)))

    interp = ReactantInterpreter(; world)

    sig = Tuple{args...}

    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))

    lookup_result = lookup_world(sig, world, Core.Compiler.method_table(interp), min_world, max_world)
    
    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]

    # No method could be found (including in our method table), bail with an error
    if lookup_result == nothing
        return stub(world, source, method_error)
        tmp_min_world = Ref{UInt}(typemin(UInt))
        tmp_max_world = Ref{UInt}(typemax(UInt))
        match = ccall(:jl_gf_invoke_lookup_worlds, Any,
                      (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
                      Tuple{typeof(throw_method_error), sig}, #=mt=# nothing, world, tmp_min_world, tmp_max_world)
        @assert match !== nothing

        # look up the method and code instance
        mi = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                   (Any, Any, Any), match.method, match.spec_types, match.sparams)
     
        ci = Core.Compiler.retrieve_code_info(mi, world)::Core.Compiler.CodeInfo

        src = copy(ci)
        src.slotnames = Any[:call_with_reactant, REDUB_ARGUMENTS_NAME]

        src.edges = Any[ccall(:jl_method_table_for, Any, (Any,), sig)::Core.MethodTable, sig]
        src.min_world = min_world[]
        src.max_world = max_world[]

        push!(overdubbed_code, :($(Base.getindex)($(Core.Argument(2)), 1)))
        push!(overdubbed_codelocs, 0)

        expr_fn = Core.SSAValue(length(overdubbed_code))


        push!(overdubbed_code, :($(Base.lastindex)($(Core.Argument(2)))))
        push!(overdubbed_codelocs, 0)

        expr_lastindex = Core.SSAValue(length(overdubbed_code))


        push!(overdubbed_code, :(2:$expr_lastindex))
        push!(overdubbed_codelocs, 0)

        expr_slice = Core.SSAValue(length(overdubbed_code))

        push!(overdubbed_code, :($(Base.getindex)($(Core.Argument(2)), $expr_slice)))
        push!(overdubbed_codelocs, 0)

        expr_args = Core.SSAValue(length(overdubbed_code))

        push!(overdubbed_code, :($(Base.MethodError)($expr_fn, $expr_args, $world)))
        push!(overdubbed_codelocs, 0)

        expr_method = Core.SSAValue(length(overdubbed_code))

        push!(overdubbed_code, :($(Base.throw)($expr_method)))
        push!(overdubbed_codelocs, 0)

        push!(
            overdubbed_code,
            Core.ReturnNode(Core.SSAValue(length(overdubbed_code)))
        )
        push!(overdubbed_codelocs, 0)

        src.code = overdubbed_code
        src.codelocs = overdubbed_codelocs
        src.ssavaluetypes = length(overdubbed_code)
        src.ssaflags = [0x00 for _ in 1:length(overdubbed_code)] # XXX we need to copy flags that are set for the original code

        return src
    end

    match = lookup_result::Core.MethodMatch
    # look up the method and code instance
    mi = ccall(
        :jl_specializations_get_linfo,
        Ref{Core.MethodInstance},
        (Any, Any, Any),
        match.method,
        match.spec_types,
        match.sparams,
    )

    result = Core.Compiler.InferenceResult(mi, Core.Compiler.typeinf_lattice(interp))
    frame = Core.Compiler.InferenceState(result, :local, interp) #=cache_mode=#
    @assert frame !== nothing
    Core.Compiler.typeinf(interp, frame)
    @static if VERSION >= v"1.11"
        # `typeinf` doesn't update the cfg. We need to do it manually.
        frame.cfg = Core.Compiler.compute_basic_blocks(frame.src.code)
    end
    @assert Core.Compiler.is_inferred(frame)

    method = match.method

    # The original julia code (on 1.11+) has the potential constprop, for now
    # we assume this outermost function does not constprop, for ease.
    #if Core.Compiler.result_is_constabi(interp, frame.result)
    #    rt = frame.result.result::Core.Compiler.Const
    #    src = Core.Compiler.codeinfo_for_const(interp, frame.linfo, rt.val)
    #else
    opt = Core.Compiler.OptimizationState(frame, interp)

    caller = frame.result
    @static if VERSION < v"1.11-"
        ir = Core.Compiler.run_passes(opt.src, opt, caller)
    else
        ir = Core.Compiler.run_passes_ipo_safe(opt.src, opt, caller)
        Core.Compiler.ipo_dataflow_analysis!(interp, ir, caller)
    end
    
    # Rewrite type unstable calls to recurse into call_with_reactant to ensure
    # they continue to use our interpreter. Reset the derived return type
    # to Any if our interpreter would change the return type of any result.
    # Also rewrite invoke (type stable call) to be :call, since otherwise apparently
    # screws up type inference after this (TODO this should be fixed).
    any_changed = false
    for (i, inst) in enumerate(ir.stmts)
        @static if VERSION < v"1.11"
            changed, next = rewrite_inst(inst[:inst], ir, interp)
            Core.Compiler.setindex!(ir.stmts[i], next, :inst)
        else
            changed, next = rewrite_inst(inst[:stmt], ir, interp)
            Core.Compiler.setindex!(ir.stmts[i], next, :stmt)
        end
        if changed
            any_changed = true
            Core.Compiler.setindex!(ir.stmts[i], Any, :type)
        end
    end
    Core.Compiler.finish(interp, opt, ir, caller)

    src = Core.Compiler.ir_to_codeinf!(opt)
    
    # prepare a new code info
    code_info = copy(src)
    static_params = match.sparams
    signature = sig

    # propagate edge metadata, this method is invalidated if the original function we are calling
    # is invalidated
    code_info.edges = Core.MethodInstance[mi]
    code_info.min_world = min_world[]
    code_info.max_world = max_world[]

    # Rewrite the arguments to this function, to prepend the two new arguments, the function :call_with_reactant,
    # and the REDUB_ARGUMENTS_NAME tuple of input arguments
    code_info.slotnames = Any[:call_with_reactant, REDUB_ARGUMENTS_NAME]
    code_info.slotflags = UInt8[0x00, 0x00]
    n_prepended_slots = 2
    overdub_args_slot = Core.SlotNumber(n_prepended_slots)

    # For the sake of convenience, the rest of this pass will translate `code_info`'s fields
    # into these overdubbed equivalents instead of updating `code_info` in-place. Then, at
    # the end of the pass, we'll reset `code_info` fields accordingly.
    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]
    # Rewire the arguments from our tuple input of fn and args, to the corresponding calling convention
    # required by the base method.

    # destructure the generated argument slots into the overdubbed method's argument slots.

    offset = 1
    fn_args = Any[]
    n_method_args = method.nargs
    n_actual_args = length(redub_arguments)

    tys = []
    
    iter_args = n_actual_args
    if method.isva
        iter_args = min(n_actual_args, n_method_args-1)
    end
        
    for i in 1:iter_args
        actual_argument = Expr(
            :call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset
        )
        push!(overdubbed_code, actual_argument)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        offset += 1
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
        push!(tys, redub_arguments[i])
    end


    # If `method` is a varargs method, we have to restructure the original method call's
    # trailing arguments into a tuple and assign that tuple to the expected argument slot.
    if method.isva
        trailing_arguments = Expr(:call, Core.GlobalRef(Core, :tuple))
        for i in n_method_args:n_actual_args
            push!(
                overdubbed_code,
                Expr(:call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset),
            )
            push!(overdubbed_codelocs, code_info.codelocs[1])
            push!(trailing_arguments.args, Core.SSAValue(length(overdubbed_code)))
            offset += 1
        end

        push!(
            overdubbed_code, trailing_arguments
        )
        push!(overdubbed_codelocs, code_info.codelocs[1])
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
        push!(tys, Tuple{redub_arguments[n_method_args:n_actual_args]...})
    end

    rt = Base.Experimental.compute_ir_rettype(ir)
    
    # ocva = method.isva

    ocva = false # method.isva

    ocnargs = method.nargs - 1
    # octup = Tuple{mi.specTypes.parameters[2:end]...}
    # octup = Tuple{method.sig.parameters[2:end]...}
    octup = Tuple{tys[2:end]...}
    ocva = false

    # jl_new_opaque_closure forcibly executes in the current world... This means that we won't get the right
    # inner code during compilation without special handling (i.e. call_in_world_total).
    # Opaque closures also require takign the function argument. We can work around the latter
    # if the function is stateless. But regardless, to work around this we sadly create/compile the opaque closure
    oc = if false && Base.issingletontype(args[1])
        Core._call_in_world_total(world, make_oc, octup, rt, src, ocnargs, ocva, args[1].instance)::Core.OpaqueClosure
    else
        farg = fn_args[1]
        push!(overdubbed_code,
            Expr(:call,
                make_oc,
                octup,
                rt,
                src,
                ocnargs,
                ocva,
                farg
                )
                )
        push!(overdubbed_codelocs, code_info.codelocs[1])
        Core.SSAValue(length(overdubbed_code))
    end

    push!(
        overdubbed_code,
        Expr(
            :(call),
            oc,
            fn_args[2:end]...
        ),
    )

    push!(overdubbed_codelocs, code_info.codelocs[1])

    push!(
        overdubbed_code,
        Core.ReturnNode(Core.SSAValue(length(overdubbed_code)))
    )
    push!(overdubbed_codelocs, code_info.codelocs[1])

    #=== set `code_info`/`reflection` fields accordingly ===#

    if code_info.method_for_inference_limit_heuristics === nothing
        code_info.method_for_inference_limit_heuristics = method
    end

    code_info.code = overdubbed_code
    code_info.codelocs = overdubbed_codelocs
    code_info.ssavaluetypes = length(overdubbed_code)
    code_info.ssaflags = [0x00 for _ in 1:length(overdubbed_code)] # XXX we need to copy flags that are set for the original code
    
    return code_info
end

@eval function call_with_reactant($REDUB_ARGUMENTS_NAME...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, call_with_reactant_generator))
end
