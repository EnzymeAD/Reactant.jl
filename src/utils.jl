
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

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Union{Nothing,Core.MethodTable},
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
    res = ccall(
        :jl_gf_invoke_lookup_worlds,
        Any,
        (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
        sig,
        mt,
        world,
        min_world,
        max_world,
    )
    return res
end

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Core.Compiler.InternalMethodTable,
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
    res = lookup_world(sig, mt.world, nothing, min_world, max_world)
    return res
end

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Core.Compiler.OverlayMethodTable,
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
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
        if hasfield(typeof(ft), :name) &&
            hasfield(typeof(ft.name), :name) &&
            isdefined(ft.name, :name)
            namestr = String(ft.name.name)
            if startswith(namestr, "##(overlay (. Reactant (inert REACTANT_METHOD_TABLE)")
                return false
            end
        end

        # We need this for closures to work
        if hasfield(typeof(ft), :name) && hasfield(typeof(ft.name), :module)
            mod = ft.name.module
            # Don't rewrite primitive ops, tracing utilities, or any MLIR-based functions
            if has_ancestor(mod, Reactant.Ops) ||
                has_ancestor(mod, Reactant.TracedUtils) ||
                has_ancestor(mod, Reactant.MLIR) ||
                has_ancestor(mod, Reactant.TracedRandom)
                return false
            end
            if string(mod) == "CUDA"
                if ft.name.name == Symbol("#launch_configuration")
                    return false
                end
                if ft.name.name == Symbol("cudaconvert")
                    return false
                end
            end
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

    # Avoid the 1.10 stackoverflow
    if ft <: typeof(Base.typed_hvcat)
        return false
    end
    if ft <: typeof(Base.hvcat)
        return false
    end
    if ft <: typeof(Core.Compiler.concrete_eval_eligible)
        return false
    end
    if ft <: typeof(Core.Compiler.typeinf_type) || ft <: typeof(Core.Compiler.typeinf_ext)
        return false
    end

    # Don't rewrite traced constructors
    if ft <: Type{<:TracedRArray} ||
        ft <: Type{<:TracedRNumber} ||
        ft === Type{MLIR.IR.Location} ||
        ft === Type{MLIR.IR.Block}
        return false
    end

    # Perf optimizations
    if ft <: typeof(Core.Compiler.return_type)
        return false
    end

    # Perf optimizations
    if ft <: typeof(Base.typemax) ||
        ft <: typeof(Base.typemin) ||
        ft <: typeof(Base.getproperty) ||
        ft <: typeof(Base.vect) ||
        ft <: typeof(Base.eltype) ||
        ft <: typeof(Base.argtail) ||
        ft <: typeof(Base.identity) ||
        ft <: typeof(Base.print) ||
        ft <: typeof(Base.println) ||
        ft <: typeof(Adapt.adapt_structure)
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

struct MustThrowError end

@generated function applyiterate_with_reactant(
    iteratefn, applyfn, args::Vararg{Any,N}
) where {N}
    if iteratefn != typeof(Base.iterate)
        return quote
            error("Unhandled apply_iterate with iteratefn=$iteratefn")
        end
    end
    newargs = Vector{Expr}(undef, N)
    for i in 1:N
        @inbounds newargs[i] = :(args[$i]...)
    end
    quote
        Base.@_inline_meta
        call_with_reactant(applyfn, $(newargs...))
    end
end

@generated function applyiterate_with_reactant(
    mt::MustThrowError, iteratefn, applyfn, args::Vararg{Any,N}
) where {N}
    @assert iteratefn == typeof(Base.iterate)
    newargs = Vector{Expr}(undef, N)
    for i in 1:N
        @inbounds newargs[i] = :(args[$i]...)
    end
    quote
        Base.@_inline_meta
        call_with_reactant(mt, applyfn, $(newargs...))
    end
end

function certain_error()
    throw(
        AssertionError(
            "The inferred code was guaranteed to throw this error. And yet, it didn't. So here we are...",
        ),
    )
end

function rewrite_inst(inst, ir, interp, RT, guaranteed_error)
    if Meta.isexpr(inst, :call)
        # Even if type unstable we do not want (or need) to replace intrinsic
        # calls or builtins with our version.
        ft = Core.Compiler.widenconst(maybe_argextype(inst.args[1], ir))
        if ft == typeof(Core.kwcall)
            ft = Core.Compiler.widenconst(maybe_argextype(inst.args[3], ir))
        end
        if ft == typeof(Core._apply_iterate)
            ft = Core.Compiler.widenconst(maybe_argextype(inst.args[3], ir))
            if Base.invokelatest(should_rewrite_ft, ft)
                if RT === Union{}
                    rep = Expr(
                        :call,
                        applyiterate_with_reactant,
                        MustThrowError(),
                        inst.args[2:end]...,
                    )
                    return true, rep, Union{}
                else
                    rep = Expr(:call, applyiterate_with_reactant, inst.args[2:end]...)
                    return true, rep, Any
                end
            end
        elseif Base.invokelatest(should_rewrite_ft, ft)
            if RT === Union{}
                rep = Expr(:call, call_with_reactant, MustThrowError(), inst.args...)
                return true, rep, Union{}
            else
                rep = Expr(:call, call_with_reactant, inst.args...)
                return true, rep, Any
            end
        end
    end
    if Meta.isexpr(inst, :invoke)
        omi = inst.args[1]::Core.MethodInstance
        sig = omi.specTypes
        ft = sig.parameters[1]
        if ft == typeof(Core.kwcall)
            ft = sig.parameters[3]
        end
        if Base.invokelatest(should_rewrite_ft, ft) && !is_reactant_method(omi)
            method = omi.def::Core.Method

            min_world = Ref{UInt}(typemin(UInt))
            max_world = Ref{UInt}(typemax(UInt))

            # RT = Any

            if !method.isva || !Base.isvarargtype(sig.parameters[end])
                if RT === Union{}
                    sig2 = Tuple{
                        typeof(call_with_reactant),MustThrowError,sig.parameters...
                    }
                else
                    sig2 = Tuple{typeof(call_with_reactant),sig.parameters...}
                end
            else
                vartup = inst.args[end]
                ns = Type[]
                eT = sig.parameters[end].T
                for i in 1:(length(inst.args) - 1 - (length(sig.parameters) - 1))
                    push!(ns, eT)
                end
                if RT === Union{}
                    sig2 = Tuple{
                        typeof(call_with_reactant),
                        MustThrowError,
                        sig.parameters[1:(end - 1)]...,
                        ns...,
                    }
                else
                    sig2 = Tuple{
                        typeof(call_with_reactant),sig.parameters[1:(end - 1)]...,ns...
                    }
                end
            end

            lookup_result = lookup_world(
                sig2, interp.world, Core.Compiler.method_table(interp), min_world, max_world
            )

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
            n_method_args = method.nargs
            if RT === Union{}
                rep = Expr(
                    :invoke, mi, call_with_reactant, MustThrowError(), inst.args[2:end]...
                )
                return true, rep, Union{}
            else
                rep = Expr(:invoke, mi, call_with_reactant, inst.args[2:end]...)
                return true, rep, Any
            end
        end
    end
    if isa(inst, Core.ReturnNode) && (!isdefined(inst, :val) || guaranteed_error)
        min_world = Ref{UInt}(typemin(UInt))
        max_world = Ref{UInt}(typemax(UInt))

        sig2 = Tuple{typeof(certain_error)}

        lookup_result = lookup_world(
            sig2, interp.world, Core.Compiler.method_table(interp), min_world, max_world
        )

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
        rep = Expr(:invoke, mi, certain_error)
        return true, rep, Union{}
    end
    return false, inst, RT
end

const oc_capture_vec = Vector{Any}()

# Caching is both good to reducing compile times and necessary to work around julia bugs
# in OpaqueClosure's: https://github.com/JuliaLang/julia/issues/56833
function make_oc_dict(
    @nospecialize(oc_captures::Dict{FT,Core.OpaqueClosure}),
    @nospecialize(sig::Type),
    @nospecialize(rt::Type),
    @nospecialize(src::Core.CodeInfo),
    nargs::Int,
    isva::Bool,
    @nospecialize(f::FT)
)::Core.OpaqueClosure where {FT}
    key = f
    if haskey(oc_captures, key)
        oc = oc_captures[key]
        oc
    else
        ores = ccall(
            :jl_new_opaque_closure_from_code_info,
            Any,
            (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any, Cint),
            sig,
            rt,
            rt,
            @__MODULE__,
            src,
            0,
            nothing,
            nargs,
            isva,
            f,
            true,
        )::Core.OpaqueClosure
        oc_captures[key] = ores
        return ores
    end
end

function make_oc_ref(
    oc_captures::Base.RefValue{Core.OpaqueClosure},
    @nospecialize(sig::Type),
    @nospecialize(rt::Type),
    @nospecialize(src::Core.CodeInfo),
    nargs::Int,
    isva::Bool,
    @nospecialize(f)
)::Core.OpaqueClosure
    if Base.isassigned(oc_captures)
        return oc_captures[]
    else
        ores = ccall(
            :jl_new_opaque_closure_from_code_info,
            Any,
            (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any, Cint),
            sig,
            rt,
            rt,
            @__MODULE__,
            src,
            0,
            nothing,
            nargs,
            isva,
            f,
            true,
        )::Core.OpaqueClosure
        oc_captures[] = ores
        return ores
    end
end

function safe_print(name, x)
    return ccall(:jl_, Cvoid, (Any,), name * " " * string(x))
end

const DEBUG_INTERP = Ref(false)

# Rewrite type unstable calls to recurse into call_with_reactant to ensure
# they continue to use our interpreter. Reset the derived return type
# to Any if our interpreter would change the return type of any result.
# Also rewrite invoke (type stable call) to be :call, since otherwise apparently
# screws up type inference after this (TODO this should be fixed).
function rewrite_insts!(ir, interp, guaranteed_error)
    any_changed = false
    for (i, inst) in enumerate(ir.stmts)
        # Explicitly skip any code which returns Union{} so that we throw the error
        # instead of risking a segfault
        RT = inst[:type]
        @static if VERSION < v"1.11"
            changed, next, RT = rewrite_inst(inst[:inst], ir, interp, RT, guaranteed_error)
            Core.Compiler.setindex!(ir.stmts[i], next, :inst)
        else
            changed, next, RT = rewrite_inst(inst[:stmt], ir, interp, RT, guaranteed_error)
            Core.Compiler.setindex!(ir.stmts[i], next, :stmt)
        end
        if changed
            any_changed = true
            Core.Compiler.setindex!(ir.stmts[i], RT, :type)
        end
    end
    return ir, any_changed
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
    if DEBUG_INTERP[]
        safe_print("args", args)
    end

    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(:call_with_reactant, REDUB_ARGUMENTS_NAME), Core.svec()
    )

    fn = args[1]
    sig = Tuple{args...}

    guaranteed_error = false
    if fn === MustThrowError
        guaranteed_error = true
        fn = args[2]
        sig = Tuple{args[2:end]...}
    end

    # look up the method match
    builtin_error =
        :(throw(AssertionError("Unsupported call_with_reactant of builtin $fn")))

    if fn <: Core.Builtin
        return stub(world, source, builtin_error)
    end

    if guaranteed_error
        method_error = :(throw(
            MethodError($REDUB_ARGUMENTS_NAME[2], $REDUB_ARGUMENTS_NAME[3:end], $world)
        ))
    else
        method_error = :(throw(
            MethodError($REDUB_ARGUMENTS_NAME[1], $REDUB_ARGUMENTS_NAME[2:end], $world)
        ))
    end

    interp = ReactantInterpreter(; world)

    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))

    lookup_result = lookup_world(
        sig, world, Core.Compiler.method_table(interp), min_world, max_world
    )

    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]

    # No method could be found (including in our method table), bail with an error
    if lookup_result === nothing
        return stub(world, source, method_error)
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
    method = mi.def

    @static if VERSION < v"1.11"
        # For older Julia versions, we vendor in some of the code to prevent
        # having to build the MethodInstance twice.
        result = CC.InferenceResult(mi, CC.typeinf_lattice(interp))
        frame = CC.InferenceState(result, :no, interp)
        @assert !isnothing(frame)
        CC.typeinf(interp, frame)
        ir = CC.run_passes(frame.src, CC.OptimizationState(frame, interp), result, nothing)
        rt = CC.widenconst(CC.ignorelimited(result.result))
    else
        ir, rt = CC.typeinf_ircode(interp, mi, nothing)
    end

    if guaranteed_error
        if rt !== Union{}
            safe_print("Inconsistent guaranteed error IR", ir)
        end
        rt = Union{}
    end

    if DEBUG_INTERP[]
        safe_print("ir", ir)
    end

    if !is_reactant_method(mi::Core.MethodInstance) || guaranteed_error
        ir, any_changed = rewrite_insts!(ir, interp, guaranteed_error)
    end

    src = ccall(:jl_new_code_info_uninit, Ref{CC.CodeInfo}, ())
    src.slotnames = fill(:none, length(ir.argtypes) + 1)
    src.slotflags = fill(zero(UInt8), length(ir.argtypes))
    src.slottypes = copy(ir.argtypes)
    src.rettype = rt
    src = CC.ir_to_codeinf!(src, ir)

    if DEBUG_INTERP[]
        safe_print("src", src)
    end

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
    if guaranteed_error
        offset += 1
        n_actual_args -= 1
    end

    tys = []

    iter_args = n_actual_args
    if method.isva
        iter_args = min(n_actual_args, n_method_args - 1)
    end

    for i in 1:iter_args
        actual_argument = Expr(
            :call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset
        )
        push!(overdubbed_code, actual_argument)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        offset += 1
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
        push!(tys, redub_arguments[i + (guaranteed_error ? 1 : 0)])

        if DEBUG_INTERP[]
            push!(
                overdubbed_code,
                Expr(
                    :call,
                    safe_print,
                    "fn arg[" * string(length(fn_args)) * "]",
                    fn_args[end],
                ),
            )
            push!(overdubbed_codelocs, code_info.codelocs[1])
        end
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

        push!(overdubbed_code, trailing_arguments)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
        push!(
            tys,
            Tuple{
                redub_arguments[(n_method_args:n_actual_args) .+ (guaranteed_error ? 1 : 0)]...,
            },
        )

        if DEBUG_INTERP[]
            push!(
                overdubbed_code,
                Expr(
                    :call,
                    safe_print,
                    "fn arg[" * string(length(fn_args)) * "]",
                    fn_args[end],
                ),
            )
            push!(overdubbed_codelocs, code_info.codelocs[1])
        end
    end

    # ocva = method.isva

    ocva = false # method.isva

    ocnargs = method.nargs - 1
    # octup = Tuple{mi.specTypes.parameters[2:end]...}
    # octup = Tuple{method.sig.parameters[2:end]...}
    octup = Tuple{tys[2:end]...}
    ocva = false

    # jl_new_opaque_closure forcibly executes in the current world... This means that we won't get the right
    # inner code during compilation without special handling (i.e. call_in_world_total).
    # Opaque closures also require taking the function argument. We can work around the latter
    # if the function is stateless. But regardless, to work around this we sadly create/compile the opaque closure

    dict, make_oc = if Base.issingletontype(fn)
        Base.Ref{Core.OpaqueClosure}(), make_oc_ref
    else
        Dict{fn,Core.OpaqueClosure}(), make_oc_dict
    end

    push!(oc_capture_vec, dict)

    oc = if false && Base.issingletontype(fn)
        res = Core._call_in_world_total(
            world, make_oc, dict, octup, rt, src, ocnargs, ocva, fn.instance
        )::Core.OpaqueClosure

    else
        farg = fn_args[1]
        rep = Expr(:call, make_oc, dict, octup, rt, src, ocnargs, ocva, farg)
        push!(overdubbed_code, rep)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        Core.SSAValue(length(overdubbed_code))
    end

    push!(overdubbed_code, Expr(:call, oc, fn_args[2:end]...))
    push!(overdubbed_codelocs, code_info.codelocs[1])

    ocres = Core.SSAValue(length(overdubbed_code))

    if DEBUG_INTERP[]
        push!(overdubbed_code, Expr(:call, safe_print, "ocres", ocres))
        push!(overdubbed_codelocs, code_info.codelocs[1])
    end

    push!(overdubbed_code, Core.ReturnNode(ocres))
    push!(overdubbed_codelocs, code_info.codelocs[1])

    #=== set `code_info`/`reflection` fields accordingly ===#

    if code_info.method_for_inference_limit_heuristics === nothing
        code_info.method_for_inference_limit_heuristics = method
    end

    code_info.code = overdubbed_code
    code_info.codelocs = overdubbed_codelocs
    code_info.ssavaluetypes = length(overdubbed_code)
    code_info.ssaflags = [0x00 for _ in 1:length(overdubbed_code)] # XXX we need to copy flags that are set for the original code

    if DEBUG_INTERP[]
        safe_print("code_info", code_info)
    end

    return code_info
end

@eval function call_with_reactant($REDUB_ARGUMENTS_NAME...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, call_with_reactant_generator))
end
