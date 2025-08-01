
function apply(f::F, args...; kwargs...) where {F}
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

const __skip_rewrite_func_set_lock = ReentrantLock()
const __skip_rewrite_func_set = Set([
    # Avoid the 1.10 stackoverflow
    typeof(Base.typed_hvcat),
    typeof(Base.hvcat),
    typeof(Core.Compiler.concrete_eval_eligible),
    typeof(Core.Compiler.typeinf_type),
    typeof(Core.Compiler.typeinf_ext),
    # TODO: perhaps problematic calls in `traced_call`
    # should be moved to TracedUtils.jl:
    typeof(Reactant.ReactantCore.traced_call),
    typeof(ReactantCore.is_traced),
    # Perf optimization
    typeof(Base.typemax),
    typeof(Base.typemin),
    typeof(Base.getproperty),
    typeof(Base.vect),
    typeof(Base.eltype),
    typeof(Base.argtail),
    typeof(Base.identity),
    typeof(Base.print),
    typeof(Base.println),
    typeof(Base.show),
    typeof(Base.show_delim_array),
    typeof(Base.sprint),
    typeof(Adapt.adapt_structure),
    typeof(Core.is_top_bit_set),
    typeof(Base.setindex_widen_up_to),
    typeof(Base.typejoin),
    typeof(Base.argtype_decl),
    typeof(Base.arg_decl_parts),
    typeof(Base.StackTraces.show_spec_sig),
    typeof(Core.Compiler.return_type),
    typeof(Core.throw_inexacterror),
    typeof(Base.throw_boundserror),
    typeof(Base._shrink),
    typeof(Base._shrink!),
    typeof(Base.ht_keyindex),
    typeof(Base.checkindex),
    typeof(Base.to_index),
    @static(
        if VERSION >= v"1.11.0"
            typeof(Base.memoryref)
        end
    ),
    typeof(Reactant.materialize_traced_array),
])

"""
    @skip_rewrite_func f

Mark function `f` so that Reactant's IR rewrite mechanism will skip it.
This can improve compilation time if it's safe to assume that no call inside `f`
will need a `@reactant_overlay` method.

!!! info
    Note that this marks the whole function, not a specific method with a type
    signature.

!!! warning
    The macro call should be inside the `__init__` function. If you want to
    mark it for precompilation, you must add the macro call in the global scope
    too.

See also: [`@skip_rewrite_type`](@ref)
"""
macro skip_rewrite_func(fname)
    quote
        @lock $(Reactant.__skip_rewrite_func_set_lock) push!(
            $(Reactant.__skip_rewrite_func_set), typeof($(esc(fname)))
        )
    end
end

const __skip_rewrite_type_constructor_list_lock = ReentrantLock()
const __skip_rewrite_type_constructor_list = [
    # Don't rewrite Val
    Type{Base.Val},
    # Don't rewrite exception constructors
    Type{<:Core.Exception},
    # Don't rewrite traced constructors
    Type{<:TracedRArray},
    Type{<:TracedRNumber},
    Type{MLIR.IR.Location},
    Type{MLIR.IR.Block},
]

"""
    @skip_rewrite_type MyStruct
    @skip_rewrite_type Type{<:MyStruct}

Mark the construct function of `MyStruct` so that Reactant's IR rewrite mechanism
will skip it. It does the same as [`@skip_rewrite_func`](@ref) but for type
constructors.

If you want to mark the set of constructors over it's type parameters or over its
abstract type, you should use then the `Type{<:MyStruct}` syntax.

!!! warning
    The macro call should be inside the `__init__` function. If you want to
    mark it for precompilation, you must add the macro call in the global scope
    too.
"""
macro skip_rewrite_type(typ)
    typ = if Base.isexpr(typ, :curly) && typ.args[1] === :Type
        typ
    else
        Expr(:curly, :Type, typ)
    end
    return quote
        @lock $(Reactant.__skip_rewrite_type_constructor_list_lock) push!(
            $(Reactant.__skip_rewrite_type_constructor_list), $(esc(typ))
        )
    end
end

function should_rewrite_call(@nospecialize(ft))
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
                has_ancestor(mod, Reactant.MLIR)
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

    # `ft isa Type` is for performance as it avoids checking against all the list, but can be removed if problematic
    if ft isa Type && any(t -> ft <: t, __skip_rewrite_type_constructor_list)
        return false
    end

    if ft in __skip_rewrite_func_set
        return false
    end

    # Default assume all functions need to be reactant-ified
    return true
end

# by default, same as `should_rewrite_call`
function should_rewrite_invoke(@nospecialize(ft), @nospecialize(args))
    # TODO how can we extend `@skip_rewrite` to methods?
    if ft <: typeof(repeat) && (args == Tuple{String,Int64} || args == Tuple{Char,Int64})
        return false
    end
    return should_rewrite_call(ft)
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

function rewrite_inst(inst, ir::CC.IRCode, interp, RT, guaranteed_error)
    if Meta.isexpr(inst, :call)
        # Even if type unstable we do not want (or need) to replace intrinsic
        # calls or builtins with our version.
        ft = Core.Compiler.widenconst(maybe_argextype(inst.args[1], ir))
        if ft == typeof(Core.kwcall)
            ft = Core.Compiler.widenconst(maybe_argextype(inst.args[3], ir))
        end
        if ft == typeof(Core._apply_iterate)
            ft = Core.Compiler.widenconst(maybe_argextype(inst.args[3], ir))
            if Base.invokelatest(should_rewrite_call, ft)
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
        elseif Base.invokelatest(should_rewrite_call, ft)
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
        argsig = sig.parameters[2:end]
        if ft == typeof(Core.kwcall)
            ft = sig.parameters[3]
            argsig = sig.parameters[4:end]
        end
        argsig = Core.apply_type(Core.Tuple, argsig...)
        if Base.invokelatest(should_rewrite_invoke, ft, argsig) && !is_reactant_method(omi)
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

function safe_print(name, x)
    return ccall(:jl_, Cvoid, (Any,), name * " " * string(x))
end

const DEBUG_INTERP = Ref(false)

# Rewrite type unstable calls to recurse into call_with_reactant to ensure
# they continue to use our interpreter. Reset the derived return type
# to Any if our interpreter would change the return type of any result.
# Also rewrite invoke (type stable call) to be :call, since otherwise apparently
# screws up type inference after this (TODO this should be fixed).
function rewrite_insts!(ir::CC.IRCode, interp, guaranteed_error)
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

using GPUCompiler
using GPUCompiler: AbstractCompilerParams, CompilerJob, NativeCompilerTarget

struct CompilerParams <: AbstractCompilerParams
    function CompilerParams()
        return new()
    end
end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,CompilerParams}

GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = false
GPUCompiler.can_throw(@nospecialize(job::NativeCompilerJob)) = true
GPUCompiler.needs_byval(@nospecialize(job::NativeCompilerJob)) = false

function GPUCompiler.optimize!(
    @nospecialize(job::NativeCompilerJob), mod::GPUCompiler.LLVM.Module; opt_level
)
    return nothing #TODO: add all except GPU stuff passes
end

using Enzyme
ReactantInter = Enzyme.Compiler.Interpreter.EnzymeInterpreter{
    typeof(Reactant.set_reactant_abi)
}

function GPUCompiler.get_interpreter(@nospecialize(job::NativeCompilerJob))
    return Reactant.ReactantInterpreter(; world=job.world)
end
function GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob))
    return CC.method_table(GPUCompiler.get_interpreter(job))
end

GPUCompiler.llvm_debug_info(@nospecialize(::NativeCompilerJob)) = GPUCompiler.LLVM.API.LLVMDebugEmissionKindNoDebug

function CC.optimize(
    interp::ReactantInter, opt::CC.OptimizationState, caller::CC.InferenceResult
)
    CC.@timeit "optimizer" ir = CC.run_passes_ipo_safe(opt.src, opt, caller)
    CC.ipo_dataflow_analysis!(interp, ir, caller)

    mi = caller.linfo
    if false &&
        mi in mi_set &&
        !(
            is_reactant_method(mi) || (
                mi.def.sig isa DataType &&
                !should_rewrite_invoke(
                    mi.def.sig.parameters[1], Tuple{mi.def.sig.parameters[2:end]...}
                )
            )
        )
        @info ir
        ir, has_changed = rewrite_insts!(ir, interp, false)
        @info ir
        has_changed && @info "rewrite instruction $mi"
    end

    return CC.finish(interp, opt, ir, caller)
end



function GPUCompiler.ci_cache_populate(
    interp::Reactant.ReactantInter,
    cache::CC.WorldView{CC.InternalCodeCache},
    mi::Core.MethodInstance,
    min_world::UInt64,
    max_world::UInt64,
)
    @warn mi min_world max_world CC.get_inference_world(interp)
    @invoke GPUCompiler.ci_cache_populate(
        interp::CC.AbstractInterpreter, cache, mi, min_world, max_world
    )
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
        identity, Core.svec(:call_with_reactant, Reactant.REDUB_ARGUMENTS_NAME), Core.svec()
    )

    guaranteed_error = false
    if args[1] === Reactant.MustThrowError
        guaranteed_error = true
    end
    offset_error = guaranteed_error ? 1 : 0
    fn = args[1 + offset_error]

    if fn <: Core.Builtin
        builtin_error =
            :(throw(AssertionError("Unsupported call_with_reactant of builtin $fn")))
        return stub(world, source, builtin_error)
    end

    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]

    if guaranteed_error
        if rt !== Union{}
            safe_print("Inconsistent guaranteed error IR", ir)
        end
        rt = Union{}
    end

    mi = GPUCompiler.methodinstance(
        fn, Base.to_tuple_type(args[(2 + offset_error):end]), world
    )
    if mi === nothing
        method_error = :(throw(
            MethodError(
                $REDUB_ARGUMENTS_NAME[1 + offset_error],
                $REDUB_ARGUMENTS_NAME[(2 + offset_error):end],
                $world,
            ),
        ))
        return stub(world, mi, method_error)
    end
    config = CompilerConfig(
        Reactant.NativeCompilerTarget(),
        Reactant.CompilerParams();
        kernel=false,
        libraries=false,
        toplevel=true,
        validate=false,
        strip=true,
        entry_abi=:func,
    )

    job = GPUCompiler.CompilerJob(mi, config, world)

    llvm_module, meta_ = Reactant.JuliaContext() do _ctx
        GPUCompiler.compile(:llvm, job)
    end
    mm = meta_.compiled[job.source]

    #CodeInfo placehold
    code_info = begin
        ir = CC.IRCode()
        src = ccall(:jl_new_code_info_uninit, Ref{CC.CodeInfo}, ())
        src.slotnames = fill(:none, length(ir.argtypes) + 1)
        src.slotflags = fill(zero(UInt8), length(ir.argtypes))
        src.slottypes = copy(ir.argtypes)
        src.rettype = Int
        CC.ir_to_codeinf!(src, ir)
    end

    rt = mm.ci.rettype
    code_info.rettype = rt

    # propagate edge metadata, this method is invalidated if the original function we are calling
    # is invalidated
    code_info.edges = Core.MethodInstance[job.source]
    code_info.min_world = typemin(UInt)
    code_info.max_world = typemax(UInt)

    # Rewrite the arguments to this function, to prepend the two new arguments, the function :call_with_reactant,
    # and the REDUB_ARGUMENTS_NAME tuple of input arguments
    code_info.slotnames = Any[:call_with_reactant, Reactant.REDUB_ARGUMENTS_NAME]
    code_info.slotflags = UInt8[0x00, 0x00]
    n_prepended_slots = 2
    overdub_args_slot = Core.SlotNumber(n_prepended_slots)

    # For the sake of convenience, the rest of this pass will translate `code_info`'s fields
    # into these overdubbed equivalents instead of updating `code_info` in-place. Then, at
    # the end of the pass, we'll reset `code_info` fields accordingly.
    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]
    function push_inst!(inst)
        push!(overdubbed_code, inst)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        return Core.SSAValue(length(overdubbed_code))
    end
    # Rewire the arguments from our tuple input of fn and args, to the corresponding calling convention
    # required by the base method.

    # destructure the generated argument slots into the overdubbed method's argument slots.

    offset = 2
    fn_args = Any[]
    method = job.source.def
    n_actual_args = length(redub_arguments)
    offset += offset_error
    n_actual_args -= offset_error

    tys = []
    for i in 2:n_actual_args
        type = redub_arguments[i + (guaranteed_error ? 1 : 0)]
        actual_argument = Expr(
            :call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset
        )
        arg = push_inst!(actual_argument)
        push!(fn_args, arg)
        push!(tys,  Base.RefValue{type})
        offset += 1
    end

    #TODO: replace 0 with function pointer using mi.cache.invoke
    @warn mi.cache.invoke
    fn_args_tuple = push_inst!(Expr(:call, GlobalRef(Base, :vect), fn_args...))
    boxed_res = push_inst!(
        Expr(
            :call,
            GlobalRef(Base, :llvmcall),
            (string(llvm_module), mm.func),
            Base.RefValue{rt},
            Tuple{Base.RefValue{fn},Vector{Any}, Int32},
            0, fn_args_tuple, Int32(length(fn_args))
        ),
    )
    res = push_inst!(Expr(:call, GlobalRef(Base, :getfield), boxed_res, QuoteNode(:x)))
    push_inst!(Core.ReturnNode(res))

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

@eval function call_with_reactant($(Reactant.REDUB_ARGUMENTS_NAME)...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, call_with_reactant_generator))
end


@static if isdefined(Core, :BFloat16)
    nmantissa(::Type{Core.BFloat16}) = 7
end
nmantissa(::Type{Float16}) = 10
nmantissa(::Type{Float32}) = 23
nmantissa(::Type{Float64}) = 52