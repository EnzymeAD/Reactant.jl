using LLVM: LLVM
using GPUCompiler: GPUCompiler

struct CompilerParams <: GPUCompiler.AbstractCompilerParams
    use_native_interp::Bool
end

NativeCompilerJob = GPUCompiler.CompilerJob{GPUCompiler.NativeCompilerTarget,CompilerParams}
GPUCompiler.can_throw(@nospecialize(job::NativeCompilerJob)) = true
function GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob))
    return Core.Compiler.method_table(GPUCompiler.get_interpreter(job))
end

const DEBUG_INTERP = Ref(false)

ReactantInterp = Enzyme.Compiler.Interpreter.EnzymeInterpreter{
    typeof(Reactant.set_reactant_abi)
}
function GPUCompiler.get_interpreter(@nospecialize(job::NativeCompilerJob))
    return if job.config.params.use_native_interp
        Core.Compiler.NativeInterpreter(job.world)
    else
        Reactant.ReactantInterpreter(; world=job.world)
    end
end

function Core.Compiler.optimize(
    interp::ReactantInterp, opt::Core.Compiler.OptimizationState, caller::Core.Compiler.InferenceResult
)
    @static if VERSION < v"1.11"
        Core.Compiler.@timeit "optimizer" ir = Core.Compiler.run_passes(opt.src, opt, caller)
    elseif VERSION < v"1.12"
        Core.Compiler.@timeit "optimizer" ir = Core.Compiler.run_passes_ipo_safe(opt.src, opt, caller)
        Core.Compiler.ipo_dataflow_analysis!(interp, ir, caller)
    else
        Core.Compiler.@timeit "optimizer" ir = Core.Compiler.run_passes_ipo_safe(opt.src, opt)
        Core.Compiler.ipo_dataflow_analysis!(interp, opt, ir, caller)
    end
    mi = opt.linfo
		if DEBUG_INTERP[]
		   safe_print("pre rewrite_insts", ir)
		end
    ir, _ = rewrite_insts!(ir, interp)
		if DEBUG_INTERP[]
		   safe_print("post rewrite_insts", ir)
		end
    Core.Compiler.verify_ir(ir)
    res = Core.Compiler.finish(interp, opt, ir, caller)

    return res
end

@noinline call_with_native(@nospecialize(f), @nospecialize(args...); @nospecialize(kwargs...)) = f(args...; kwargs...)

struct CallWithReactant{F} <: Function
    f::F
end

function Base.reducedim_init(f::F, op::CallWithReactant, A::AbstractArray, region) where {F}
    return Base.reducedim_init(f, op.f, A, region)
end

function (f::CallWithReactant{F})(args...; kwargs...) where {F}
    if isempty(kwargs)
        return call_with_reactant(f.f, args...)
    else
        return call_with_reactant(Core.kwcall, NamedTuple(kwargs), f.f, args...)
    end
end

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

# Defined in KernelAbstractions Ext
function ka_with_reactant end

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
    typeof(call_with_native),
    # Avoid the 1.10 stackoverflow
    typeof(Base.typed_hvcat),
    typeof(Base.hvcat),
    typeof(Core.Compiler.concrete_eval_eligible),
    typeof(Core.Compiler.typeinf_type),
    typeof(Core.Compiler.typeinf_ext),
    # TODO: perhaps problematic calls in `traced_call`
    # should be moved to TracedUtils.jl:
    typeof(ReactantCore.traced_call),
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
    typeof(Base.print_to_string),
    typeof(Base.println),
    typeof(Base.Filesystem.joinpath),
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
    typeof(materialize_traced_array),
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

const no_rewrite_ancestor_modules = Module[Reactant.MLIR]
function should_rewrite_call(@nospecialize(ft))
    # Don't rewrite builtin or intrinsics, unless they are apply iter or kwcall
    if ft === typeof(Core.kwcall) || ft === typeof(Core._apply_iterate)
        return true
    end
    if ft <: Core.IntrinsicFunction || ft <: Core.Builtin
        return false
    end
    if ft === typeof(Base.string)
        return false
    end
    if ft === typeof(call_with_reactant)
        return false
    end
    if ft === typeof(Core.Intrinsics.llvmcall)
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
	    for nrwmod in no_rewrite_ancestor_modules
		if has_ancestor(mod, nrwmod)
			return false
		end
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

struct EnsureReturnType{T}
end

@generated function applyiterate_with_reactant(
    ert::EnsureReturnType, iteratefn, applyfn, args::Vararg{Any,N}
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
        call_with_reactant(ert, applyfn, $(newargs...))
    end
end

function rewrite_inst(inst, ir, interp, RT)
    RT0 = Core.Compiler.widenconst(RT)
    if Meta.isexpr(inst, :call)
        # Even if type unstable we do not want (or need) to replace intrinsic
        # calls or builtins with our version.
        ft = Core.Compiler.widenconst(maybe_argextype(inst.args[1], ir))
        if ft == typeof(Core.kwcall)
            ft = Core.Compiler.widenconst(maybe_argextype(inst.args[3], ir))
        end
        if ft == typeof(Core._apply_iterate)
            ft = Core.Compiler.widenconst(maybe_argextype(inst.args[3], ir))
            if Core._call_in_world_total(interp.world, should_rewrite_call, ft)
	       rep = Expr(:call, applyiterate_with_reactant, EnsureReturnType{RT0}(), inst.args[2:end]...)
                return true, rep, RT0
            end
        elseif Core._call_in_world_total(interp.world, should_rewrite_call, ft)
	       rep = Expr(:call, call_with_reactant, EnsureReturnType{RT0}(), inst.args...)
            return true, rep, RT0
        end
    end
    if Meta.isexpr(inst, :invoke)
        omi = if inst.args[1] isa Core.MethodInstance
            inst.args[1]
        else
            (inst.args[1]::Core.CodeInstance).def
        end
        sig = omi.specTypes
        ft = sig.parameters[1]
        argsig = sig.parameters[2:end]
        if ft == typeof(Core.kwcall)
            ft = sig.parameters[3]
            argsig = sig.parameters[4:end]
        end
        argsig = Core.apply_type(Core.Tuple, argsig...)
        if Core._call_in_world_total(interp.world, should_rewrite_invoke, ft, argsig) &&
            !is_reactant_method(omi)
            method = omi.def::Core.Method

            min_world = Ref{UInt}(typemin(UInt))
            max_world = Ref{UInt}(typemax(UInt))

            # RT = Any

            if !method.isva || !Base.isvarargtype(sig.parameters[end])
                sig2 = Tuple{typeof(call_with_reactant),EnsureReturnType{RT0}, sig.parameters...}
            else
                vartup = inst.args[end]
                ns = Type[]
                eT = sig.parameters[end].T
                for i in 1:(length(inst.args) - 1 - (length(sig.parameters) - 1))
                    push!(ns, eT)
                end
                sig2 = Tuple{
                    typeof(call_with_reactant),EnsureReturnType{RT0}, sig.parameters[1:(end - 1)]...,ns...
                }
            end
	    
	    all_datatype = true
	    for T in sig2.parameters
		if !(T <: DataType)
		    all_datatype = false
		    break
		end
	    end
	    if !all_datatype
	       rep = Expr(:call, call_with_reactant, EnsureReturnType{RT0}(), inst.args[2:end]...)
               return true, rep, RT0
	    end

            lookup_result = Enzyme.lookup_world(
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
	    if is_reactant_method(mi)
	       rep = Expr(:call, call_with_reactant, EnsureReturnType{RT0}(), inst.args[2:end]...)
               return true, rep, RT0
	    end
            n_method_args = method.nargs
	    rep = Expr(:invoke, mi, call_with_reactant, EnsureReturnType{RT0}(), inst.args[2:end]...)
            return true, rep, RT0
        end
    end
    if false && isa(inst, Core.ReturnNode) && (!isdefined(inst, :val))
        min_world = Ref{UInt}(typemin(UInt))
        max_world = Ref{UInt}(typemax(UInt))

        sig2 = Tuple{typeof(certain_error)}

        lookup_result = Enzyme.lookup_world(
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

function safe_print2(name, x)
    ccall(:jl_, Cvoid, (Any,), name * "=\\n")
    ccall(:jl_, Cvoid, (Any,), x)
end

# Rewrite type unstable calls to recurse into call_with_reactant to ensure
# they continue to use our interpreter. Reset the derived return type
# to Any if our interpreter would change the return type of any result.
# Also rewrite invoke (type stable call) to be :call, since otherwise apparently
# screws up type inference after this (TODO this should be fixed).
function rewrite_insts!(ir, interp)
    any_changed = false
    for (i, inst) in enumerate(ir.stmts)
        # Explicitly skip any code which returns Union{} so that we throw the error
        # instead of risking a segfault
        RT = inst[:type]
        @static if VERSION < v"1.11"
            changed, next, RT = rewrite_inst(inst[:inst], ir, interp, RT)
            Core.Compiler.setindex!(ir.stmts[i], next, :inst)
        else
            changed, next, RT = rewrite_inst(inst[:stmt], ir, interp, RT)
            Core.Compiler.setindex!(ir.stmts[i], next, :stmt)
        end
        if changed
            any_changed = true
            Core.Compiler.setindex!(ir.stmts[i], RT, :type)
        end
    end
    return ir, any_changed
end

function rewrite_argnumbers_by_one!(ir)
    # Add one dummy argument at the beginning
    pushfirst!(ir.argtypes, Nothing)

    # Re-write all references to existing arguments to their new index (N + 1)
    for idx in 1:length(ir.stmts)
        urs = Core.Compiler.userefs(ir.stmts[idx][:inst])
        changed = false
        it = Base.iterate(urs)
        while it !== nothing
            (ur, next) = it
            old = Base.getindex(ur)
            if old isa Core.Argument
                # Replace the Argument(n) with Argument(n + 1)
                Core.Compiler.setindex!(ur, Core.Argument(old.n + 1))
                changed = true
            end
            it = Core.Compiler.iterate(urs, next)
        end
        if changed
            @static if VERSION < v"1.11"
                Core.Compiler.setindex!(ir.stmts[idx], Core.Compiler.getindex(urs), :inst)
            else
                Core.Compiler.setindex!(ir.stmts[idx], Core.Compiler.getindex(urs), :stmt)
            end
        end
    end

    return nothing
end

const call_with_reactant_lock = ReentrantLock()
const call_with_reactant_cache = Dict{UInt,Tuple{String, Type, Vector{Any}}}()

@inline function push_inst!(overdubbed_code::Vector{Any}, @nospecialize(inst::Any))
    push!(overdubbed_code, inst)
    return Core.SSAValue(length(overdubbed_code))
end

struct ReactantRuntimeException <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::ReactantRuntimeException)
    print(io, "ReactantRuntimeException: Reactant interpretation failed.\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct ReactantPrecompilationException <: Base.Exception
   str::String
end

function Base.showerror(io::IO, ece::ReactantPrecompilationException)
   print(io, "ReactantPrecopmilationException: Precompilation not supported due to null global: $(ece.str)\n")
end

# Generator function which ensures that all calls to the function are executed within the ReactantInterpreter
# In particular this entails two pieces:
#   1) We enforce the use of the ReactantInterpreter method table when generating the original methodinstance
#   2) Post type inference (using of course the reactant interpreter), all type unstable call functions are
#      replaced with calls to `call_with_reactant`. This allows us to circumvent long standing issues in Julia
#      using a custom interpreter in type unstable code.
# `redub_arguments` is `(typeof(original_function), map(typeof, original_args_tuple)...)`
function call_llvm_generator(world::UInt, source, self, ::Type{typeof(Reactant.call_with_reactant)}, @nospecialize(args::Tuple{Vararg{DataType}}))
    RT = nothing
    if args[1] <: EnsureReturnType
        RT = args[1].parameters[1]
	args = args[2:end]
    end
    f = args[1]
    tt = Tuple{f,args[2:end]...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))

    use_native_interpreter =
        !Core._call_in_world_total(world, should_rewrite_invoke, f, Tuple{args[2:end]...})
    interp = if use_native_interpreter
        Core.Compiler.NativeInterpreter(world)
    else
        ReactantInterpreter(; world)
    end

    if DEBUG_INTERP[]
	safe_print("tt, native_interp", (tt, use_native_interpreter))
    end

    lookup_result = Enzyme.lookup_world(
        tt, world, Core.Compiler.method_table(interp), min_world, max_world
    )

    match = lookup_result::Core.MethodMatch

    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(:call_with_reactant, REDUB_ARGUMENTS_NAME), Core.svec()
    )

    if isnothing(match)
        method_error = :(throw(
            MethodError($REDUB_ARGUMENTS_NAME[1], $REDUB_ARGUMENTS_NAME[2:end], $world)
        ))
        return stub(world, source, method_error)
    end

    mi = ccall(
        :jl_specializations_get_linfo,
        Ref{Core.MethodInstance},
        (Any, Any, Any),
        match.method,
        match.spec_types,
        match.sparams,
    )
    method = mi.def

    if DEBUG_INTERP[]
	safe_print("mi", mi)
    end

    slotnames = Any[:call_llvm_generator, REDUB_ARGUMENTS_NAME]
    overdubbed_code = Any[]

    if !use_native_interpreter
       push_inst!(overdubbed_code, Expr(:meta, :force_compile))
    end

    fn_args = Core.SSAValue[]
    f_arg = push_inst!(
       overdubbed_code, Expr(:call, Core.GlobalRef(Core, :getfield), Core.SlotNumber(2), 1 + (RT !== nothing))
    )
    if DEBUG_INTERP[]
       push_inst!(
			  overdubbed_code, Expr(:call, safe_print2, "f_arg", f_arg))
    end
    for i in 2:length(args)
        named_tuple_ssa = push_inst!(overdubbed_code, Expr(
	       :call, Core.GlobalRef(Core, :getfield), Core.SlotNumber(2), i + (RT !== nothing)
        ))
        if DEBUG_INTERP[]
          push_inst!(
		     overdubbed_code, Expr(:call, safe_print2, "args[$i]", named_tuple_ssa))
        end
        push!(fn_args, named_tuple_ssa)
    end


    if use_native_interpreter
        result = push_inst!(overdubbed_code, Expr(:call, f_arg, fn_args...))
        if DEBUG_INTERP[]
          push_inst!(
		     overdubbed_code, Expr(:call, safe_print2, "resultNative", result))
        end
    	if RT !== nothing
	    result = push_inst!(overdubbed_code, Expr(:call, Core.typeassert, result, RT))
	end

        push_inst!(overdubbed_code, Core.ReturnNode(result))

        code_info = Enzyme.create_fresh_codeinfo(
            call_with_reactant, source, world, slotnames, overdubbed_code
        )

	if code_info.method_for_inference_limit_heuristics === nothing
	    code_info.method_for_inference_limit_heuristics = method
    	end
    
	if VERSION >= v"1.12-"
            code_info.isva = true
	end

        code_info.min_world = min_world[]
        code_info.max_world = max_world[]

        edges = Any[mi]

        for gen_sig in (
            Tuple{typeof(should_rewrite_invoke),Type,Type},
            Tuple{typeof(should_rewrite_call),Type},
        )
            Enzyme.add_edge!(edges, gen_sig)
        end

        code_info.edges = edges
        
	if DEBUG_INTERP[]
		safe_print("code_infoNative", code_info)
        end
    
        return code_info
    end

    config = GPUCompiler.CompilerConfig(
        GPUCompiler.NativeCompilerTarget(; jlruntime=true, llvm_always_inline=false),
        CompilerParams(use_native_interpreter);
        kernel=false,
        libraries=true,
        toplevel=true,
        optimize=false,
	cleanup=false,
	only_entry=false,
	validate=false,
        entry_abi=:func,
    )

    job = GPUCompiler.CompilerJob(mi, config, world)
    key = hash(job)

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(call_with_reactant_lock)
    cached_compilation = try
        obj = Base.get(call_with_reactant_cache, key, nothing)
        if obj === nothing
            ts_ctx = GPUCompiler.JuliaContext()
            ctx = LLVM.context(ts_ctx)
            LLVM.activate(ctx)
            obj = try
                llvm_module, p = GPUCompiler.emit_llvm(job)
		gmap = Dict{String, UInt}()
		for g in LLVM.globals(llvm_module)
		    if haskey(LLVM.metadata(g), "julia.constgv") && !LLVM.isnull(LLVM.initializer(g))
		       addr = LLVM.initializer(g)
		       addr, _ = Enzyme.Compiler.get_base_and_offset(addr; offsetAllowed=false, inttoptr=true)
		       @assert isa(addr, LLVM.ConstantInt)
		       gmap[LLVM.name(g)] = convert(UInt, addr)
		       LLVM.linkage!(g, LLVM.API.LLVMExternalLinkage)
		       LLVM.initializer!(g, LLVM.null(LLVM.value_type(LLVM.initializer(g))))
		    end
		end
	
		llvm_fn_name = LLVM.name(p.entry)
		
		jlvaluet = convert(LLVM.LLVMType, Any; allow_boxed=true)
		ptrt = convert(LLVM.LLVMType, Core.LLVMPtr{Any, 0}; allow_boxed=true)
		wrapper_ft = LLVM.FunctionType(jlvaluet, LLVM.LLVMType[jlvaluet, ptrt, LLVM.Int32Type(), ptrt])
		wrapper_f = LLVM.Function(llvm_module, "entry", wrapper_ft)

		sfn = LLVM.subprogram(p.entry)
	    if sfn !== nothing
		   LLVM.set_subprogram!(wrapper_f, sfn)
	    end


		for f in LLVM.functions(llvm_module)
		    for b in LLVM.blocks(f)
			term = LLVM.terminator(b)
			if isa(term, LLVM.UnreachableInst)
			    shouldemit = true
			    tmp = term
			    while true
				tmp = LLVM.API.LLVMGetPreviousInstruction(tmp)
				if tmp == C_NULL
				    break
				end
				tmp = LLVM.Instruction(tmp)
				if isa(tmp, LLVM.CallInst)
				    cf = LLVM.called_operand(tmp)
				    if isa(cf, LLVM.Function)
					nm = LLVM.name(cf)
					if nm == "gpu_signal_exception" ||
					   nm == "gpu_report_exception" ||
					   nm == "ijl_throw" ||
					   nm == "jl_throw"
					    shouldemit = false
					    break
					end
				    end
				end
			    end

			    if shouldemit
				b = LLVM.IRBuilder()
				LLVM.position!(b, term)

                if LLVM.subprogram(f) !== nothing
                    LLVM.debuglocation!(b, LLVM.DILocation(0, 0, LLVM.subprogram(f)))
                end
				Enzyme.Compiler.emit_error(
				    b,
				    term,
				    "Reactant: The original primal code hits this error condition, thus differentiating it does not make sense",
				    ReactantRuntimeException
				)
			    end
			end
		    end
		    if false && !isempty(LLVM.blocks(f))
			 LLVM.name!(f, "reactant\$"*LLVM.name(f))
			if Enzyme.Compiler.has_fn_attr(f, LLVM.EnumAttribute("optnone"))
			    delete!(LLVM.function_attributes(f), LLVM.EnumAttribute("optnone"))
			end
			if Enzyme.Compiler.has_fn_attr(f, LLVM.EnumAttribute("noinline"))
			    delete!(LLVM.function_attributes(f), LLVM.EnumAttribute("noinline"))
			end
			LLVM.linkage!(f, LLVM.API.LLVMInternalLinkage)
		    end
		end
		

		builder = LLVM.IRBuilder()
		entry = LLVM.BasicBlock(wrapper_f, "entry")
        if LLVM.subprogram(wrapper_f) !== nothing
            LLVM.debuglocation!(builder, LLVM.DILocation(0, 0, LLVM.subprogram(wrapper_f)))
        end
		LLVM.position!(builder, entry)

		args = collect(LLVM.Value, LLVM.parameters(wrapper_f))
		args[2] = LLVM.bitcast!(builder, args[2], LLVM.PointerType(jlvaluet))
		args[4] = LLVM.bitcast!(builder, args[4], LLVM.PointerType(jlvaluet))
		
		globals = Any[]
		for g in LLVM.globals(llvm_module)
		    if LLVM.initializer(g) !== nothing
			LLVM.linkage!(g, LLVM.API.LLVMInternalLinkage)
		    end
		    if !haskey(LLVM.metadata(g), "julia.constgv")
			continue
		    end
		    if !haskey(gmap, LLVM.name(g))
			if Reactant.precompiling()
				throw(ReactantPrecompilationException(string(g)))
			end
			continue
		    end
		    gval = LLVM.load!(builder, jlvaluet, LLVM.gep!(builder, jlvaluet, args[4], LLVM.Value[LLVM.ConstantInt(length(globals))]))
		    push!(globals, unsafe_pointer_to_objref(Base.reinterpret(Ptr{Cvoid}, gmap[LLVM.name(g)])))
		    LLVM.store!(builder, gval, LLVM.bitcast!(builder, g, LLVM.PointerType(jlvaluet)))
		end
		res = LLVM.call!(builder, LLVM.function_type(p.entry), p.entry, args[1:3])
		LLVM.ret!(builder, res)
		push!(LLVM.function_attributes(wrapper_f), LLVM.EnumAttribute("alwaysinline"))

		LLVM.run!(LLVM.GlobalOptPass(), llvm_module)

        # Required for windows
        for f in LLVM.functions(llvm_module)
            if isempty(LLVM.blocks(f))
                continue
            end
            if !Enzyme.Compiler.has_fn_attr(f, LLVM.StringAttribute("frame-pointer"))
                push!(LLVM.function_attributes(f), LLVM.StringAttribute("frame-pointer", "all"))
            end
        end

		if DEBUG_INTERP[]
			Enzyme.API.EnzymeDumpModuleRef(llvm_module.ref)
		end
		mod = string(llvm_module)
		if VERSION < v"1.11" && occursin("inttoptr", mod) && Reactant.precompiling()
		   throw(ReactantPrecompilationException("Baked in global"))
		end
		mod, p.compiled[mi].ci.rettype, globals
            finally
                LLVM.deactivate(ctx)
                LLVM.dispose(ts_ctx)
            end
            call_with_reactant_cache[key] = obj
        end
        obj
    finally
        unlock(call_with_reactant_lock)
    end

    mod, rt, globals = cached_compilation::Tuple{String, Type, Vector{Any}}

    args_vec = push_inst!(
        overdubbed_code,
	Expr(:call, Vector{Any}, undef, length(fn_args)))

    for (i, v) in enumerate(fn_args)
    push_inst!(overdubbed_code, Expr(:call,
	       GlobalRef(Base, :setindex!), args_vec, v, i))
    end

    preserve = push_inst!(overdubbed_code, Expr(:gc_preserve_begin, args_vec))
    preserve2 = push_inst!(overdubbed_code, Expr(:gc_preserve_begin, globals))
    args_vec = push_inst!(overdubbed_code, Expr(:call, GlobalRef(Base, :pointer), args_vec))
    args_vec = push_inst!(overdubbed_code, Expr(:call, GlobalRef(Base, :reinterpret), Core.LLVMPtr{Any, 0}, args_vec))
    
    globals_vec = push_inst!(overdubbed_code, Expr(:call, GlobalRef(Base, :pointer), globals))
    globals_vec = push_inst!(overdubbed_code, Expr(:call, GlobalRef(Base, :reinterpret), Core.LLVMPtr{Any, 0}, globals_vec))
    n_args = length(fn_args)

    result = push_inst!(
        overdubbed_code,
        Expr(
            :call,
            GlobalRef(Base, :llvmcall),
            (mod, "entry"),
            Any,
	    Tuple{Any,Core.LLVMPtr{Any, 0},Int32, Core.LLVMPtr{Any, 0}},
            f_arg,
            args_vec,
	    Int32(n_args),
	    globals_vec
        ),
    )

    push_inst!(overdubbed_code, Expr(:gc_preserve_end, preserve))
    push_inst!(overdubbed_code, Expr(:gc_preserve_end, preserve2))
    if RT !== nothing
        if !(rt <: RT)
    		push_inst!(overdubbed_code, Expr(:call, Core.typeassert, result, RT))
        end
    end
    result = push_inst!(overdubbed_code, Expr(:call, Core.typeassert, result, rt))
        if DEBUG_INTERP[]
          push_inst!(
		     overdubbed_code, Expr(:call, safe_print2, "result", result))
        end
    push_inst!(overdubbed_code, Core.ReturnNode(result))

    code_info = Enzyme.create_fresh_codeinfo(
        call_with_reactant, source, world, slotnames, overdubbed_code
    )
	
    if code_info.method_for_inference_limit_heuristics === nothing
        code_info.method_for_inference_limit_heuristics = method
    end

    if VERSION >= v"1.12-"
        code_info.isva = true
    end

    code_info.min_world = min_world[]
    code_info.max_world = max_world[]

    edges = Any[mi]

    for gen_sig in (
        Tuple{typeof(should_rewrite_invoke),Type,Type},
        Tuple{typeof(should_rewrite_call),Type},
    )
        Enzyme.add_edge!(edges, gen_sig)
    end

    code_info.edges = edges
    code_info.rettype = rt
        if DEBUG_INTERP[]
		safe_print("code_info", code_info)
		safe_print("method nargs, isva", (method.nargs, method.isva))
        end
    return code_info
end

@eval function call_with_reactant($(REDUB_ARGUMENTS_NAME)::Vararg{Any,N}) where N
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, call_llvm_generator))
end

@static if isdefined(Core, :BFloat16)
    nmantissa(::Type{Core.BFloat16}) = 7
end
nmantissa(::Type{Float16}) = 10
nmantissa(::Type{Float32}) = 23
nmantissa(::Type{Float64}) = 52

_unwrap_val(::Val{T}) where {T} = T
