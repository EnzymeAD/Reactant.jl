using GPUCompiler
CC = Core.Compiler

#leak each argument to a global variable
macro lk(args...)
    quote
        $([:(
            let val = $(esc(p))
                global $(esc(p)) = val
            end
        ) for p in args]...)
    end
end

Base.Experimental.@MethodTable(REACTANT_METHOD_TABLE)

function var"@reactant_overlay"(__source__::LineNumberNode, __module__::Module, def)
    return Base.Experimental.var"@overlay"(
        __source__, __module__, :(Reactant.REACTANT_METHOD_TABLE), def
    )
end

function call_with_reactant() end

@noinline call_with_native(@nospecialize(f), @nospecialize(args...)) =
    Base.inferencebarrier(f)(args...)

const __skip_rewrite_func_set = Set([
    typeof(call_with_reactant),
    typeof(call_with_native),
    typeof(task_local_storage),
    typeof(getproperty),
    typeof(invokelatest),
    typeof(objectid)
])
const __skip_rewrite_func_set_lock = ReentrantLock()

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

const __skip_files = Set([Symbol("sysimg.jl"), Symbol("boot.jl")])

struct CompilerParams <: AbstractCompilerParams
    function CompilerParams()
        return new()
    end
end

@kwdef struct MetaData end

@kwdef struct DebugData
    enable_log::Bool = true
    enable_runtime_log::Bool = true
    rewrite_call::Set = Set()
    non_rewrite_call::Set = Set()
end

struct ReactantToken end

@kwdef struct ReactantInterpreter <: CC.AbstractInterpreter
    token::ReactantToken = ReactantToken()
    # Cache of inference results for this particular interpreter
    local_cache::Vector{CC.InferenceResult} = CC.InferenceResult[]
    # The world age we're working inside of
    world::UInt = Base.get_world_counter()

    # Parameters for inference and optimization
    inf_params::CC.InferenceParams = CC.InferenceParams()
    opt_params::CC.OptimizationParams = CC.OptimizationParams()

    meta_data::Ref{MetaData} = Ref(MetaData())
    debug_data::Ref{DebugData} = Ref(DebugData())
end

log(interp::ReactantInterpreter)::Bool = interp.debug_data[].enable_log
runtime_log(interp::ReactantInterpreter)::Bool = interp.debug_data[].enable_runtime_log
reset_debug_data(interp::ReactantInterpreter) = interp.debug_data[] = DebugData();

NativeCompilerJob = CompilerJob{NativeCompilerTarget,CompilerParams}
GPUCompiler.can_throw(@nospecialize(job::NativeCompilerJob)) = true
function GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob))
    return CC.method_table(GPUCompiler.get_interpreter(job))
end

current_interpreter = Ref{Union{Nothing,ReactantInterpreter}}(nothing)

function GPUCompiler.get_interpreter(@nospecialize(job::NativeCompilerJob))
    isnothing(current_interpreter[]) &&
        (return current_interpreter[] = ReactantInterpreter(; world=job.world))

    if job.world == current_interpreter[].world
        current_interpreter[]
    else
        (; meta_data, debug_data) = current_interpreter[]
        current_interpreter[] = ReactantInterpreter(;
            world=job.world, meta_data, debug_data
        )
    end
end

@noinline barrier(@nospecialize(x), @nospecialize(T::Type = Any)) =
    Core.Compiler.inferencebarrier(x)::T

CC.InferenceParams(@nospecialize(interp::ReactantInterpreter)) = interp.inf_params
CC.OptimizationParams(@nospecialize(interp::ReactantInterpreter)) = interp.opt_params
CC.get_inference_world(@nospecialize(interp::ReactantInterpreter)) = interp.world
CC.get_inference_cache(@nospecialize(interp::ReactantInterpreter)) = interp.local_cache
CC.cache_owner(@nospecialize(interp::ReactantInterpreter)) = interp.token
function CC.method_table(@nospecialize(interp::ReactantInterpreter))
    return CC.OverlayMethodTable(CC.get_inference_world(interp), REACTANT_METHOD_TABLE)
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
is_base_or_core(t::TypeVar) = begin
    println("TypeVar ", t)
    return false
end
is_base_or_core(t::Core.TypeofVararg) = is_base_or_core(t.T)
is_base_or_core(m::Module) = has_ancestor(m, Core) || has_ancestor(m, Base)
is_base_or_core(@nospecialize(u::Union)) = begin
    u == Union{} && return true
    is_base_or_core(u.a) && is_base_or_core(u.b)
end
is_base_or_core(u::UnionAll) = is_base_or_core(Base.unwrap_unionall(u))
is_base_or_core(@nospecialize(ty::Type)) = is_base_or_core(parentmodule(ty))

function skip_rewrite(mi::Core.MethodInstance)::Bool
    mod = mi.def.module
    mi.def.file in __skip_files && return true
    @lk mi
    ft = Base.unwrap_unionall(mi.specTypes).parameters[1]
    ft in __skip_rewrite_func_set && return true

    (
        has_ancestor(mod, Reactant.Ops) ||
        has_ancestor(mod, Reactant.TracedUtils) ||
        has_ancestor(mod, Reactant.MLIR)
    ) && return true

    if is_base_or_core(mod)
        modules = is_base_or_core.(Base.unwrap_unionall(mi.specTypes).parameters[2:end])
        all(modules) && return true
    end
    return false
end

disable_call_with_reactant = false
vv = []
vb = []
@inline function typeinf_local(interp::CC.AbstractInterpreter, frame::CC.InferenceState)
    @invoke CC.typeinf_local(interp::CC.AbstractInterpreter, frame)
end

function CC.typeinf_local(interp::ReactantInterpreter, frame::CC.InferenceState)
    mi = frame.linfo
    global disable_call_with_reactant
    disable_cwr = disable_call_with_reactant ? false : skip_rewrite(mi)
    disable_cwr && (disable_call_with_reactant = true)
    disable_call_with_reactant || push!(vb, (mi, CC.copy(frame.src)))
    tl = typeinf_local(interp, frame)
    disable_call_with_reactant || push!(vv, (mi, CC.copy(frame.src)))
    disable_cwr && (disable_call_with_reactant = false)
    return tl
end

lead_to_dynamic_call(@nospecialize(ty)) = begin
    isconcretetype(ty) && return false
    ty == Union{} && return false
    Base.isvarargtype(ty) && return true
    (ty <: Type || ty <: Tuple) && return false
    return true
end

# Rewrite type unstable calls to recurse into call_with_reactant to ensure
# they continue to use our interpreter.
function need_rewrite_call(interp, @nospecialize(fn), @nospecialize(args))
    #UnionAll constructor cannot get a singleton type, and are not handled by the call_with_reactant macro: degradate type inference
    isnothing(fn) && return false
    #ignore constructor
    fn isa Type && return false

    ft = typeof(fn)
    (ft <: Core.IntrinsicFunction || ft <: Core.Builtin) && return false
    ft in __skip_rewrite_func_set && return false
    #Base.isstructtype(ft) && return false
    if hasfield(typeof(ft), :name) && hasfield(typeof(ft.name), :module)
        mod = ft.name.module
        # Don't rewrite primitive ops, tracing utilities, or any MLIR-based functions
        if has_ancestor(mod, Reactant.Ops) ||
            has_ancestor(mod, Reactant.TracedUtils) ||
            has_ancestor(mod, Reactant.MLIR) ||
            has_ancestor(mod, Core.Compiler)
            return false
        end
    end
    #ft isa Type && any(t -> ft <: t, __skip_rewrite_type_constructor_list) && return false
    #ft in __skip_rewrite_func_set && return false

    #ft<: typeof(Core.kwcall) && return true
    tt = Tuple{ft,args...}
    match = CC._findsup(tt, REACTANT_METHOD_TABLE, CC.get_inference_world(interp))[1]
    !isnothing(match) && return true
    match = CC._findsup(tt, nothing, CC.get_inference_world(interp))[1]
    isnothing(match) && return true
    startswith(
        string(match.method.name), "#(overlay (. Reactant (inert REACTANT_METHOD_TABLE))"
    ) && return false

    # Avoid recursively interpreting into methods we define explicitly
    # as overloads, which we assume should handle the entirety of the
    # translation (and if not they can use call_in_reactant).
    isdefined(match.method, :external_mt) &&
        match.method.external_mt === REACTANT_METHOD_TABLE &&
        return false

    match.method.file in __skip_files && return false

    #Dynamic dispatch handler
    types = if match.method.nospecialize != 0
        match.method.sig
    else
        mi = CC.specialize_method(match)
        mi.specTypes
    end

    mask = lead_to_dynamic_call.(Base.unwrap_unionall(types).parameters)
    #@error string(ft) mask types
    return any(mask)
end

function CC.abstract_eval_call(
    interp::ReactantInterpreter,
    e::Expr,
    vtypes::Union{CC.VarTable,Nothing},
    sv::CC.AbsIntState,
)
    if !(sv isa CC.IRInterpretationState) #during type inference, rewrite dynamic call with call_with_reactant
        global disable_call_with_reactant
        if !disable_call_with_reactant
            argtypes = CC.collect_argtypes(interp, e.args, vtypes, sv)
            args = CC.argtypes_to_type(argtypes).parameters
            fn = CC.singleton_type(argtypes[1])
            if need_rewrite_call(interp, fn, args[2:end])
                @error fn string(argtypes) sv.linfo
                log(interp) && push!(
                    interp.debug_data[].rewrite_call,
                    (fn, args[2:end], sv.linfo), #CC.copy(sv.src)
                )
                e = Expr(:call, GlobalRef(@__MODULE__, :call_with_reactant), e.args...)
                expr = sv.src.code[sv.currpc]
                sv.src.code[sv.currpc] = if expr.head == :call
                    e
                else
                    @assert expr.head == :(=) #CodeInfo slot write
                    Expr(:(=), expr.args[1], e)
                end
            end
        else
            log(interp) && push!(
                interp.debug_data[].non_rewrite_call,
                (sv.linfo, CC.collect_argtypes(interp, e.args, vtypes, sv)),
            )
        end
    end

    return @invoke CC.abstract_eval_call(
        interp::CC.AbstractInterpreter,
        e::Expr,
        vtypes::Union{CC.VarTable,Nothing},
        sv::CC.AbsIntState,
    )
end

using LLVM, LLVM.Interop

struct CompilerInstance
    lljit::LLVM.JuliaOJIT
    lctm::LLVM.LazyCallThroughManager
    ism::LLVM.IndirectStubsManager
end
const jit = Ref{CompilerInstance}()

function get_trampoline(job)
    (; lljit, lctm, ism) = jit[]
    jd = JITDylib(lljit)

    target_sym = String(gensym(string(job.source)))

    # symbol flags (callable + exported)
    flags = LLVM.API.LLVMJITSymbolFlags(
        LLVM.API.LLVMJITSymbolGenericFlagsCallable |
        LLVM.API.LLVMJITSymbolGenericFlagsExported,
        0,
    )

    sym = Ref(LLVM.API.LLVMOrcCSymbolFlagsMapPair(mangle(lljit, target_sym), flags))

    # materialize callback: compile/emit module when symbols requested
    function materialize(mr)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job; validate=false)
            runtime_log(GPUCompiler.get_interpreter(job)) && @warn "materialize" job
            @lk ir
            # Ensure the module's entry has the target name we declared
            LLVM.name!(meta.entry, target_sym)
            r_symbols = string.(LLVM.get_requested_symbols(mr))
            #expose only the function defined in job
            for f in LLVM.functions(ir)
                isempty(LLVM.blocks(f)) && continue #declare functions
                LLVM.name(f) in r_symbols && continue
                LLVM.linkage!(f, LLVM.API.LLVMPrivateLinkage)
            end

            #convert global alias to private linkage in order to not be relocatable
            for g in LLVM.globals(ir)
                ua = LLVM.API.LLVMGetUnnamedAddress(g)
                (ua == LLVM.API.LLVMLocalUnnamedAddr || ua == LLVM.API.LLVMNoUnnamedAddr) ||
                    continue
                LLVM.isconstant(g) && continue
                LLVM.API.LLVMSetUnnamedAddress(g, LLVM.API.LLVMNoUnnamedAddr)
                LLVM.linkage!(g, LLVM.API.LLVMPrivateLinkage)
            end
            # serialize the module IR into a memory buffer
            buf = convert(MemoryBuffer, ir)
            # deserialize under a thread-safe context and emit via IRCompileLayer
            ThreadSafeContext() do ts_ctx
                tsm = context!(context(ts_ctx)) do
                    mod = parse(LLVM.Module, buf)
                    ThreadSafeModule(mod)
                end

                il = LLVM.IRCompileLayer(lljit)
                # Emit the ThreadSafeModule for the responsibility mr.
                LLVM.emit(il, mr, tsm)
            end
        end
        return nothing
    end

    # discard callback (no-op for now)
    function discard(jd_arg, sym)
        @error "discard" sym
    end

    # Create a single CustomMaterializationUnit that declares both entry and target.
    # Name it something descriptive (e.g., the entry_sym)
    mu = LLVM.CustomMaterializationUnit("MU_" * target_sym, sym, materialize, discard)

    # Define the MU in the JITDylib (declares the symbols as owned by this MU)
    LLVM.define(jd, mu)

    # Lookup the entry address (this will trigger materialize if needed)
    addr = lookup(lljit, target_sym)
    return addr
end
import GPUCompiler: deferred_codegen_jobs

function ccall_deferred(ptr::Ptr{Cvoid})
    return ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), ptr)
end

"""
    Reactant.REDUB_ARGUMENTS_NAME

The variable name bound to `call_with_reactant`'s tuple of arguments in its
`@generated` method definition.

This binding can be used to manually reference/destructure `call_with_reactants` arguments

This is required because user arguments could have a name which clashes with whatever name we choose for
our argument. Thus we gensym to create it.

This originates from 
    https://github.com/JuliaLabs/Cassette.jl/blob/c29b237c1ec0deda3a1037ec519eebe216952bfe/src/overdub.jl#L154
    https://github.com/JuliaGPU/GPUCompiler.jl/blob/master/examples/jit.jl
"""
const REDUB_ARGUMENTS_NAME = gensym("redub_arguments")

function deferred_call_with_reactant(
    world::UInt, source::LineNumberNode, self, @nospecialize(args)
)
    f = args[1]
    tt = Tuple{f,args[2:end]...}
    match = CC._findsup(tt, REACTANT_METHOD_TABLE, world)
    match = isnothing(match[1]) ? CC._findsup(tt, nothing, world) : match

    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(:call_with_reactant, REDUB_ARGUMENTS_NAME), Core.svec()
    )

    if isnothing(match[1])
        method_error = :(throw(
            MethodError($REDUB_ARGUMENTS_NAME[1], $REDUB_ARGUMENTS_NAME[2:end], $world)
        ))
        return stub(world, source, method_error)
    end

    mi = CC.specialize_method(match[1])

    target = NativeCompilerTarget(; jlruntime=true, llvm_always_inline=false)
    config = CompilerConfig(
        target,
        CompilerParams();
        kernel=false,
        libraries=false,
        toplevel=true,
        validate=false,
        strip=false,
        optimize=true,
        entry_abi=:func,
    )
    job = CompilerJob(mi, config, world)
    interp = GPUCompiler.get_interpreter(job)

    ci = CC.typeinf_ext(interp, mi)
    @assert !isnothing(ci)
    rt = ci.rettype
    @lk ci job
    runtime_log(interp) && @warn "ci rt" job ci rt

    addr = get_trampoline(job)
    trampoline = pointer(addr)
    id = Base.reinterpret(Int, trampoline)

    deferred_codegen_jobs[id] = job

    #build CodeInfo directly 
    code_info = begin
        ir = CC.IRCode()
        src = @ccall jl_new_code_info_uninit()::Ref{CC.CodeInfo}
        src.slotnames = fill(:none, length(ir.argtypes) + 1)
        src.slotflags = fill(zero(UInt8), length(ir.argtypes))
        src.slottypes = copy(ir.argtypes)
        src.rettype = UInt64
        CC.ir_to_codeinf!(src, ir)
    end

    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]
    function push_inst!(inst)
        push!(overdubbed_code, inst)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        return Core.SSAValue(length(overdubbed_code))
    end
    code_info.edges = Core.MethodInstance[job.source]
    code_info.rettype = rt

    ptr = push_inst!(Expr(:call, :ccall_deferred, trampoline))

    fn_args = []
    for i in 2:length(args)
        named_tuple_ssa = Expr(
            :call, Core.GlobalRef(Core, :getfield), Core.SlotNumber(2), i
        )
        arg = push_inst!(named_tuple_ssa)
        push!(fn_args, arg)
    end

    f_arg = push_inst!(Expr(:call, Core.GlobalRef(Core, :getfield), Core.SlotNumber(2), 1))

    args_vec = push_inst!(
        Expr(:call, GlobalRef(Base, :getindex), GlobalRef(Base, :Any), fn_args...)
    )

    runtime_log(interp) && push_inst!(
        Expr(
            :call,
            GlobalRef(Base, :println),
            "before call_with_reactant ",
            f_arg,
            "(",
            args_vec,
            ")",
        ),
    )
    preserve = push_inst!(Expr(:gc_preserve_begin, args_vec))
    args_vec = push_inst!(Expr(:call, GlobalRef(Base, :pointer), args_vec))
    n_args = length(fn_args)

    #Use ccall internal directly to call the wrapped llvm function
    result = push_inst!(
        Expr(
            :foreigncall,
            ptr,
            Ptr{rt},
            Core.svec(Any, Ptr{Any}, Int),
            0,
            QuoteNode(:ccall),
            f_arg,
            args_vec,
            n_args,
            n_args,
            args_vec,
            f_arg,
        ),
    )

    result = push_inst!(Expr(:call, GlobalRef(Base, :unsafe_pointer_to_objref), result))
    push_inst!(Expr(:gc_preserve_end, preserve))
    result = push_inst!(Expr(:call, GlobalRef(@__MODULE__, :barrier), result, rt))
    runtime_log(interp) && push_inst!(
        Expr(
            :call,
            GlobalRef(Base, :println),
            "after call_with_reactant ",
            f_arg,
            " ",
            result,
        ),
    )
    push_inst!(Core.ReturnNode(result))

    code_info.min_world = typemin(UInt)
    code_info.max_world = typemax(UInt)
    code_info.slotnames = Any[:call_with_reactant_, REDUB_ARGUMENTS_NAME]
    code_info.slotflags = UInt8[0x00, 0x00]
    code_info.code = overdubbed_code
    code_info.codelocs = overdubbed_codelocs
    code_info.ssavaluetypes = length(overdubbed_code)
    code_info.ssaflags = [0x00 for _ in 1:length(overdubbed_code)]
    return code_info
end

@eval function call_with_reactant($(REDUB_ARGUMENTS_NAME)...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, deferred_call_with_reactant))
end
const jd_main = Ref{Any}()
function init_jit()
    lljit = JuliaOJIT()
    jd_main[] = JITDylib(lljit)
    prefix = LLVM.get_prefix(lljit)

    dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
    add!(jd_main[], dg)

    es = ExecutionSession(lljit)

    lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
    ism = LLVM.LocalIndirectStubsManager(triple(lljit))

    jit[] = CompilerInstance(lljit, lctm, ism)
    atexit() do
        (; lljit, lctm, ism) = jit[]
        dispose(ism)
        dispose(lctm)
        dispose(lljit)
    end
end

function ir_to_codeinfo!(ir::CC.IRCode)::CC.CodeInfo
    code_info = begin
        src = ccall(:jl_new_code_info_uninit, Ref{CC.CodeInfo}, ())
        src.slotnames = fill(:none, length(ir.argtypes) + 1)
        src.slotflags = fill(zero(UInt8), length(ir.argtypes))
        src.slottypes = copy(ir.argtypes)
        src.rettype = Int
        CC.ir_to_codeinf!(src, ir)
        src.ssavaluetypes = length(src.ssavaluetypes)
        src
    end
    return code_info
end

struct FakeOc
    f::Vector
    ci::Vector{CC.CodeInfo}
end

fake_oc_dict = FakeOc([], [])

fake_oc(ir::CC.IRCode, return_type=Any) = begin
    src = ir_to_codeinfo!(ir)
    fake_oc(src, return_type)
end

function fake_oc(src::CC.CodeInfo, return_type=Any; args=nothing)
    @assert !isnothing(current_interpreter[])
    types = isnothing(args) ? src.slottypes[2:end] : args
    global fake_oc_dict
    index = findfirst(==(src), fake_oc_dict.ci)
    !isnothing(index) && return fake_oc_dict.f[index]

    expr = (Expr(:(::), Symbol("arg_$i"), type) for (i, type) in enumerate(types))
    args = Expr(:tuple, (Symbol("arg_$i") for (i, type) in enumerate(types))...)
    fn_name = gensym(:fake_oc)
    call_expr = Expr(:call, fn_name, expr...)
    f_expr = Expr(
        :(=),
        call_expr,
        quote
            Reactant.barrier($args, $return_type)
        end,
    )
    f = @eval @noinline $f_expr
    mi = Base.method_instance(f, types)
    @assert !isnothing(mi)
    mi.def.source = CC.maybe_compress_codeinfo(current_interpreter[], mi, src)
    push!(fake_oc_dict.f, f)
    push!(fake_oc_dict.ci, src)
    return f
end
