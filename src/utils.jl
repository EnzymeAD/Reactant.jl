function mlir_type(x::RArray{T,N}) where {T,N}
    return MLIR.IR.TensorType(size(x), MLIR.IR.Type(T))
end

mlir_type(::RNumber{T}) where {T} = MLIR.IR.TensorType((), MLIR.IR.Type(T))

mlir_type(::MissingTracedValue) = MLIR.IR.TensorType((), MLIR.IR.Type(Bool))

function mlir_type(::Type{<:RArray{T,N}}, shape) where {T,N}
    @assert length(shape) == N
    return MLIR.IR.TensorType(shape, MLIR.IR.Type(T))
end

function mlir_type(::Type{<:RNumber{T}}) where {T}
    return MLIR.IR.TensorType((), MLIR.IR.Type(T))
end

function mlir_type(::Type{<:MissingTracedValue})
    return MLIR.IR.TensorType((), MLIR.IR.Type(Bool))
end

function batch_ty(width, mlirty)
    return MLIR.IR.TensorType([width, size(mlirty)...], eltype(mlirty))
end

function transpose_ty(mlirty)
    return MLIR.IR.TensorType([reverse(size(mlirty))...], eltype(mlirty))
end
function transpose_val(val)
    attr = MLIR.IR.DenseArrayAttribute(
        Int64[reverse(0:(length(size(MLIR.IR.type(val))) - 1))...]
    )
    return MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(val; permutation=attr), 1)
end

function apply(f, args...; kwargs...)
    return f(args...; kwargs...)
end

function call_with_reactant end

# generate a LineInfoNode for the current source code location
macro LineInfoNode(method)
    return Core.LineInfoNode(
        __module__, method, __source__.file, Int32(__source__.line), Int32(0)
    )
end

function maybe_argextype(@nospecialize(x), src)
    return try
        Core.Compiler.argextype(x, src)
    catch err
        !(err isa Core.Compiler.InvalidIRError) && rethrow()
        nothing
    end
end

function rewrite_inst(inst, ir)
    if Meta.isexpr(inst, :call)
        # Even if type unstable we do not want (or need) to replace intrinsic
        # calls or builtins with our version.
        ft = Core.Compiler.widenconst(maybe_argextype(inst.args[1], ir))
        if !(ft <: Core.IntrinsicFunction) && !(ft <: Core.Builtin)
            rep = Expr(:call, call_with_reactant, inst.args...)
            return true, rep
        end
    end
    if Meta.isexpr(inst, :invoke)
    #    return false, Expr(:call, inst.args[2:end]...)
    end
    return false, inst
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
    ccall(:jl_, Any, (Any,), "pre mt "*string(world)*" mnw="*string(min_world)*" mxw"*string(max_world))
    res = ccall(:jl_gf_invoke_lookup_worlds, Any,
                  (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
                  sig, mt, world, min_world, max_world)
    ccall(:jl_, Any, (Any,), "post mt "*string(world)*" mnw="*string(min_world)* " mxw"*string(max_world))
    return res
end

@inline function lookup_world(@nospecialize(sig::Type), world::UInt, mt::Core.Compiler.InternalMethodTable, min_world::Ref{UInt}, max_world::Ref{UInt})
    @show "pre imt", world, min_world, max_world
    res = lookup_world(sig, mt.world, nothing, min_world, max_world)
    @show "imt", res, world, min_world, max_world
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


# HACK: in all versions of Julia, `jl_new_opaque_closure_from_code_info` doesn't take a world argument
#       but instead always generates code for the current world. note that this doesn't
#       actually change the world age, but just spoofs the counter `jl_create_native` reads.
# XXX: Base.get_world_counter is supposed to be monotonically increasing and is runtime global.
macro in_world(world, ex)
    quote
        actual_world = Base.get_world_counter()
        world_counter = cglobal(:jl_world_counter, Csize_t)
        unsafe_store!(world_counter, $(esc(world)))
        try
            $(esc(ex))
        finally
            unsafe_store!(world_counter, actual_world)
        end
    end
end

#define jl_current_task (container_of(jl_get_pgcstack(), jl_task_t, gcstack))


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

    ccall(:jl_, Any, (Any,), string(world)*" args="*string(args))

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
    
    ccall(:jl_, Any, (Any,), string(lookup_result)*" sig="*string(sig)*" mw="*string(min_world)*" "*string(max_world)*" "*string(Base.get_world_counter()))

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

        @show src
        @show src.edges
        @show typeof(src)

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

    ccall(:jl_, Any, (Any,), ("method=")*string(method))
    ccall(:jl_, Any, (Any,), ("va=")*string(method.isva))

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

    ccall(:jl_, Any, (Any,), ("ir=")*string(ir))

    # Rewrite type unstable calls to recurse into call_with_reactant to ensure
    # they continue to use our interpreter. Reset the derived return type
    # to Any if our interpreter would change the return type of any result.
    # Also rewrite invoke (type stable call) to be :call, since otherwise apparently
    # screws up type inference after this (TODO this should be fixed).
    any_changed = false
    for (i, inst) in enumerate(ir.stmts)
        @static if VERSION < v"1.11"
            changed, next = rewrite_inst(inst[:inst], ir)
            Core.Compiler.setindex!(ir.stmts[i], next, :inst)
        else
            changed, next = rewrite_inst(inst[:stmt], ir)
            Core.Compiler.setindex!(ir.stmts[i], next, :stmt)
        end
        if changed
            any_changed = true
            Core.Compiler.setindex!(ir.stmts[i], Any, :type)
        end
    end
    Core.Compiler.finish(interp, opt, ir, caller)

    src = Core.Compiler.ir_to_codeinf!(opt)

    ccall(:jl_, Any, (Any,), ("src=")*string(src))

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
    for i in 1:length(redub_arguments)
        actual_argument = Expr(
            :call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset
        )
        push!(overdubbed_code, actual_argument)
        push!(overdubbed_codelocs, code_info.codelocs[1])
        offset += 1
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
    end

    rt = Base.Experimental.compute_ir_rettype(ir)
    
    # jl_new_opaque_closure forcibly executes in the current world... This means that we won't get the right
    # inner code during compilation without special handling (i.e. call_in_world_total).
    # Opaque closures also require takign the function argument. We can work around the latter
    # if the function is stateless. But regardless, to work around this we sadly create/compile the opaque closure
    oc = if Base.issingletontype(args[1])
        Core._call_in_world_total(world, make_oc, Tuple{sig.parameters[2:end]...}, rt, src, method.nargs - 1, method.isva, args[1].instance)::Core.OpaqueClosure
    else
        farg = fn_args[1]
        push!(overdubbed_code,
            Expr(:call,
                make_oc,
                Tuple{sig.parameters[2:end]...},
                rt,
                src,
                method.nargs-1,
                method.isva,
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

    ccall(:jl_, Any, (Any,), "code_info="*string(code_info))
    return code_info
end

@eval function call_with_reactant($REDUB_ARGUMENTS_NAME...)
    $(Expr(:meta, :generated_only))
    return $(Expr(:meta, :generated, call_with_reactant_generator))
end

function make_mlir_fn(
    f,
    args,
    kwargs,
    name="main",
    concretein=true;
    toscalar=false,
    return_dialect=:func,
    no_args_in_result::Bool=false,
    construct_function_without_args::Bool=false,
    do_transpose=true,
)
    if sizeof(typeof(f)) != 0 || f isa BroadcastFunction
        return (
            true,
            make_mlir_fn(
                apply,
                (f, args...),
                kwargs,
                name,
                concretein;
                toscalar,
                return_dialect,
                no_args_in_result,
                construct_function_without_args,
                do_transpose,
            )[2:end]...,
        )
    end

    N = length(args)
    seen_args = OrderedIdDict()
    traced_args = ntuple(N) do i
        return make_tracer(
            seen_args,
            args[i],
            (:args, i),
            concretein ? ConcreteToTraced : TracedSetPath;
            toscalar,
            track_numbers=construct_function_without_args ? (Number,) : (),
        )
    end

    linear_args = TracedType[]
    for (k, v) in seen_args
        v isa TracedType || continue
        push!(linear_args, v)
    end

    in_tys = if toscalar
        [MLIR.IR.TensorType((), MLIR.IR.Type(eltype(arg))) for arg in linear_args]
    elseif do_transpose
        [transpose_ty(mlir_type(arg)) for arg in linear_args]
    else
        [mlir_type(arg) for arg in linear_args]
    end

    sym_visibility = nothing
    if !concretein
        sym_visibility = MLIR.IR.Attribute("private")
    end

    mod = MLIR.IR.mmodule()
    func = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=name * "_tmp",
            function_type=MLIR.IR.FunctionType(in_tys, []),
            body=MLIR.IR.Region(),
        )
    end

    if construct_function_without_args
        fnbody = MLIR.IR.Block()
    else
        fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in linear_args])
    end
    push!(MLIR.IR.region(func, 1), fnbody)

    @assert MLIR.IR._has_block()

    result = MLIR.IR.block!(fnbody) do
        for (i, arg) in enumerate(linear_args)
            if construct_function_without_args
                arg.mlir_data = args[i].mlir_data
            else
                raw_arg = MLIR.IR.argument(fnbody, i)
                row_maj_arg = do_transpose ? transpose_val(raw_arg) : raw_arg
                arg.mlir_data = row_maj_arg
            end
        end

        # TODO fix it for kwargs	
        call_with_reactant(f, traced_args...)
    end

    seen_results = OrderedIdDict()

    traced_result = make_tracer(
        seen_results,
        result,
        (:result,),
        concretein ? TracedTrack : TracedSetPath;
        track_numbers=construct_function_without_args ? (Number,) : (),
    )

    # marks buffers to be donated
    for i in 1:N
        make_tracer(
            seen_results, traced_args[i], concretein ? (:resargs, i) : (), TracedTrack
        )
    end

    linear_results = TracedType[]

    for (k, v) in seen_results
        v isa TracedType || continue
        (no_args_in_result && length(v.paths) > 0 && v.paths[1][1] == :args) && continue
        push!(linear_results, v)
    end

    out_tys = [transpose_ty(mlir_type(arg)) for arg in linear_results]

    ret = MLIR.IR.block!(fnbody) do
        vals = MLIR.IR.Value[]
        for res in linear_results
            col_maj = if res isa MissingTracedValue
                broadcast_to_size(false, ()).mlir_data
            elseif construct_function_without_args || !do_transpose
                res.mlir_data
            elseif do_transpose
                transpose_val(res.mlir_data)
            end
            push!(vals, col_maj)
        end
        !no_args_in_result && @assert length(vals) == length(linear_results)

        dialect = getfield(MLIR.Dialects, return_dialect)
        return dialect.return_(vals)
    end

    name2 = name

    tab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
    for i in 0:10000
        name2 = if i == 0
            name
        else
            name * string(i)
        end
        if MLIR.IR.mlirIsNull(MLIR.API.mlirSymbolTableLookup(tab, name2))
            break
        end
    end

    func2 = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.func.func_(;
            sym_name=name2,
            function_type=MLIR.IR.FunctionType(in_tys, out_tys),
            body=MLIR.IR.Region(),
            sym_visibility,
        )
    end
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func2, 1), MLIR.IR.region(func, 1))

    MLIR.API.mlirOperationDestroy(func.operation)
    func.operation = MLIR.API.MlirOperation(C_NULL)
    return (
        false,
        func2,
        traced_result,
        result,
        seen_args,
        ret,
        linear_args,
        in_tys,
        linear_results,
    )
end

const DEBUG_MODE::Ref{Bool} = Ref(false)

function with_debug(f)
    old = DEBUG_MODE[]
    DEBUG_MODE[] = true
    try
        return f()
    finally
        DEBUG_MODE[] = old
    end
end

function mlir_stacktrace(name, file, line)::MLIR.IR.Location
    # calling `stacktrace` can add a lot of time overhead, so let's avoid adding debug info if not used
    if DEBUG_MODE[]
        return MLIR.IR.Location(name, MLIR.IR.Location(file, line, 0))
    end

    # retrieve current stacktrace, remove this function's frame and translate to MLIR Location
    st = stacktrace()
    deleteat!(st, 1)
    return mapfoldl(MLIR.IR.Location, st) do stackframe
        name = string(stackframe.func)
        file = stackframe.file
        line = stackframe.line
        return MLIR.IR.Location(name, MLIR.IR.Location(file, line, 0))
    end
end
