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

    ccall(:jl_, Any, (Any,), "args=")
    ccall(:jl_, Any, (Any,), args)

    stub = Core.GeneratedFunctionStub(
        identity, Core.svec(:call_with_reactant, REDUB_ARGUMENTS_NAME), Core.svec()
    )

    # look up the method match
    builtin_error = :(throw(
        AssertionError("Unsupported call_with_reactant of builtin $redub_arguments")
    ))

    if args[1] <: Core.Builtin
        ccall(:jl_, Any, (Any,), "builtin-ret")
        return stub(world, source, builtin_error)
    end
    method_error = :(throw(MethodError(args[1], args[2:end], $world)))

    interp = ReactantInterpreter(; world)

    sig = Tuple{args...}
    lookup_result = Core.Compiler.findall(sig, Core.Compiler.method_table(interp))
    @static if VERSION < v"1.11-"
        lookup_result = lookup_result.matches
    end

    if lookup_result === nothing || lookup_result === missing
        return stub(world, source, method_error)
    end

    matches = lookup_result.matches

    # No method could be found (including in our method table), bail with an error
    if length(matches) != 1
        return stub(world, source, method_error)
    end

    match = matches[1]::Core.MethodMatch
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

    # Julia hits various internal errors trying to re-perform type inference
    # on type infered code (that we undo inference of), if there is no type unstable
    # code to be rewritten, just use the default methodinstance (still using our methodtable),
    # to improve compatibility as these bugs are fixed upstream.
    # Just kidding we can't do this, since otherwise the inferred code won't guarantee to run
    # within our interpreter, so we must use our generated IR here.
    # if !any_changed
    #     src = Core.Compiler.retrieve_code_info(mi, world)
    # end

    # prepare a new code info
    code_info = copy(src)
    static_params = match.sparams
    signature = sig
    is_invoke = args[1] === typeof(Core.invoke)

    # propagate edge metadata, this method is invalidated if the original function we are calling
    # is invalidated
    code_info.edges = Core.MethodInstance[mi]
    code_info.min_world = lookup_result.valid_worlds.min_world
    code_info.max_world = lookup_result.valid_worlds.max_world

    # Rewrite the arguments to this function, to prepend the two new arguments, the function :call_with_reactant,
    # and the REDUB_ARGUMENTS_NAME tuple of input arguments
    code_info.slotnames = Any[
        :call_with_reactant, REDUB_ARGUMENTS_NAME, code_info.slotnames...
    ]
    code_info.slotflags = UInt8[0x00, 0x00, code_info.slotflags...]
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
    n_actual_args = fieldcount(signature)
    n_method_args = Int(method.nargs)
    offset = 1
    fn_args = Any[]
    for i in 1:n_method_args
        if is_invoke && (i == 1 || i == 2)
            # With an invoke call, we have: 1 is invoke, 2 is f, 3 is Tuple{}, 4... is args.
            # In the first loop iteration, we should skip invoke and process f.
            # In the second loop iteration, we should skip the Tuple type and process args[1].
            offset += 1
        end
        slot = i + n_prepended_slots
        actual_argument = Expr(
            :call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset
        )
        push!(overdubbed_code, :($(Core.SlotNumber(slot)) = $actual_argument))
        push!(overdubbed_codelocs, code_info.codelocs[1])
        code_info.slotflags[slot] |= 0x02 # ensure this slotflag has the "assigned" bit set
        offset += 1

        #push!(overdubbed_code, actual_argument)
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
    end

    # If `method` is a varargs method, we have to restructure the original method call's
    # trailing arguments into a tuple and assign that tuple to the expected argument slot.
    if method.isva
        if !isempty(overdubbed_code)
            # remove the final slot reassignment leftover from the previous destructuring
            pop!(overdubbed_code)
            pop!(overdubbed_codelocs)
            pop!(fn_args)
        end
        trailing_arguments = Expr(:call, Core.GlobalRef(Core, :tuple))
        for i in n_method_args:n_actual_args
            push!(
                overdubbed_code,
                Expr(:call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset - 1),
            )
            push!(overdubbed_codelocs, code_info.codelocs[1])
            push!(trailing_arguments.args, Core.SSAValue(length(overdubbed_code)))
            offset += 1
        end
        push!(
            overdubbed_code,
            Expr(
                :(=), Core.SlotNumber(n_method_args + n_prepended_slots), trailing_arguments
            ),
        )
        push!(overdubbed_codelocs, code_info.codelocs[1])
        push!(fn_args, Core.SSAValue(length(overdubbed_code)))
    end

    rt = Base.Experimental.compute_ir_rettype(ir)
    
    
    oc = if Base.issingletontype(args[1])
        ccall(:jl_new_opaque_closure_from_code_info, Any, (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any, Cint),
        Tuple{sig.parameters[2:end]...}, rt, rt, @__MODULE__, src, 0, nothing, method.nargs-1, method.isva, args[1].instance, true)
    else
        push!(overdubbed_code,
        quote
            args[1]
        end
        )
        push!(overdubbed_codelocs, code_info.codelocs[1])
        farg = Core.SSAValue(length(overdubbed_code))
        push!(overdubbed_code,
        quote
            ccall(:jl_new_opaque_closure_from_code_info, Any, (Any, Any, Any, Any, Any, Cint, Any, Cint, Cint, Any, Cint),
            $(Tuple{sig.parameters[2:end]...}), $rt, $rt, $(@__MODULE__), $src, 0, nothing, $(method.nargs-1), $(method.isva), $farg, true)
        end
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
