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
    Core.LineInfoNode(__module__, method, __source__.file, Int32(__source__.line), Int32(0))
end


const REDUB_ARGUMENTS_NAME = gensym("redub_arguments")

function call_with_reactant_generator(world::UInt, source::LineNumberNode, self, @nospecialize(args))
    @nospecialize
    
    @show args

    stub = Core.GeneratedFunctionStub(identity, Core.svec(:call_with_reactant, REDUB_ARGUMENTS_NAME), Core.svec())

    # look up the method match
    builtin_error = :(throw(AssertionError("Unsupported call_with_reactant of builtin $args")))
    
    if args[1] <: Core.Builtin
        return stub(world, source, builtin_error)
    end
    
    method_error = :(throw(MethodError(args[1], args[2:end], $world)))

    interp = ReactantInterpreter(; world)
    
    sig = Tuple{args...}
    lookup_result = Core.Compiler.findall(sig, Core.Compiler.method_table(interp)).matches

    if lookup_result === nothing || lookup_result === missing
        return stub(world, source, method_error)
    end

    matches = lookup_result.matches

    if length(matches) != 1
        return stub(world, source, method_error)
    end

    match = matches[1]::Core.MethodMatch
    
    # look up the method and code instance
    mi = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
               (Any, Any, Any), match.method, match.spec_types, match.sparams)
 
    result = Core.Compiler.InferenceResult(mi, Core.Compiler.typeinf_lattice(interp))
    src = Core.Compiler.retrieve_code_info(mi, world)

    # prepare a new code info
    code_info = copy(src)
    method = match.method
    static_params = match.sparams
    signature = sig
    is_invoke = args[1] === typeof(Core.invoke)

    # propagate edge metadata
    code_info.edges = Core.MethodInstance[mi]
    code_info.min_world = lookup_result.valid_worlds.min_world
    code_info.max_world = lookup_result.valid_worlds.max_world

    code_info.slotnames = Any[:call_with_reactant, REDUB_ARGUMENTS_NAME, code_info.slotnames...]
    code_info.slotflags = UInt8[0x00, 0x00, code_info.slotflags...]
    #code_info.slotnames = Any[:call_with_reactant, REDUB_ARGUMENTS_NAME] #code_info.slotnames...]
    #code_info.slotflags = UInt8[0x00, 0x00] # code_info.slotflags...]
    n_prepended_slots = 2
    overdub_args_slot = Core.SlotNumber(n_prepended_slots)

    # For the sake of convenience, the rest of this pass will translate `code_info`'s fields
    # into these overdubbed equivalents instead of updating `code_info` in-place. Then, at
    # the end of the pass, we'll reset `code_info` fields accordingly.
    overdubbed_code = Any[]
    overdubbed_codelocs = Int32[]

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
        actual_argument = Expr(:call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset)
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
            push!(overdubbed_code, Expr(:call, Core.GlobalRef(Core, :getfield), overdub_args_slot, offset - 1))
            push!(overdubbed_codelocs, code_info.codelocs[1])
            push!(trailing_arguments.args, Core.SSAValue(length(overdubbed_code)))
            offset += 1
        end
        push!(overdubbed_code, Expr(:(=), Core.SlotNumber(n_method_args + n_prepended_slots), trailing_arguments))
        push!(overdubbed_codelocs, code_info.codelocs[1])   
	push!(fn_args, Core.SSAValue(length(overdubbed_code)))
    end

    #=== finish initialization of `overdubbed_code`/`overdubbed_codelocs` ===#

    # substitute static parameters, offset slot numbers by number of added slots, and
    # offset statement indices by the number of additional statements
    @show code_info.code

    @show n_prepended_slots
    Base.Meta.partially_inline!(code_info.code, fn_args, method.sig, Any[static_params...],
                                n_prepended_slots, length(overdubbed_code), :propagate)
    @show code_info.code

    #callexpr = Expr(:call, Core.OpaqueClosure(ir), fn_args...)
    #push!(overdubbed_code, callexpr)
    #push!(overdubbed_codelocs, code_info.codelocs[1])
    
    #push!(new_ci.code, Core.Compiler.ReturnNode(Core.SSAValue(length(overdubbed_code))))
    #push!(overdubbed_codelocs, code_info.codelocs[1])

    # original_code_start_index = length(overdubbed_code) + 1

    append!(overdubbed_code, code_info.code)
    append!(overdubbed_codelocs, code_info.codelocs)

    @show overdubbed_code

    for i in eachindex(overdubbed_code)
	prev = overdubbed_code[i]
	if Base.Meta.isexpr(prev, :call)
	   @show prev
	   @show prev.args[1]
	   @show prev.args[1] isa Core.IntrinsicFunction
	   if !(prev.args[1] isa Core.IntrinsicFunction)
		   overdubbed_code[i] = Expr(:call, GlobalRef(Reactant, :call_with_reactant), prev.args...)
		   @show "post", overdubbed_code[i]
	   end
	end
    end

    #=== set `code_info`/`reflection` fields accordingly ===#

    if code_info.method_for_inference_limit_heuristics === nothing
        code_info.method_for_inference_limit_heuristics = method
    end

    code_info.code = overdubbed_code
    code_info.codelocs = overdubbed_codelocs
    code_info.ssavaluetypes = length(overdubbed_code)
    code_info.ssaflags = [0x00 for _ in 1:length(overdubbed_code)] # XXX we need to copy flags that are set for the original code
    self_result = Core.Compiler.InferenceResult(self_mi, Core.Compiler.typeinf_lattice(interp))

    @show code_info

    @show self
    self_meths = Base._methods_by_ftype(Tuple{self, Vararg{Any}}, -1, world)
    @show self_meths
    self_method = (self_meths[1]::Core.MethodMatch).method
    self_mi = Core.Compiler.specialize_method(self_method, Tuple{typeof(Reactant.call_with_reactant), sig.parameters...}, Core.svec())
    @show self_mi
    self_result = Core.Compiler.InferenceResult(self_mi, Core.Compiler.typeinf_lattice(interp))
    frame = Core.Compiler.InferenceState(self_result, code_info, #=cache_mode=#:global, interp)
    @assert frame !== nothing
    Core.Compiler.typeinf(interp, frame)
    @assert Core.Compiler.is_inferred(frame)

    #if Core.Compiler.result_is_constabi(interp, frame.result)
    #    rt = frame.result.result::Core.Compiler.Const
    #    src = Core.Compiler.codeinfo_for_const(interp, frame.linfo, rt.val)
    #else
        opt = Core.Compiler.OptimizationState(frame, interp)

    ir = opt.src
    @show ir
    for (i, stmt) in enumerate(ir.stmts)
      @show stmt

    end

    @show ir

	caller = frame.result
	@static if VERSION < v"1.11-"
	  ir = Core.Compiler.run_passes(ir, opt, caller)
	else
	  ir = Core.Compiler.run_passes_ipo_safe(ir, opt, caller)
	  Core.Compiler.ipo_dataflow_analysis!(interp, opt, ir, caller)
	end
    	Core.Compiler.finish(interp, opt, ir, caller)

        src = Core.Compiler.ir_to_codeinf!(opt)
    #end

    src = copy(src)
    src.ssavaluetypes = length(src.code)

    @show src

    return src
end

@eval function call_with_reactant($REDUB_ARGUMENTS_NAME...)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, call_with_reactant_generator))
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

        interp = ReactantInterpreter()

        # TODO replace with `Base.invoke_within` if julia#52964 lands        
        # TODO fix it for kwargs	
        if f === Reactant.apply
            call_with_reactant(f, traced_args[1], (traced_args[2:end]...,))
        else
            call_with_reactant(f, traced_args...)
        end
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
