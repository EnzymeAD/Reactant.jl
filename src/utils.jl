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

function rewrite_inst(inst)
  @show inst
  if Meta.isexpr(inst, :call)
    rep = Expr(:call, call_with_reactant, inst.args...)
    @show rep
    return rep
  end
  return inst
end

function call_with_reactant_generator(world::UInt, source::LineNumberNode, @nospecialize(F::Type), @nospecialize(N::Int), self, @nospecialize(f::Type), @nospecialize(args))
    @nospecialize
    @show f, args

    stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :f, :args), Core.svec())

    # look up the method match
    method_error = :(throw(MethodError(f, args, $world)))

    interp = ReactantInterpreter(; world)
    
    mt = interp.method_table

    sig = Tuple{F, args...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    match = ccall(:jl_gf_invoke_lookup_worlds, Any,
                  (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
                  sig, mt, world, min_world, max_world)
    match === nothing && return stub(world, source, method_error)

    # look up the method and code instance
    mi = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
               (Any, Any, Any), match.method, match.spec_types, match.sparams)
  
    result = Core.Compiler.InferenceResult(mi, Core.Compiler.typeinf_lattice(interp))
    frame = Core.Compiler.InferenceState(result, #=cache_mode=#:global, interp)
    @assert frame !== nothing
    Core.Compiler.typeinf(interp, frame)
    @assert Core.Compiler.is_inferred(frame)

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
	  Core.Compiler.ipo_dataflow_analysis!(interp, opt, ir, caller)
	end
	@show ir
	  for (i, inst) in enumerate(ir.stmts)
	     @static if VERSION < v"1.11"
	        Core.Compiler.setindex!(ir.stmts[i], rewrite_inst(inst[:inst]), :inst)
	     else
	        Core.Compiler.setindex!(ir.stmts[i], rewrite_inst(inst[:stmt]), :stmt)
	     end
	  end
	@show ir
    	Core.Compiler.finish(interp, opt, ir, caller)
        src = Core.Compiler.ir_to_codeinf!(opt)
    #end

    new_ci = copy(src)
    new_ci.slotnames = Symbol[Symbol("#self#"), :f, :args]
    new_ci.edges = Core.MethodInstance[mi]
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]

    return new_ci
end

@eval function call_with_reactant(f::F, args::Vararg{Any, N}) where {F, N}
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
	oc = call_with_reactant # Core.OpaqueClosure(ir)

        if f === Reactant.apply
            oc(f, traced_args[1], (traced_args[2:end]...,))
        else
            oc(f, traced_args...)
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
