function mlir_type(x::RArray{T,N}) where {T,N}
    return MLIR.IR.TensorType(size(x), MLIR.IR.Type(T))
end

function mlir_type(::Type{<:RArray{T,N}}, shape) where {T,N}
    @assert length(shape) == N
    return MLIR.IR.TensorType(shape, MLIR.IR.Type(T))
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

function make_mlir_fn(f, args, kwargs, name="main", concretein=true; toscalar=false)
    if sizeof(typeof(f)) != 0 || f isa BroadcastFunction
        return (true, make_mlir_fn(apply, (f, args...), kwargs, name, concretein)[2:end]...)
    end

    N = length(args)
    seen_args = IdDict()
    traced_args = ntuple(N) do i
        return make_tracer(
            seen_args,
            args[i],
            (:args, i),
            concretein ? ConcreteToTraced : TracedSetPath;
            toscalar,
        )
    end

    linear_args = TracedRArray[]
    for (k, v) in seen_args
        if !(v isa TracedRArray)
            continue
        end
        push!(linear_args, v)
    end

    in_tys = if toscalar
        [MLIR.IR.TensorType((), MLIR.IR.Type(eltype(arg))) for arg in linear_args]
    else
        [transpose_ty(mlir_type(arg)) for arg in linear_args]
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

    fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in linear_args])
    push!(MLIR.IR.region(func, 1), fnbody)

    @assert MLIR.IR._has_block()

    result = MLIR.IR.block!(fnbody) do
        for (i, arg) in enumerate(linear_args)
            raw_arg = MLIR.IR.argument(fnbody, i)
            row_maj_arg = transpose_val(raw_arg)
            arg.mlir_data = row_maj_arg
        end

        # NOTE an `AbstractInterpreter` cannot process methods with more recent world-ages than it
        # solution is to use a new interpreter, but we reuse the `code_cache` to minimize comptime in Julia <= 1.10
        @static if !HAS_INTEGRATED_CACHE
            interp = ReactantInterpreter(; code_cache=REACTANT_CACHE)
        else
            interp = ReactantInterpreter()
        end

        # TODO replace with `Base.invoke_within` if julia#52964 lands
        ir = first(only(
            # TODO fix it for kwargs
            Base.code_ircode(f, map(typeof, traced_args); interp),
        ))

        oc = Core.OpaqueClosure(ir)

        if f === Reactant.apply
            oc(traced_args[1], (traced_args[2:end]...,))
        else
            oc(traced_args...)
        end
    end

    seen_results = IdDict()

    traced_result = make_tracer(
        seen_results, result, (:result,), concretein ? TracedTrack : TracedSetPath
    )

    # marks buffers to be donated
    for i in 1:N
        make_tracer(
            seen_results, traced_args[i], concretein ? (:resargs, i) : (), TracedTrack
        )
    end

    linear_results = TracedRArray[]

    for (k, v) in seen_results
        if !(v isa TracedRArray)
            continue
        end

        push!(linear_results, v)
    end

    out_tys = [transpose_ty(mlir_type(arg)) for arg in linear_results]

    ret = MLIR.IR.block!(fnbody) do
        vals = MLIR.IR.Value[]
        for res in linear_results
            col_maj = transpose_val(res.mlir_data)
            push!(vals, col_maj)
        end
        @assert length(vals) == length(linear_results)
        return MLIR.Dialects.func.return_(vals)
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
    return false,
    func2, traced_result, result, seen_args, ret, linear_args, in_tys,
    linear_results
end
