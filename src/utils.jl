
function transpose_ty(mlirty)
    return MLIR.IR.TensorType([reverse(size(mlirty))...,], eltype(mlirty))
end
function transpose_val(val)
    attr = MLIR.IR.DenseArrayAttribute(Int64[reverse(0:length(size(MLIR.IR.type(val)))-1)...])
    return MLIR.IR.result(MLIR.Dialects.stablehlo.transpose(val, permutation=attr), 1)
end

function apply(f, args...; kwargs)
    f(args...; kwargs...)
end

function make_mlir_fn(mod, f, args, kwargs, name="main", concretein=true)
    if sizeof(typeof(f)) != 0
        return (true, make_mlir_fn(mod, apply, (f, args...), kwargs, name, concretein)[2:end]...)
    end

    N = length(args)
    seen_args = IdDict()
    traced_args = ntuple(Val(N)) do i
        Base.@_inline_meta
        make_tracer(seen_args, args[i], ("args", i,), concretein ? ConcreteToTraced : TracedSetPath, #=data=#nothing)
    end

    linear_args = TracedRArray[]
    for (k, v) in seen_args
        if !(v isa TracedRArray)
            continue
        end
        push!(linear_args, v)
    end

    in_tys = [transpose_ty(mlir_type(arg)) for arg in linear_args]

    func = MLIR.Dialects.func.func_(; sym_name=name*"_tmp", function_type=MLIR.IR.FunctionType(in_tys, []), body=MLIR.IR.Region())

    fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in linear_args])
    push!(MLIR.IR.region(func, 1), fnbody)

    result = MLIR.IR.block!(fnbody) do
        for (i, arg) in enumerate(linear_args)
            raw_arg = MLIR.IR.argument(fnbody, i)
            row_maj_arg = transpose_val(raw_arg)
            arg.mlir_data = row_maj_arg
        end

        f(traced_args...; kwargs...)
    end

    seen_results = IdDict()

    traced_result = make_tracer(seen_results, result, ("result",), concretein ? TracedTrack : TracedSetPath, #=data=#nothing)

    retraced_args = ntuple(Val(N)) do i
        Base.@_inline_meta
        make_tracer(seen_results, traced_args[i], concretein ? ("resargs", i,) : (), TracedTrack, #=data=#nothing)
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
        MLIR.Dialects.func.return_(vals)
    end

    func2 = MLIR.Dialects.func.func_(; sym_name=name, function_type=MLIR.IR.FunctionType(in_tys, out_tys), body=MLIR.IR.Region())
    MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func2, 1), MLIR.IR.region(func, 1))

    modbody = MLIR.IR.body(mod)
    push!(modbody, func2)

    return false, func2, traced_result, result, seen_args, ret, linear_args, in_tys, linear_results
end
