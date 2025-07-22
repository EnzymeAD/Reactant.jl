"""
    lattice_call(f::F, args...; kwargs...) where {F}

Calls a function `f` with the provided arguments.
"""
function lattice_call(f, args...; kwargs...)
    mod = MLIR.IR.mmodule()

    # TODO: arbitrary structures support
    # TODO: any constant created here must be an arith.constant if using make_mlir_fn

    lattice_traced_types = Vector{Any}(undef, length(args))
    linear_args = TracedRNumber[]
    in_tys = Vector{MLIR.IR.Type}(undef, 0)
    for (i, arg) in enumerate(args)
        if arg isa TracedRArray || arg isa TracedRNumber
            lattice_traced_types[i] = TracedRNumber{TTPtr{unwrapped_eltype(arg)}}(
                (), nothing
            )
            push!(linear_args, lattice_traced_types[i])
            push!(in_tys, MLIR.IR.Type(TTPtr{unwrapped_eltype(arg)}))
        else
            lattice_traced_types[i] = arg
        end
    end

    fnbody = MLIR.IR.Block(
        in_tys, MLIR.IR.Location[MLIR.IR.Location() for _ in 1:length(in_tys)]
    )

    func = MLIR.IR.block!(MLIR.IR.body(mod)) do
        return MLIR.Dialects.tt.func(;
            sym_name=string(f) * "_lattice_kernel",
            function_type=MLIR.IR.FunctionType(in_tys, MLIR.IR.Type[]),
            body=MLIR.IR.Region(),
        )
    end
    push!(MLIR.IR.region(func, 1), fnbody)
    MLIR.IR.activate!(fnbody)

    for (i, arg) in enumerate(linear_args)
        block_arg = MLIR.IR.argument(fnbody, i)
        # We require zero-ranked tensors, so we splat this
        block_arg = MLIR.IR.result(
            MLIR.Dialects.tt.splat(
                block_arg;
                result=MLIR.IR.TensorType(Int[], MLIR.IR.Type(unwrapped_eltype(arg))),
            ),
        )
        TracedUtils.set_mlir_data!(arg, block_arg)
    end

    res = try
        tmp_result = f(lattice_traced_types...; kwargs...)
        MLIR.Dialects.tt.return_(MLIR.IR.Value[])
        tmp_result
    catch err
        @error "error in lattice_call" exception = (err, catch_backtrace())
    finally
        MLIR.IR.deactivate!(fnbody)
    end
    @assert res === nothing "`f` to lattice_call must return `nothing`"

    # TODO: emit a custom_call here

    return nothing
end
