module ReactantcuTileExt

using Reactant
using Reactant: @reactant_overlay, make_tracer
using Reactant.MLIR: IR
using Reactant.Ops: mlir_stacktrace, mlir_type
import cuTile as ct

@reactant_overlay function ct.launch(@nospecialize(f), grid, args...; kwargs...)
    # generate MLIR with cuTile.jl
    argtypes = Tuple(typeof.(args))
    job = ct.tile_job(f, argtypes; kwargs...)
    io = IOBuffer()
    ct.code_tiled(io, job; debuginfo=false, remarks=false)
    code = takestring!(io)

    # parse `code` into Reactant's current MLIR module
    mod = IR.current_module()
    ct_mod = parse(IR.Operation, code; block=IR.body(mod))

    # linearize kernel arguments
    mlir_args = IR.Value[]
    restys = IR.Type[]
    aliases = IR.Attribute[]
    seen = Reactant.OrderedIdDict()
    kernelargsym = gensym("kernelarg")
    
    for (i, prev) in enumerate(Any[func.f, args...])
        make_tracer(seen, prev, (kernelargsym, i), Reactant.NoStopTracedTrack)
    end

    argidx = 1
    for arg in values(seen)
        if !(arg isa TracedRArray || arg isa TracedRNumber)
            continue
        end

        paths = Reactant.TracedUtils.get_paths(arg)
        arg = arg.mlir_data
        arg = Reactant.TracedUtils.transpose_val(arg)
        push!(restys, MLIR.IR.type(arg))
        push!(mlir_args, arg)

        ctx = MLIR.IR.current_context()
        out_tup = Ref{Int64}(argidx - 1)
        push!(
            aliases,
            MLIR.IR.Attribute(
                GC.@preserve ctx out_tup MLIR.API.stablehloOutputOperandAliasGet(
                    ctx,
                    length(wrapper_tys) == 1 ? 0 : 1,
                    pointer_from_objref(out_tup),
                    argidx - 1,
                    0,
                    C_NULL,
                )
            ),
        )

        argidx += 1
    end

    blk_operands = IR.Value[]
    for idx in grid
        # TODO
    end

    # TODO emit a kernel_call?
    call = Reactant.MLIR.Dialects.enzymexla.kernel_call(
        blk_operands...;
        inputs=mlir_args,
        result_0=restys,
        fn=IR.FlatSymbolRefAttribute(string(f)),
        output_operand_aliases=IR.Attribute(aliases),
        xla_side_effect_free=IR.UnitAttribute(),
        location=mlir_stacktrace("enzymexla.kernel_call", @__FILE__, @__LINE__),
    )

    return nothing
end

end
