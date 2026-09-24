# Debugging instrumentation that reports where NaNs first show up in gradients.
#
# When `DEBUG_NAN_GRADIENTS[]` is set, every `enzyme.set` that stores a floating point
# gradient is preceded by
#
#   %nan = math.isnan %value
#   %any = stablehlo.reduce(%nan init: false) applies stablehlo.or
#   stablehlo.if %any { enzymexla.jit_call @__reactant_nan_report_N() }
#
# where `@__reactant_nan_report_N` is a host function printing the location of the
# corresponding `enzyme.get` (i.e. the location of the primal op whose derivative is being
# accumulated). The pass has to run on the output of the `enzyme` pass but before
# `remove-unnecessary-enzyme-ops` promotes the gradient slots to SSA values.
#
# External (Julia) passes cannot be registered in MLIR's pass registry, so they cannot be
# referenced from the textual `postpasses` of the `enzyme` pass. Instead, the pipeline
# string contains `NAN_GRADIENT_CHECK_PASS` as a placeholder that `run_pass_pipeline!`
# replaces with the Julia pass, see `add_pipeline_with_julia_passes!`.
#
# Since `remove-unnecessary-enzyme-ops` then only runs after the whole `enzyme` pass, nested
# differentiation fails with this instrumentation enabled: the inner derivative still
# contains `enzyme.get`/`enzyme.set` when the outer one differentiates it.

const DEBUG_NAN_GRADIENTS = Ref(false)

# Placeholder for the pass in textual pipelines: `reactant-nan-gradient-check{backend=cuda}`
const NAN_GRADIENT_CHECK_PASS = "reactant-nan-gradient-check"
const NAN_GRADIENT_CHECK_PASS_REGEX = r"reactant-nan-gradient-check(?:\{backend=(\w+)\})?"

struct NaNGradientCheckPass <: MLIR.IR.AbstractPass
    # `lower-jit{backend=cuda}` requires host functions to follow the CUDA ABI
    cuda_abi::Bool
end

MLIR.IR.opname(::NaNGradientCheckPass) = "builtin.module"

function MLIR.IR.pass_run(ctx::MLIR.IR.Context, pass::NaNGradientCheckPass, mod_op)
    MLIR.IR.@with_context ctx begin
        instrument_nan_gradients!(mod_op; pass.cuda_abi)
    end
    return nothing
end

function add_pipeline_with_julia_passes!(pm, opm, pass_pipeline)
    offset = 1
    for m in eachmatch(NAN_GRADIENT_CHECK_PASS_REGEX, pass_pipeline)
        add_textual_pipeline!(opm, pass_pipeline[offset:(m.offset - 1)])
        pass = MLIR.IR.create_external_pass!(
            pm,
            NaNGradientCheckPass(something(m[1], "cuda") == "cuda"),
            "NaNGradientCheck",
            NAN_GRADIENT_CHECK_PASS,
            "Report the location of gradients that contain NaNs",
        )
        MLIR.IR.add_owned_pass!(opm, pass)
        offset = m.offset + ncodeunits(m.match)
    end
    add_textual_pipeline!(opm, pass_pipeline[offset:end])
    return opm
end

function add_textual_pipeline!(opm, pass_pipeline)
    pass_pipeline = strip(pass_pipeline, ',')
    isempty(pass_pipeline) || MLIR.IR.add_pipeline!(opm, String(pass_pipeline))
    return opm
end

function collect_ops!(ops, op::MLIR.IR.Operation, opname)
    for region in op, block in region, inner in block
        MLIR.IR.name(inner) == opname && push!(ops, inner)
        collect_ops!(ops, inner, opname)
    end
    return ops
end

defining_op(value) = MLIR.IR.is_op_res(value) ? MLIR.IR.op_owner(value) : nothing

function location_string(loc::MLIR.IR.Location)
    frames = _collect_location_frames(loc)
    isempty(frames) && return " unknown location"
    io = IOBuffer()
    println(io)
    for (j, frame) in enumerate(frames)
        _print_frame(io, j, frame, length(frames))
    end
    return rstrip(String(take!(io)))
end

# Accumulation into a gradient is emitted as `%g = get %grad; %v = add %g, %d; set %grad, %v`.
# Report the location of that `get` (falling back to the `set`), and of the op defining the
# accumulated value `%d`, which is usually the op that introduced the NaN.
function report_message(set_op)
    gradient, value = MLIR.IR.operand(set_op, 1), MLIR.IR.operand(set_op, 2)
    loc = MLIR.IR.location(set_op)
    addend_loc = nothing
    add_op = defining_op(value)
    if add_op !== nothing && MLIR.IR.name(add_op) == "stablehlo.add"
        operands = MLIR.IR.operands(add_op)
        for (i, operand) in enumerate(operands)
            get_op = defining_op(operand)
            if get_op !== nothing &&
                MLIR.IR.name(get_op) == "enzyme.get" &&
                MLIR.IR.operand(get_op, 1) == gradient
                loc = MLIR.IR.location(get_op)
                addend_op = defining_op(operands[3 - i])
                addend_op === nothing || (addend_loc = MLIR.IR.location(addend_op))
                break
            end
        end
    end
    msg = "Reactant: NaN detected in gradient of$(location_string(loc))"
    if addend_loc !== nothing && addend_loc != loc
        msg *= "\n  accumulated from$(location_string(addend_loc))"
    end
    return msg
end

function should_check(set_op)
    type = MLIR.IR.type(MLIR.IR.operand(set_op, 2))
    MLIR.IR.istensor(type) && MLIR.IR.hasstaticshape(type) || return false
    MLIR.API.mlirTypeIsAFloat(eltype(type)) || return false
    # resetting a gradient to a constant (typically zero) can't introduce a NaN
    op = defining_op(MLIR.IR.operand(set_op, 2))
    return op === nothing || MLIR.IR.name(op) != "stablehlo.constant"
end

# Escape `str` for an MLIR string literal
function escape_mlir_string(str)
    io = IOBuffer()
    for byte in codeunits(str)
        if 0x20 <= byte < 0x7f && byte != UInt8('"') && byte != UInt8('\\')
            write(io, byte)
        else
            print(io, '\\', string(byte; base=16, pad=2))
        end
    end
    return String(take!(io))
end

# Emit, for each message, a host function printing it (plus a newline) to stdout.
function emit_report_functions!(mod_op, messages; cuda_abi::Bool)
    symbol_table = MLIR.IR.SymbolTable(mod_op)
    is_free(name) = MLIR.IR.lookup(symbol_table, name) === nothing

    io = IOBuffer()
    println(io, "module {")
    for callee in ("puts", "fflush")
        is_free(callee) && println(io, "  llvm.func @$callee(!llvm.ptr) -> i32")
    end
    fn_attrs = cuda_abi ? "attributes {enzymexla.device_abi = \"cuda\"}" : ""
    fnames = String[]
    i = 0
    for message in messages
        while !is_free("__reactant_nan_report_$i") ||
              !is_free("__reactant_nan_report_$(i)_msg")
            i += 1
        end
        fname = "__reactant_nan_report_$i"
        gname = fname * "_msg"
        i += 1
        push!(fnames, fname)
        print(
            io,
            """
              llvm.mlir.global private constant @$gname("$(escape_mlir_string(message))\\00") {addr_space = 0 : i32} : !llvm.array<$(ncodeunits(message) + 1) x i8>
              llvm.func private @$fname() $(fn_attrs) {
                %0 = llvm.mlir.addressof @$gname : !llvm.ptr
                %1 = llvm.call @puts(%0) : (!llvm.ptr) -> i32
                %2 = llvm.mlir.zero : !llvm.ptr
                %3 = llvm.call @fflush(%2) : (!llvm.ptr) -> i32
                llvm.return
              }
            """,
        )
    end
    println(io, "}")

    parsed = parse(MLIR.IR.Module, String(take!(io)))
    body = first(MLIR.IR.region(mod_op, 1))
    for op in collect(MLIR.IR.body(parsed))
        MLIR.IR.rmfromparent!(op)
        push!(body, op)
    end
    return fnames
end

function emit_nan_check!(set_op, fname)
    loc = MLIR.IR.location(set_op)
    value = MLIR.IR.operand(set_op, 2)
    type = MLIR.IR.type(value)
    i1 = MLIR.IR.Type(Bool)
    scalar_i1 = MLIR.IR.TensorType(Int[], i1)
    rank = ndims(type)

    scratch = MLIR.IR.Block(MLIR.IR.Type[], MLIR.IR.Location[])
    MLIR.IR.@with_block scratch begin
        isnan = MLIR.IR.create_operation(
            "math.isnan",
            loc;
            operands=[value],
            results=[MLIR.IR.TensorType(collect(Int, size(type)), i1)],
            result_inference=false,
        )
        any_nan = if rank == 0
            MLIR.IR.result(isnan)
        else
            init = MLIR.IR.result(
                MLIR.Dialects.stablehlo.constant(;
                    value=MLIR.IR.DenseElementsAttribute(fill(false)), location=loc
                ),
            )
            reducer = MLIR.IR.Block([scalar_i1, scalar_i1], [loc, loc])
            MLIR.IR.@with_block reducer begin
                res = MLIR.Dialects.stablehlo.or(
                    MLIR.IR.argument(reducer, 1),
                    MLIR.IR.argument(reducer, 2);
                    result=scalar_i1,
                    location=loc,
                )
                MLIR.Dialects.stablehlo.return_([MLIR.IR.result(res)]; location=loc)
            end
            body = MLIR.IR.Region()
            push!(body, reducer)
            MLIR.IR.result(
                MLIR.Dialects.stablehlo.reduce(
                    [MLIR.IR.result(isnan)],
                    [init];
                    result_0=[scalar_i1],
                    dimensions=MLIR.IR.DenseArrayAttribute(collect(Int64, 0:(rank - 1))),
                    body,
                    location=loc,
                ),
            )
        end

        true_block = MLIR.IR.Block(MLIR.IR.Type[], MLIR.IR.Location[])
        MLIR.IR.@with_block true_block begin
            MLIR.Dialects.enzymexla.jit_call(
                MLIR.IR.Value[];
                result_0=MLIR.IR.Type[],
                fn=MLIR.IR.FlatSymbolRefAttribute(fname),
                location=loc,
            )
            MLIR.Dialects.stablehlo.return_(MLIR.IR.Value[]; location=loc)
        end
        false_block = MLIR.IR.Block(MLIR.IR.Type[], MLIR.IR.Location[])
        MLIR.IR.@with_block false_block begin
            MLIR.Dialects.stablehlo.return_(MLIR.IR.Value[]; location=loc)
        end
        true_branch, false_branch = MLIR.IR.Region(), MLIR.IR.Region()
        push!(true_branch, true_block)
        push!(false_branch, false_block)
        MLIR.Dialects.stablehlo.if_(
            any_nan; result_0=MLIR.IR.Type[], true_branch, false_branch, location=loc
        )
    end

    for op in collect(scratch)
        MLIR.IR.move_before!(op, set_op)
    end
    return nothing
end

function instrument_nan_gradients!(mod_op; cuda_abi::Bool)
    set_ops = filter!(should_check, collect_ops!(MLIR.IR.Operation[], mod_op, "enzyme.set"))
    isempty(set_ops) && return nothing

    fnames = emit_report_functions!(mod_op, map(report_message, set_ops); cuda_abi)
    for (set_op, fname) in zip(set_ops, fnames)
        emit_nan_check!(set_op, fname)
    end
    return nothing
end
