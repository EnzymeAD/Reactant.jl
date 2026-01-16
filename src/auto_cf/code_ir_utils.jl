"""
    method_instance(f::Function, sign::Tuple{Vararg{Type}}, world) -> Union{Base.MethodInstance, Nothing}

Same as `Base.method_instance` except it can work in generated function such as `call_with_reactant`
"""
function method_instance(f::Function, sign::Tuple{Vararg{Type}}, world)
    tt = Base.signature_type(f, sign)
    match, _ = Core.Compiler._findsup(tt, nothing, world)
    isnothing(match) && return nothing
    mi = Core.Compiler.specialize_method(match)
    return mi
end

"""
    change_stmt!(ir::Core.Compiler.IRCode, ssa::Int, stmt, return_type::Type) -> Core.Compiler.Instruction

Change the `ir` at position `ssa` by the statement `stmt` with a `return_type`
"""
function change_stmt!(ir::Core.Compiler.IRCode, ssa::Int, stmt, return_type::Type)
    return Core.Compiler.inst_from_newinst!(
        ir[Core.SSAValue(ssa)],
        Core.Compiler.NewInstruction(stmt, return_type),
        Int32(0),
        UInt32(0),
    )
end

"""
    change_stmt!(ir::Core.Compiler.IRCode, ssa::Int, goto::Core.GotoNode, return_type::Type) -> Core.Compiler.Instruction
Specialization of [`change_stmt!`](@ref) for `Core.GotoNode` to deal with control flow graph changes 
"""
function change_stmt!(
    ir::Core.Compiler.IRCode, ssa::Int, goto::Core.GotoNode, return_type::Type
)
    bb::Int64 = Core.Compiler.block_for_inst(ir, ssa)
    succs = ir.cfg.blocks[bb].succs
    empty!(succs)
    push!(succs, goto.label)
    push!(ir.cfg.blocks[goto.label].preds, bb)
    @invoke change_stmt!(ir, ssa, goto::Any, return_type)
end

"""
    clear_block_ir!(ir::Core.Compiler.IRCode, blocks::Set{Int})
    Replace in BB `blocks` of `ir` each instruction by nothing
"""
function clear_block_ir!(ir::Core.Compiler.IRCode, blocks::Set{Int})
    for block in blocks
        stmt_range::Core.Compiler.StmtRange = ir.cfg.blocks[block].stmts
        (f, l) = (first(stmt_range), last(stmt_range))
        for i in f:l
            change_stmt!(ir, i, nothing, Nothing)
        end
    end
end

"""
    type_from_ssa(ir::Core.Compiler.IRCode, argtypes::Vector, v::Vector)::Vector
    For each stmt in `v` in `ir` get its type 
"""
function type_from_ssa(ir::Core.Compiler.IRCode, argtypes::Vector, v)
    cir = ir
    @lk cir
    return [
        begin
            if e isa Core.SSAValue
                ir.stmts.type[e.id]
            elseif e isa Core.Argument
                argtypes[e.n]
            else
                typeof(e)
            end
        end for e in v
    ]
end

"""
    apply_map(array, block_map::Dict)::Vector
    For each element of `array`, get the value associated in the dictionnary `block_map`
"""
function apply_map(array, block_map)
    return [block_map[a] for a in array if haskey(block_map, a)]
end

"""
    new_cfg(ir, to_extract::Vector, block_map)::Core.Compiler.CFG
    Get the new CFG of `ir` after the extraction of `to_extract` blocks
"""
function new_cfg(ir, to_extract, block_map)
    n = 1
    bbs = Core.Compiler.BasicBlock[]
    index = Int64[]
    for b in to_extract
        bb = ir.cfg.blocks[b]
        (; start, stop) = bb.stmts
        diff = stop - start
        push!(
            bbs,
            Core.Compiler.BasicBlock(
                Core.Compiler.StmtRange(n, diff + n),
                apply_map(bb.preds, block_map),
                apply_map(bb.succs, block_map),
            ),
        )
        n += diff + 1
        push!(index, n)
    end
    return Core.Compiler.CFG(bbs, index)
end

"""
    WipExtracting
    struct used for an extracted IRCode which is not fully constructed
"""
struct WipExtracting
    ir::Core.Compiler.IRCode
end

"""
    is_a_terminator(stmt)
    Check if `stmt` is a terminator
"""
function is_a_terminator(stmt)
    return stmt isa Union{Core.GotoNode,Core.ReturnNode,Core.GotoIfNot}
end

"""
    offset_stmt!(dict::Dict, stmt, offset::Int, ir::Core.Compiler.IRCode, bb_map)
    internal recursive function of [`extract_multiple_block_ir`](@ref) to shift SSAValue/Argument/BasicBlock in `ir`
"""
function offset_stmt!(dict::Dict, stmt, offset::Dict, ir::Core.Compiler.IRCode, bb_map)
    if stmt isa Expr
        Expr(stmt.head, (offset_stmt!(dict, a, offset, ir, bb_map) for a in stmt.args)...)
    elseif stmt isa Core.Argument
        tmp = Core.Argument(length(dict) + 2)
        get!(dict, stmt, (tmp, ir.argtypes[stmt.n]))[1]
    elseif stmt isa Core.ReturnNode
        Core.ReturnNode(offset_stmt!(dict, stmt.val, offset, ir, bb_map))
    elseif stmt isa Core.SSAValue
        stmt_bb = Core.Compiler.block_for_inst(ir, stmt.id)
        if stmt_bb in keys(offset) #TODO: remove? && stmt.id > offset[stmt_bb]
            Core.SSAValue(stmt.id - offset[stmt_bb])
        else
            #the stmt is transformed to an IR argument 
            tmp = Core.Argument(length(dict) + 2)
            get!(dict, stmt, (tmp, ir.stmts.type[stmt.id]))[1]
        end
    elseif stmt isa Core.GotoNode
        Core.GotoNode(get(bb_map, stmt.label, 0))
    elseif stmt isa Core.GotoIfNot
        Core.GotoIfNot(
            offset_stmt!(dict, stmt.cond, offset, ir, bb_map), get(bb_map, stmt.dest, 0)
        )
    elseif stmt isa Core.PhiNode
        Core.PhiNode(
            Int32[bb_map[edge] for edge in stmt.edges],
            Any[offset_stmt!(dict, value, offset, ir, bb_map) for value in stmt.values],
        )
    elseif stmt isa Core.PiNode
        Core.PiNode(offset_stmt!(dict, stmt.val, offset, ir, bb_map), stmt.typ)
    else
        stmt
    end
end


"""
    extract_multiple_block_ir(ir, to_extract_set::Set, args::Dict, new_returns::Vector)::WipExtracting
    Extract from `ir` a list of blocks `to_extract_set`, creating an new independant IR containing only these blocks.
    All unlinked SSA are added to the `args` dictionnary and all values of `new_returns` are returned by the new IR. 
"""
function extract_multiple_block_ir(
    ir::Core.Compiler.IRCode, to_extract_set::Set{Int}, args::Dict, new_returns::Vector
)::WipExtracting
    @assert isempty(ir.new_nodes.stmts)
    to_extract = sort(collect(to_extract_set))
    
    #for each extracted basic block, get the new offset.
    #useful to deal with non-contiguous extraction because in this case, the offset doesn't follow `ir` block offset anymore
    bb_offset::Dict{Int,Int} = Dict()
    new_n_stmt = 0
    if !isempty(to_extract)
        cumulative_offset = (first(ir.cfg.blocks[first(to_extract)].stmts)) - 1
        for bb in minimum(to_extract):maximum(to_extract)
            n_stmt = length(ir.cfg.blocks[bb].stmts)
            if bb in to_extract
                bb_offset[bb] = cumulative_offset
                new_n_stmt += n_stmt
            else
                cumulative_offset += n_stmt
            end
        end
    end

    block_map = Dict()
    for (i, b) in enumerate(to_extract)
        block_map[b] = i
    end
    cfg = new_cfg(ir, to_extract, block_map)

    #PhiNode uses the global IR, either shift it or add it to the new IR argument
    for (i, rb) in enumerate(new_returns)
        rb isa Union{Core.SSAValue,Core.Argument} || continue
        new_returns[i] = offset_stmt!(args, rb, bb_offset, ir, block_map)
    end

    #recreate instruction_stream of the block
    instruction_stream = Core.Compiler.InstructionStream(new_n_stmt)
    dico = Dict()
    new_stmt = 0
    for bb in to_extract
        range_bb = ir.cfg.blocks[bb].stmts[[1, end]]
        for old_stmt in range_bb[1]:range_bb[2]
            new_stmt += 1
            Core.Compiler.setindex!(instruction_stream, ir.stmts[old_stmt], new_stmt) #TODO: check if needed
            #ssa offset
            instruction_stream.stmt[new_stmt] = offset_stmt!(
                args, ir.stmts.stmt[old_stmt], bb_offset, ir, block_map
            )
            #line_info
            line_info = ir.stmts.line[old_stmt]
            line_info == 0 && continue
            instruction_stream.line[new_stmt] = get!(dico, line_info, length(dico) + 1)
        end
    end

    linetable = ir.linetable[sort(collect(keys(dico)))]
    linetable = [
        Core.LineInfoNode(l.module, l.method, l.file, l.line, Int32(0)) for l in linetable
    ]
    #Build the new IR argtypes from args dictionnary
    (_, argtypes) = vec_args(ir, args)
    #JuliaIR block can end without a terminator
    new_ir, has_terminator, n_ssa = if !isempty(instruction_stream)
        (Core.Compiler.IRCode(
            instruction_stream,
            cfg,
            linetable,
            argtypes,
            Expr[],
            Core.Compiler.VarState[],
        ), is_a_terminator(instruction_stream.stmt[end]), length(instruction_stream))
    else
        new_ir = CC.IRCode()
        empty!(new_ir.argtypes)
        push!(new_ir.argtypes, argtypes...)
        (new_ir, true, 1)
    end

    @lk new_returns args
    @error "" args argtypes
    retu = if length(new_returns) > 1
        tuple = Core.Compiler.NewInstruction(
            Expr(:call, Core.GlobalRef(Core, :tuple), new_returns...),
            Tuple{type_from_ssa(new_ir, argtypes, new_returns)...},
        )
        Core.Compiler.insert_node!(
            new_ir, Core.Compiler.SSAValue(n_ssa), tuple, !has_terminator
        )
    else
        length(new_returns) == 1 ? only(new_returns) : nothing
    end

    if has_terminator
        change_stmt!(new_ir, n_ssa, Core.ReturnNode(retu), Nothing)
    else
        terminator = Core.Compiler.NewInstruction(Core.ReturnNode(retu), Nothing)
        @lk new_ir n_ssa terminator
        Core.Compiler.insert_node!(new_ir, Core.Compiler.SSAValue(n_ssa), terminator, true)
    end
    return WipExtracting(Core.Compiler.compact!(new_ir, true))
end

function mlir_type(x)
    return Reactant.MLIR.IR.TensorType(
        size(x), Reactant.MLIR.IR.Type(Reactant.unwrapped_eltype(x))
    )
end

"""
    vec_args(ir::Core.Compiler.IRCode, new_args::Dict)::Vector
    Construct args Vector from `new_args` dictionnary
"""
function vec_args(ir::Core.Compiler.IRCode, new_args::Dict)
    argtypes = Vector(undef, length(new_args) + 1)
    argtypes[1] = Core.Const("opaque")
    value = Vector(undef, length(new_args))
    for (arg, index) in new_args
        value[index[1].n - 1] = arg
        argtypes[index[1].n] = if arg isa Core.Argument #TODO: reuse function
            index[2]
        else
            ir.stmts.type[arg.id]
        end
    end
    return (value, argtypes)
end

"""
    typeof_ir(ir::CC.IRCode, e::Union{Core.Argument, Core.SSAValue})
    Return the type of a stmt in `ir`
    TODO: replace by CC.argextype 
"""
function typeof_ir(ir::CC.IRCode, e::Union{Core.Argument,Core.SSAValue})
    if e isa Core.Argument
        ir.argtypes[e.n]
    else
        ir.stmts.type[e.id]
    end
end

"""
    finish(wir::WipExtracting, new_args::Vector)::Code.Compiler.IRCode

    Constructing the extracted IR by applying the full arguments list
"""
function finish(wir::WipExtracting, new_args::Vector)
    (; ir) = wir
    empty!(ir.argtypes)
    append!(ir.argtypes, new_args)
    ir = rewrite_insts!(ir, current_interpreter[], false)[1]
    return ir
end

"""
    add_phi_value!(v::Vector, phi::Core.PhiNode, edge::Set{Int})

    Add `Core.PhiNode` values to `v` for each `edge` in the set.
    If phi node contains `header_bb` and no element in `edge` then insert the value associated with the header.
"""
function add_phi_value!(v::Vector, phi::Core.PhiNode, edge::Set{Int}, header_bb::Int)
    header_index = nothing
    find_element = false
    for (i, e) in enumerate(phi.edges)
        e == header_bb && (header_index = i)
        e in edge || continue
        find_element = true
        push!(v, phi.values[i]) #TODO: add break after
    end
    (!find_element && !isnothing(header_index)) && (push!(v, phi.values[header_index]))
end

"""
    cond_ssa(ir::CC.IRCode, bb::Int)

    Return the SSA value in a traced GotoIfNot
"""
function cond_ssa(ir::CC.IRCode, bb::Int)
    ti = terminator_index(ir, bb)
    terminator = ir.stmts.stmt[ti]
    terminator isa Core.GotoIfNot || return nothing
    protection = ir.stmts.stmt[terminator.cond.id]
    (
        protection isa Expr &&
        protection.head == :call &&
        protection.args[1] == Core.GlobalRef(@__MODULE__, :traced_protection)
    ) || return nothing
    return protection.args[2]
end

"""
    check_integrity(ir::CC.IRCode)::Bool
    check if `unreachable` is present in the IR, return true if none
"""
function check_integrity(ir::CC.IRCode)::Bool
    return !any(ir.stmts.stmt .== [Core.ReturnNode()])
end