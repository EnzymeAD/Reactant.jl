struct ShiftedSSA
    e::Int
end

struct ShiftedCF
    e::Int
end

function offset_stmt!(stmt, index, next_bb = true)
    if stmt isa Expr
        Expr(
            stmt.head, (offset_stmt!(a, index) for a in stmt.args)...)
    elseif stmt isa Core.ReturnNode
        Core.ReturnNode(offset_stmt!(stmt.val, index))
    elseif stmt isa Core.SSAValue
        Core.SSAValue(offset_stmt!(ShiftedSSA(stmt.id), index))
    elseif stmt isa Core.GotoIfNot
        Core.GotoIfNot(offset_stmt!(stmt.cond, index), offset_stmt!(ShiftedCF(stmt.dest), index, next_bb))
    elseif stmt isa Core.GotoNode
        Core.GotoNode(offset_stmt!(ShiftedCF(stmt.label), index, next_bb))
    elseif stmt isa ShiftedSSA
        stmt.e + (stmt.e < index ? 0 : 1)
    elseif stmt isa ShiftedCF
        stmt.e + (stmt.e < index + (next_bb ? 1 : 0) ? 0 : 1)
    else
        stmt
    end
end

#insert stmt in frame after index
function add_instruction!(frame, index, stmt; type=CC.NotFound(), next_bb = true)
    add_instruction!(frame.src, index, stmt; type, next_bb)
    frame.ssavalue_uses = CC.find_ssavalue_uses(frame.src.code, length(frame.src.code)) #TODO: more fine graine change here
    insert!(frame.stmt_info, index + 1, CC.NoCallInfo())
    insert!(frame.stmt_edges, index + 1, nothing)
    insert!(frame.handler_at, index + 1, (0,0))
    frame.cfg = CC.compute_basic_blocks(frame.src.code)
    Core.SSAValue(index + 1)
end

function modify_instruction!(frame, index, stmt)
    frame.src.code[index] = stmt
    frame.ssavalue_uses = CC.find_ssavalue_uses(frame.src.code, length(frame.src.code)) #TODO: refine this
end

"""
    add_instruction!(ir::CC.CodeInfo, index, stmt; next_bb)

"""
function add_instruction!(ir::CC.CodeInfo, index, stmt; type=CC.NotFound(), next_bb=true)
    for (i, c) in enumerate(ir.code)
        ir.code[i] = offset_stmt!(c, index + 1, next_bb)
    end
    insert!(ir.code, index + 1, stmt)
    insert!(ir.codelocs, index + 1, 0)
    insert!(ir.ssaflags, index + 1, 0x00000000)
    if ir.ssavaluetypes isa Int
        ir.ssavaluetypes = ir.ssavaluetypes + 1
    else
        insert!(ir.ssavaluetypes, index + 1, type)
    end
end


function create_slot!(ir::CC.CodeInfo)::Core.SlotNumber
    push!(ir.slotflags, 0x00)
    push!(ir.slotnames, Symbol(""))
    Core.SlotNumber(length(ir.slotflags))
end

function create_slot!(frame)::Core.SlotNumber
    push!(frame.slottypes, Union{})
    for s in frame.bb_vartables
        isnothing(s) && continue
        push!(s, CC.VarState(Union{}, true))
    end
    create_slot!(frame.src)
end

add_slot_change!(ir::CC.CodeInfo, index, old_slot::Int) = add_slot_change!(ir, index, Core.SlotNumber(old_slot))

function add_slot_change!(ir::CC.CodeInfo, index, old_slot::Core.SlotNumber)
    push!(ir.slotflags, 0x00)
    push!(ir.slotnames, Symbol(""))
    new_slot = Core.SlotNumber(length(ir.slotflags))
    add_instruction!(frame, index, Expr(:(=), new_slot, Expr(:call, GlobalRef(@__MODULE__, :upgrade), old_slot)))
    update_ir_new_slot(ir, index, old_slot, new_slot)
end

function update_ir_new_slot(ir, index, old_slot, new_slot)
    for i in index+2:length(ir.code) #TODO: probably need to refine this
        ir.code[i] = replace_slot_stmt(ir.code[i], old_slot, new_slot)
    end
end

function replace_slot_stmt(stmt, old_slot, new_slot)
    if stmt isa Core.NewvarNode
        stmt
    elseif stmt isa Expr
        Expr(stmt.head, (replace_slot_stmt(e, old_slot, new_slot) for e in stmt.args)...)
    elseif stmt isa Core.SlotNumber
        stmt == old_slot ? new_slot : stmt
    else
        stmt
    end
end