#simple version of `CC.scan_slot_def_use`
function fill_slot_definition_map(frame)
    n_slot = length(frame.src.slotnames)
    n_args = length(frame.linfo.specTypes.types)
    v = [0 for _ in 1:n_slot]
    for (i, stmt) in enumerate(frame.src.code)
        stmt isa Expr || continue
        stmt.head == :(=) || continue
        slot = stmt.args[1]
        slot isa Core.SlotNumber || continue
        slot.id > n_args || continue
        v[slot.id] = v[slot.id] == 0 ? i : v[slot.id]
    end
    return v
end

function fill_slot_usage_map(frame)
    n_slot = length(frame.src.slotnames)
    v = [Set() for _ in 1:n_slot]
    for (pos, stmt) in enumerate(frame.src.code)
        get_slot(v, stmt, frame, pos)
    end
    return v
end

function get_slot(vec, stmt, frame, pos)
    if stmt isa Expr
        stmt.head == :(=) && return get_slot(vec, stmt.args[2], frame, pos)
        for e in stmt.args
            get_slot(vec, e, frame, pos)
        end
    elseif stmt isa Core.SlotNumber
        push!(vec[stmt.id], CC.block_for_inst(frame.cfg, pos))
    else
        stmt
    end
end

#an = Analysis(Tree(nothing, [], Ref{Tree}()), nothing, nothing, nothing, nothing)

function update_tree!(an::Analysis, bb::Int)
    tree = an.tree
    if tree.node isa LoopStructure && tree.node.kind == While
        tree.node.header_bb == bb && return true
    end
    for c in tree.children
        c.node.header_bb == bb || continue
        an.pending_tree = c
        return true
    end
    return false
end

function add_tree!(an::Analysis, tl)
    parent = an.tree
    t = Tree(tl, [], Ref{Tree}(parent))
    push!(parent.children, t)
    return an.pending_tree = t
end

#Several TCF can end in the same bb
function up_tree!(an::Analysis, bb)
    terminal = false
    while is_terminal_bb(an.tree, bb)
        an.tree = an.tree.parent[]
        terminal = true
    end
    terminal && return nothing
    #terminal bb is not always reach: for instance, if bodies are more precisely inferred and nothing change 
    while !isnothing(an.tree.node)
        in_header(bb, an.tree.node) && break
        an.tree = an.tree.parent[]
    end
end

function down_tree!(an::Analysis, bb)
    for child in an.tree.children
        if child.node.header_bb == bb
            an.tree = child
            break
        end
    end
end

function is_terminal_bb(tree::Tree, bb)
    isnothing(tree.node) && return false
    return tree.node.terminal_bb == bb
end

Base.in(bb::Int, is::IfStructure) = bb in is.true_bbs || bb in is.false_bbs
Base.in(bb::Int, is::LoopStructure) = bb in is.body_bbs

function in_header(bb::Int, is::IfStructure)
    return bb in is.true_bbs || bb in is.false_bbs || bb == is.header_bb
end
function in_header(bb::Int, is::LoopStructure)
    return bb in is.body_bbs || bb == is.header_bb || bb == is.latch_bb
end

function in_stack(tree::Tree, bb::Int)
    while !isnothing(tree.node)
        in_header(bb, tree.node) && return true
        tree = tree.parent[]
    end
    return false
end

function protect_goto_if_not!(frame, an, bb, cond) #TODO: remove cond
    goto_if_not_index = terminator_index(frame.cfg, bb)
    ssa = add_instruction!(
        frame,
        goto_if_not_index - 1,
        Expr(:call, GlobalRef(@__MODULE__, :traced_protection), cond),
    )
    invalidate_slot_definition_analysis!(an)
    (; dest::Int) = frame.src.code[goto_if_not_index + 1]::Core.GotoIfNot #shifted because of the insertion
    return modify_instruction!(frame, goto_if_not_index + 1, Core.GotoIfNot(ssa, dest))
end

#TODO: don't recompute TCF each time
function add_cf!(an, frame, currbb, currpc, condt)
    update_tree!(an, frame.currbb) && return false
    Debugger.@bp
    tl = is_a_traced_loop(an, frame.src, frame.cfg, frame.currbb)
    if tl !== nothing
        add_tree!(an, tl)
        if tl.kind == While
            Debugger.@bp
            terminator_pos = terminator_index(frame.cfg, tl.header_bb)
            (; cond) = frame.src.code[terminator_pos]::Core.GotoIfNot
            protect_goto_if_not!(frame, an, tl.header_bb, cond)
            return true
        end
        return false
    end

    tl = is_a_traced_if(an, frame, frame.currbb, condt)
    if tl !== nothing
        add_tree!(an, tl)
        !tl.legalize[] || return false
        #legalize if by inserting a call
        protect_goto_if_not!(frame, an, frame.currbb, tl.ssa_cond)
        tl.legalize[] = true
        return true
    end
    return false
end

@noinline traced_protection(x::Reactant.TracedRNumber{Bool}) = CC.inferencebarrier(x)::Bool
Reactant.@skip_rewrite_func traced_protection

@noinline upgrade(x) = Reactant.Ops.constant(x)
@noinline upgrade(x::Union{Reactant.TracedRNumber,Reactant.TracedRArray}) = x

Reactant.@skip_rewrite_func upgrade
#TODO: need a new traced mode Julia Type Non-concrete -> Traced
upgrade_traced_type(t::Core.Const) = upgrade_traced_type(CC.widenconst(t))
upgrade_traced_type(t::Type{<:Number}) = Reactant.TracedRNumber{t}
upgrade_traced_type(t::Type{<:Reactant.TracedRNumber}) = t

in_tcf(an::Analysis) = begin
    !isnothing(an.tree.node)
end

invalidate_slot_definition_analysis!(an) = an.slotanalysis = nothing

function if_type_passing!(an, frame)
    in_tcf(an) || return false
    last_cf = an.tree.node
    last_cf isa IfStructure || return false
    last_cf.header_bb == frame.currbb || return false
    !last_cf.legalize[] || return false
    protect_goto_if_not!(frame, an, frame.currbb, last_cf.ssa_cond)
    last_cf.legalize[] = true
    #update frame
    return true
end

function can_upgrade_loop(an, rt)
    in_tcf(an) || return false
    last_cf = an.tree.node
    last_cf isa LoopStructure || return false
    last_cf.state == Maybe || return false
    is_traced(rt) || return false
    return true
end

# a = expr
# =>
# a = upgrade(expr)
# do nothing if expr is already an upgrade call
#TODO: rt suspicious
function apply_slot_upgrade!(frame, pos::Int, rt)::Bool
    @warn "upgrade slot $pos $rt"
    stmt = frame.src.code[pos]
    @assert Base.isexpr(stmt, :(=)) "$stmt"
    r = stmt.args[2]
    #TODO: iterate can be upgraded to a traced iterate. SSAValue, slots & literal only need stmt change. Others need a new stmt
    if Base.isexpr(r, :call)
        r.args[1] == GlobalRef(@__MODULE__, :upgrade) && return false
        if r.args[1] == GlobalRef(Base, :iterate)
            new_type = traced_iterator(rt)
            frame.src.code[pos] = Expr(
                :(=),
                stmt.args[1],
                Expr(:call, GlobalRef(Base, :iterate), new_type, r.args[2:end]...),
            )
            return true
        end
        frame.src.code[pos] = stmt.args[2]
        add_instruction!(
            frame,
            pos,
            Expr(
                :(=),
                stmt.args[1],
                Expr(:call, GlobalRef(@__MODULE__, :upgrade), Core.SSAValue(pos)),
            );
            next_bb=false,
        )
    elseif r isa Core.SlotNumber || r isa Core.SSAValue || true #TODO: for expr we must create a new call and the expr 
        frame.src.code[pos] = Expr(
            :(=), stmt.args[1], Expr(:call, GlobalRef(@__MODULE__, :upgrade), stmt.args[2])
        )
    else
        error("unsupported slot upgrade $stmt")
    end
    return true
end

function current_top_struct(tree)
    top_struct = nothing
    while !isnothing(tree.node)
        top_struct = tree.node
        tree = tree.parent[]
    end
    return top_struct
end

function get_root(tree)
    while !isnothing(tree.node)
        tree = tree.parent[]
    end
    return tree
end

#TODO: remove
function get_first_slot_read_stack(frame, tree, slot::Core.SlotNumber, stop::Int)
    node = current_top_struct(tree)
    start_stmt = frame.cfg.blocks[node.header_bb].stmts.start
    for stmt_index in start_stmt:stop
        s = frame.src.code[stmt_index]
        s isa Core.SlotNumber || continue
        s.id == slot.id && return CC.block_for_inst(frame.cfg.index, stmt_index)
    end
    return nothing
end

@inline function check_and_upgrade_slot!(an, frame, stmt, rt, currstate)
    in_tcf(an) || return (NoUpgrade,)
    stmt isa Expr || return (NoUpgrade,)
    stmt.head == :(=) || return (NoUpgrade,)
    last_cf = an.tree.node
    rt_traced = is_traced(rt)
    slot = stmt.args[1].id
    slot_type::Type = CC.widenconst(currstate[slot].typ)

    #If the stmt is traced: if the slot is traced or not set, don't need to upgrade the slot
    #TODO: Nothing suspicions
    rt_traced &&
        (is_traced(slot_type) || slot_type === Union{} || slot_type == Nothing) &&
        return (NoUpgrade,)

    if last_cf isa IfStructure
        (frame.currbb in last_cf.true_bbs || frame.currbb in last_cf.false_bbs) ||
            return (NoUpgrade,)
        #inside a traced_if, slot must be upgraded to a traced type
        sa = get_slot_analysis(an, frame)::SlotAnalysis
        #TODO: approximation: use liveness analysis to precise promote local slot
        # if traced
        isempty(
            setdiff(sa.slot_bb_usage[slot], union(last_cf.true_bbs, last_cf.false_bbs))
        ) && return (NoUpgrade,)

        #invalidate_slot_definition_analysis!(an)
        return if apply_slot_upgrade!(frame, frame.currpc, rt)
            (UpgradeLocally,)
        else
            (NoUpgrade,)
        end

        #no need to change frame furthermore
    elseif last_cf isa LoopStructure
        (last_cf.state == Traced || last_cf.state == Upgraded) || return (NoUpgrade,)
        Debugger.@bp
        if (!rt_traced && is_traced(slot_type))
            return if apply_slot_upgrade!(frame, frame.currpc, rt)
                (UpgradeLocally,)
            else
                (NoUpgrade,)
            end
        end
        sa = get_slot_analysis(an, frame)::SlotAnalysis
        slot_definition_pos = sa.slot_stmt_def[slot]
        slot_definition_bb = CC.block_for_inst(frame.cfg, slot_definition_pos)
        #local slot doesn't need to be upgrade TODO: suspicious
        slot_definition_bb in last_cf.body_bbs && return (NoUpgrade,)
        for_stack_cond =
            (last_cf.kind == For) && (
                slot_definition_bb == last_cf.header_bb ||
                in_stack(an.tree, slot_definition_bb)
            )
        while_stack_cond =
            (last_cf.kind == While) && (slot_definition_bb + 1 == last_cf.header_bb) #While loops to header block: slot upgrade must be done before this block
        if for_stack_cond || while_stack_cond
            #stack upgrade
            #the slot has been upgraded: find read of the slot inside the current traced stack: if any, we must restart the inference from there
            return if apply_slot_upgrade!(frame, slot_definition_pos, rt)
                (UpgradeDefinition, stmt.args[1])
            else
                (NoUpgrade,)
            end
        else
            #global upgrade: add a new slot
            new_slot_def_pos = if last_cf.header_bb == 1
                #first block contains argument to slot write: new instructions must be placed after (otherwise all the IR is dead)
                new_index = 0
                for i in frame.cfg.blocks[1].stmts
                    local_stmt = frame.src.code[i]
                    local_stmt isa Expr &&
                        local_stmt.head == :(=) &&
                        typeof.(frame.src.code[i].args) ==
                        [Core.SlotNumber, Core.SlotNumber] &&
                        continue
                    local_stmt isa Core.NewvarNode && continue
                    new_index = i
                    break
                end
                new_index
            else
                frame.cfg.blocks[last_cf.header_bb].stmts.start - 1
            end
            #add_slot_change!(frame.src, new_slot_def_pos, slot)
            slot = stmt.args[1]
            #CodeInfo: Cannot use a slot inside a call
            add_instruction!(frame, new_slot_def_pos, slot)
            add_instruction!(
                frame,
                new_slot_def_pos + 1,
                Expr(
                    :(=),
                    slot,
                    Expr(
                        :call,
                        GlobalRef(@__MODULE__, :upgrade),
                        Core.SSAValue(new_slot_def_pos + 1),
                    ),
                ),
            )
            invalidate_slot_definition_analysis!(an)
            return (UpgradeDefinitionGlobal,)
        end
        return (UpgradeDefinition,)
    end
end

terminator_index(ir::Core.Compiler.IRCode, bb::Int) = terminator_index(ir.cfg, bb)
terminator_index(cfg::CC.CFG, bb::Int) = cfg.blocks[bb].stmts.stop
start_index(ir::CC.IRCode, bb::Int) = start_index(ir.cfg, bb)
start_index(cfg::CC.CFG, bb::Int) = bb == 1 ? 1 : cfg.index[bb - 1]

#TODO: proper support this by walking the IR
function is_traced_loop_iterator(src::CC.CodeInfo, cfg::CC.CFG, bb::Int)
    terminator_pos = terminator_index(cfg, bb)
    iterator_index = src.code[terminator_pos].cond.id - 3
    iterator_type = src.ssavaluetypes[iterator_index]
    return is_traced(iterator_type)
end

is_traced(t::Type) = parentmodule(t) == Reactant
is_traced(::Core.TypeofBottom) = false
is_traced(t::UnionAll) = is_traced(CC.unwrap_unionall(t))
is_traced(u::Union) = (|)(is_traced.(Base.uniontypes(u))...)
function is_traced(t::Type{<:Tuple})
    t isa Union && return @invoke is_traced(t::Union)
    t = Base.unwrap_unionall(t)
    t isa UnionAll && return is_traced(Base.unwrap_unionall(t))
    if typeof(t) == UnionAll
        t = t.body
    end
    return (|)(is_traced.(t.types)...)
end
is_traced(::Type{Tuple{}}) = false
is_traced(t) = false

function is_for_last_body_block(node, bb)::Bool
    isnothing(node) && return false
    node isa Reactant.LoopStructure || return false
    node.kind == For || return false
    max(node.body_bbs...) == bb
end

#TODO: add support to while loop / general loop
function is_a_traced_loop(an, src::CC.CodeInfo, cfg::CC.CFG, bb_header)
    bb_body_first = min(cfg.blocks[bb_header].succs...)
    can_be_for = max(cfg.blocks[bb_body_first].preds...) > bb_body_first
    begin
        an.tree.node
    end
    can_be_while = !is_for_last_body_block(an.tree.node, bb_header) && max(cfg.blocks[bb_header].preds...) > bb_header
    #detect cycle in the cfg
    if can_be_for || can_be_while
        bb_end = max(cfg.blocks[bb_header].succs...)
        latch = max(cfg.blocks[bb_body_first].preds...)
        @error bb_header latch
        bb_body_last = cfg.blocks[latch].preds
        #the latch is present only in for loop: detect if the final body block can go between the latch and the end block
        if length(bb_body_last) == 1 && bb_end == max(cfg.blocks[bb_body_last[1]].succs...)
            can_be_for && return LoopStructure(
                For,
                (),
                bb_header,
                latch,
                bb_end,
                Set(bb_body_first:bb_body_last[1]),
                is_traced_loop_iterator(src, cfg, bb_header) ? Traced : Maybe,
            )
        else
            bb_body_max = max(cfg.blocks[bb_header].preds...)
            terminator_pos = terminator_index(cfg, bb_header)
            cond_index = src.code[terminator_pos].cond.id
            can_be_while && return LoopStructure(
                While,
                (),
                bb_header,
                0,
                bb_end,
                Set(bb_body_first:bb_body_max),
                is_traced(src.ssavaluetypes[cond_index]) ? Traced : Maybe,
            )
        end
    end
    return nothing
end

function bb_owned_branch(domtree, bb::Int)::Set{Int}
    bbs = Set(bb)
    for c in domtree[bb].children
        bbs = union(bbs, bb_owned_branch(domtree, c))
    end
    return bbs
end

function bb_branch(cfg, bb::Int, t_bb::Int)::Set{Int}
    bbs = Set()
    work = [bb]
    while !isempty(work)
        c_bb = pop!(work)
        (c_bb in bbs || c_bb == t_bb) && continue
        push!(bbs, c_bb)
        for s in cfg.blocks[c_bb].succs
            push!(work, s)
        end
    end
    return bbs
end

function get_doms(an, frame)
    if an.domtree === nothing
        an.domtree = CC.construct_domtree(frame.cfg).nodes
        an.postdomtree = CC.construct_postdomtree(frame.cfg).nodes
    end
    return (an.domtree, an.postdomtree)
end

function get_slot_analysis(an::Analysis, frame)::SlotAnalysis
    if an.slotanalysis === nothing
        an.slotanalysis = SlotAnalysis(
            fill_slot_definition_map(frame), fill_slot_usage_map(frame)
        )
    end
    return an.slotanalysis
end

#TODO:remove currbb
function is_a_traced_if(an, frame, currbb, condt)
    condt == Reactant.TracedRNumber{Bool} || return nothing
    (domtree, postdomtree) = get_doms(an, frame) #compute dominance analysis only when needed
    bb = frame.cfg.blocks[currbb]
    succs::Vector{Int64} = bb.succs
    if_goto_stmt::Core.GotoIfNot = frame.src.code[last(bb.stmts)]
    #CodeInfo GotoIfNot.dest is a stmt
    first_false_bb = CC.block_for_inst(frame.cfg.index, if_goto_stmt.dest)
    first_true_bb = succs[1] == first_false_bb ? succs[2] : succs[1]
    last_child = last(domtree[currbb].children)
    is_diamond = currbb in postdomtree[last_child].children
    final_bb = if is_diamond
        last_child
    else
        if_final_bb = nothing
        for (final_bb, nodes) in enumerate(postdomtree)
            if currbb in nodes.children
                if_final_bb = final_bb
                break
            end
        end
        @assert !isnothing(if_final_bb)
        if_final_bb
    end
    true_bbs = bb_branch(frame.cfg, first_true_bb, final_bb)
    false_bbs = bb_branch(frame.cfg, first_false_bb, final_bb)
    all_owned = bb_owned_branch(domtree, currbb)
    true_owned_bbs = intersect(bb_owned_branch(domtree, first_true_bb), all_owned, true_bbs)
    false_owned_bbs = intersect(
        bb_owned_branch(domtree, first_false_bb), all_owned, false_bbs
    )
    return IfStructure(
        if_goto_stmt.cond,
        currbb,
        final_bb,
        true_bbs,
        false_bbs,
        true_owned_bbs,
        false_owned_bbs,
        Ref{Bool}(false),
        Set(),
    )
end

#HACK: add a general T to Traced{T} conversion
function traced_iterator(::Type{Union{Nothing,Tuple{T,T}}}) where {T}
    is_traced(T) && return T
    Tout = Reactant.TracedRNumber{T}
    return Union{Nothing,Tuple{Tout,Nothing}} #TODO: replace INT -> Nothing
end

traced_iterator(t::Type{Tuple{T,T}}) where {T} = traced_iterator(Union{Nothing,t})

traced_iterator(t) = begin
    if !is_traced(t)
        error("fallback $t")
    end
    t
end

function get_new_iterator_type(src::CC.CodeInfo, cfg::CC.CFG, bb::Int)
    terminator_pos = terminator_index(cfg, bb)
    iterator_index = src.code[terminator_pos].cond.id - 3
    iterator_type = src.ssavaluetypes[iterator_index]
    iterator_type = CC.widenconst(iterator_type)
    return traced_iterator(iterator_type)
end

#TODO: proper check if the iterator exists and replace -3
function rewrite_iterator(src::CC.CodeInfo, cfg::CC.CFG, bb::Int, new_type::Type)
    terminator_pos = terminator_index(cfg, bb)
    iterator_index = src.code[terminator_pos].cond.id - 3
    iterator = src.code[iterator_index]
    iterator_arg = iterator.args[end].args[end]
    iterator.args[end] = Expr(:call, GlobalRef(Base, :iterate), new_type, iterator_arg)
    return iterator.args[1]
end

function reset_slot!(state::Union{Nothing,Vector{Core.Compiler.VarState}}, slot::Int)
    return isnothing(state) ? state : state[slot] = CC.VarState(Union{}, true)
end

function reset_slot!(
    state::Union{Nothing,Vector{Core.Compiler.VarState}}, slot::Core.SlotNumber
)
    return reset_slot!(state, slot.id)
end

function reset_slot!(states)
    for i in eachindex(states)
        states[i] = nothing
    end
end

function reset_slot!(states, fs::LoopStructure, slot::Core.SlotNumber)
    reset_slot!(states[fs.header_bb], slot)
    for bb in fs.body_bbs
        reset_slot!(states[bb], slot)
    end
    reset_slot!(states[fs.latch_bb], slot)
    return reset_slot!(states[fs.terminal_bb], slot)
end

#upgrade a while loop cond by upgrading each slot used in the cond expression
function while_cond_upgrade!(frame, an, cfg, bb)
    src = frame.src
    terminator_pos = terminator_index(cfg, bb)
    (; cond) = src.code[terminator_pos]::Core.GotoIfNot
    to_upgrade_slots = Dict{Int,Core.SlotNumber}()
    to_visit = Any[cond]
    current_pos = terminator_pos
    while !isempty(to_visit)
        e = pop!(to_visit)
        if e isa Expr
            push!(to_visit, e.args[2:end]...)
        elseif e isa Core.SlotNumber
            to_upgrade_slots[current_pos] = e  #TODO: pos before header must be handled
        elseif e isa Core.SSAValue
            current_pos = e.id
            push!(to_visit, src.code[current_pos])
        end
    end

    for e in sort(to_upgrade_slots)
        pos = e.first
        frame.src.code[pos] = Expr(:call, GlobalRef(@__MODULE__, :upgrade), e.second)
    end
    return protect_goto_if_not!(frame, an, bb, cond)
end

#TODO: stack -> branch
function rewrite_loop_stack!(an::Analysis, frame, states, currstate)
    (; src::CC.CodeInfo, cfg::CC.CFG) = frame
    ct = an.tree
    top_loop_tcf = nothing
    while !isnothing(ct.node)
        node = ct.node
        ct = ct.parent[]
        node isa LoopStructure || continue
        node.state == Maybe || continue
        if node.kind == For
            new_iterator_type = get_new_iterator_type(src, cfg, node.header_bb)
            slot = rewrite_iterator(src, cfg, node.header_bb, new_iterator_type)
            last_for_bb = last(sort(collect(node.body_bbs)))
            slot = rewrite_iterator(frame.src, frame.cfg, last_for_bb, new_iterator_type)
        else
            while_cond_upgrade!(frame, an, cfg, node.header_bb)
            Debugger.@bp
        end
        top_loop_tcf = ct
        node.state = Upgraded
    end
    @assert(!isnothing(top_loop_tcf))
    return top_loop_tcf
    #restart type inference from: top_header_rewritten
end

#Transform an n-terminator bb IR to an 1-terminator bb IR
#TODO: improve algo: remove frame in the loop
function normalize_exit!(frame)
    terminator_bbs = findall(isempty.(getfield.(frame.cfg.blocks, :succs)))
    length(terminator_bbs) <= 1 && return nothing
    new_slot = create_slot!(frame)
    add_instruction!(frame, 0, Core.NewvarNode(new_slot))

    n = length(frame.src.code)
    add_instruction!(frame, n, new_slot)
    add_instruction!(frame, n + 1, Core.ReturnNode(Core.SSAValue(n + 1)))
    push!(frame.bb_vartables, nothing)
    offset = 0
    tis = [terminator_index(frame.cfg, tbb) for tbb in terminator_bbs]
    for tbb in tis
        return_index = offset + tbb
        return_ = frame.src.code[return_index]
        @assert(return_ isa Core.ReturnNode)
        exit_bb_start_pos = terminator_index(frame.cfg, length(frame.cfg.blocks))
        offset += if return_.val isa Core.SSAValue
            temp = frame.src.code[return_.val.id]
            frame.src.code[return_.val.id] = Expr(:(=), new_slot, temp)
            frame.src.code[return_index] = Core.GotoNode(exit_bb_start_pos)
            0
        else
            add_instruction!(
                frame, return_index, Core.GotoNode(exit_bb_start_pos); next_bb=false
            )
            frame.src.code[return_index] = Expr(:(=), new_slot, return_.val)
            1
        end
    end
    return frame.cfg = CC.compute_basic_blocks(frame.src.code)
end

#=
    CC.typeinf_local(interp::Reactant.ReactantInterp, frame::CC.InferenceState)

    Specialize type inference to support control flow aware tracing type inferency
    TODO: enable this only for usercode because the new type inference is costly now (several type inference can be needed for a same function)
=#
function CC.typeinf_local(interp::Reactant.ReactantInterp, frame::CC.InferenceState)
    mod = frame.mod
    if @static (VERSION < v"1.12" && VERSION > v"1.11") &&
        has_ancestor(mod, Main) &&
        #is_traced(frame.linfo.specTypes) &&
        !has_ancestor(mod, Core) &&
        !has_ancestor(mod, Base) &&
        !has_ancestor(mod, Reactant)
        @info "auto control flow tracing enabled: $(frame.linfo)"
        normalize_exit!(frame)
        an = Analysis(Tree(nothing, [], Ref{Tree}()), nothing, nothing, nothing, nothing)
        typeinf_local_traced(interp, frame, an)
        @error frame.src
        isempty(an.tree) ||
            (get_meta(interp).traced_tree_map[mi_key(frame.linfo)] = an.tree)
    else
        @invoke CC.typeinf_local(interp::CC.AbstractInterpreter, frame::CC.InferenceState)
    end
end

function update_context!(an::Analysis, currbb::Int)
    isnothing(an.pending_tree) && return nothing
    currbb in an.pending_tree.node || return nothing
    an.tree = an.pending_tree
    return an.pending_tree = nothing
end

function handle_different_branches() end

#=
    typeinf_local_traced(interp::ReactantInterp, frame::CC.InferenceState)

    type infer the `frame` using a Reactant interpreter; notably detect traced control-flow and upgrade traced slot 
=#
function typeinf_local_traced(
    interp::Reactant.ReactantInterp, frame::CC.InferenceState, an::Analysis
)
    @assert !CC.is_inferred(frame)
    frame.dont_work_on_me = true # mark that this function is currently on the stack
    W = frame.ip
    ssavaluetypes = frame.ssavaluetypes
    bbs = frame.cfg.blocks
    nbbs = length(bbs)
    𝕃ᵢ = CC.typeinf_lattice(interp)

    currbb = frame.currbb
    if currbb != 1
        currbb = frame.currbb = CC._bits_findnext(W.bits, 1)::Int # next basic block
    end

    states = frame.bb_vartables
    init_state = CC.copy(states[currbb])
    currstate = CC.copy(states[currbb]::CC.VarTable)
    terminal_block_if::Union{Nothing,IfStructure} = nothing

    while currbb <= nbbs
        CC.delete!(W, currbb)
        bbstart = first(bbs[currbb].stmts)
        bbend = last(bbs[currbb].stmts)
        currpc = bbstart - 1
        terminal_block_if =
            if an.tree.node isa IfStructure && is_terminal_bb(an.tree, currbb)
                an.tree.node
            else
                nothing
            end
        update_context!(an, currbb)
        up_tree!(an, currbb)
        @warn frame.linfo currbb an.tree.node get_root(an.tree)
        while currpc < bbend
            currpc += 1
            frame.currpc = currpc
            CC.empty_backedges!(frame, currpc)
            stmt = frame.src.code[currpc]
            # If we're at the end of the basic block ...
            if currpc == bbend
                # Handle control flow
                if isa(stmt, Core.GotoNode)
                    succs = bbs[currbb].succs
                    @assert length(succs) == 1
                    nextbb = succs[1]
                    ssavaluetypes[currpc] = Any
                    CC.handle_control_backedge!(interp, frame, currpc, stmt.label)
                    CC.add_curr_ssaflag!(frame, CC.IR_FLAG_NOTHROW)
                    @goto branch
                elseif isa(stmt, Core.GotoIfNot)
                    condx = stmt.cond
                    condxslot = CC.ssa_def_slot(condx, frame)
                    condt = CC.abstract_eval_value(interp, condx, currstate, frame)
                    @error condx condxslot condt
                    if add_cf!(an, frame, currbb, currpc, condt)
                        @goto reset_inference
                    end

                    if condt === CC.Bottom
                        ssavaluetypes[currpc] = CC.Bottom
                        CC.empty!(frame.pclimitations)
                        @goto find_next_bb
                    end
                    orig_condt = condt
                    if !(isa(condt, Core.Const) || isa(condt, CC.Conditional)) &&
                        isa(condxslot, Core.SlotNumber)
                        # if this non-`Conditional` object is a slot, we form and propagate
                        # the conditional constraint on it
                        condt = CC.Conditional(
                            condxslot, Core.Const(true), Core.Const(false)
                        )
                    end
                    condval = CC.maybe_extract_const_bool(condt)
                    nothrow = (condval !== nothing) || CC.:(⊑)(𝕃ᵢ, orig_condt, Bool)
                    if nothrow
                        CC.add_curr_ssaflag!(frame, CC.IR_FLAG_NOTHROW)
                    else
                        CC.update_exc_bestguess!(interp, TypeError, frame)
                        CC.propagate_to_error_handler!(currstate, frame, 𝕃ᵢ)
                        CC.merge_effects!(interp, frame, CC.EFFECTS_THROWS)
                    end

                    if !CC.isempty(frame.pclimitations)
                        # we can't model the possible effect of control
                        # dependencies on the return
                        # directly to all the return values (unless we error first)
                        condval isa Bool ||
                            CC.union!(frame.limitations, frame.pclimitations)
                        empty!(frame.pclimitations)
                    end
                    ssavaluetypes[currpc] = Any
                    if condval === true
                        @goto fallthrough
                    else
                        if !nothrow && !CC.hasintersect(CC.widenconst(orig_condt), Bool)
                            ssavaluetypes[currpc] = CC.Bottom
                            @goto find_next_bb
                        end

                        succs = bbs[currbb].succs
                        if length(succs) == 1
                            @assert condval === false || (stmt.dest === currpc + 1)
                            nextbb = succs[1]
                            @goto branch
                        end
                        @assert length(succs) == 2
                        truebb = currbb + 1
                        falsebb = succs[1] == truebb ? succs[2] : succs[1]
                        if condval === false
                            nextbb = falsebb
                            CC.handle_control_backedge!(interp, frame, currpc, stmt.dest)
                            @goto branch
                        end
                        # We continue with the true branch, but process the false
                        # branch here.
                        if isa(condt, CC.Conditional)
                            else_change = CC.conditional_change(
                                𝕃ᵢ, currstate, condt.elsetype, condt.slot
                            )
                            if else_change !== nothing
                                false_vartable = CC.stoverwrite1!(
                                    copy(currstate), else_change
                                )
                            else
                                false_vartable = currstate
                            end
                            changed = CC.update_bbstate!(𝕃ᵢ, frame, falsebb, false_vartable)
                            then_change = CC.conditional_change(
                                𝕃ᵢ, currstate, condt.thentype, condt.slot
                            )

                            if then_change !== nothing
                                CC.stoverwrite1!(currstate, then_change)
                            end
                        else
                            changed = CC.update_bbstate!(𝕃ᵢ, frame, falsebb, currstate)
                        end
                        if changed
                            CC.handle_control_backedge!(interp, frame, currpc, stmt.dest)
                            CC.push!(W, falsebb)
                        end
                        @goto fallthrough
                    end
                elseif isa(stmt, Core.ReturnNode)
                    rt = CC.abstract_eval_value(interp, stmt.val, currstate, frame)
                    if CC.update_bestguess!(interp, frame, currstate, rt)
                        CC.update_cycle_worklists!(
                            frame
                        ) do caller::CC.InferenceState, caller_pc::Int
                            # no reason to revisit if that call-site doesn't affect the final result
                            return caller.ssavaluetypes[caller_pc] !== Any
                        end
                    end
                    ssavaluetypes[frame.currpc] = Any
                    @goto find_next_bb
                elseif isa(stmt, Core.EnterNode)
                    ssavaluetypes[currpc] = Any
                    CC.add_curr_ssaflag!(frame, CC.IR_FLAG_NOTHROW)
                    if isdefined(stmt, :scope)
                        scopet = CC.abstract_eval_value(
                            interp, stmt.scope, currstate, frame
                        )
                        handler = frame.handlers[frame.handler_at[frame.currpc + 1][1]]
                        @assert handler.scopet !== nothing
                        if !CC.:(⊑)(𝕃ᵢ, scopet, handler.scopet)
                            handler.scopet = CC.tmerge(𝕃ᵢ, scopet, handler.scopet)
                            if isdefined(handler, :scope_uses)
                                for bb in handler.scope_uses
                                    push!(W, bb)
                                end
                            end
                        end
                    end
                    @goto fallthrough
                elseif CC.isexpr(stmt, :leave)
                    ssavaluetypes[currpc] = Any
                    @goto fallthrough
                end
                # Fall through terminator - treat as regular stmt
            end

            # Process non control-flow statements
            (; changes, rt, exct) = CC.abstract_eval_basic_statement(
                interp, stmt, currstate, frame
            )
            if !CC.has_curr_ssaflag(frame, CC.IR_FLAG_NOTHROW)
                if exct !== Union{}
                    CC.update_exc_bestguess!(interp, exct, frame)
                    # TODO: assert that these conditions match. For now, we assume the `nothrow` flag
                    # to be correct, but allow the exct to be an over-approximation.
                end
                CC.propagate_to_error_handler!(currstate, frame, 𝕃ᵢ)
            end

            if !isnothing(terminal_block_if)
                #check if both branches handle correctly the slot, otherwise an upgrade call must be inserted before the if
                # if b c = 10 end => c is present only in one branch
                if !(stmt isa Core.SlotNumber)
                    #consider only if expression slots.
                    terminal_block_if = nothing
                else
                    if rt isa Union
                        union_types = Base.uniontypes(rt)
                        if length(union_types) > 1
                            #TODO: proper check: must change an union only in case like this: Union{Int, Traced{Int}}
                            if length(is_traced.(union_types)) == 1
                                stmt in terminal_block_if.unbalanced_slots &&
                                    error("cannot support $stmt : $rt")
                                push!(terminal_block_if.unbalanced_slots, stmt)
                                if_head_block = terminal_block_if.header_bb
                                terminator_pos = frame.cfg.blocks[if_head_block].stmts.stop
                                upgrade_pos = terminator_pos - 2
                                add_instruction!(frame, upgrade_pos, stmt)
                                @error "missing slot in if branch: upgrade $stmt : $rt"
                                @error frame.src
                                add_instruction!(
                                    frame,
                                    upgrade_pos + 1,
                                    Expr(
                                        :(=),
                                        stmt,
                                        Expr(
                                            :call,
                                            GlobalRef(@__MODULE__, :upgrade),
                                            Core.SSAValue(upgrade_pos + 1),
                                        ),
                                    ),
                                )
                                invalidate_slot_definition_analysis!(an)
                                @goto reset_inference
                            end
                        end
                    end
                end
            end

            #upgrade maybe for loop here: eagerly restart type inference if we detect an traced type
            #NOTE: must be placed before CC.Bottom check: in a traced context, an iterator with invalid arguments still should be upgraded
            upgrade_result = check_and_upgrade_slot!(an, frame, stmt, rt, currstate)
            slot_state = first(upgrade_result)
            if slot_state === UpgradeDefinition #Slot Upgrade ...
                bbs = frame.cfg.blocks
                bbend = last(bbs[currbb].stmts)
                @goto reset_inference
                continue
            elseif slot_state === UpgradeDefinitionGlobal
                @goto reset_inference
            elseif slot_state === UpgradeLocally
                @goto reset_inference
            end

            if rt === CC.Bottom
                ssavaluetypes[currpc] = CC.Bottom
                # Special case: Bottom-typed PhiNodes do not error (but must also be unused)
                if isa(stmt, Core.PhiNode)
                    continue
                end
                @goto find_next_bb
            end

            #Slot upgrade must be placed before any slot/ssa table change
            if changes !== nothing
                CC.stoverwrite1!(currstate, changes)
            end
            if rt === nothing
                ssavaluetypes[currpc] = Any
                continue
            end

            if can_upgrade_loop(an, rt)
                rewrite_loop_stack!(an, frame, states, currstate)
                @goto reset_inference
            end

            # IMPORTANT: set the type
            CC.record_ssa_assign!(𝕃ᵢ, currpc, rt, frame)
        end # while currpc < bbend

        # Case 1: Fallthrough termination
        begin
            @label fallthrough
            nextbb = currbb + 1
        end

        # Case 2: Directly branch to a different BB
        begin
            @label branch
            if CC.update_bbstate!(𝕃ᵢ, frame, nextbb, currstate)
                CC.push!(W, nextbb)
            end
        end

        # Case 3: Control flow ended along the current path (converged, return or throw)
        begin
            @label find_next_bb
            currbb = frame.currbb = CC._bits_findnext(W.bits, 1)::Int # next basic block
            currbb == -1 && break # the working set is empty
            currbb > nbbs && break
            nexttable = states[currbb]
            if nexttable === nothing
                CC.init_vartable!(currstate, frame)
            else
                CC.stoverwrite!(currstate, nexttable)
            end
        end

        begin
            continue
            @label reset_inference
            CC.empty!(W)
            currbb = 1
            frame.currbb = 1
            currpc = 1
            frame.currpc = 1
            reset_slot!(states)
            an.tree = get_root(an.tree)
            states[1] = copy(init_state)
            currstate = copy(init_state)
            for i in eachindex(frame.ssavaluetypes)
                frame.ssavaluetypes[i] = CC.NotFound()
            end
            bbs = frame.cfg.blocks
            nbbs = length(bbs)
            ssavaluetypes = frame.ssavaluetypes
        end
    end # while currbb <= nbbs
    @lk an
    frame.dont_work_on_me = false
    return nothing
end
