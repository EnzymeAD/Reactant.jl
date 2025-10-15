@enum UpgradeSlot NoUpgrade UpgradeLocally UpgradeDefinition UpgradeDefinitionGlobal

@enum State Traced Upgraded Maybe NotTraced

mutable struct ForStructure
    accus::Tuple
    header_bb::Int
    latch_bb::Int
    terminal_bb::Int
    body_bbs::Set{Int}
    state::State
end

struct IfStructure
    ssa_cond
    header_bb::Int
    terminal_bb::Int
    true_bbs::Set{Int}
    false_bbs::Set{Int}
    owned_true_bbs::Set{Int}
    owned_false_bbs::Set{Int}
    legalize::Ref{Bool} #inform that the if traced GotoIfNot can pass type inference
end

mutable struct SlotAnalysis
    slot_stmt_def::Vector{Integer} #0 for argument
    slot_bb_usage::Vector{Set{Int}}
end


CFStructure = Union{IfStructure,ForStructure}
mutable struct Tree
    node::Union{Nothing,Base.uniontypes(CFStructure)...}
    children::Vector{Tree}
    parent::Ref{Tree}
end

Base.isempty(tree::Tree) = isnothing(tree.node) && length(tree.children) == 0

Base.show(io::IO, t::Tree) = begin
    Base.print(io, '(')
    Base.show(io, t.node)
    Base.print(io, ',')
    Base.show(io, t.children)
    Base.print(io, ')')
end

mutable struct Analysis
    tree::Tree
    domtree::Union{Nothing,Vector{CC.DomTreeNode}}
    postdomtree::Union{Nothing,Vector{CC.DomTreeNode}}
    slotanalysis::Union{Nothing,SlotAnalysis}
    pending_tree::Union{Nothing,Tree}
end

