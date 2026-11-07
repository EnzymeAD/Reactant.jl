# Resolve operand locations once per method, rather than dispatching on Julia
# IR nodes at every statement. Globals remain reads: a non-const binding can
# change without advancing the method world.
@enum OperandKind::UInt8 LITERAL SSA ARGUMENT BLOCK_ARGUMENT GLOBAL UNDEFINED

struct PreparedOperand
    kind::OperandKind
    index::Int
    value::Any
end

prepare_operand(@nospecialize(x)) = PreparedOperand(LITERAL, 0, x)
prepare_operand(x::Core.SSAValue) = PreparedOperand(SSA, x.id, nothing)
prepare_operand(x::Core.Argument) = PreparedOperand(ARGUMENT, x.n, nothing)
prepare_operand(x::BlockArgument) = PreparedOperand(BLOCK_ARGUMENT, x.id, nothing)
prepare_operand(x::Core.PiNode) = prepare_operand(x.val)
prepare_operand(x::QuoteNode) = PreparedOperand(LITERAL, 0, x.value)
prepare_operand(x::GlobalRef) = PreparedOperand(GLOBAL, 0, x)
prepare_operand(x::Undef) = PreparedOperand(UNDEFINED, 0, x)

const NO_VALUE = prepare_operand(nothing)
const NO_OPERANDS = PreparedOperand[]

# OTHER statements are emitted by dispatch on the statement itself.
@enum StatementKind::UInt8 VALUE CALL NEW OTHER

struct PreparedStatement
    kind::StatementKind
    value::PreparedOperand              # the value, the callee or the type
    operands::Vector{PreparedOperand}   # the arguments or the fields
    traced::Bool                        # a call inferred to return a traced value
end
PreparedStatement(kind, value, operands) = PreparedStatement(kind, value, operands, false)

@enum TerminatorKind::UInt8 RETURN YIELD CONTINUE BREAK CONDITION UNREACHABLE UNKNOWN

struct PreparedTerminator
    kind::TerminatorKind
    value::PreparedOperand
    operands::Vector{PreparedOperand}
end

struct PreparedBlock
    statements::Vector{PreparedStatement}
    terminator::PreparedTerminator
end

function prepare_statement(@nospecialize(stmt), @nospecialize(type))
    if stmt isa Expr && stmt.head in (:call, :invoke, :new)
        first = stmt.head === :invoke ? 2 : 1
        args = PreparedOperand[
            prepare_operand(stmt.args[i]) for i in (first + 1):length(stmt.args)
        ]
        kind = stmt.head === :new ? NEW : CALL
        traced = kind === CALL && involves_reactant(CC.widenconst(type))
        return PreparedStatement(kind, prepare_operand(stmt.args[first]), args, traced)
    elseif stmt isa Union{
        Expr,
        ControlFlowOp,
        Core.PhiNode,
        Core.GotoNode,
        Core.GotoIfNot,
        Core.PhiCNode,
        Core.UpsilonNode,
        Core.EnterNode,
        Core.ReturnNode,
    }
        return PreparedStatement(OTHER, NO_VALUE, NO_OPERANDS)
    end
    return PreparedStatement(VALUE, prepare_operand(stmt), NO_OPERANDS)
end

function prepare_terminator(@nospecialize(t))
    if t isa Core.ReturnNode
        isdefined(t, :val) || return PreparedTerminator(UNREACHABLE, NO_VALUE, NO_OPERANDS)
        return PreparedTerminator(RETURN, prepare_operand(t.val), NO_OPERANDS)
    elseif t isa YieldOp
        return PreparedTerminator(YIELD, NO_VALUE, map(prepare_operand, t.values))
    elseif t isa ContinueOp
        return PreparedTerminator(CONTINUE, NO_VALUE, map(prepare_operand, t.values))
    elseif t isa BreakOp
        return PreparedTerminator(BREAK, NO_VALUE, map(prepare_operand, t.values))
    elseif t isa ConditionOp
        operands = map(prepare_operand, t.args)
        return PreparedTerminator(CONDITION, prepare_operand(t.condition), operands)
    end
    return PreparedTerminator(UNKNOWN, NO_VALUE, NO_OPERANDS)
end

function prepare(sci::StructuredIRCode)
    prepared = IdDict{Block,PreparedBlock}()
    for block in eachblock(sci.entry)
        body = block.body
        statements = PreparedStatement[
            prepare_statement(s, t) for (s, t) in zip(body.stmts, body.types)
        ]
        prepared[block] = PreparedBlock(statements, prepare_terminator(block.terminator))
    end
    return prepared
end
