module StructuredReactant

using CompilerCaching: CompilerCaching, CacheView

using Reactant: Reactant, Ops, TracedRArray, TracedRNumber, TracedType
using ReactantCore: ReactantCore, MissingTracedValue
using IRStructurizer:
    IRStructurizer,
    StructuredIRCode,
    Block,
    BlockArgument,
    Undef,
    ControlFlowOp,
    IfOp,
    WhileOp,
    ForOp,
    LoopOp,
    YieldOp,
    ContinueOp,
    BreakOp,
    ConditionOp,
    blocks,
    eachblock

const CC = Core.Compiler
const TracedEnum = Reactant.TracedEnum

# The world the compiler runs in, fixed at load so that later package loads do
# not invalidate its native code; user code is inferred in the traced program's.
# `invoke_in_world` clamps the precompilation value to the current world.
const COMPILER_WORLD = Ref{UInt}(typemax(UInt))

function __init__()
    COMPILER_WORLD[] = Base.get_world_counter()
    return nothing
end

include("errors.jl")
public FrontendError

include("interpreter.jl")

include("prepare.jl")

include("code.jl")

include("program.jl")
export structured

include("entrypoints.jl")
public @compile, @jit, @code_hlo, compile, code_hlo

include("emit.jl")

include("intrinsics.jl")

include("regions.jl")
public UNROLL_WARNING

include("precompile.jl")

end
