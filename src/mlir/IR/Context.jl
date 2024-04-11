struct Context
    context::API.MlirContext

    function Context(context)
        @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
        new(context)
    end
end

"""
    Context()

Creates an MLIR context and transfers its ownership to the caller.
"""
function Context()
    context = API.mlirContextCreate()
    context = Context(context)
    activate!(context)
    context
end

function Context(f::Core.Function)
    ctx = Context()
    try
        f(ctx)
    finally
        dispose!(ctx)
    end
end

Base.convert(::Core.Type{API.MlirContext}, c::Context) = c.context

# Global state

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(ctx::Context)
    stack = get!(task_local_storage(), :mlir_context_stack) do
        Context[]
    end
    Base.push!(stack, ctx)
    return
end

function deactivate!(ctx::Context)
    context() == ctx || error("Deactivating wrong context")
    Base.pop!(task_local_storage(:mlir_context_stack))
end

function dispose!(ctx::Context)
    deactivate!(ctx)
    API.mlirContextDestroy(ctx.context)
end

_has_context() = haskey(task_local_storage(), :mlir_context_stack) && !Base.isempty(task_local_storage(:mlir_context_stack))

function context(; throw_error::Core.Bool=true)
    if !_has_context()
        throw_error && error("No MLIR context is active")
        return nothing
    end
    last(task_local_storage(:mlir_context_stack))
end

function context!(f, ctx::Context)
    activate!(ctx)
    try
        f()
    finally
        deactivate!(ctx)
    end
end

function enable_multithreading!(enable::Bool=true; context::Context=context())
    API.mlirContextEnableMultithreading(context, enable)
    context
end

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a, b)
