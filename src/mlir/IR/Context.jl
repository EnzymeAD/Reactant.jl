struct Context
    context::API.MlirContext

    function Context(context)
        @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
        return new(context)
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
    return context
end

function Context(f::Core.Function)
    ctx = Context()
    try
        f(ctx)
    finally
        dispose!(ctx)
    end
end

Context(threading::Bool) = Context(API.mlirContextCreateWithThreading(threading))
function Context(registry, threading)
    return Context(API.mlirContextCreateWithRegistry(registry, threading))
end

Base.convert(::Core.Type{API.MlirContext}, c::Context) = c.context

# Global state

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(ctx::Context)
    stack = get!(task_local_storage(), :mlir_context_stack) do
        return Context[]
    end
    Base.push!(stack, ctx)
    return nothing
end

function deactivate!(ctx::Context)
    context() == ctx || error("Deactivating wrong context")
    return Base.pop!(task_local_storage(:mlir_context_stack))
end

function dispose!(ctx::Context)
    deactivate!(ctx)
    return API.mlirContextDestroy(ctx.context)
end

function _has_context()
    return haskey(task_local_storage(), :mlir_context_stack) &&
           !Base.isempty(task_local_storage(:mlir_context_stack))
end

function context(; throw_error::Core.Bool=true)
    if !_has_context()
        throw_error && error("No MLIR context is active")
        return nothing
    end
    return last(task_local_storage(:mlir_context_stack))
end

function context!(f, ctx::Context)
    activate!(ctx)
    try
        f()
    finally
        deactivate!(ctx)
    end
end

function with_context(f; allow_use_existing=false)
    delete_context = false
    if allow_use_existing && _has_context()
        ctx = context()
    else
        delete_context = true
        ctx = Context(Reactant.registry[], false)
        Reactant.Compiler.context_gc_vector[ctx] = Vector{
            Union{Reactant.TracedRArray,Reactant.TracedRNumber}
        }(
            undef, 0
        )
        @ccall API.mlir_c.RegisterDialects(ctx::API.MlirContext)::Cvoid
    end

    activate!(ctx)
    result = try
        f(ctx)
    finally
        deactivate!(ctx)
    end

    delete_context && Base.delete!(Reactant.Compiler.context_gc_vector, ctx)

    return result
end

function enable_multithreading!(enable::Bool=true; context::Context=context())
    API.mlirContextEnableMultithreading(context, enable)
    return context
end

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a, b)
