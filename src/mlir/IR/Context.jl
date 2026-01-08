struct Context
    context::API.MlirContext

    """
        Context(context::API.MlirContext)

    Wraps a given MlirContext in a Context struct and transfers ownership to the caller.
    """
    function Context(context)
        @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
        return new(context)
    end
end

"""
    Context()

Creates an MLIR context and transfers its ownership to the caller.
"""
Context() = Context(API.mlirContextCreate())

"""
    Context(threading::Bool)

Creates an MLIR context with or without multithreading support.
"""
function Context(threading::Bool)
    return Context(API.mlirContextCreateWithThreading(threading))
end

"""
    Context(registry::DialectRegistry, threading::Bool)

Creates an MLIR context with the given dialect registry and with or without multithreading support.
"""
function Context(registry, threading)
    return Context(API.mlirContextCreateWithRegistry(registry, threading))
end

"""
    dispose!(ctx::Context)

Disposes the given context and releases its resources.
After calling this function, the context must not be used anymore.
"""
function dispose!(ctx::Context)
    @assert !mlirIsNull(ctx.context) "Context already disposed"
    API.mlirContextDestroy(ctx.context)
end

Base.cconvert(::Core.Type{API.MlirContext}, c::Context) = c.context
Base.unsafe_convert(::Core.Type{API.MlirContext}, c::Context) = c.context

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a, b)

function enable_multithreading!(enable::Bool=true; context::Context=context())
    API.mlirContextEnableMultithreading(context, enable)
    return context
end

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

function has_active_context()
    return haskey(task_local_storage(), :mlir_context_stack) &&
           !Base.isempty(task_local_storage(:mlir_context_stack))
end
const _has_context = has_active_context

function context(; throw_error::Core.Bool=true)
    if !has_active_context()
        throw_error && error("No MLIR context is active")
    end
    return last(task_local_storage(:mlir_context_stack))
end

"""
    with_context(f, ctx::Context)

Executes function `f` with the given MLIR context `ctx` activated.
"""
function with_context(f, ctx::Context)
    depwarn("`with_context` is deprecated, use `@scope` instead.", :with_block)
    @scope ctx f()
end

# TODO try to remove it
"""
    with_context(f; allow_use_existing=false)

Executes function `f` with an active MLIR context. If `allow_use_existing` is true and there is already an active
context, that context is used. Otherwise, a new context is created for the duration of `f`.
"""
function with_context(f; allow_use_existing=false)
    delete_context = false
    if allow_use_existing && has_active_context()
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

    result = with_context(f, ctx)

    delete_context && Base.delete!(Reactant.Compiler.context_gc_vector, ctx)

    return result
end
