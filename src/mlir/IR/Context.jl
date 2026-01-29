@checked struct Context
    ref::API.MlirContext
end

"""
    Context(registry=DialectRegistry(); threading = false)

Creates an MLIR context.
"""
function Context(registry=DialectRegistry(); threading::Bool=false)
    return Context(mark_alloc(API.mlirContextCreateWithRegistry(registry, threading)))
end

"""
    dispose!(ctx::Context)

Disposes the given context and releases its resources.
After calling this function, the context must not be used anymore.
"""
function dispose(ctx::Context)
    # deactivate!(ctx)
    return mark_dispose(API.mlirContextDestroy, ctx)
end

Base.cconvert(::Core.Type{API.MlirContext}, c::Context) = c
Base.unsafe_convert(::Core.Type{API.MlirContext}, c::Context) = c.ref

# Global state

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate!(ctx::Context)
    stack = get!(task_local_storage(), :mlir_context_stack) do
        return Context[]
    end::Vector{Context}
    Base.push!(stack, ctx)
    return nothing
end

function deactivate!(ctx::Context)
    current_context() == ctx || error("Deactivating wrong context")
    return Base.pop!(task_local_storage(:mlir_context_stack)::Vector{Context})
end

function dispose!(ctx::Context)
    deactivate!(ctx)
    return API.mlirContextDestroy(ctx)
end

function has_context()
    return haskey(task_local_storage(), :mlir_context_stack) &&
           !Base.isempty(task_local_storage(:mlir_context_stack)::Vector{Context})
end

function current_context(; throw_error::Core.Bool=true)
    if !has_context()
        throw_error && error("No MLIR context is active")
        return nothing
    end
    return last(task_local_storage(:mlir_context_stack)::Vector{Context})
end

function with_context(f, ctx::Context)
    activate!(ctx)
    try
        f()
    finally
        deactivate!(ctx)
    end
end

# TODO replace this method on all call sites for the one accepting a context argument
function with_context(f; allow_use_existing=false)
    if allow_use_existing && has_context()
        ctx = current_context()
    else
        ctx = Context(Reactant.registry[])
        @ccall API.mlir_c.RegisterDialects(ctx::API.MlirContext)::Cvoid
    end

    activate!(ctx)
    try
        return f(ctx)
    finally
        deactivate!(ctx)
    end
end

function enable_multithreading!(enable::Bool=true; context::Context=current_context())
    API.mlirContextEnableMultithreading(context, enable)
    return context
end

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a, b)
