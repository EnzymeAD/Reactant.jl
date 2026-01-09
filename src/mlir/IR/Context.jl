@checked struct Context
    ref::API.MlirContext
end

"""
    Context(registry = default_registry[]; threading = false)

Creates an MLIR context.

If `isnothing(registry)`, then it will use Reactant's registry of dialects.
If you want to use a custom or empty registry, just pass it as the first argument.

```julia
Context(DialectRegistry())
```
"""
function Context(registry=default_registry[]; threading::Bool=false)
    if isnothing(registry)
        registry = DialectRegistry()
    end
    return Context(mark_alloc(API.mlirContextCreateWithRegistry(registry, threading)))
end

"""
    dispose!(ctx::Context)

Disposes the given context and releases its resources.
After calling this function, the context must not be used anymore.
"""
function dispose!(ctx::Context)
    # deactivate!(ctx)
    return mark_dispose(API.mlirContextDestroy, ctx)
end

Base.cconvert(::Core.Type{API.MlirContext}, c::Context) = mark_use(c).ref

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
    end::Vector{Context}
    Base.push!(stack, ctx)
    return nothing
end

function deactivate!(ctx::Context)
    context() == ctx || error("Deactivating wrong context")
    return Base.pop!(task_local_storage(:mlir_context_stack)::Vector{Context})
end

function has_context()
    return haskey(task_local_storage(), :mlir_context_stack) &&
           !Base.isempty(task_local_storage(:mlir_context_stack)::Vector{Context})
end
const _has_context = has_context

function context(; throw_error::Core.Bool=true)
    if !has_context()
        throw_error && error("No MLIR context is active")
    end
    return last(task_local_storage(:mlir_context_stack)::Vector{Context})
end

"""
    with_context(f, ctx::Context)

Executes function `f` with the given MLIR context `ctx` activated.
"""
function with_context(f, ctx::Context)
    depwarn("`with_context` is deprecated, use `@scope` instead.", :with_context)
    @scope ctx f()
end

# TODO try to remove it
"""
    with_context(f; allow_use_existing=false)

Executes function `f` with an active MLIR context. If `allow_use_existing` is true and there is already an active
context, that context is used. Otherwise, a new context is created for the duration of `f`.
"""
function with_context(f; allow_use_existing=false)
    if allow_use_existing && has_context()
        ctx = context()
    else
        ctx = Context(Reactant.registry[], false)
        @ccall API.mlir_c.RegisterDialects(ctx::API.MlirContext)::Cvoid
    end

    return with_context(f, ctx)
end
