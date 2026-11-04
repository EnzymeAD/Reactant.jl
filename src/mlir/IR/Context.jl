@checked struct Context
    ref::API.MlirContext
end

"""
    Context(registry=DialectRegistry(); threading = false)

Creates an MLIR context.
"""
function Context(registry=DialectRegistry(); threading::Bool=false)
    ctx = Context(mark_alloc(API.mlirContextCreateWithRegistry(registry, threading)))
    register_live!(ctx)
    return ctx
end

"""
    dispose(ctx::Context)

Disposes the given context and releases its resources.
After calling this function, the context must not be used anymore.
"""
function dispose(ctx::Context)
    # deactivate(ctx)
    unregister_live!(ctx)
    return mark_dispose(API.mlirContextDestroy, ctx)
end

# The contexts created here that are not yet disposed, each with a generation
# that tells it apart from a later one at the same address, so that an object
# of a disposed context (a location in an error shown after its compilation
# failed, say) can be shown without reading freed memory.
const LIVE_CONTEXTS = Dict{Ptr{Cvoid},UInt}()
const LIVE_CONTEXTS_LOCK = ReentrantLock()
const CONTEXT_GENERATION = Ref{UInt}(0)

function register_live!(ctx::Context)
    @lock LIVE_CONTEXTS_LOCK begin
        LIVE_CONTEXTS[ctx.ref.ptr] = (CONTEXT_GENERATION[] += 1)
    end
    return nothing
end

function unregister_live!(ctx::Context)
    @lock LIVE_CONTEXTS_LOCK delete!(LIVE_CONTEXTS, ctx.ref.ptr)
    return nothing
end

# The generation of the live context at `ref`, or 0 when it is not one made here.
function live_generation(ref::API.MlirContext)
    return @lock LIVE_CONTEXTS_LOCK get(LIVE_CONTEXTS, ref.ptr, UInt(0))
end

Base.cconvert(::Core.Type{API.MlirContext}, c::Context) = c
Base.unsafe_convert(::Core.Type{API.MlirContext}, c::Context) = mark_use(c).ref

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a, b)

function enable_multithreading!(enable::Bool=true; context::Context=current_context())
    API.mlirContextEnableMultithreading(context, enable)
    return context
end

# Global state

# to simplify the API, we maintain a stack of contexts in task local storage
# and pass them implicitly to MLIR API's that require them.
function activate(ctx::Context)
    stack = get!(task_local_storage(), :mlir_context_stack) do
        return Context[]
    end::Vector{Context}
    Base.push!(stack, ctx)
    return nothing
end

function deactivate(ctx::Context)
    current_context() == ctx || error("Deactivating wrong context")
    return Base.pop!(task_local_storage(:mlir_context_stack)::Vector{Context})
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

macro with_context(ctx, body)
    quote
        ctx = $(esc(ctx))
        $activate(ctx)
        try
            $(esc(body))
        finally
            $deactivate(ctx)
        end
    end
end
