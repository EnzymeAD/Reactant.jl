using ScopedValues

mutable struct Context
    context::API.MlirContext
    owned::Bool

    """
        Context(context::API.MlirContext)

    Wraps a given MlirContext in a Context struct and transfers ownership to the caller.
    """
    function Context(context::API.MlirContext; owned::Bool=true)
        @assert !mlirIsNull(context) "cannot create Context with null MlirContext"
        obj = new(context, owned)
        return finalizer(obj) do ctx
            if ctx.owned
                API.mlirContextDestroy(ctx.context)
                ctx.context = API.MlirContext(C_NULL)
            end
        end
    end
end

"""
    Context()

Creates an MLIR context and transfers its ownership to the caller.
"""
Context() = Context(API.mlirContextCreate(); owned=true)

"""
    Context(threading::Bool)

Creates an MLIR context with or without multithreading support.
"""
function Context(threading::Bool)
    return Context(API.mlirContextCreateWithThreading(threading); owned=true)
end

"""
    Context(registry::DialectRegistry, threading::Bool)

Creates an MLIR context with the given dialect registry and with or without multithreading support.
"""
function Context(registry, threading)
    return Context(API.mlirContextCreateWithRegistry(registry, threading); owned=true)
end

Base.convert(::Core.Type{API.MlirContext}, c::Context) = c.context

const scoped_context = ScopedValue{Union{Nothing,Context}}(nothing)

has_active_context() = !isnothing(scoped_context[])

"""
    context(; throw_error::Bool=true)

Returns the currently active MLIR context.
If no context is active and `throw_error` is true, an error is thrown.
"""
function context(; throw_error::Bool=true)
    ctx = scoped_context[]

    if isnothing(ctx)
        throw_error && error("No MLIR context is active")
    end

    return ctx::Context
end

"""
    with_context(f, ctx::Context)

Executes function `f` with the given MLIR context `ctx` activated.
"""
with_context(f, ctx::Context) = with(f, scoped_context => ctx)

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

    with_context(f, ctx)

    delete_context && Base.delete!(Reactant.Compiler.context_gc_vector, ctx)

    return result
end

function enable_multithreading!(enable::Bool=true; context::Context=context())
    API.mlirContextEnableMultithreading(context, enable)
    return context
end

Base.:(==)(a::Context, b::Context) = API.mlirContextEqual(a, b)
