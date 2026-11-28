@checked struct ThreadPool
    ref::API.MlirLlvmThreadPool
end

"""
    ThreadPool(; max_threads::Integer=0)

Creates an LLVM thread pool with at most `max_threads` concurrent workers.
`max_threads = 0` uses the hardware concurrency.

The pool is not owned by the contexts it is attached to with
[`set_thread_pool!`](@ref): it must outlive them and be disposed explicitly.
"""
function ThreadPool(; max_threads::Integer=0)
    max_threads >= 0 || throw(ArgumentError("max_threads must be non-negative"))
    return ThreadPool(mark_alloc(API.mlirLlvmThreadPoolCreateWithMaxThreads(max_threads)))
end

"""
    dispose(pool::ThreadPool)

Disposes the given thread pool, joining its workers.
All contexts using the pool must be disposed before.
"""
dispose(pool::ThreadPool) = mark_dispose(API.mlirLlvmThreadPoolDestroy, pool)

Base.cconvert(::Core.Type{API.MlirLlvmThreadPool}, pool::ThreadPool) = pool
function Base.unsafe_convert(::Core.Type{API.MlirLlvmThreadPool}, pool::ThreadPool)
    return mark_use(pool).ref
end

"""
    max_concurrency(pool::ThreadPool)

Returns the maximum number of threads in the thread pool.
"""
max_concurrency(pool::ThreadPool) = Int(API.mlirLlvmThreadPoolGetMaxConcurrency(pool))

"""
    set_thread_pool!(pool::ThreadPool; context::Context=current_context())

Makes `context` run its multithreaded work on `pool` and enables multithreading.
The context does not take ownership of `pool`.
"""
function set_thread_pool!(pool::ThreadPool; context::Context=current_context())
    # MLIR requires multithreading to be disabled when setting a pool
    API.mlirContextEnableMultithreading(context, false)
    API.mlirContextSetThreadPool(context, pool)
    return context
end

"""
    num_threads(; context::Context=current_context())

Returns the number of threads `context` uses, or 1 if multithreading is disabled.
"""
function num_threads(; context::Context=current_context())
    return Int(API.mlirContextGetNumThreads(context))
end
