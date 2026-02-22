mutable struct Future <: XLA.AbstractFuture
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        return finalizer(free_future, new(future))
    end
end

@inline function free_future(future::Future)
    return MLIR.API.FreeFuture(future.future)
end

function Base.isready(future::Future)
    return (MLIR.API.FutureIsReady(future.future)) != 0
end

@inline function Base.wait(future::Future)::Nothing
    MLIR.API.FutureAwait(future.future)
    return nothing
end
