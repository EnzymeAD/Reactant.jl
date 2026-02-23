mutable struct Future <: XLA.AbstractFuture
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        return finalizer(free_future, new(future))
    end
end

@inline free_future(future::Future) = MLIR.API.FreeFuture(future.future)

Base.isready(future::Future) = MLIR.API.FutureIsReady(future.future)

@inline Base.wait(future::Future) = MLIR.API.FutureAwait(future.future)
