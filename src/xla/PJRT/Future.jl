mutable struct Future <: XLA.AbstractFuture
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        return finalizer(free_future, new(future))
    end
end

@inline function free_future(future::Future)
    GC.@preserve future begin
        MLIR.API.FreeFuture(future.future)
    end
end

function Base.isready(future::Future)
    GC.@preserve future begin
        res = MLIR.API.FutureIsReady(future.future)
    end
    return res != 0
end

@inline function Base.wait(future::Future)::Nothing
    GC.@preserve future begin
        MLIR.API.FutureAwait(future.future)
    end
    return nothing
end
