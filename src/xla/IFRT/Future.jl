mutable struct Future <: XLA.AbstractFuture
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        # TODO(#2235): double free issue?? potentiialy due to wrapper over PJRTFuture?
        # return finalizer(free_future, new(future))
        return new(future)
    end
end

@inline function free_future(future::Future)
    return MLIR.API.ifrt_free_future(future.future)
end

function Base.isready(future::Future)
    return (MLIR.API.ifrt_future_is_ready(future.future)) != 0
end

@inline function Base.wait(future::Future)::Nothing
    MLIR.API.ifrt_future_await(future.future)
    return nothing
end
