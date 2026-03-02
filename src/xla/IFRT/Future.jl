mutable struct Future <: XLA.AbstractFuture
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        # TODO(#2235): double free issue?? potentiialy due to wrapper over PJRTFuture?
        # return finalizer(free_future, new(future))
        return new(future)
    end
end

@inline free_future(future::Future) = MLIR.API.ifrt_free_future(future.future)

function Base.isready(future::Future)
    GC.@preserve future begin
        res = MLIR.API.mlir_c.ifrt_future_is_ready(future.future)
    end
    return res != 0
end

@inline function Base.wait(future::Future)::Nothing
    GC.@preserve future begin
        MLIR.API.ifrt_future_await(future.future)
    end
    return nothing
end
