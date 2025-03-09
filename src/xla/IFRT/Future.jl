mutable struct Future <: XLA.AbstractFuture
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        # XXX: double free issue?? potentiialy due to wrapper over PJRTFuture?
        # return finalizer(free_future, new(future))
        return new(future)
    end
end

@inline function free_future(future::Future)
    @ccall MLIR.API.mlir_c.ifrt_free_future(future.future::Ptr{Cvoid})::Cvoid
end

function Base.isready(future::Future)
    GC.@preserve future begin
        return (@ccall MLIR.API.mlir_c.ifrt_future_is_ready(
            future.future::Ptr{Cvoid}
        )::UInt8) != 0
    end
end

@inline function Base.wait(future::Future)::Nothing
    GC.@preserve future begin
        @ccall MLIR.API.mlir_c.ifrt_future_await(future.future::Ptr{Cvoid})::Cvoid
    end
    return nothing
end
