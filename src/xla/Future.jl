@inline function free_future(future)
    @ccall MLIR.API.mlir_c.FreeFuture(future.future::Ptr{Cvoid})::Cvoid
end

mutable struct Future
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        # @assert future != C_NULL
        return finalizer(free_future, new(future))
    end
end

function is_ready(future::Future)
    GC.@preserve future begin
        return (@ccall MLIR.API.mlir_c.FutureIsReady(future.future::Ptr{Cvoid})::UInt8) != 0
    end
end

@inline function await(future::Future)::Nothing
    GC.@preserve future begin
        @ccall MLIR.API.mlir_c.FutureAwait(future.future::Ptr{Cvoid})::Cvoid
    end
    return nothing
end
