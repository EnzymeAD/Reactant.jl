# currently, only supports IFRT-PjRt
mutable struct LoadedExecutable
    ptr::Ptr{Cvoid}

    function LoadedExecutable(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_exec, new(ptr))
    end
end

@inline function free_exec(exec)
    @ccall MLIR.API.mlir_c.ifrt_pjrt_FreeLoadedExecutable(exec.ptr::Ptr{Cvoid})::Cvoid
end

# TODO execute
