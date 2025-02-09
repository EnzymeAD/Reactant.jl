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

function execute(exec::LoadedExecutable, args::NTuple{N,Ptr{Cvoid}}, donated_mask::NTuple{N,UInt8}, ::Val{n_results}) where {N,n_results}
    results = Ref{NTuple{n_results, Ptr{Cvoid}}}()
    has_future = Ref{UInt8}()
    status = Ref{NTuple{1, Ptr{Cvoid}}}() # unused right now

    args = Base.RefValue(args)
    donated_mask = Base.RefValue(donated_mask)

    GC.@preserve exec args donated_mask results has_future status begin
        @ccall MLIR.API.mlir_c.ifrt_Execute(
            exec.ptr::Ptr{Cvoid},
            N::Cint,
            args::Ptr{Cvoid},
            donated_mask::Ptr{Cvoid},
            n_results::Cint,
            Base.unsafe_convert(Ptr{Cvoid}, results)::Ptr{Cvoid},
            has_future::Ptr{Cvoid},
            status::Ptr{Cvoid},
        )::Cvoid
    end

    @assert has_future[] == true

    results = results[]

    return ntuple(Val(n_results)) do i
        return Array(results[i])
    end
end
