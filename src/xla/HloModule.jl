mutable struct HloModule
    ptr::Ptr{Cvoid}

    function HloModule(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_hlo_module, new(ptr))
    end
end

function free_hlo_module(hlo_module)
    @ccall MLIR.API.mlir_c.FreeHloModule(hlo_module.ptr::Ptr{Cvoid})::Cvoid
end

function HloModule(mod::MLIR.IR.Module)
    return HloModule(
        @ccall MLIR.API.mlir_c.convertMlirModuleToHloModule(
            mod::MLIR.API.MlirModule
        )::Ptr{Cvoid}
    )
end

function Base.show(io::IO, hlo_module::HloModule)
    GC.@preserve hlo_module begin
        str = @ccall MLIR.API.mlir_c.HloModuleToString(hlo_module.ptr::Ptr{Cvoid})::Cstring
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end
