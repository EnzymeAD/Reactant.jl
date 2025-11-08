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

function _iobuffer_to_hlo_print_options(io::IO)
    get(io, :compact, false) && return Int32(1) # ShortParsable
    get(io, :canonical, false) && return Int32(2) # Canonical
    get(io, :fingerprint, false) && return Int32(3) # Fingerprint
    get(io, :module_fingerprint, false) && return Int32(4) # ModuleFingerprint
    return Int32(0) # Default
end

function Base.show(io::IO, hlo_module::HloModule)
    GC.@preserve hlo_module begin
        str = @ccall MLIR.API.mlir_c.HloModuleToString(
            hlo_module.ptr::Ptr{Cvoid}, _iobuffer_to_hlo_print_options(io)::Int32
        )::Cstring
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end

function Base.parse(::Type{HloModule}, str::AbstractString)
    return HloModule(
        @ccall MLIR.API.mlir_c.parseAndReturnUnverifiedHloModule(str::Cstring)::Ptr{Cvoid}
    )
end

function Base.read(filename::AbstractString, ::Type{HloModule})
    return parse(HloModule, read(filename, String))
end

function Base.getproperty(hlo_module::HloModule, sym::Symbol)
    if sym === :entry_computation
        return HloComputation(
            @ccall MLIR.API.mlir_c.hloModuleGetEntryComputation(
                hlo_module.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
    return getfield(hlo_module, sym)
end
