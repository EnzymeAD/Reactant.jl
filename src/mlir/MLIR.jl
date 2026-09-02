module MLIR

using ..Reactant

module API
    using CEnum: @cenum
    using Preferences: Preferences
    using Reactant_jll: Reactant_jll
    using Libdl: Libdl

    # MLIR C API
    let
        include("libMLIR_h.jl")
    end

    # Additional functions
    function EnzymeJaXMapSymbol(name, sym)
        @ccall Reactant_jll.libReactantExtra.EnzymeJaXMapSymbol(
            name::Cstring, sym::Ptr{Cvoid}
        )::Cvoid
    end
    function EnzymeJaXMapSymbol(name, sym::Integer)
        @ccall Reactant_jll.libReactantExtra.EnzymeJaXMapSymbol(
            name::Cstring, sym::Int64
        )::Cvoid
    end

    function RegisterEnzymeXLACPUHandler()
        @ccall Reactant_jll.libReactantExtra.RegisterEnzymeXLACPUHandler()::Cvoid
    end

    function RegisterEnzymeXLAGPUHandler()
        @ccall Reactant_jll.libReactantExtra.RegisterEnzymeXLAGPUHandler()::Cvoid
    end

    function registerEnzymeJaXXLAFFI()
        if Libdl.dlsym(
            Reactant_jll.libReactantExtra_handle,
            :registerEnzymeJaXXLAFFI;
            throw_error=false,
        ) === nothing
            @debug "registerEnzymeJaXXLAFFI not found in libReactantExtra, skipping \
                    registration of EnzymeJaXXLAFFI. Update Reactant_jll to resolve this."
            return nothing
        end
        @ccall Reactant_jll.libReactantExtra.registerEnzymeJaXXLAFFI()::Cvoid
    end

    function ifrt_compile_with_proto(
        client, cmod, compile_options_proto::Vector{UInt8}, compile_options_proto_size
    )
        return @ccall Reactant_jll.libReactantExtra.ifrt_compile_with_proto(
            client::Ptr{Client},
            cmod::MlirModule,
            compile_options_proto::Ptr{UInt8},
            compile_options_proto_size::Csize_t,
        )::Ptr{HeldIfrtLoadedExecutable}
    end

    function ClientCompileWithProto(
        client, cmod, compile_options_proto::Vector{UInt8}, compile_options_proto_size
    )
        @ccall Reactant_jll.libReactantExtra.ClientCompileWithProto(
            client::Ptr{PjRtClient},
            cmod::MlirModule,
            compile_options_proto::Ptr{UInt8},
            compile_options_proto_size::Csize_t,
        )::Ptr{PjRtLoadedExecutable}
    end
end # module API

include("IR/IR.jl")
include("Dialects.jl")
include("Highlight.jl")

end # module MLIR
