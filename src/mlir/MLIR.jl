module MLIR

using ..Reactant

module API
    using CEnum: @cenum
    using Preferences: Preferences
    using Reactant_jll: Reactant_jll

    const mlir_c = if Reactant_jll.is_available()
        Reactant_jll.libReactantExtra
    else
        missing
    end

    # MLIR C API
    let
        include("libMLIR_h.jl")
    end

    # Additional functions
    function EnzymeJaXMapSymbol(name, sym)
        @ccall mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end
    function EnzymeJaXMapSymbol(name, sym::Integer)
        @ccall mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Int64)::Cvoid
    end

    function RegisterEnzymeXLACPUHandler()
        @ccall mlir_c.RegisterEnzymeXLACPUHandler()::Cvoid
    end

    function RegisterEnzymeXLAGPUHandler()
        @ccall mlir_c.RegisterEnzymeXLAGPUHandler()::Cvoid
    end

    function ifrt_compile_with_proto(
        client, cmod, compile_options_proto::Vector{UInt8}, compile_options_proto_size
    )
        return @ccall mlir_c.ifrt_compile_with_proto(
            client::Ptr{Client},
            cmod::MlirModule,
            compile_options_proto::Ptr{UInt8},
            compile_options_proto_size::Csize_t,
        )::Ptr{HeldIfrtLoadedExecutable}
    end

    function ClientCompileWithProto(
        client, cmod, compile_options_proto::Vector{UInt8}, compile_options_proto_size
    )
        @ccall mlir_c.ClientCompileWithProto(
            client::Ptr{PjRtClient},
            cmod::MlirModule,
            compile_options_proto::Ptr{UInt8},
            compile_options_proto_size::Csize_t,
        )::Ptr{PjRtLoadedExecutable}
    end
end # module API

include("IR/IR.jl")

include("Dialects.jl")

end # module MLIR
