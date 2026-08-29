module MLIR

using ..Reactant

module API
    using CEnum: @cenum
    using Preferences: Preferences
    using Reactant_jll: Reactant_jll
    using Libdl: Libdl

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
        @ccall mlir_c.registerEnzymeJaXXLAFFI()::Cvoid
    end

    function compile_mhlo_to_llvm_string(
        mhlo_text::String; pass_pipeline::String="", xla_runtime::Bool=false
    )
        out_str_ref = Ref{Ptr{Cchar}}(C_NULL)

        exec_ptr = ccall(
            (:ReactantCompileMhloToLLVM, Reactant_jll.libReactantExtra),
            Ptr{Cvoid},
            (Cstring, Csize_t, Ref{Ptr{Cchar}}, UInt8, Cstring),
            mhlo_text,
            sizeof(mhlo_text),
            out_str_ref,
            UInt8(xla_runtime),
            pass_pipeline,
        )

        if exec_ptr == C_NULL
            error("Reactant XLA compilation to LLVM failed.")
        end

        llvm_ir = ""
        if out_str_ref[] != C_NULL
            llvm_ir = unsafe_string(out_str_ref[])
            Libc.free(out_str_ref[])
        end

        ccall(
            (:ReactantFreeLocalExecutable, Reactant_jll.libReactantExtra),
            Cvoid,
            (Ptr{Cvoid},),
            exec_ptr,
        )

        return llvm_ir
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
include("Highlight.jl")

end # module MLIR
