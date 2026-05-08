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

    # GPU topology for AOT compilation with mock devices
    const PjRtTopologyDescription = Cvoid
    const HeldIfrtExecutable = Cvoid

    function ifrt_gpu_topology_create(
        client,
        platform_version::String,
        num_partitions::Int32,
        num_hosts_per_partition::Int32,
        num_devices_per_host::Int32,
    )
        return @ccall mlir_c.ifrt_gpu_topology_create(
            client::Ptr{Cvoid},
            platform_version::Cstring,
            num_partitions::Int32,
            num_hosts_per_partition::Int32,
            num_devices_per_host::Int32,
        )::Ptr{PjRtTopologyDescription}
    end

    function ifrt_topology_dtor(topology)
        return @ccall mlir_c.ifrt_topology_dtor(
            topology::Ptr{PjRtTopologyDescription}
        )::Cvoid
    end

    function ifrt_compile_with_topology(
        client,
        cmod,
        topology,
        compile_options_proto::Vector{UInt8},
        compile_options_proto_size,
    )
        return @ccall mlir_c.ifrt_compile_with_topology(
            client.client::Ptr{Cvoid},
            cmod::MlirModule,
            topology::Ptr{PjRtTopologyDescription},
            compile_options_proto::Ptr{UInt8},
            compile_options_proto_size::Csize_t,
        )::Ptr{HeldIfrtExecutable}
    end

    function ifrt_executable_serialize(exec)
        out_size = Ref{Csize_t}(0)
        ptr = @ccall mlir_c.ifrt_executable_serialize(
            exec::Ptr{HeldIfrtExecutable}, out_size::Ptr{Csize_t}
        )::Cstring
        bytes = unsafe_wrap(Array, Ptr{UInt8}(ptr), out_size[]; own=true)
        return bytes
    end

    function ifrt_executable_dtor(exec)
        return @ccall mlir_c.ifrt_executable_dtor(exec::Ptr{HeldIfrtExecutable})::Cvoid
    end

    function ifrt_deserialize_and_load(
        client,
        serialized_bytes::Vector{UInt8},
        serialized_bytes_size,
        compile_options_proto::Vector{UInt8},
        compile_options_proto_size,
    )
        return @ccall mlir_c.ifrt_deserialize_and_load(
            client.client::Ptr{Cvoid},
            serialized_bytes::Ptr{UInt8},
            serialized_bytes_size::Csize_t,
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
