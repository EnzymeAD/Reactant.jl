# IFRT-PjRt backend
struct PjRtBackend <: Backend end

# TODO for PjRt-IFRT backend, implement `ifrt_to_primitive_type` and `ifrt_to_dtype`

# PjRtTuple
function Tuple(::PjRtBackend, client::Client)
    return Tuple(
        PjRtBackend(),
        @ccall libxla.ifrt_pjrt_tuple_ctor(
            client::Ptr{Cvoid}, values::Vector{Ptr{Cvoid}}, length(values)::Int
        )::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Tuple)
    @ccall libxla.ifrt_pjrt_tuple_free(x::Ptr{Cvoid})::Cvoid
end

# PjRtMemory
function Memory(client::Client, mem_space::PjRt.MemorySpace)
    @assert backend(client) === PjRtBackend()
    return Memory(
        PjRtBackend(),
        @ccall libxla.ifrt_pjrt_memory_ctor(
            client::Ptr{Cvoid}, mem_space::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Memory)
    @ccall libxla.ifrt_pjrt_memory_free(x::Ptr{Cvoid})::Cvoid
end

function client(::PjRtBackend, x::Memory)
    return Client(
        PjRtBackend(), @ccall libxla.ifrt_pjrt_memory_client(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function memory_space(::PjRtBackend, x::Memory)
    return PjRt.MemorySpace(@ccall libxla.ifrt_pjrt_memory_space(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# PjRtDevice
# TODO add `attributes` argument
# TODO refactor `XLA.Device` to `PjRt.Device` when the move is done
function Device(
    client::Client,
    device_id::Int32,
    kind::String,
    to_string::String,
    debug_string::String,
    pid::Int,
    pjrt_device::XLA.Device,
)
    @assert backend(client) === PjRtBackend()
    return Device(
        PjRtBackend(),
        @ccall libxla.ifrt_pjrt_device_ctor(
            client::Ptr{Cvoid},
            device_id::Int32,
            kind::Cstring,
            to_string::Cstring,
            debug_string::Cstring,
            pid::Int,
            pjrt_device::Ptr{Cvoid},
        )::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Device)
    @ccall libxla.ifrt_pjrt_device_free(x::Ptr{Cvoid})::Cvoid
end

# TODO refactor `XLA.Device` to `PjRt.Device` when the move is done
function XLA.Device(x::Device)
    @assert backend(x) === PjRtBackend()
    return PjRt.Device(
        @ccall libxla.ifrt_pjrt_device_pjrt_device(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

# PjRtArray
function free(::PjRtBackend, x::Array)
    @ccall libxla.ifrt_pjrt_array_free(x::Ptr{Cvoid})::Cvoid
end
# PjRtTopology
function Topology(topology::PjRt.TopologyDescription)
    return Topology(
        PjRtBackend(),
        @ccall libxla.ifrt_pjrt_topology_ctor(
            client::Ptr{Cvoid}, topology::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Topology)
    @ccall libxla.ifrt_pjrt_topology_free(x::Ptr{Cvoid})::Cvoid
end

function PjRt.TopologyDescription(x::Topology)
    @assert backend(x) === PjRtBackend()
    return PjRt.TopologyDescription(
        @ccall libxla.ifrt_pjrt_topology_pjrt_topology(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

# PjRtClient
function Client(x::XLA.Client)
    return Client(
        PjRtBackend(), @ccall libxla.ifrt_pjrt_client_ctor(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Client)
    @ccall libxla.ifrt_pjrt_client_free(x::Ptr{Cvoid})::Cvoid
end

# struct PjRtHostSendAndRecvLoadedHostCallback <: AbstractLoadedHostCallback
#     ptr::Ptr{Cvoid}
#     function PjRtHostSendAndRecvLoadedHostCallback(x)
#         @assert x != C_NULL
#         return new(x)
#     end
# end

# function free(::PjRtBackend, x::HostSendAndRecvLoadedHostCallback)
#     @ccall libxla.ifrt_pjrt_hostsendandrecv_loadhostcallback_free(x::Ptr{Cvoid})::Cvoid
# end

# PjRtExecutable
function Executable(executable::PjRt.Executable)
    return Executable(
        PjRtBackend(),
        @ccall libxla.ifrt_pjrt_executable_ctor(executable::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Executable)
    @ccall libxla.ifrt_pjrt_executable_free(x::Ptr{Cvoid})::Cvoid
end

function PjRt.Executable(x::Executable)
    @assert backend(x) === PjRtBackend()
    return PjRt.Executable(
        @ccall libxla.ifrt_pjrt_executable_pjrt_executable(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

# PjRtLoadedExecutable
function LoadedExecutable(
    client::Client, mlir_mod::MLIR.IR.Module, comp_options::PjRt.CompileOptions
)
    @assert backend(client) === PjRtBackend()
    return LoadedExecutable(
        PjRtBackend(),
        @ccall libxla.ifrt_pjrt_loadedexecutable_ctor(
            client::Ptr{Cvoid}, mlir_mod::Ptr{Cvoid}, comp_options::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::LoadedExecutable)
    @ccall libxla.ifrt_pjrt_loadedexecutable_free(x::Ptr{Cvoid})::Cvoid
end

# PjRtCompiler
function Compiler(::PjRtBackend, x::Client)
    return Compiler(
        PjRtBackend(), @ccall libxla.ifrt_pjrt_compiler_ctor(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

function free(::PjRtBackend, x::Compiler)
    @ccall libxla.ifrt_pjrt_compiler_free(x::Ptr{Cvoid})::Cvoid
end
