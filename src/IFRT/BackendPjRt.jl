# IFRT-PjRt backend
struct PjRtBackend <: Backend end

struct PjRtTuple <: AbstractTuple
    ptr::Ptr{Cvoid}
    function PjRtTuple(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtTuple)
    @ccall libxla.ifrt_pjrt_tuple_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtTuple) = x.ptr

struct PjRtMemory <: AbstractMemory
    ptr::Ptr{Cvoid}
    function PjRtMemory(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtMemory)
    @ccall libxla.ifrt_pjrt_memory_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtMemory) = x.ptr

struct PjRtDevice <: AbstractDevice
    ptr::Ptr{Cvoid}
    function PjRtDevice(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtDevice)
    @ccall libxla.ifrt_pjrt_device_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtDevice) = x.ptr

struct PjRtArray <: AbstractArray
    ptr::Ptr{Cvoid}
    function PjRtArray(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtArray)
    @ccall libxla.ifrt_pjrt_array_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtArray) = x.ptr

struct PjRtTopology <: AbstractTopology
    ptr::Ptr{Cvoid}
    function PjRtTopology(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtTopology)
    @ccall libxla.ifrt_pjrt_topology_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtTopology) = x.ptr

struct PjRtClient <: AbstractClient
    ptr::Ptr{Cvoid}
    function PjRtClient(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtClient)
    @ccall libxla.ifrt_pjrt_client_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtClient) = x.ptr

struct PjRtHostSendAndRecvLoadedHostCallback <: AbstractLoadedHostCallback
    ptr::Ptr{Cvoid}
    function PjRtHostSendAndRecvLoadedHostCallback(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtHostSendAndRecvLoadedHostCallback)
    @ccall libxla.ifrt_pjrt_hostsendandrecv_loadhostcallback_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtHostSendAndRecvLoadedHostCallback) = x.ptr

struct PjRtExecutable <: AbstractExecutable
    ptr::Ptr{Cvoid}
    function PjRtExecutable(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtExecutable)
    @ccall libxla.ifrt_pjrt_executable_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtExecutable) = x.ptr

struct PjRtLoadedExecutable <: AbstractLoadedExecutable
    ptr::Ptr{Cvoid}
    function PjRtLoadedExecutable(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtLoadedExecutable)
    @ccall libxla.ifrt_pjrt_loadedexecutable_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtLoadedExecutable) = x.ptr

struct PjRtCompiler <: AbstractCompiler
    ptr::Ptr{Cvoid}
    function PjRtCompiler(x)
        @assert x != C_NULL
        return new(x)
    end
end

function free(x::PjRtCompiler)
    @ccall libxla.ifrt_pjrt_compiler_free(x::Ptr{Cvoid})::Cvoid
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::PjRtCompiler) = x.ptr

# TODO for PjRt-IFRT backend, implement `ifrt_to_primitive_type` and `ifrt_to_dtype`

function PjRtTuple(client::PjRtClient, values::Vector{Value})
    return PjRtTuple(
        @ccall libxla.ifrt_pjrt_tuple_ctor(
            client::Ptr{Cvoid}, values::Vector{Ptr{Cvoid}}, length(values)::Int
        )::Ptr{Cvoid}
    )
end

function PjRtMemory(client::PjRtClient, mem_space::PjRt.MemorySpace)
    return PjRtMemory(
        @ccall libxla.ifrt_pjrt_memory_ctor(
            client::Ptr{Cvoid}, mem_space::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function client(x::PjRtMemory)
    return PjRtClient(@ccall libxla.ifrt_pjrt_memory_client(x::Ptr{Cvoid})::Ptr{Cvoid})
end

function memory_space(x::PjRtMemory)
    return PjRt.MemorySpace(@ccall libxla.ifrt_pjrt_memory_space(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# PjRtDevice
# TODO add `attributes` argument
# TODO refactor `XLA.Device` to `PjRt.Device` when the move is done
function PjRtDevice(
    client::PjRtClient,
    device_id::Int32,
    kind::String,
    to_string::String,
    debug_string::String,
    pid::Int,
    pjrt_device::XLA.Device,
)
    return PjRtDevice(
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

# TODO refactor `XLA.Device` to `PjRt.Device` when the move is done
function XLA.Device(x::PjRtDevice)
    return PjRt.Device(
        @ccall libxla.ifrt_pjrt_device_pjrt_device(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

# TODO PjRtArray
# TODO PjRtTopology

function PjRtClient(x::XLA.Client)
    return PjRtClient(@ccall libxla.ifrt_pjrt_client_ctor(x::Ptr{Cvoid})::Ptr{Cvoid})
end

# TODO PjRtHostSendAndRecvLoadedHostCallback

# PjRtExecutable
function PjRtExecutable(executable::PjRt.Executable, compile_options::PjRt.CompileOptions)
    return PjRtExecutable(
        @ccall libxla.ifrt_pjrt_executable_ctor(
            executable::Ptr{Cvoid}, compile_options::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

function PjRt.Executable(x::PjRtExecutable)
    return PjRt.Executable(
        @ccall libxla.ifrt_pjrt_executable_pjrt_executable(x::Ptr{Cvoid})::Ptr{Cvoid}
    )
end

# TODO PjRtLoadedExecutable

# PjRtCompiler
function PjRtCompiler(x::PjRtClient)
    return PjRtCompiler(@ccall libxla.ifrt_pjrt_compiler_ctor(x::Ptr{Cvoid})::Ptr{Cvoid})
end
