abstract type AbstractDevice end

function Base.show(io::IO, ::MIME"text/plain", device::D) where {D<:AbstractDevice}
    print(io, "$(parentmodule(D)).Device($(device.device), \"$(string(device))\")")
    return nothing
end

function device end
function get_local_device_id end
function device_kind end
function default_memory end
function memories end
function is_addressable end
function get_local_hardware_id end

"""
    device_ordinal(device::Device)

Given the device, return the corresponding global device ordinal in the client.
"""
function device_ordinal end

function Base.string(device::AbstractDevice)
    _client = client(device)
    pname = platform_name(_client)
    return "$(uppercase(pname)):$(device_ordinal(device)) $(device_kind(device))"
end

# Fallback method, preferably all device implementations overload this
function is_addressable(device::AbstractDevice)
    return device ∈ addressable_devices(client(device))
end

# Keep in sync with API.cpp
struct DeviceProperties
    total_global_mem::Csize_t
    shared_mem_per_block::Csize_t
    regs_per_block::Cint
    warp_size::Cint
    max_threads_per_block::Cint
    max_threads_dim::NTuple{3,Cint}
    max_grid_size::NTuple{3,Cint}
    total_const_mem::Csize_t
    major::Cint
    minor::Cint
    multi_processor_count::Cint
    can_map_host_memory::Cint
    l2_cache_size::Cint
    max_threads_per_multiprocessor::Cint
end

const DEVICE_PROPERTIES_CACHE = Dict{Tuple{Int,String},DeviceProperties}()

"""
    device_properties(device::AbstractDevice)

Get a struct containing device properties. Which exact fields are populated relies on the
underlying device implementation.
"""
function device_properties(device::AbstractDevice)
    pname = platform_name(client(device))
    local_hardware_id = get_local_hardware_id(device)

    if haskey(DEVICE_PROPERTIES_CACHE, (local_hardware_id, pname))
        return DEVICE_PROPERTIES_CACHE[(local_hardware_id, pname)]
    end

    jldevprops = Ref{DeviceProperties}()
    if pname == "cuda"
        GC.@preserve jldevprops begin
            @ccall MLIR.API.mlir_c.ReactantCudaDeviceGetProperties(
                jldevprops::Ptr{Cvoid}, local_hardware_id::Cint
            )::Cvoid
        end
    else
        @warn "`get_properties` not implemented for platform: $(pname)" maxlog = 1
    end
    DEVICE_PROPERTIES_CACHE[(local_hardware_id, pname)] = jldevprops[]
    return jldevprops[]
end

function Base.show(io::IO, ::MIME"text/plain", props::DeviceProperties)
    return print(
        io,
        """
        DeviceProperties
        ----------------
        Total Global Mem: $(_format_bytes(props.total_global_mem))
        Shared Mem Per Block: $(_format_bytes(props.shared_mem_per_block))
        Regs Per Block: $(props.regs_per_block)
        Warp Size: $(props.warp_size)
        Max Threads Per Block: $(props.max_threads_per_block)
        Max Threads Dim: $(props.max_threads_dim)
        Max Grid Size: $(props.max_grid_size)
        Total Const Mem: $(_format_bytes(props.total_const_mem))
        Version: $(VersionNumber(props.major, props.minor))
        Multi Processor Count: $(props.multi_processor_count)
        Can Map Host Memory: $(props.can_map_host_memory)
        L2 Cache Size: $(props.l2_cache_size)
        Max Threads Per Multiprocessor: $(props.max_threads_per_multiprocessor)
        """,
    )
end

# only for streaming executors like CUDA / ROCM
mutable struct StreamExecutorDeviceDescription
    ptr::Ptr{Cvoid}

    function StreamExecutorDeviceDescription(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return new(ptr)
    end
end

function StreamExecutorDeviceDescription(device::AbstractDevice)
    panme = platform_name(client(device))
    local_hardware_id = get_local_hardware_id(device)

    if panme == "cuda"
        return StreamExecutorDeviceDescription(
            @ccall MLIR.API.mlir_c.CudaGetStreamExecutorDeviceDescription(
                local_hardware_id::Int32
            )::Ptr{Cvoid}
        )
    else
        error("Unsupported platform: $(panme)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", props::StreamExecutorDeviceDescription)
    GC.@preserve props begin
        str = @ccall MLIR.API.mlir_c.deviceDescriptionToString(
            props.ptr::Ptr{Cvoid}
        )::Cstring
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end
