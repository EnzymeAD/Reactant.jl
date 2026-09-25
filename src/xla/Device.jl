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

const DEVICE_PROPERTIES_CACHE = Dict{Tuple{Int,String},MLIR.API.DeviceProperties}()

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

    jldevprops = Ref{MLIR.API.DeviceProperties}()
    if pname == "cuda"
        GC.@preserve jldevprops begin
            MLIR.API.ReactantCudaDeviceGetProperties(jldevprops, local_hardware_id)
        end
    elseif pname == "rocm"
        if _has_hip_device_properties()
            GC.@preserve jldevprops begin
                MLIR.API.ReactantHipDeviceGetProperties(jldevprops, local_hardware_id)
            end
        else
            # Older builds of libReactantExtra predate the HIP property query;
            # fill CDNA-flavored defaults rather than hand back an unfilled Ref.
            @warn "This build of libReactantExtra cannot query ROCm device \
                   properties; using defaults" maxlog = 1
            jldevprops[] = _default_rocm_device_properties()
        end
    else
        # Erroring beats the alternative: an unfilled Ref would hand the caller
        # (and the cache) a struct of uninitialized memory.
        error("`device_properties` is not implemented for platform: $(pname)")
    end
    DEVICE_PROPERTIES_CACHE[(local_hardware_id, pname)] = jldevprops[]
    return jldevprops[]
end

function _has_hip_device_properties()
    return Libdl.dlsym(
        Reactant_jll.libReactantExtra_handle,
        :ReactantHipDeviceGetProperties;
        throw_error=false,
    ) !== nothing
end

# Conservative CDNA (Instinct-class) defaults, for libReactantExtra builds that
# predate `ReactantHipDeviceGetProperties`. Sizes we cannot guess are 0.
function _default_rocm_device_properties()
    return MLIR.API.DeviceProperties(
        0,                          # totalGlobalMem (unknown)
        65536,                      # sharedMemPerBlock: 64 KiB LDS
        65536,                      # regsPerBlock
        64,                         # warpSize: CDNA wavefront
        1024,                       # maxThreadsPerBlock
        (1024, 1024, 1024),         # maxThreadsDim
        (2147483647, 65536, 65536), # maxGridSize
        2147483647,                 # totalConstMem
        9,                          # major (gfx9xx)
        0,                          # minor
        0,                          # multiProcessorCount (unknown)
        1,                          # canMapHostMemory
        0,                          # l2CacheSize (unknown)
        2048,                       # maxThreadsPerMultiProcessor
    )
end

"""
    device_gcn_arch(device::AbstractDevice)

Return the LLVM target id of a ROCm device, e.g. `"gfx950:sramecc+:xnack-"`.
Falls back to a generic default (with a warning) on builds of libReactantExtra
that cannot query it.
"""
function device_gcn_arch(device::AbstractDevice)
    pname = platform_name(client(device))
    pname == "rocm" || error("`device_gcn_arch` requires a ROCm device, got: $(pname)")

    if Libdl.dlsym(
        Reactant_jll.libReactantExtra_handle,
        :ReactantHipDeviceGetGCNArchName;
        throw_error=false,
    ) !== nothing
        str = MLIR.API.ReactantHipDeviceGetGCNArchName(get_local_hardware_id(device))
        return unsafe_string_and_free(str)
    end
    @warn "This build of libReactantExtra cannot query the ROCm gcn arch; \
           defaulting to gfx942" maxlog = 1
    return "gfx942"
end

function Base.show(io::IO, ::MIME"text/plain", props::MLIR.API.DeviceProperties)
    return print(
        io,
        """
        DeviceProperties
        ----------------
        Total Global Mem: $(_format_bytes(props.totalGlobalMem))
        Shared Mem Per Block: $(_format_bytes(props.sharedMemPerBlock))
        Regs Per Block: $(props.regsPerBlock)
        Warp Size: $(props.warpSize)
        Max Threads Per Block: $(props.maxThreadsPerBlock)
        Max Threads Dim: $(props.maxThreadsDim)
        Max Grid Size: $(props.maxGridSize)
        Total Const Mem: $(_format_bytes(props.totalConstMem))
        Version: $(VersionNumber(props.major, props.minor))
        Multi Processor Count: $(props.multiProcessorCount)
        Can Map Host Memory: $(props.canMapHostMemory)
        L2 Cache Size: $(props.l2CacheSize)
        Max Threads Per Multiprocessor: $(props.maxThreadsPerMultiProcessor)
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
            MLIR.API.CudaGetStreamExecutorDeviceDescription(local_hardware_id)
        )
    else
        error("Unsupported platform: $(panme)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", props::StreamExecutorDeviceDescription)
    GC.@preserve props begin
        str = MLIR.API.deviceDescriptionToString(props.ptr)
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end
