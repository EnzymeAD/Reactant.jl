module JuliaCPU

using Base: unsafe_convert
using ..Reactant: Reactant
using Reactant.MLIR: IR, API
using Reactant.XLA: CAPI

using Reactant.XLA.PJRT.Plugin:
    AbstractPlugin,
    Device,
    DeviceDescription,
    Event,
    Executable,
    LoadedExecutable,
    Memory,
    PluginApi,
    ProfilerExtension,
    TiledMemoryLayout,
    invalid_argument,
    make_api,
    npartitions,
    nreplicas,
    pjrt_elsize,
    resource_exhausted,
    unimplemented

# interface functions extended by this plugin
import Reactant.XLA.PJRT.Plugin:
    Buffer,
    Client,
    Profiler,
    addressable_device_logical_ids,
    addressable_devices,
    addressable_memories,
    attributes,
    bytes_in_use,
    clear_memory_stats!,
    codesize,
    compile,
    compile_options,
    copyto,
    default_memory,
    description,
    device,
    device_assignment,
    devices,
    dynamic_dims,
    executable,
    execute,
    fingerprint,
    id,
    isaddressable,
    isdeleted,
    isoncpu,
    kind,
    kind_id,
    largest_alloc_size,
    local_hardware_id,
    memory,
    memory_layout,
    nallocs,
    name,
    optimized_program,
    output_eltypes,
    peak_bytes_in_use,
    pjrt_eltype,
    platform_name,
    platform_version,
    process_index,
    ready_event,
    serialize,
    start!,
    stop!,
    to_host_buffer!,
    topology_description

using LinearAlgebra: BLAS, BlasFloat, Hermitian, cholesky, issuccess, mul!

include("Memory.jl")
include("DeviceDescription.jl")
include("Device.jl")
include("Buffer.jl")
include("Executable.jl")
include("LoadedExecutable.jl")
include("Ops.jl")
include("Codegen.jl")
include("Client.jl")
include("Plugin.jl")
include("Extensions/Profiler.jl")

const plugin_api = Ref{PluginApi}()

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert node_id == 0
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert isnothing(distributed_runtime_client) "`make_pjrt_client` does not \
                                                    support distributed_runtime_client"
    @assert isnothing(allowed_devices)

    errstr = Ref{Cstring}()
    api = unsafe_convert(Ptr{CAPI.PJRT_Api}, plugin_api[])
    ptr = API.MakeClientFromApi(api, "julia-cpu", "JULIA-CPU", errstr)
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.PJRT.Client(ptr)
end

function make_ifrt_client(; kwargs...)
    pjrt_client = make_pjrt_client(; kwargs...)
    errstr = Ref{Cstring}()
    ptr = GC.@preserve pjrt_client errstr begin
        API.ifrt_pjrt_make_client_with_default_kv_store(
            pjrt_client.client, 0, 1, C_NULL, errstr, "julia-cpu"
        )
    end
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.IFRT.Client(ptr)
end

function __init__()
    plugin_api[] = make_api(HostPlugin(); extensions=[ProfilerExtension])

    Reactant.Accelerators.Registration.register_backend(
        "julia-cpu";
        priority=100,
        pjrt_initialize_function=make_pjrt_client,
        ifrt_initialize_function=make_ifrt_client,
    )

    return nothing
end

end
