module ReactantMetalExt

using Base: unsafe_convert
using Reactant: Reactant
using Reactant.MLIR: IR, API
using Reactant.XLA: CAPI
using Reactant.XLA.PJRT: Plugin
using Reactant.Accelerators.JuliaCPU: decode_attribute

using Reactant.XLA.PJRT.Plugin:
    AbstractPlugin,
    Buffer,
    Client,
    Device,
    DeviceDescription,
    Event,
    Executable,
    LoadedExecutable,
    Memory,
    PluginApi,
    TiledMemoryLayout,
    invalid_argument,
    make_api,
    npartitions,
    nreplicas,
    pjrt_elsize,
    resource_exhausted,
    unimplemented

# interface functions extended by this plugin.
# `Plugin.id` is extended by its qualified name, since `id` is the type of the pointers to Objective-C objects
import Reactant.XLA.PJRT.Plugin:
    addressable_device_logical_ids,
    addressable_devices,
    addressable_memories,
    attributes,
    buffer_from_host_buffer,
    bytes_in_use,
    codesize,
    compile,
    compile_options,
    copyto,
    create_client,
    create_uninitialized_buffer,
    default_memory,
    description,
    device,
    device_assignment,
    devices,
    dynamic_dims,
    executable,
    execute,
    fingerprint,
    isaddressable,
    isdeleted,
    isoncpu,
    kind,
    kind_id,
    local_hardware_id,
    memory,
    memory_layout,
    name,
    optimized_program,
    output_eltypes,
    output_sizes,
    pjrt_eltype,
    platform_name,
    platform_version,
    process_index,
    ready_event,
    serialize,
    to_host_buffer!,
    topology_description

using Metal: Metal, MTL, MPS, MPSGraphs
using Metal.MTL: MTLBuffer, MTLDevice
using Metal.MPS: MPSDataType, MPSShape
using Metal.MPSGraphs:
    MPSGraph,
    MPSGraphConvolution3DOpDescriptor,
    MPSGraphPooling4DOpDescriptor,
    MPSGraphRandomOpDescriptor,
    MPSGraphTensor,
    MPSGraphTensorData
using ObjectiveC: @objc, id, nil
using ObjectiveC.Foundation:
    @autoreleasepool,
    NSArray,
    NSData,
    NSDictionary,
    NSInteger,
    NSNumber,
    NSString,
    NSUInteger,
    release

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
    ptr = API.MakeClientFromApi(api, "metal", "METAL", errstr)
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.PJRT.Client(ptr)
end

function __init__()
    Metal.functional() || return nothing

    plugin_api[] = make_api(MetalPlugin())

    Reactant.Accelerators.Registration.register_backend(
        "metal";
        priority=75, # lower than "cpu": most programs need ops or element types that Metal doesn't support yet
        pjrt_initialize_function=make_pjrt_client,
    )

    return nothing
end

end
