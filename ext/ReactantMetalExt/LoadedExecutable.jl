# `graph` is `@main` compiled for the shapes of its arguments. It takes the arguments `inputs` and returns the results
# `outputs` (indices into those of `@main`), in this order
mutable struct MetalLoadedExecutable <: LoadedExecutable
    const executable::MetalExecutable
    const graph::MPSGraphExecutable
    const inputs::Vector{Int}
    const outputs::Vector{Int}
    const devices::Vector{Device}
    const logical_ids::Vector{CAPI.PJRT_LogicalDeviceIds}
    const device_assignment::Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}
    deleted::Bool
end

executable(loaded_exe::MetalLoadedExecutable) = loaded_exe.executable

function Base.delete!(loaded_exe::MetalLoadedExecutable)
    loaded_exe.deleted = true
    return nothing
end

isdeleted(loaded_exe::MetalLoadedExecutable) = loaded_exe.deleted
addressable_devices(loaded_exe::MetalLoadedExecutable) = loaded_exe.devices
addressable_device_logical_ids(loaded_exe::MetalLoadedExecutable) = loaded_exe.logical_ids
device_assignment(loaded_exe::MetalLoadedExecutable) = loaded_exe.device_assignment

function execute(
    loaded_exe::MetalLoadedExecutable, arguments::Vector{Vector{Buffer}}, dev, _context
)
    length(arguments) == 1 ||
        throw(invalid_argument("expected arguments for 1 device, got $(length(arguments))"))
    out_dev = loaded_exe.devices[1]
    isnothing(dev) ||
        dev === out_dev ||
        throw(invalid_argument("cannot execute on another device"))

    exe = loaded_exe.executable
    buffers = only(arguments)
    nargs = length(exe.input_types)
    length(buffers) == nargs ||
        throw(invalid_argument("expected $nargs arguments, got $(length(buffers))"))
    for (i, buffer) in enumerate(buffers)
        buffer isa MetalBuffer ||
            throw(invalid_argument("argument $i is not a `MetalBuffer`"))
        isdeleted(buffer) && throw(invalid_argument("argument $i is deleted"))
        T, dims = exe.input_types[i], exe.input_sizes[i]
        if pjrt_eltype(buffer) != pjrt_eltype(T) || buffer.dims != dims
            throw(invalid_argument("argument $i must be a `$T` buffer of size $dims"))
        end
    end

    out_mem = default_memory(out_dev)
    outputs = Buffer[
        MetalBuffer(pjrt_eltype(T), dims, out_dev, out_mem) for
        (T, dims) in zip(exe.output_types, exe.output_sizes)
    ]
    @autoreleasepool begin
        # argument buffers are fed to the graph without copying, and it writes its results directly into the output
        # buffers
        feeds = map(loaded_exe.inputs) do i
            MPSGraphTensorData(
                buffers[i].data,
                convert(MPSShape, exe.input_sizes[i]),
                exe.input_types[i],
            )
        end
        results = map(loaded_exe.outputs) do i
            MPSGraphTensorData(
                outputs[i].data,
                convert(MPSShape, exe.output_sizes[i]),
                exe.output_types[i],
            )
        end

        # TODO return before the GPU finishes, and make `ready_event` and the host accessors of the outputs wait for it
        graph = loaded_exe.graph
        queue = Metal.global_queue(out_dev.mtl)
        cmdbuf = MPS.MPSCommandBuffer(queue)
        @objc [
            graph::id{MPSGraphExecutable} encodeToCommandBuffer:(cmdbuf::id{MPS.MPSCommandBuffer})
            inputsArray:(NSArray(feeds)::id{NSArray})
            resultsArray:(NSArray(results)::id{NSArray})
            executionDescriptor:(nil::id{MPSGraphExecutableExecutionDescriptor})
        ]::id{NSArray}
        MTL.commit!(cmdbuf)
        Metal.synchronize(queue)

        # tensor data retains its buffer, so it's released now instead of whenever the GC finalizes it
        foreach(release, feeds)
        foreach(release, results)
    end
    return [outputs]
end
