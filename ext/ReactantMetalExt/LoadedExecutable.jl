# `inputs` are the placeholders of the arguments of `@main` and `outputs` the tensors that it returns
mutable struct MetalLoadedExecutable <: LoadedExecutable
    const executable::MetalExecutable
    const graph::MPSGraph
    const inputs::Vector{MPSGraphTensor}
    const outputs::Vector{MPSGraphTensor}
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
    nargs = length(loaded_exe.inputs)
    length(buffers) == nargs ||
        throw(invalid_argument("expected $nargs arguments, got $(length(buffers))"))

    out_mem = default_memory(out_dev)
    outputs = Buffer[]
    @autoreleasepool begin
        # argument buffers are fed to the graph without copying
        feeds = Dict{MPSGraphTensor,MPSGraphTensorData}()
        for (i, buffer) in enumerate(buffers)
            buffer isa MetalBuffer ||
                throw(invalid_argument("argument $i is not a `MetalBuffer`"))
            isdeleted(buffer) && throw(invalid_argument("argument $i is deleted"))
            T, dims = exe.input_types[i], exe.input_sizes[i]
            if pjrt_eltype(buffer) != pjrt_eltype(T) || buffer.dims != dims
                throw(invalid_argument("argument $i must be a `$T` buffer of size $dims"))
            end
            feeds[loaded_exe.inputs[i]] = MPSGraphTensorData(
                buffer.data, convert(MPSShape, buffer.dims), T
            )
        end

        # the graph writes its results directly into the output buffers
        results = Dict{MPSGraphTensor,MPSGraphTensorData}()
        for (tensor, T, dims) in zip(loaded_exe.outputs, exe.output_types, exe.output_sizes)
            buffer = MetalBuffer(pjrt_eltype(T), dims, out_dev, out_mem)
            push!(outputs, buffer)
            results[tensor] = MPSGraphTensorData(buffer.data, convert(MPSShape, dims), T)
        end

        # TODO return before the GPU finishes, and make `ready_event` and the host accessors of the outputs wait for it
        queue = Metal.global_queue(out_dev.mtl)
        cmdbuf = MPS.MPSCommandBuffer(queue)
        MPS.encode!(
            cmdbuf,
            loaded_exe.graph,
            NSDictionary(feeds),
            NSDictionary(results),
            nil,
            MPSGraphs.default_exec_desc(),
        )
        MTL.commit!(cmdbuf)
        Metal.synchronize(queue)

        # tensor data retains its buffer, so it's released now instead of whenever the GC finalizes it
        foreach(release, values(feeds))
        foreach(release, values(results))
    end
    return [outputs]
end
