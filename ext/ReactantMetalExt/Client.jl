mutable struct MetalClient <: Client
    devices::Vector{Device}
    memories::Vector{Memory}
end

function MetalClient()
    devs = Vector{Device}(undef, 1)
    mems = Vector{Memory}(undef, 1)

    mtl = Metal.device()
    dev = MetalDevice(MetalDeviceDescription(0, 0, String(mtl.name)), mtl, mems)
    mem = MetalMemory(0, "device", devs)

    devs[1] = dev
    mems[1] = mem

    return MetalClient(devs, mems)
end

platform_name(::MetalClient) = "metal"
platform_version(::MetalClient) = string("macOS ", Metal.macos_version())
process_index(::MetalClient) = 0
devices(client::MetalClient) = client.devices
addressable_devices(client::MetalClient) = devices(client)
addressable_memories(client::MetalClient) = client.memories

# XLA tolerates an error here (it only fails if the topology is used later), but not a missing method
function topology_description(::MetalClient)
    throw(unimplemented("`MetalClient` has no topology description yet"))
end

function buffer_from_host_buffer(
    ::MetalClient,
    dev::MetalDevice,
    mem::MetalMemory,
    data::Ptr,
    _eltype,
    _size,
    host_layout,
    _layout,
    _host_buffer_semantics,
)
    # TODO support strided host data, layout and host buffer semantics

    # only dense major-to-minor host data is supported; strides of size-1 dims are irrelevant
    if !isnothing(host_layout)
        stride = pjrt_elsize(_eltype)
        for i in length(_size):-1:1
            _size[i] != 1 &&
                host_layout.byte_strides[i] != stride &&
                throw(unimplemented("only dense major-to-minor host data is supported"))
            stride *= _size[i]
        end
    end

    # the copy is synchronous, so all `host_buffer_semantics` are honored
    buffer = MetalBuffer(_eltype, _size, dev, mem)
    Libc.memcpy(pointer(buffer), data, sizeof(buffer))
    return buffer
end

# like `buffer_from_host_buffer`, buffers are dense in major-to-minor order and `layout` is not checked
function create_uninitialized_buffer(
    ::MetalClient, dev::MetalDevice, mem::MetalMemory, _eltype, _size, _layout
)
    return MetalBuffer(_eltype, _size, dev, mem)
end

function compile(client::MetalClient, code, options)
    # `code` lives in a context owned by the caller, so only copies of its contents are kept
    main = IR.lookup(IR.SymbolTable(code), "main")
    isnothing(main) && throw(invalid_argument("program has no `main` function"))
    block = IR.first_block(IR.region(main, 1))
    ftype = IR.FunctionType(main)
    inputs = [tensor_type(IR.type(IR.argument(block, i))) for i in 1:IR.nargs(block)]
    outputs = [tensor_type(IR.result(ftype, i)) for i in 1:IR.nresults(ftype)]

    name_attr = IR.getattr(IR.Operation(code), "sym_name")
    _name = isnothing(name_attr) ? "main" : String(name_attr)

    exe = MetalExecutable(
        _name,
        string(code),
        [input.eltype for input in inputs],
        [input.size for input in inputs],
        [output.eltype for output in outputs],
        [output.size for output in outputs],
        options,
    )
    if nreplicas(exe) * npartitions(exe) != 1
        throw(unimplemented("`MetalClient` only supports single-device executables"))
    end

    # run on the device given by the device assignment, or on the first one if there is none
    build_options = options.executable_build_options
    assignment = isnothing(build_options) ? nothing : build_options.device_assignment
    dev = if isnothing(assignment)
        first(client.devices)
    else
        dev_id = only(only(assignment.computation_devices).replica_device_ids)
        idx = findfirst(d -> Plugin.id(description(d)) == dev_id, client.devices)
        if isnothing(idx)
            throw(invalid_argument("device $dev_id is not addressable by `MetalClient`"))
        end
        client.devices[idx]
    end

    # translate `@main` op by op into a graph, whose placeholders are fed with the argument buffers by `execute`.
    # like MLIR tensors, the tensors of the graph are row-major, so dimensions and shapes are not reversed.
    graph = MPSGraph()
    placeholders = MPSGraphTensor[]
    compiled, input_order, output_order = @autoreleasepool begin
        vars = Dict{Ptr{Cvoid},MPSGraphTensor}()
        for (i, input) in enumerate(inputs)
            placeholder = MPSGraphs.placeholderTensor(
                graph, convert(MPSShape, input.size), input.eltype, string("arg", i - 1)
            )
            vars[IR.argument(block, i).ref.ptr] = placeholder
            push!(placeholders, placeholder)
        end

        # `execute` writes each output into a buffer of its own, so they must be distinct tensors (the same value
        # may be returned twice) and none can be a placeholder
        tensors = MPSGraphTensor[
            MPSGraphs.identityWithTensor(graph, tensor) for
            tensor in codegen!(graph, block, vars)
        ]

        # the graph is compiled once for the shapes of the arguments, with the optimization level of
        # `MPSGraphs.default_exec_desc` (which the handles are tested with)
        feeds = Dict(
            placeholder => MPSGraphShapedType(convert(MPSShape, input.size), input.eltype)
            for (placeholder, input) in zip(placeholders, inputs)
        )
        desc = MPSGraphCompilationDescriptor()
        desc.optimizationLevel = MPSGraphs.MPSGraphOptimizationLevel0
        ptr = @objc [
            graph::id{MPSGraph} compileWithDevice:(MPSGraphDevice(dev.mtl)::id{MPSGraphDevice})
            feeds:(NSDictionary(feeds)::id{NSDictionary})
            targetTensors:(NSArray(tensors)::id{NSArray})
            targetOperations:(nil::id{NSArray})
            compilationDescriptor:(desc::id{MPSGraphCompilationDescriptor})
        ]::id{MPSGraphExecutable}
        foreach(release, values(feeds))
        release(desc)
        if reinterpret(Ptr{Cvoid}, ptr) == C_NULL
            throw(
                Plugin.Error(
                    CAPI.PJRT_Error_Code_INTERNAL, "MPSGraph failed to compile `$_name`"
                ),
            )
        end
        # the executable is autoreleased, so it's retained to outlive this pool
        compiled = MPSGraphExecutable(ptr)
        retain(compiled)

        # the executable takes the arguments in the order of `feedTensors` and returns the results in the order of
        # `targetTensors`, as indices into the arguments and results of `@main`
        input_order = [
            findfirst(p -> p.ptr == t.ptr, placeholders) for t in compiled.feedTensors
        ]
        output_order = [
            findfirst(o -> o.ptr == t.ptr, tensors) for t in compiled.targetTensors
        ]
        if any(isnothing, input_order) ||
            any(isnothing, output_order) ||
            length(output_order) != length(tensors)
            throw(
                Plugin.Error(
                    CAPI.PJRT_Error_Code_INTERNAL,
                    "MPSGraph compiled `$_name` with unknown tensors",
                ),
            )
        end
        (compiled, Vector{Int}(input_order), Vector{Int}(output_order))
    end

    # single-device executables run as replica 0 of partition 0
    logical_ids = [CAPI.PJRT_LogicalDeviceIds(0, 0)]
    return MetalLoadedExecutable(
        exe,
        compiled,
        input_order,
        output_order,
        Device[dev],
        logical_ids,
        assignment,
        false,
    )
end
