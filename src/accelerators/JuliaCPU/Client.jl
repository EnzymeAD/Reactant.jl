mutable struct JuliaClient <: Client
    devices::Vector{Device}
    memories::Vector{Memory}
end

function JuliaClient()
    devs = Vector{Device}(undef, 1)
    mems = Vector{Memory}(undef, 1)

    dev = JuliaDevice(JuliaDeviceDescription(), -1, mems)
    mem = JuliaMemory(0, "device", devs)

    devs[1] = dev
    mems[1] = mem

    return JuliaClient(devs, mems)
end

platform_name(::JuliaClient) = "julia-cpu"
platform_version(::JuliaClient) = string(VERSION)
process_index(::JuliaClient) = 0
devices(client::JuliaClient) = client.devices
addressable_devices(client::JuliaClient) = devices(client)
addressable_memories(client::JuliaClient) = client.memories

# XLA tolerates an error here (it only fails if the topology is used later), but not a missing method
function topology_description(::JuliaClient)
    throw(unimplemented("`JuliaClient` has no topology description yet"))
end

function buffer_from_host_buffer(
    ::JuliaClient,
    dev::JuliaDevice,
    mem::JuliaMemory,
    data::Ptr,
    _eltype,
    _size,
    host_layout,
    _layout,
    _host_buffer_semantics,
)
    _elsize = pjrt_elsize(_eltype)
    _ndims = length(_size)

    size_in_bytes = prod(_size; init=_elsize)
    minor_to_major = collect(_ndims:-1:1)

    # TODO support strided host data, layout and host buffer semantics

    # only dense major-to-minor host data is supported; strides of size-1 dims are irrelevant
    if !isnothing(host_layout)
        stride = _elsize
        for i in _ndims:-1:1
            _size[i] != 1 &&
                host_layout.byte_strides[i] != stride &&
                throw(unimplemented("only dense major-to-minor host data is supported"))
            stride *= _size[i]
        end
    end

    # relayout is not supported, so `device_layout` must be dense major-to-minor
    # if device_layout != C_NULL
    #     layout_type = unsafe_load(device_layout.type)
    #     layout_type != CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled && throw(unimplemented())

    #     tiled = unsafe_load(device_layout.tiled)
    #     tiled.num_tiles != 0 && throw(unimplemented())
    #     tiled.minor_to_major_size != num_dims && throw(invalid_argument())
    #     for i in 1:num_dims
    #         unsafe_load(tiled.minor_to_major, i) != num_dims - i && throw(unimplemented())
    #     end
    # end

    # the copy is synchronous, so all `host_buffer_semantics` are honored
    new_data = Libc.malloc(size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())

    Libc.memcpy(new_data, data, size_in_bytes)

    return JuliaBuffer(new_data, size_in_bytes, _eltype, _size, minor_to_major, dev, mem)
end

# like `buffer_from_host_buffer`, buffers are dense in major-to-minor order and `layout` is not checked
function create_uninitialized_buffer(
    ::JuliaClient, dev::JuliaDevice, mem::JuliaMemory, _eltype, _size, _layout
)
    size_in_bytes = prod(_size; init=pjrt_elsize(_eltype))
    minor_to_major = collect(length(_size):-1:1)
    # `malloc(0)` may return `C_NULL`, which would mark the buffer as deleted
    data = Libc.malloc(max(size_in_bytes, 1))
    data == C_NULL && throw(resource_exhausted())
    return JuliaBuffer(data, size_in_bytes, _eltype, _size, minor_to_major, dev, mem)
end

function compile(client::JuliaClient, code, options)
    # `code` lives in a context owned by the caller, so only copies of its contents are kept
    main = IR.lookup(IR.SymbolTable(code), "main")
    isnothing(main) && throw(invalid_argument("program has no `main` function"))
    ftype = IR.FunctionType(main)
    output_types = map(1:IR.nresults(ftype)) do i
        IR.julia_type(eltype(IR.result(ftype, i)))
    end

    name_attr = IR.getattr(IR.Operation(code), "sym_name")
    _name = isnothing(name_attr) ? "main" : String(name_attr)

    exe = JuliaExecutable(_name, string(code), output_types, options)
    if nreplicas(exe) * npartitions(exe) != 1
        throw(unimplemented("`JuliaClient` only supports single-device executables"))
    end

    # run on the device given by the device assignment, or on the first one if there is none
    build_options = options.executable_build_options
    assignment = isnothing(build_options) ? nothing : build_options.device_assignment
    dev = if isnothing(assignment)
        first(client.devices)
    else
        dev_id = only(only(assignment.computation_devices).replica_device_ids)
        idx = findfirst(d -> id(description(d)) == dev_id, client.devices)
        if isnothing(idx)
            throw(invalid_argument("device $dev_id is not addressable by `JuliaClient`"))
        end
        client.devices[idx]
    end

    # translate `@main` op by op into the body of an `execute` method specific to this executable.
    # buffers are row-major, so they are seen from Julia as column-major arrays with reversed dimensions.
    block = IR.first_block(IR.region(main, 1))
    nargs = IR.nargs(block)
    vars = Dict{Ptr{Cvoid},Symbol}()
    body = quote
        length(arguments) == 1 || throw(
            invalid_argument(
                string("expected arguments for 1 device, got ", length(arguments))
            ),
        )
        isnothing(dev) ||
            dev === loaded_exe.devices[1] ||
            throw(invalid_argument("cannot execute on another device"))
        buffers = only(arguments)
        length(buffers) == $nargs || throw(
            invalid_argument(
                string("expected ", $nargs, " arguments, got ", length(buffers))
            ),
        )
    end

    # wrap operand buffers into Array without copying
    # TODO support wrapping other MLIR types apart of `tensor`
    for i in 1:nargs
        arg = IR.argument(block, i)
        T = IR.julia_type(eltype(IR.type(arg)))
        dims = reverse(size(IR.type(arg)))
        var = vars[arg.ref.ptr] = Symbol("%arg", i - 1)
        push!(
            body.args,
            quote
                isdeleted(buffers[$i]) &&
                    throw(invalid_argument(string("argument ", $i, " is deleted")))
                $var = unsafe_wrap(Array, Ptr{$T}((buffers[$i]::JuliaBuffer).data), $dims)
            end,
        )
    end

    # each op becomes a call to its handler
    outputs = codegen!(body, block, vars, Ref(0))

    # outputs are copied into new buffers, since they may alias inputs or embedded constants
    # TODO support buffer donation?
    push!(
        body.args,
        quote
            out_dev = loaded_exe.devices[1]
            out_mem = default_memory(out_dev)
            results = Buffer[]
            for out in ($(outputs...),)
                nbytes = sizeof(out::Array)
                data = Libc.malloc(max(nbytes, 1))
                data == C_NULL && throw(resource_exhausted())
                GC.@preserve out unsafe_copyto!(
                    Ptr{UInt8}(data), Ptr{UInt8}(pointer(out)), nbytes
                )
                dims = collect(Int64, reverse(size(out)))
                minor_to_major = collect(Int64, ndims(out):-1:1)
                push!(
                    results,
                    JuliaBuffer(
                        data,
                        nbytes,
                        pjrt_eltype(eltype(out)),
                        dims,
                        minor_to_major,
                        out_dev,
                        out_mem,
                    ),
                )
            end
            return [results]
        end,
    )

    tag = gensym(:main)
    @eval function execute(
        loaded_exe::JuliaLoadedExecutable{$(QuoteNode(tag))},
        arguments::Vector{Vector{Buffer}},
        dev,
        _context,
    )
        return $body
    end

    # single-device executables run as replica 0 of partition 0
    logical_ids = [CAPI.PJRT_LogicalDeviceIds(0, 0)]
    return JuliaLoadedExecutable{tag}(exe, Device[dev], logical_ids, assignment, false)
end
