module XLA

import ...MLIR

function RunPassPipeline(pass_pipeline, mod::MLIR.IR.Module)
    GC.@preserve pass_pipeline mod begin
         @ccall MLIR.API.mlir_c.RunPassPipeline(pass_pipeline::Cstring, mod.module_::MLIR.API.MlirModule)::Cvoid
    end
end
mutable struct Client
    client::Ptr{Cvoid}

    function Client(client::Ptr{Cvoid})
        return new(client)
        #@assert client != C_NULL
        #finalizer(new(client)) do client
        #    @ccall MLIR.API.mlir_c.FreeClient(client.client::Ptr{Cvoid})::Cvoid
        #end
    end
end


function to_row_major(x::Array{T, N}) where {T, N}
    return permutedims(x, reverse(Base.OneTo(N)))
end

function to_row_major(x::Vector{T}) where T
    return x
end

function to_row_major(x::Matrix{T}) where T
    return Matrix{T}(transpose(x))
end


function from_row_major(x::Array{T, N}) where {T, N}
    return permutedims(x, reverse(Base.OneTo(N)))
end

function from_row_major(x::Vector{T}) where T
    return x
end

function from_row_major(x::Matrix{T}) where T
    return transpose(x)
end

const cpuclientcount = Ref(0)
function CPUClient(asynchronous=true, node_id=0, num_nodes=1)
    global cpuclientcount
    @assert cpuclientcount[] == 0
    cpuclientcount[]+=1
    client = @ccall MLIR.API.mlir_c.MakeCPUClient(asynchronous::UInt8, node_id::Cint, num_nodes::Cint)::Ptr{Cvoid}
    return Client(client)
end

function GPUClient(node_id=0, num_nodes=1, platform="cuda")
    allowed_devices = [-1]
    GC.@preserve allowed_devices begin
        client = @ccall MLIR.API.mlir_c.MakeGPUClient(node_id::Cint, num_nodes::Cint, pointer(allowed_devices)::Ptr{Cvoid}, length(allowed_devices)::Cint, platform::Cstring)::Ptr{Cvoid}
    end
    return Client(client)
end

const backends = Dict{String, Client}()
const default_backend = Ref{Client}()
const default_device_idx = Ref{Int}(0)
function __init__()
    cpu = CPUClient()
    backends["cpu"] = cpu
    default_backend[] = cpu
end

mutable struct LoadedExecutable
    exec::Ptr{Cvoid}

    function LoadedExecutable(exec::Ptr{Cvoid})
        @assert exec != C_NULL
        finalizer(new(exec)) do exec
            @ccall MLIR.API.mlir_c.ExecutableFree(exec.exec::Ptr{Cvoid})::Cvoid
        end
    end
end



mutable struct Future
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        @assert future != C_NULL
        finalizer(new(future)) do future
            @ccall MLIR.API.mlir_c.FreeFuture(future.future::Ptr{Cvoid})::Cvoid
        end
    end
end


mutable struct Buffer
    buffer::Ptr{Cvoid}

    function Buffer(buffer::Ptr{Cvoid})
        finalizer(new(buffer)) do buffer
            if buffer != C_NULL
                @ccall MLIR.API.mlir_c.PjRtBufferFree(buffer.buffer::Ptr{Cvoid})::Cvoid
            end
        end
    end
end


struct Device
    device::Ptr{Cvoid}
end

struct AsyncBuffer
    buffer::Buffer
    future::Union{Future, Nothing}
end

function device(buffer::Buffer)
    GC.@preserve buffer begin
        return Device(@ccall MLIR.API.mlir_c.BufferToDevice(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid})
    end
end
function client(buffer::Buffer)
    GC.@preserve buffer begin
        return Client(@ccall MLIR.API.mlir_c.BufferToClient(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid})
    end
end
function device(buffer::AsyncBuffer)
    return device(buffer.buffer)
end
function client(buffer::AsyncBuffer)
    return client(buffer.buffer)
end
function client(device::Device)
    GC.@preserve device begin
        return Client(@ccall MLIR.API.mlir_c.DeviceToClient(device.device::Ptr{Cvoid})::Ptr{Cvoid})
    end
end

function ArrayFromHostBuffer(client::Client, array::Array{T, N}, device) where {T, N}
    buffer = MLIR.IR.context!(MLIR.IR.Context()) do
        dtype = MLIR.IR.Type(T)
        sizear = Int64[s for s in size(array)]
        GC.@preserve array sizear begin
            @ccall MLIR.API.mlir_c.ArrayFromHostBuffer(client.client::Ptr{Cvoid}, pointer(array)::Ptr{T}, dtype::MLIR.API.MlirType, N::Csize_t, pointer(sizear)::Ptr{Int64}, device.device::Ptr{Cvoid})::Ptr{Cvoid}
        end
    end
    return Buffer(buffer)
end

function BufferToHost(buffer::Buffer, data::Ptr)
    GC.@preserve buffer begin
        @ccall MLIR.API.mlir_c.BufferToHost(buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid})::Cvoid
    end
end

# TODO users themselves need to gc preserve here
function UnsafeBufferPointer(buffer::Buffer)
    @ccall MLIR.API.mlir_c.UnsafeBufferPointer(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
end

function CopyBufferToDevice(buffer::Buffer, device::Device)
    GC.@preserve buffer device begin
        Buffer(@ccall MLIR.API.mlir_c.CopyBufferToDevice(buffer.buffer::Ptr{Cvoid}, device.device::Ptr{Cvoid})::Ptr{Cvoid})
    end
end

function BufferOnCPU(buffer::Buffer)
    GC.@preserve buffer begin
        (@ccall MLIR.API.mlir_c.BufferOnCPU(buffer.buffer::Ptr{Cvoid})::UInt8) != 0
    end
end


function ExecutableCall(exec::LoadedExecutable, inputs::Vector{Ptr{Cvoid}}, donated_args::Vector{UInt8}, ::Val{n_outs}) where n_outs
    outputs = Vector{Ptr{Cvoid}}(undef, n_outs)
    future_res = Vector{Future}(undef, n_outs)
    futures = Ref{UInt8}(0)

    GC.@preserve inputs donated_args outputs futures future_res begin
        @ccall MLIR.API.mlir_c.XLAExecute(exec.exec::Ptr{Cvoid}, length(inputs)::Cint, pointer(inputs)::Ptr{Cvoid}, pointer(donated_args)::Ptr{UInt8}, n_outs::Cint, pointer(outputs)::Ptr{Cvoid}, Base.unsafe_convert(Ptr{UInt8}, futures)::Ptr{UInt8}, pointer(future_res)::Ptr{Cvoid})::Cvoid
    end

    future = futures[] != 0

    return ntuple(Val(n_outs)) do i
        Base.@_inline_meta
        AsyncBuffer(Buffer(@inbounds outputs[i]), future ? (@inbounds future_res[i]) : nothing)
    end
end


function Compile(client::Client, mod::MLIR.IR.Module)
    GC.@preserve client mod begin
        executable = LoadedExecutable(@ccall MLIR.API.mlir_c.ClientCompile(client.client::Ptr{Cvoid}, mod.module_::MLIR.API.MlirModule)::Ptr{Cvoid})
    end
end

function ClientNumDevices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumDevices(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientNumAddressableDevices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumAddressableDevices(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientProcessIndex(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientProcessIndex(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientGetDevice(client::Client, idx)
    GC.@preserve client begin
        return Device(@ccall MLIR.API.mlir_c.ClientGetDevice(client.client::Ptr{Cvoid}, idx::Cint)::Ptr{Cvoid})
    end
end

function ClientGetAddressableDevice(client::Client, idx)
    GC.@preserve client begin
        return Device(@ccall MLIR.API.mlir_c.ClientGetAddressableDevice(client.client::Ptr{Cvoid}, idx::Cint)::Ptr{Cvoid})
    end
end

function is_ready(future::Future)
    GC.@preserve future begin
        return (@ccall MLIR.API.mlir_c.FutureIsReady(future.future::Ptr{Cvoid})::UInt8) != 0
    end
end

function await(future::Future)
    GC.@preserve future begin
        return @ccall MLIR.API.mlir_c.FutureAwait(future.future::Ptr{Cvoid})::Cvoid
    end
end


function is_ready(buffer::AsyncBuffer)
    if future === nothing
        return true
    else
        return is_ready(buffer.future)
    end
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

function await(buffer::AsyncBuffer)
    if buffer.future === nothing
        return
    else
        await(buffer.future)
        buffer.future = nothing
    end
end


function synced_buffer(buffer::AsyncBuffer)
    if buffer.future !== nothing
        await(buffer.future)
        buffer.future = nothing
    end
    return buffer.buffer
end

function synced_buffer(buffer::Buffer)
    return buffer
end

end
