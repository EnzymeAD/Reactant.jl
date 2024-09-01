module XLA

import ...MLIR

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

function to_row_major(x::Array{T,N}) where {T,N}
    return permutedims(x, reverse(Base.OneTo(N)))
end

function to_row_major(x::Vector{T}) where {T}
    return x
end

function to_row_major(x::Matrix{T}) where {T}
    return Matrix{T}(transpose(x))
end

function from_row_major(x::Array{T,N}) where {T,N}
    return permutedims(x, reverse(Base.OneTo(N)))
end

function from_row_major(x::Vector{T}) where {T}
    return x
end

function from_row_major(x::Matrix{T}) where {T}
    return transpose(x)
end

const cpuclientcount = Ref(0)
# TODO synchronization when async is not working because `future` in `ConcreteRArray` is always `nothing`
function CPUClient(asynchronous=false, node_id=0, num_nodes=1)
    global cpuclientcount
    @assert cpuclientcount[] == 0
    cpuclientcount[] += 1

    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeCPUClient")
    client = ccall(f, Ptr{Cvoid}, (UInt, Cint, Cint), asynchronous, node_id, num_nodes)
    #client = @ccall MLIR.API.mlir_c.MakeCPUClient(asynchronous::UInt8, node_id::Cint, num_nodes::Cint)::Ptr{Cvoid}
    return Client(client)
end

function GPUClient(node_id=0, num_nodes=1, platform="gpu")
    #allowed_devices = [-1]
    # GC.@preserve allowed_devices begin
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeGPUClient")
    refstr = Ref{Cstring}()
    client = ccall(
        f,
        Ptr{Cvoid},
        (Cint, Cint, Ptr{Cvoid}, Cint, Cstring, Ptr{Cstring}),
        node_id,
        num_nodes,
        C_NULL,
        0,
        platform,
        refstr,
    )
    if client == C_NULL
        throw(AssertionError(unsafe_string(refstr[])))
    end
    return Client(client)
end

function TPUClient(tpu_path::String)
    f = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "MakeTPUClient")
    refstr = Ref{Cstring}()
    client = ccall(f, Ptr{Cvoid}, (Cstring, Ptr{Cstring}), tpu_path, refstr)
    if client == C_NULL
        throw(AssertionError(unsafe_string(refstr[])))
    end
    return Client(client)
end

const backends = Dict{String,Client}()
const default_backend = Ref{Client}()
const default_device_idx = Ref{Int}(0)
using Reactant_jll
using Libdl
function __init__()
    initLogs = Libdl.dlsym(Reactant_jll.libReactantExtra_handle, "InitializeLogs")
    ccall(initLogs, Cvoid, ())
    cpu = CPUClient()
    backends["cpu"] = cpu
    default_backend[] = cpu
    @static if !Sys.isapple()
        if isfile("/usr/lib/libtpu.so")
            try
                tpu = TPUClient(
                    "/home/wmoses/.local/lib/python3.8/site-packages/libtpu/libtpu.so"
                )
                backends["tpu"] = tpu
                default_backend[] = tpu
            catch e
                println(stdout, e)
            end
        else
            try
                gpu = GPUClient()
                backends["gpu"] = gpu
            catch e
                println(stdout, e)
            end
        end
    end
    return nothing
end

@inline function free_exec(exec)
    @ccall MLIR.API.mlir_c.ExecutableFree(exec.exec::Ptr{Cvoid})::Cvoid
end

mutable struct LoadedExecutable
    exec::Ptr{Cvoid}

    function LoadedExecutable(exec::Ptr{Cvoid})
        @assert exec != C_NULL
        return finalizer(free_exec, new(exec))
    end
end

@inline function free_future(future)
    @ccall MLIR.API.mlir_c.FreeFuture(future.future::Ptr{Cvoid})::Cvoid
end

mutable struct Future
    future::Ptr{Cvoid}

    function Future(future::Ptr{Cvoid})
        # @assert future != C_NULL
        return finalizer(free_future, new(future))
    end
end

@inline function free_buffer(buffer)
    sbuffer = buffer.buffer
    if sbuffer != C_NULL
        @ccall MLIR.API.mlir_c.PjRtBufferFree(sbuffer::Ptr{Cvoid})::Cvoid
    end
end

mutable struct Buffer
    buffer::Ptr{Cvoid}
    function Buffer(buffer::Ptr{Cvoid})
        return finalizer(free_buffer, new(buffer))
    end
end

struct Device
    device::Ptr{Cvoid}
end

mutable struct AsyncBuffer
    buffer::Buffer
    future::Union{Future,Nothing}
end

function device(buffer::Buffer)
    GC.@preserve buffer begin
        return Device(
            @ccall MLIR.API.mlir_c.BufferToDevice(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end
function client(buffer::Buffer)
    GC.@preserve buffer begin
        return Client(
            @ccall MLIR.API.mlir_c.BufferToClient(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
        )
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
        return Client(
            @ccall MLIR.API.mlir_c.DeviceToClient(device.device::Ptr{Cvoid})::Ptr{Cvoid}
        )
    end
end

# https://github.com/openxla/xla/blob/4bfb5c82a427151d6fe5acad8ebe12cee403036a/xla/xla_data.proto#L29
@inline primitive_type(::Type{Bool}) = 1

@inline primitive_type(::Type{Int8}) = 2
@inline primitive_type(::Type{UInt8}) = 6

@inline primitive_type(::Type{Int16}) = 3
@inline primitive_type(::Type{UInt16}) = 7

@inline primitive_type(::Type{Int32}) = 4
@inline primitive_type(::Type{UInt32}) = 8

@inline primitive_type(::Type{Int64}) = 5
@inline primitive_type(::Type{UInt64}) = 9

@inline primitive_type(::Type{Float16}) = 10
@inline primitive_type(::Type{Float32}) = 11

# @inline primitive_type(::Type{BFloat16}) = 16

@inline primitive_type(::Type{Float64}) = 12

@inline primitive_type(::Type{Complex{Float32}}) = 24
@inline primitive_type(::Type{Complex{Float64}}) = 25

function ArrayFromHostBuffer(client::Client, array::Array{T,N}, device) where {T,N}
    sizear = Int64[s for s in reverse(size(array))]
    buffer = GC.@preserve array sizear begin
        @ccall MLIR.API.mlir_c.ArrayFromHostBuffer(
            client.client::Ptr{Cvoid},
            pointer(array)::Ptr{T},
            primitive_type(T)::UInt64,
            N::Csize_t,
            pointer(sizear)::Ptr{Int64},
            device.device::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end
    return Buffer(buffer)
end

function BufferToHost(buffer::Buffer, data)
    GC.@preserve buffer begin
        @ccall MLIR.API.mlir_c.BufferToHost(
            buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
        )::Cvoid
    end
end

# TODO users themselves need to gc preserve here
function UnsafeBufferPointer(buffer::Buffer)
    @ccall MLIR.API.mlir_c.UnsafeBufferPointer(buffer.buffer::Ptr{Cvoid})::Ptr{Cvoid}
end

function CopyBufferToDevice(buffer::Buffer, device::Device)
    GC.@preserve buffer device begin
        Buffer(
            @ccall MLIR.API.mlir_c.CopyBufferToDevice(
                buffer.buffer::Ptr{Cvoid}, device.device::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function BufferOnCPU(buffer::Buffer)
    GC.@preserve buffer begin
        (@ccall MLIR.API.mlir_c.BufferOnCPU(buffer.buffer::Ptr{Cvoid})::UInt8) != 0
    end
end

function execute_ir(N, n_outs, fn)
    ptr = sizeof(Int) == sizeof(Int64) ? "i64" : "i32"
    cint = sizeof(Cint) == sizeof(Int64) ? "i64" : "i32"
    res = """define { [$n_outs x $ptr], [$n_outs x $ptr], i8 } @f($ptr %exec, [$N x $ptr] %inps, [$N x i8] %donated) alwaysinline {
entry:
    %inpa = alloca [$N x $ptr]
    %outa = alloca [$n_outs x $ptr]
    %futpa = alloca [$n_outs x $ptr]
    store [$N x $ptr] %inps, [$N x $ptr]* %inpa
    %dona = alloca [$N x i8]
    store [$N x i8] %donated, [$N x i8]* %dona
    %futa = alloca i8
    call void inttoptr ($ptr $fn to void ($ptr, $cint, [$N x $ptr]*, [$N x i8]*, $cint, [$n_outs x $ptr]*, i8*, [$n_outs x $ptr]*)*)($ptr %exec, $cint $N, [$N x $ptr]* nocapture readonly %inpa, [$N x i8]* nocapture readonly %dona, $cint $n_outs, [$n_outs x $ptr]* nocapture writeonly %outa, i8* nocapture writeonly %futa, [$n_outs x $ptr]* nocapture writeonly %futpa)
    %out = load [$n_outs x $ptr], [$n_outs x $ptr]* %outa
    %fut = load i8, i8* %futa
    %futp = load [$n_outs x $ptr], [$n_outs x $ptr]* %futpa
    %fca.0.insert = insertvalue { [$n_outs x $ptr], [$n_outs x $ptr], i8 } undef, [$n_outs x $ptr] %out, 0
    %fca.1.insert = insertvalue { [$n_outs x $ptr], [$n_outs x $ptr], i8 } %fca.0.insert, [$n_outs x $ptr] %futp, 1
    %fca.2.insert = insertvalue { [$n_outs x $ptr], [$n_outs x $ptr], i8 } %fca.1.insert, i8 %fut, 2
    ret { [$n_outs x $ptr], [$n_outs x $ptr], i8 } %fca.2.insert
}
"""
    return res
end

@generated function ExecutableCall(
    exec::LoadedExecutable,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
) where {N,n_outs}
    sym0 = dlsym(Reactant_jll.libReactantExtra_handle, "XLAExecute")
    xla_execute_fn = reinterpret(UInt, sym0)
    ir = execute_ir(N, n_outs, xla_execute_fn)
    results = []
    for i in 1:n_outs
        push!(
            results,
            :(AsyncBuffer(Buffer(outputs[$i]), future ? Future(future_res[$i]) : nothing)),
        )
    end
    return quote
        Base.@_inline_meta
        exec = exec.exec
        GC.@preserve exec begin
            outputs, future_res, future = Base.llvmcall(
                ($ir, "f"),
                Tuple{NTuple{n_outs,Ptr{Cvoid}},NTuple{n_outs,Ptr{Cvoid}},Bool},
                Tuple{Ptr{Cvoid},NTuple{N,Ptr{Cvoid}},NTuple{N,UInt8}},
                exec,
                inputs,
                donated_args,
            )
        end
        return ($(results...),)
    end
end

@inline function ExecutableCall0(
    exec::LoadedExecutable,
    inputs::NTuple{N,Ptr{Cvoid}},
    donated_args::NTuple{N,UInt8},
    ::Val{n_outs},
) where {N,n_outs}
    outputs = Ref{NTuple{n_outs,Ptr{Cvoid}}}()
    future_res = Ref{NTuple{n_outs,Ptr{Cvoid}}}()
    futures = Ref{UInt8}(0)

    inputs = Base.RefValue(inputs)
    donated_args = Base.RefValue(donated_args)
    GC.@preserve inputs donated_args outputs futures future_res begin
        @ccall MLIR.API.mlir_c.XLAExecute(
            exec.exec::Ptr{Cvoid},
            N::Cint,
            inputs::Ptr{Cvoid},
            donated_args::Ptr{UInt8},
            n_outs::Cint,
            Base.unsafe_convert(Ptr{Cvoid}, outputs)::Ptr{Cvoid},
            Base.unsafe_convert(Ptr{UInt8}, futures)::Ptr{UInt8},
            Base.unsafe_convert(Ptr{Cvoid}, future_res)::Ptr{Cvoid},
        )::Cvoid
    end

    outputs = outputs[]
    future_res = future_res[]
    future = futures[] != 0

    return ntuple(Val(n_outs)) do i
        Base.@_inline_meta
        return AsyncBuffer(Buffer(outputs[i]), future ? Future(future_res[i]) : nothing)
    end
end

function Compile(client::Client, mod::MLIR.IR.Module)
    GC.@preserve client mod begin
        executable = LoadedExecutable(
            @ccall MLIR.API.mlir_c.ClientCompile(
                client.client::Ptr{Cvoid}, mod.module_::MLIR.API.MlirModule
            )::Ptr{Cvoid}
        )
    end
end

function ClientNumDevices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumDevices(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientNumAddressableDevices(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientNumAddressableDevices(
            client.client::Ptr{Cvoid}
        )::Cint
    end
end

function ClientProcessIndex(client::Client)
    GC.@preserve client begin
        return @ccall MLIR.API.mlir_c.ClientProcessIndex(client.client::Ptr{Cvoid})::Cint
    end
end

function ClientGetDevice(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function ClientGetAddressableDevice(client::Client, idx)
    GC.@preserve client begin
        return Device(
            @ccall MLIR.API.mlir_c.ClientGetAddressableDevice(
                client.client::Ptr{Cvoid}, idx::Cint
            )::Ptr{Cvoid}
        )
    end
end

function is_ready(future::Future)
    GC.@preserve future begin
        return (@ccall MLIR.API.mlir_c.FutureIsReady(future.future::Ptr{Cvoid})::UInt8) != 0
    end
end

@inline function await(future::Future)::Nothing
    GC.@preserve future begin
        @ccall MLIR.API.mlir_c.FutureAwait(future.future::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

function is_ready(buffer::AsyncBuffer)::Bool
    future = buffer.future
    if isnothing(future)
        return true
    else
        return is_ready(future)
    end
end

const AsyncEmptyBuffer = AsyncBuffer(Buffer(C_NULL), nothing)

@inline function await(buffer::AsyncBuffer)::Nothing
    if buffer.future == nothing
        return nothing
    else
        future = buffer.future
        buffer.future = nothing
        await(future::Future)
    end
    return nothing
end

@inline function synced_buffer(buffer::AsyncBuffer)
    if buffer.future != nothing
        future = buffer.future
        buffer.future = nothing
        await(future::Future)
    end
    return buffer.buffer
end

@inline function synced_buffer(buffer::Buffer)
    return buffer
end

end
