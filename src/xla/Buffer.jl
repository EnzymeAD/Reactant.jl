abstract type AbstractBuffer end

function free_buffer end
function synced_buffer end
function buffer_on_cpu end
function to_host end
function unsafe_buffer_pointer end
function copy_buffer_to_device end
function sharding end

Base.isempty(buffer::AbstractBuffer) = buffer.buffer == C_NULL

@inline function client(
    buffers::Union{Array{<:AbstractBuffer},NTuple{<:Any,AbstractBuffer}}
)
    all_clients = map(client, buffers)
    @assert allequal(all_clients) "All buffers must have the same client"
    return first(all_clients)
end

@inline function synced_buffer(
    buffers::Union{AbstractArray{<:AbstractBuffer},NTuple{<:Any,<:AbstractBuffer}}
)
    return map(synced_buffer, buffers)
end

# Async Buffers
abstract type AbstractAsyncBuffer <: AbstractBuffer end

Base.isempty(buffer::AbstractAsyncBuffer) = isempty(buffer.buffer)

for op in (:(Base.ndims), :(Base.size), :(Base.eltype), :device, :client, :sharding)
    @eval $op(buffer::AbstractAsyncBuffer) = $op(buffer.buffer)
end

function synced_buffer(buffer::AbstractAsyncBuffer)
    wait(buffer)
    return buffer.buffer
end

function Base.wait(buffer::AbstractAsyncBuffer)
    buffer.future === nothing && return nothing
    future = buffer.future
    buffer.future = nothing
    wait(future)
    return nothing
end

function Base.isready(buffer::AbstractAsyncBuffer)
    buffer.future === nothing && return true
    return Base.isready(buffer.future)
end

buffer_on_cpu(buffer::AbstractAsyncBuffer) = buffer_on_cpu(buffer.buffer)

function to_host(buffer::AbstractAsyncBuffer, data, sharding)
    wait(buffer)
    to_host(buffer.buffer, data, sharding)
    return nothing
end
