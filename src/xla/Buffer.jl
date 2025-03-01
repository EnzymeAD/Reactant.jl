abstract type AbstractBuffer end

function synced_buffer end
function buffer_on_cpu end
function to_host end
function unsafe_buffer_pointer end
function copy_buffer_to_device end
function sharding end

Base.convert(::Type{Array}, buffer::AbstractBuffer) = convert(Array{eltype(buffer)}, buffer)

function Base.convert(::Type{<:Array{T}}, buffer::AbstractBuffer) where {T}
    arr = zeros(T, reverse(size(buffer))...)
    XLA.to_host(buffer, arr)
    return arr
end

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

function Base.show(io::IO, mime::MIME"text/plain", buffer::B) where {B<:AbstractBuffer}
    print(io, "$(B) storing ")
    show(io, mime, convert(Array, buffer))
    return nothing
end

# Async Buffers
abstract type AbstractAsyncBuffer <: AbstractBuffer end

Base.isempty(buffer::AbstractAsyncBuffer) = buffer.buffer.buffer == C_NULL

function Base.convert(T::Type{Array}, buffer::AbstractAsyncBuffer)
    wait(buffer)
    return convert(T, buffer.buffer)
end

function Base.convert(T::Type{<:Array{T1}}, buffer::AbstractAsyncBuffer) where {T1}
    wait(buffer)
    return convert(T, buffer.buffer)
end

for op in (:(Base.ndims), :(Base.size), :(Base.eltype), :device, :client, :sharding)
    @eval $op(buffer::AbstractAsyncBuffer) = $op(buffer.buffer)
end

function XLA.synced_buffer(buffer::AbstractAsyncBuffer)
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

XLA.buffer_on_cpu(buffer::AbstractAsyncBuffer) = XLA.buffer_on_cpu(buffer.buffer)
