abstract type AbstractBuffer end

function free_buffer end

"""
    XLA.free_buffer!(buffer::XLA.AbstractBuffer)
    XLA.free_buffer!(buffer::XLA.AbstractAsyncBuffer)

Immediately release the runtime-side handle held by `buffer` (e.g. the `PJRT.Buffer` or
`IFRT.Array` backing a `ConcreteRArray`), independently of the Julia GC. After the call
the buffer is in the empty state (`isempty(buffer) == true`); its finalizer becomes a
no-op, so double frees are impossible. The call is idempotent.

The runtime defers actual device-memory reclamation until any in-flight computation
using the buffer completes (identical to what the buffer's finalizer would do — just at a
deterministic point in time).

See also [`Reactant.free!`](@ref) for the user-facing API over `ConcreteRArray`s.
"""
function free_buffer! end

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

# Shared error for operations that require a live (non-empty) buffer.
function throw_empty_buffer(op::AbstractString)
    return error(
        "Cannot $(op) an empty XLA buffer. The buffer was either already ",
        "freed (via `Reactant.free!` or `XLA.free_buffer!`) or was never populated.",
    )
end

function synced_buffer(buffer::AbstractAsyncBuffer)
    isempty(buffer) && throw_empty_buffer("use")
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
# Frees the underlying buffer in place; the pending `future` (if any) is left alone and
# is cleaned up by its own finalizer.
function free_buffer!(buffer::AbstractAsyncBuffer)
    return free_buffer!(buffer.buffer)
end

function to_host(buffer::AbstractAsyncBuffer, data, sharding)
    isempty(buffer) && throw_empty_buffer("copy to host from")
    wait(buffer)
    to_host(buffer.buffer, data, sharding)
    return nothing
end
