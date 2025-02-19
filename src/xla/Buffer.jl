abstract type AbstractBuffer end

function synced_buffer end
function buffer_on_cpu end
function to_host end
function unsafe_buffer_pointer end
function copy_buffer_to_device end

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
