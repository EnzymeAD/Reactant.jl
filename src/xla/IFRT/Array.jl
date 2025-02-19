mutable struct Array <: XLA.AbstractBuffer
    buffer::Ptr{Cvoid}

    function Array(buffer::Ptr{Cvoid})
        return finalizer(free_ifrt_array, new(buffer))
    end
end

# function Array(client::Client, array::Base.Array{T,N}, device::Device) where {T,N}
#     sizear = collect(Int64, reverse(size(array)))
#     buffer = GC.@preserve array sizear begin
#         @ccall MLIR.API.mlir_c.ArrayFromHostBuffer(
#             client.client::Ptr{Cvoid},
#             pointer(array)::Ptr{T},
#             XLA.primitive_type(T)::UInt64,
#             N::Csize_t,
#             pointer(sizear)::Ptr{Int64},
#             device.device::Ptr{Cvoid},
#         )::Ptr{Cvoid}
#     end
#     return Array(buffer)
# end

@inline function free_ifrt_array(buffer::Array)
    sbuffer = buffer.buffer
    if sbuffer != C_NULL
        @ccall MLIR.API.mlir_c.ifrt_free_array(sbuffer::Ptr{Cvoid})::Cvoid
    end
end

function Base.ndims(buffer::Array)
    GC.@preserve buffer begin
        return @ccall MLIR.API.mlir_c.ifrt_array_ndims(buffer.buffer::Ptr{Cvoid})::Int64
    end
end

function Base.size(buffer::Array)
    GC.@preserve buffer begin
        sz = @ccall MLIR.API.mlir_c.ifrt_array_shape(buffer.buffer::Ptr{Cvoid})::Ptr{Int64}
    end
    return Tuple(unsafe_wrap(Array, sz, ndims(buffer)))
end

function Base.eltype(buffer::Array)
    GC.@preserve buffer begin
        return XLA.julia_type(
            @ccall MLIR.API.mlir_c.ifrt_array_eltype(buffer.buffer::Ptr{Cvoid})::Cint
        )
    end
end

function XLA.device(::Array)
    return error("IFRT.Array can be sharded/replicated across multiple devices. Hence, \
                  `XLA.device` is not defined.")
end

function XLA.client(buffer::Array)
    GC.@preserve buffer begin
        return Client(
            @ccall MLIR.API.mlir_c.ifrt_array_to_client(
                buffer.buffer::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

XLA.synced_buffer(buffer::Array) = buffer

function XLA.buffer_on_cpu(::Array)
    return error("IFRT.Array does not support `XLA.buffer_on_cpu`")
end

function XLA.to_host(buffer::Array, data)
    error("TODO: not implemented")
    # GC.@preserve buffer begin
    #     @ccall MLIR.API.mlir_c.BufferToHost(
    #         buffer.buffer::Ptr{Cvoid}, data::Ptr{Cvoid}
    #     )::Cvoid
    # end
end

function XLA.unsafe_buffer_pointer(buffer::Array)
    return error("IFRT.Array does not support `XLA.unsafe_buffer_pointer`")
end

function XLA.copy_buffer_to_device(buffer::Array, dev::Device)
    return error("IFRT.Array does not support `XLA.copy_buffer_to_device`")
end

function XLA.sharding(buffer::Array)
    GC.@preserve buffer begin
        return Sharding(
            @ccall MLIR.API.mlir_c.ifrt_array_to_sharding(
                buffer.buffer::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end
