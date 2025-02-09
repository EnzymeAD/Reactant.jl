@cenum ArrayCopySemantics::UInt32 begin
    AlwaysCopy = 0
    ReuseInput = 1
    DonateInput = 2
end

# currently, only supports IFRT-PjRt
mutable struct Array
    ptr::Ptr{Cvoid}

    function Array(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return finalizer(free_array, new(ptr))
    end
end

function free_array(array)
    @ccall MLIR.API.mlir_c.reactant_release_ifrt_pjrt_array(array.ptr::Ptr{Cvoid})::Cvoid
end

function Array(client::Client, buffer::XLA.Buffer)
    hold!(buffer)
    GC.@preserve client buffer begin
        return Array(
            @ccall MLIR.API.mlir_c.ifrt_pjrt_ArrayFromHostBuffer(
                client.ptr::Ptr{Cvoid}, buffer.holded::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
end

function CopyArrayToHostBuffer(array::Array, data)
    GC.@preserve array begin
        @ccall MLIR.API.mlir_c.ifrt_CopyArrayToHostBuffer(
            array.ptr::Ptr{Cvoid}, data::Ptr{Cvoid}, AlwaysCopy::Cuint
        )::Cvoid
    end
end
