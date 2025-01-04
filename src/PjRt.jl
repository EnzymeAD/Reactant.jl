module PjRt

using CEnum

@cenum HostBufferSemantics::Int begin
    # The runtime may not hold references to `data` after the call to
    # `BufferFromHostBuffer` completes. The caller promises that `data` is
    # immutable and will not be freed only for the duration of the
    # BufferFromHostBuffer call. `on_done_with_host_buffer` will be called
    # before `BufferFromHostBuffer` returns.
    HostBufferSemanticsImmutableOnlyDuringCall

    # The runtime may hold onto `data` after the call to `BufferFromHostBuffer`
    # returns while the runtime completes a transfer to the device. The caller
    # promises not to mutate or free `data` until the transfer completes, at
    # which point the runtime will call `on_done_with_host_buffer`. It is also
    # correct to wait on the host (directly or indirectly) for the buffer's
    # definition event to complete.
    HostBufferSemanticsImmutableUntilTransferCompletes

    # The PjRtBuffer may alias `data` internally and the runtime may use the
    # `data` contents as long as the buffer is alive. The runtime promises not
    # to mutate contents of the buffer (i.e. it will not use it for aliased
    # output buffers). The caller promises to keep `data` alive and also not to
    # mutate its contents as long as the buffer is alive; to notify the caller
    # that the buffer may be freed, the runtime will call
    # `on_done_with_host_buffer` when the PjRtBuffer is freed. On non-CPU
    # platforms this acts identically to kImmutableUntilTransferCompletes.
    HostBufferSemanticsImmutableZeroCopy

    # The PjRtBuffer may alias `data` internally and the runtime may use the
    # `data` contents as long as the buffer is alive. The runtime is allowed
    # to mutate contents of the buffer (i.e. use it for aliased output
    # buffers). The caller promises to keep `data` alive and not to mutate its
    # contents as long as the buffer is alive (otherwise it could be a data
    # race with the runtime); to notify the caller that the buffer may be
    # freed, the runtime will call `on_done_with_host_buffer` when the
    # PjRtBuffer is freed. On non-CPU platforms this acts identically to
    # kImmutableUntilTransferCompletes.
    HostBufferSemanticsMutableZeroCopy
end

# @generated function Base.propertynames(::Type{HostBufferSemantics})
#     Symbol.(map(x -> chopprefix(x, "HostBufferSemantics"),string.(instances(HostBufferSemantics))))
# end

struct MemorySpace
    ptr::Ptr{Cvoid}
    function MemorySpace(x)
        @assert x != C_NULL
        return new(x)
    end
end

end
