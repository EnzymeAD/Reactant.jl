# Utility functions for passing Protocol Buffer messages to C++ via ccall
#
# These functions serialize Julia proto structs to bytes that can be passed
# to C++ functions which then deserialize them using ParseFromArray.

module ProtoUtils

using ProtoBuf: ProtoBuf
const PB = ProtoBuf

function proto_to_bytes(msg)
    io = IOBuffer()
    e = PB.ProtoEncoder(io)
    PB.encode(e, msg)
    return take!(io)
end

function proto_from_bytes(::Type{T}, data::Vector{UInt8}) where {T}
    io = IOBuffer(data)
    d = PB.ProtoDecoder(io)
    return PB.decode(d, T)
end

end # module ProtoUtils
