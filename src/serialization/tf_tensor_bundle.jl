# Pure-Julia TensorFlow V2 Checkpoint / TensorBundle writer (.index SSTable + .data shard).
# Format references:
#   tensorflow/core/util/tensor_bundle/{tensor_bundle.cc,naming.cc}
#   tsl/lib/io/{table_builder.cc,block_builder.cc,format.cc}

module TFTensorBundle

using CRC32c: crc32c
using ..TFProto:
    BUNDLE_LITTLE,
    BundleEntryProto,
    BundleHeaderProto,
    DT_STRING,
    TFDataType,
    TensorShapeProto,
    VersionDef,
    proto_to_bytes

const SSTABLE_MAGIC_NUMBER = 0xdb4775248b80fb57
const SSTABLE_BLOCK_SIZE = 4096
const SSTABLE_NO_COMPRESSION = 0x00
const CRC_MASK_DELTA = 0xa282ead8

@inline function mask_crc(crc::UInt32)::UInt32
    return ((crc >> 15) | (crc << 17)) + CRC_MASK_DELTA
end

@inline function put_varint!(buf::Vector{UInt8}, v::UInt64)
    x = v
    while x >= 0x80
        push!(buf, UInt8((x & 0x7f) | 0x80))
        x >>= 7
    end
    push!(buf, UInt8(x))
    return buf
end

@inline function append_fixed32_le!(buf::Vector{UInt8}, v::UInt32)
    le = htol(v)
    push!(
        buf,
        UInt8(le & 0xff),
        UInt8((le >> 8) & 0xff),
        UInt8((le >> 16) & 0xff),
        UInt8((le >> 24) & 0xff),
    )
    return buf
end

@inline function append_fixed64_le!(buf::Vector{UInt8}, v::UInt64)
    le = htol(v)
    for shift in 0:8:56
        push!(buf, UInt8((le >> shift) & 0xff))
    end
    return buf
end

mutable struct BlockBuilder
    const restart_interval::Int
    const buffer::Vector{UInt8}
    const restarts::Vector{UInt32}
    counter::Int
    last_key::Vector{UInt8}
end

function BlockBuilder(restart_interval::Int)
    return BlockBuilder(restart_interval, UInt8[], UInt32[0], 0, UInt8[])
end

function block_reset!(bb::BlockBuilder)
    empty!(bb.buffer)
    empty!(bb.restarts)
    push!(bb.restarts, UInt32(0))
    bb.counter = 0
    empty!(bb.last_key)
    return nothing
end

block_empty(bb::BlockBuilder)::Bool = isempty(bb.buffer)

function block_current_size_estimate(bb::BlockBuilder)::Int
    return length(bb.buffer) + length(bb.restarts) * sizeof(UInt32) + sizeof(UInt32)
end

function block_add!(bb::BlockBuilder, key::Vector{UInt8}, value::Vector{UInt8})
    shared = 0
    if bb.counter < bb.restart_interval
        min_len = min(length(bb.last_key), length(key))
        while shared < min_len && bb.last_key[shared + 1] == key[shared + 1]
            shared += 1
        end
    else
        push!(bb.restarts, UInt32(length(bb.buffer)))
        bb.counter = 0
    end
    non_shared = length(key) - shared

    put_varint!(bb.buffer, UInt64(shared))
    put_varint!(bb.buffer, UInt64(non_shared))
    put_varint!(bb.buffer, UInt64(length(value)))
    append!(bb.buffer, @view(key[(shared + 1):end]))
    append!(bb.buffer, value)

    resize!(bb.last_key, length(key))
    copyto!(bb.last_key, key)
    bb.counter += 1
    return nothing
end

function block_finish!(bb::BlockBuilder)::Vector{UInt8}
    for r in bb.restarts
        append_fixed32_le!(bb.buffer, r)
    end
    append_fixed32_le!(bb.buffer, UInt32(length(bb.restarts)))
    return copy(bb.buffer)
end

struct BlockHandle
    offset::UInt64
    size::UInt64
end

function encode_block_handle!(buf::Vector{UInt8}, bh::BlockHandle)
    put_varint!(buf, bh.offset)
    put_varint!(buf, bh.size)
    return buf
end

function encode_block_handle(bh::BlockHandle)::Vector{UInt8}
    buf = UInt8[]
    return encode_block_handle!(buf, bh)
end

function find_shortest_separator(start::Vector{UInt8}, limit::Vector{UInt8})::Vector{UInt8}
    min_len = min(length(start), length(limit))
    diff_idx = 1
    while diff_idx <= min_len && start[diff_idx] == limit[diff_idx]
        diff_idx += 1
    end
    if diff_idx <= min_len
        diff_byte = start[diff_idx]
        if diff_byte < 0xff && (diff_byte + 0x01) < limit[diff_idx]
            sep = start[1:diff_idx]
            sep[diff_idx] = diff_byte + 0x01
            return sep
        end
    end
    return copy(start)
end

function find_short_successor(key::Vector{UInt8})::Vector{UInt8}
    for i in eachindex(key)
        if key[i] != 0xff
            succ = key[1:i]
            succ[i] = key[i] + 0x01
            return succ
        end
    end
    return copy(key)
end

function write_raw_block(io::IO, block_contents::Vector{UInt8})::BlockHandle
    offset = UInt64(position(io))
    size = UInt64(length(block_contents))
    write(io, block_contents)
    write(io, SSTABLE_NO_COMPRESSION)
    crc = crc32c(UInt8[SSTABLE_NO_COMPRESSION], crc32c(block_contents))
    trailer_crc = UInt8[]
    append_fixed32_le!(trailer_crc, mask_crc(crc))
    write(io, trailer_crc)
    return BlockHandle(offset, size)
end

function write_sstable(
    path::String, sorted_entries::Vector{Pair{Vector{UInt8},Vector{UInt8}}}
)
    open(path, "w") do io
        data_block = BlockBuilder(16)
        index_block = BlockBuilder(1)
        pending_index_entry = false
        pending_handle = BlockHandle(0, 0)
        last_key = UInt8[]

        for (key, val) in sorted_entries
            if !isempty(last_key) && !(isless(last_key, key))
                error("SSTable keys must be strictly increasing")
            end
            if pending_index_entry
                sep = find_shortest_separator(last_key, key)
                block_add!(index_block, sep, encode_block_handle(pending_handle))
                pending_index_entry = false
            end

            block_add!(data_block, key, val)
            last_key = copy(key)

            if block_current_size_estimate(data_block) >= SSTABLE_BLOCK_SIZE
                pending_handle = write_raw_block(io, block_finish!(data_block))
                block_reset!(data_block)
                pending_index_entry = true
            end
        end

        if !block_empty(data_block)
            pending_handle = write_raw_block(io, block_finish!(data_block))
            block_reset!(data_block)
            pending_index_entry = true
        end

        if pending_index_entry
            succ = find_short_successor(last_key)
            block_add!(index_block, succ, encode_block_handle(pending_handle))
            pending_index_entry = false
        end

        metaindex_handle = write_raw_block(io, block_finish!(BlockBuilder(16)))
        index_handle = write_raw_block(io, block_finish!(index_block))

        footer = UInt8[]
        sizehint!(footer, 48)
        encode_block_handle!(footer, metaindex_handle)
        encode_block_handle!(footer, index_handle)
        used_len = length(footer)
        resize!(footer, 40)
        fill!(@view(footer[(used_len + 1):40]), 0x00)
        append_fixed64_le!(footer, SSTABLE_MAGIC_NUMBER)
        @assert length(footer) == 48
        write(io, footer)
    end
    return nothing
end

struct BundleTensorSpec
    key::String
    dtype::TFDataType
    tf_shape::Vector{Int64}
    array::Union{Nothing,Array}
    scalar_string::Union{Nothing,Vector{UInt8}}
end

function encode_scalar_string_bytes(str_bytes::Vector{UInt8})::Tuple{Vector{UInt8},UInt32}
    len = UInt64(length(str_bytes))
    out = UInt8[]
    put_varint!(out, len)

    len_le = UInt8[]
    if len <= typemax(UInt32)
        append_fixed32_le!(len_le, UInt32(len))
    else
        append_fixed64_le!(len_le, len)
    end
    crc = crc32c(len_le)
    len_cksum_bytes = UInt8[]
    append_fixed32_le!(len_cksum_bytes, mask_crc(crc))

    crc = crc32c(len_cksum_bytes, crc)
    crc = crc32c(str_bytes, crc)

    append!(out, len_cksum_bytes)
    append!(out, str_bytes)
    return out, mask_crc(crc)
end

function write_tensor_bundle(variables_dir::String, tensors::Vector{BundleTensorSpec})
    mkpath(variables_dir)
    data_path = joinpath(variables_dir, "variables.data-00000-of-00001")
    index_path = joinpath(variables_dir, "variables.index")

    sorted_tensors = sort(tensors; by=t -> codeunits(t.key))

    index_entries = Pair{Vector{UInt8},Vector{UInt8}}[]
    header = BundleHeaderProto(
        Int32(1), BUNDLE_LITTLE, VersionDef(; producer=1, min_consumer=0)
    )
    push!(index_entries, UInt8[] => proto_to_bytes(header))

    open(data_path, "w") do data_io
        for t in sorted_tensors
            offset = Int64(position(data_io))
            if t.dtype == DT_STRING
                str_bytes = something(t.scalar_string)
                raw_bytes, masked_crc = encode_scalar_string_bytes(str_bytes)
                write(data_io, raw_bytes)
                byte_size = Int64(length(raw_bytes))
            else
                arr = something(t.array)
                raw_bytes = if ENDIAN_BOM == 0x04030201
                    reinterpret(UInt8, vec(arr))
                else
                    reinterpret(UInt8, htol.(vec(arr)))
                end
                write(data_io, raw_bytes)
                byte_size = Int64(length(raw_bytes))
                masked_crc = mask_crc(crc32c(raw_bytes))
            end

            entry = BundleEntryProto(
                t.dtype,
                TensorShapeProto(t.tf_shape),
                Int32(0),
                offset,
                byte_size,
                masked_crc,
            )
            push!(index_entries, Vector{UInt8}(codeunits(t.key)) => proto_to_bytes(entry))
        end
    end

    write_sstable(index_path, index_entries)
    return nothing
end

end # module TFTensorBundle
