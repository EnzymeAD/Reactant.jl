"""
    MemoryLayout

Layout of an array in memory.
"""
abstract type MemoryLayout end

"""
    TiledMemoryLayout(minor_to_major, tiles=Vector{Int64}[])

Tiled layout of an array in memory, as described by `PJRT_Buffer_MemoryLayout` (of type `Tiled`).

## Fields

- `minor_to_major::Vector{Int64}`: the 0-based order of the dimensions, from the fastest-varying to the slowest-varying
  one (e.g. `[1, 0]` for a row-major matrix like C/numpy, `[0, 1]` for column-major matrix like Julia/Fortran).
- `tiles::Vector{Vector{Int64}}`: the sizes of each level of tiling, in the order they are applied to the minor-most
  dimensions (e.g. `[[8, 128], [2, 1]]` for XLA's `T(8,128)(2,1)`). Leave empty if the array is not tiled.
"""
struct TiledMemoryLayout <: MemoryLayout
    minor_to_major::Vector{Int64}
    tiles::Vector{Vector{Int64}}

    function TiledMemoryLayout(minor_to_major, tiles=Vector{Int64}[])
        # XLA can't handle such layouts, and aborts on some of them
        isperm(minor_to_major .+ 1) || throw(
            ArgumentError(
                "minor_to_major must be a permutation of 0:$(length(minor_to_major) - 1), got $minor_to_major",
            ),
        )
        all(tile -> all(>(0), tile), tiles) ||
            throw(ArgumentError("tile sizes must be positive, got $tiles"))
        return new(minor_to_major, tiles)
    end
end

function Base.:(==)(a::TiledMemoryLayout, b::TiledMemoryLayout)
    return a.minor_to_major == b.minor_to_major && a.tiles == b.tiles
end
function Base.hash(layout::TiledMemoryLayout, h::UInt)
    return hash(layout.tiles, hash(layout.minor_to_major, h))
end

"""
    StridedMemoryLayout(byte_strides)

Strided layout of an array in memory, as described by `PJRT_Buffer_MemoryLayout` (of type `Strides`):
`byte_strides::Vector{Int64}` holds the number of bytes to traverse along each dimension (e.g. `[8n, 8]` for a row-major
`m×n` matrix of `Float64`). Strides can be negative.
"""
struct StridedMemoryLayout <: MemoryLayout
    byte_strides::Vector{Int64}
end

function Base.:(==)(a::StridedMemoryLayout, b::StridedMemoryLayout)
    return a.byte_strides == b.byte_strides
end
Base.hash(layout::StridedMemoryLayout, h::UInt) = hash(layout.byte_strides, h)

# converts a layout passed by C, which must not be `C_NULL`
function MemoryLayout(c_layout::Ptr{CAPI.PJRT_Buffer_MemoryLayout})
    type = unsafe_load(c_layout.type)
    if type == CAPI.PJRT_Buffer_MemoryLayout_Type_Strides
        strides = unsafe_load(c_layout.strides)
        return StridedMemoryLayout(
            copy(unsafe_wrap(Vector{Int64}, strides.byte_strides, strides.num_byte_strides))
        )
    elseif type == CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled
        tiled = unsafe_load(c_layout.tiled)
        minor_to_major = copy(
            unsafe_wrap(Vector{Int64}, tiled.minor_to_major, tiled.minor_to_major_size)
        )
        # C passes the sizes of all tiles back to back, plus the number of dimensions of each tile
        tiles = Vector{Vector{Int64}}(undef, tiled.num_tiles)
        offset = 0
        for i in eachindex(tiles)
            n = Int(unsafe_load(tiled.tile_dim_sizes, i))
            tiles[i] = Int64[unsafe_load(tiled.tile_dims, offset + j) for j in 1:n]
            offset += n
        end
        # an invalid layout is an error of the caller
        try
            return TiledMemoryLayout(minor_to_major, tiles)
        catch err
            err isa ArgumentError && throw(invalid_argument(err.msg))
            rethrow()
        end
    else
        throw(invalid_argument("unknown layout type $type"))
    end
end
