using ..Reactant: MLIR, Profiler

function from_row_major(ptr::Ptr{T}, shape::NTuple{N,<:Integer}) where {T,N}
    if N <= 1
        return copy(unsafe_wrap(Array, ptr, shape))
    end
    reversed_shape = reverse(shape)
    transposed = unsafe_wrap(Array, ptr, reversed_shape)
    return permutedims(transposed, N:-1:1)
end

function dump(
    value_ptr::Ptr{Any},
    label_ptr::Ptr{UInt8},
    ndims_ptr::Ptr{UInt64},
    shape_ptr::Ptr{UInt64},
    width_ptr::Ptr{UInt64},
    type_kind_ptr::Ptr{UInt64},
)
    label = unsafe_string(label_ptr)
    ndims = unsafe_load(ndims_ptr)
    width = unsafe_load(width_ptr)
    type_kind = unsafe_load(type_kind_ptr)

    julia_type = if type_kind == 0
        if width == 32
            Float32
        elseif width == 64
            Float64
        else
            @ccall printf(
                "DUMP ERROR: Unsupported float width: %lld\n"::Cstring, width::Int64
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end
    elseif type_kind == 1
        if width == 1
            Bool
        elseif width == 8
            Int8
        elseif width == 16
            Int16
        elseif width == 32
            Int32
        elseif width == 64
            Int64
        else
            @ccall printf(
                "DUMP ERROR: Unsupported signed int width: %lld\n"::Cstring,
                width::Int64,
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end
    elseif type_kind == 2
        if width == 1
            Bool
        elseif width == 8
            UInt8
        elseif width == 16
            UInt16
        elseif width == 32
            UInt32
        elseif width == 64
            UInt64
        else
            @ccall printf(
                "DUMP ERROR: Unsupported unsigned int width: %lld\n"::Cstring,
                width::Int64,
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end
    else
        @ccall printf("DUMP ERROR: Unknown type kind: %lld\n"::Cstring, type_kind::Int64)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    println("═══ DUMP: $label ═══")

    if ndims == 0
        value = unsafe_load(Ptr{julia_type}(value_ptr))
        println("  Scalar ($julia_type): $value")
    else
        shape = Tuple(unsafe_wrap(Array, shape_ptr, ndims))
        value_array = from_row_major(Ptr{julia_type}(value_ptr), shape)

        println("  Shape: $shape")
        println("  Type: Array{$julia_type}")
        println("  Values:")

        total_elements = prod(shape)
        if total_elements <= 20
            println("    ", value_array)
        else
            println("    [$(total_elements) elements]")
            println("    min: $(minimum(value_array))")
            println("    max: $(maximum(value_array))")
            println("    mean: $(sum(value_array) / total_elements)")
            println("    First 10: $(value_array[1:min(10, total_elements)])")
        end
    end

    println("═══════════════════════════════════")
    return nothing
end

function __init__()
    dump_ptr = @cfunction(
        dump,
        Cvoid,
        (Ptr{Any}, Ptr{UInt8}, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_dump::Cstring, dump_ptr::Ptr{Cvoid}
    )::Cvoid

    return nothing
end
