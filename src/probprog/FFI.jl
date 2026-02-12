using ..Reactant: MLIR

const DUMP_BUFFER = Vector{Tuple{String,Any}}()
const DUMP_BUFFER_LOCK = ReentrantLock()

function from_row_major(ptr::Ptr{T}, shape::NTuple{N,<:Integer}) where {T,N}
    if N <= 1
        return copy(unsafe_wrap(Array, ptr, shape))
    end
    reversed_shape = reverse(shape)
    transposed = unsafe_wrap(Array, ptr, reversed_shape)
    return permutedims(transposed, N:-1:1)
end

# Debug only
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
            nothing
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
            nothing
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
            nothing
        end
    else
        nothing
    end

    if julia_type === nothing
        return nothing
    end

    value = if ndims == 0
        unsafe_load(Ptr{julia_type}(value_ptr))
    else
        shape = Tuple(unsafe_wrap(Array, shape_ptr, ndims))
        from_row_major(Ptr{julia_type}(value_ptr), shape)
    end

    lock(DUMP_BUFFER_LOCK) do
        push!(DUMP_BUFFER, (label, value))
    end

    return nothing
end

function clear_dump_buffer!()
    lock(DUMP_BUFFER_LOCK) do
        empty!(DUMP_BUFFER)
    end
end

function show_dumps()
    lock(DUMP_BUFFER_LOCK) do
        println("show_dumps: buffer has $(length(DUMP_BUFFER)) entries")
        for (label, value) in DUMP_BUFFER
            println("═══ DUMP: $label ═══")
            if value isa AbstractArray
                println("  Shape: $(size(value))")
                println("  Type: $(typeof(value))")
                total = length(value)
                if total <= 20
                    println("  Values: $value")
                else
                    println("  [$(total) elements]")
                    println("    min: $(minimum(value))")
                    println("    max: $(maximum(value))")
                    println("    mean: $(sum(value) / total)")
                    println("    First 10: $(value[1:min(10, total)])")
                end
            else
                println("  Scalar ($(typeof(value))): $value")
            end
            println("═══════════════════════════════════")
        end
    end
end
