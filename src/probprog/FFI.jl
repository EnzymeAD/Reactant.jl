using ..Reactant: MLIR, Profiler


function from_row_major(ptr::Ptr{T}, shape::NTuple{N,<:Integer}) where {T,N}
    if N <= 1
        return copy(unsafe_wrap(Array, ptr, shape))
    end
    reversed_shape = reverse(shape)
    transposed = unsafe_wrap(Array, ptr, reversed_shape)
    return permutedims(transposed, N:-1:1)
end

function to_row_major!(ptr::Ptr{T}, src::AbstractArray{T,N}, shape::NTuple{N,<:Integer}) where {T,N}
    if N <= 1
        dest = unsafe_wrap(Array, ptr, shape)
        copyto!(dest, src)
    else
        reversed_shape = reverse(shape)
        dest = unsafe_wrap(Array, ptr, reversed_shape)
        src_permuted = permutedims(src, N:-1:1)
        copyto!(dest, src_permuted)
    end
    return nothing
end

function initTrace(trace_ptr_ptr::Ptr{Ptr{Any}})
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.initTrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    tr = ProbProgTrace()
    _keepalive!(tr)

    unsafe_store!(trace_ptr_ptr, pointer_from_objref(tr))

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function addSampleToTrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    sample_ptr_array::Ptr{Ptr{Any}},
    num_outputs_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.addSampleToTrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    num_outputs = unsafe_load(num_outputs_ptr)
    ndims_array = unsafe_wrap(Array, ndims_array, num_outputs)
    width_array = unsafe_wrap(Array, width_array, num_outputs)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_outputs)
    sample_ptr_array = unsafe_wrap(Array, sample_ptr_array, num_outputs)

    vals = Any[]
    for i in 1:num_outputs
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        sample_ptr = sample_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %lld\n"::Cstring, width::Int64
            )::Cvoid
            return nothing
        end

        if ndims == 0
            push!(vals, unsafe_load(Ptr{julia_type}(sample_ptr)))
        else
            shape = Tuple(unsafe_wrap(Array, shape_ptr, ndims))
            push!(vals, from_row_major(Ptr{julia_type}(sample_ptr), shape))
        end
    end

    trace.choices[symbol] = tuple(vals...)

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function addSubtrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    subtrace_ptr_ptr::Ptr{Ptr{Any}},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.addSubtrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    subtrace = unsafe_pointer_to_objref(unsafe_load(subtrace_ptr_ptr))::ProbProgTrace

    trace.subtraces[symbol] = subtrace

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function addWeightToTrace(trace_ptr_ptr::Ptr{Ptr{Any}}, weight_ptr::Ptr{Any})
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.addWeightToTrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    trace.weight = unsafe_load(Ptr{Float64}(weight_ptr))

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function addRetvalToTrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    retval_ptr_array::Ptr{Ptr{Any}},
    num_results_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.addRetvalToTrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    num_results = unsafe_load(num_results_ptr)

    if num_results == 0
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    ndims_array = unsafe_wrap(Array, ndims_array, num_results)
    width_array = unsafe_wrap(Array, width_array, num_results)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_results)
    retval_ptr_array = unsafe_wrap(Array, retval_ptr_array, num_results)

    vals = Any[]
    for i in 1:num_results
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        retval_ptr = retval_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %lld\n"::Cstring, width::Int64
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end

        if ndims == 0
            push!(vals, unsafe_load(Ptr{julia_type}(retval_ptr)))
        else
            shape = Tuple(unsafe_wrap(Array, shape_ptr, ndims))
            push!(vals, from_row_major(Ptr{julia_type}(retval_ptr), shape))
        end
    end

    trace.retval = tuple(vals...)

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function getSampleFromConstraint(
    constraint_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    sample_ptr_array::Ptr{Ptr{Any}},
    num_samples_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.getSampleFromConstraint"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    constraint = unsafe_pointer_to_objref(unsafe_load(constraint_ptr_ptr))::Constraint
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    num_samples = unsafe_load(num_samples_ptr)
    ndims_array = unsafe_wrap(Array, ndims_array, num_samples)
    width_array = unsafe_wrap(Array, width_array, num_samples)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_samples)
    sample_ptr_array = unsafe_wrap(Array, sample_ptr_array, num_samples)

    tostore = get(constraint, Address(symbol), nothing)

    if tostore === nothing
        @ccall printf(
            "No constraint found for symbol: %s\n"::Cstring, string(symbol)::Cstring
        )::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    for i in 1:num_samples
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        sample_ptr = sample_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %zd\n"::Cstring, width::Csize_t
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end

        if julia_type != eltype(tostore[i])
            @ccall printf(
                "Type mismatch in constrained sample: %s != %s\n"::Cstring,
                string(julia_type)::Cstring,
                string(eltype(tostore[i]))::Cstring,
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end

        if ndims == 0
            unsafe_store!(Ptr{julia_type}(sample_ptr), tostore[i])
        else
            shape = Tuple(unsafe_wrap(Array, shape_ptr, ndims))

            if shape != size(tostore[i])
                @ccall printf(
                    "Shape mismatch in constrained sample: expected %s, got %s\n"::Cstring,
                    string(shape)::Cstring,
                    string(size(tostore[i]))::Cstring,
                )::Cvoid
                @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
                return nothing
            end

            to_row_major!(Ptr{julia_type}(sample_ptr), tostore[i], shape)
        end
    end

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function getSubconstraint(
    constraint_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    subconstraint_ptr_ptr::Ptr{Ptr{Any}},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.getSubconstraint"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    constraint = unsafe_pointer_to_objref(unsafe_load(constraint_ptr_ptr))::Constraint
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol

    subconstraint = Constraint()

    for (key, value) in constraint
        if key.path[1] == symbol
            @assert isa(key, Address) "Expected Address type for constraint key"
            @assert length(key.path) > 1 "Expected composite address with length > 1"
            tail_address = Address(key.path[2:end])
            subconstraint[tail_address] = value
        end
    end

    if isempty(subconstraint)
        @ccall printf(
            "No subconstraint found for symbol: %s\n"::Cstring, string(symbol)::Cstring
        )::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    _keepalive!(subconstraint)
    unsafe_store!(subconstraint_ptr_ptr, pointer_from_objref(subconstraint))

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function getSampleFromTrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    sample_ptr_array::Ptr{Ptr{Any}},
    num_samples_ptr::Ptr{UInt64},
    ndims_array::Ptr{UInt64},
    shape_ptr_array::Ptr{Ptr{UInt64}},
    width_array::Ptr{UInt64},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.getSampleFromTrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    num_samples = unsafe_load(num_samples_ptr)
    ndims_array = unsafe_wrap(Array, ndims_array, num_samples)
    width_array = unsafe_wrap(Array, width_array, num_samples)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_samples)
    sample_ptr_array = unsafe_wrap(Array, sample_ptr_array, num_samples)

    tostore = get(trace.choices, symbol, nothing)

    if tostore === nothing
        @ccall printf(
            "No sample found in trace for symbol: %s\n"::Cstring, string(symbol)::Cstring
        )::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    for i in 1:num_samples
        ndims = ndims_array[i]
        width = width_array[i]
        shape_ptr = shape_ptr_array[i]
        sample_ptr = sample_ptr_array[i]

        julia_type = if width == 32
            Float32
        elseif width == 64
            Float64
        elseif width == 1
            Bool
        else
            nothing
        end

        if julia_type === nothing
            @ccall printf(
                "Unsupported datatype width: %zd\n"::Cstring, width::Csize_t
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end

        if julia_type != eltype(tostore[i])
            @ccall printf(
                "Type mismatch in trace sample: %s != %s\n"::Cstring,
                string(julia_type)::Cstring,
                string(eltype(tostore[i]))::Cstring,
            )::Cvoid
            @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
            return nothing
        end

        if ndims == 0
            unsafe_store!(Ptr{julia_type}(sample_ptr), tostore[i])
        else
            shape = Tuple(unsafe_wrap(Array, shape_ptr, ndims))

            if shape != size(tostore[i])
                @ccall printf(
                    "Shape mismatch in trace sample: expected %s, got %s\n"::Cstring,
                    string(shape)::Cstring,
                    string(size(tostore[i]))::Cstring,
                )::Cvoid
                @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
                return nothing
            end

            to_row_major!(Ptr{julia_type}(sample_ptr), tostore[i], shape)
        end
    end

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function getSubtrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    subtrace_ptr_ptr::Ptr{Ptr{Any}},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.getSubtrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol

    subtrace = get(trace.subtraces, symbol, nothing)

    if subtrace === nothing
        @ccall printf(
            "No subtrace found for symbol: %s\n"::Cstring, string(symbol)::Cstring
        )::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    _keepalive!(subtrace)
    unsafe_store!(subtrace_ptr_ptr, pointer_from_objref(subtrace))

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function getWeightFromTrace(trace_ptr_ptr::Ptr{Ptr{Any}}, weight_ptr::Ptr{Any})
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.getWeightFromTrace"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("Trace dereference failure\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    unsafe_store!(Ptr{Float64}(weight_ptr), trace.weight)

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function getFlattenedSamplesFromTrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    num_addresses_ptr::Ptr{UInt64},
    total_symbols_ptr::Ptr{UInt64},
    address_lengths_ptr::Ptr{UInt64},
    flattened_symbols_ptr::Ptr{UInt64},
    position_ptr::Ptr{Any},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.getFlattenedSamplesFromTrace"::Cstring,
        Profiler.TRACE_ME_LEVEL_CRITICAL::Cint,
    )::Int64

    trace = nothing
    try
        trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    catch
        @ccall printf("No trace found\n"::Cstring)::Cvoid
        @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
        return nothing
    end

    num_addresses = unsafe_load(num_addresses_ptr)
    total_symbols = unsafe_load(total_symbols_ptr)

    address_lengths = unsafe_wrap(Array, address_lengths_ptr, num_addresses)
    flattened_symbols = unsafe_wrap(Array, flattened_symbols_ptr, total_symbols)

    addresses = Vector{Vector{Symbol}}()
    symbol_idx = 1
    for i in 1:num_addresses
        addr_len = address_lengths[i]
        address = Symbol[]
        for j in 1:addr_len
            symbol_ptr_value = flattened_symbols[symbol_idx]
            symbol = unsafe_pointer_to_objref(Ptr{Any}(symbol_ptr_value))::Symbol
            push!(address, symbol)
            symbol_idx += 1
        end
        push!(addresses, address)
    end

    flattened_values = Float64[]

    for address in addresses
        current_trace = trace

        for (idx, symbol) in enumerate(address)
            if idx < length(address)
                if !haskey(current_trace.subtraces, symbol)
                    @ccall printf(
                        "No subtrace found for symbol in address path: %s\n"::Cstring,
                        string(symbol)::Cstring,
                    )::Cvoid
                    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
                    return nothing
                end
                current_trace = current_trace.subtraces[symbol]
            else
                if !haskey(current_trace.choices, symbol)
                    @ccall printf(
                        "No sample found for symbol: %s\n"::Cstring, string(symbol)::Cstring
                    )::Cvoid
                    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
                    return nothing
                end

                sample_tuple = current_trace.choices[symbol]

                for sample_val in sample_tuple
                    if isa(sample_val, AbstractArray)
                        for val in sample_val
                            push!(flattened_values, Float64(val))
                        end
                    else
                        push!(flattened_values, Float64(sample_val))
                    end
                end
            end
        end
    end

    position_array = unsafe_wrap(
        Array, Ptr{Float64}(position_ptr), length(flattened_values)
    )
    copyto!(position_array, flattened_values)

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function dump(
    value_ptr::Ptr{Any},
    label_ptr::Ptr{UInt8},
    ndims_ptr::Ptr{UInt64},
    shape_ptr::Ptr{UInt64},
    width_ptr::Ptr{UInt64},
    type_kind_ptr::Ptr{UInt64},
)
    activity_id = @ccall MLIR.API.mlir_c.ProfilerActivityStart(
        "ProbProg.dump"::Cstring, Profiler.TRACE_ME_LEVEL_CRITICAL::Cint
    )::Int64

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

    @ccall MLIR.API.mlir_c.ProfilerActivityEnd(activity_id::Int64)::Cvoid
    return nothing
end

function __init__()
    init_trace_ptr = @cfunction(initTrace, Cvoid, (Ptr{Ptr{Any}},))
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_init_trace::Cstring, init_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_sample_to_trace_ptr = @cfunction(
        addSampleToTrace,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        )
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_sample_to_trace::Cstring, add_sample_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_subtrace_ptr = @cfunction(
        addSubtrace, Cvoid, (Ptr{Ptr{Any}}, Ptr{Ptr{Any}}, Ptr{Ptr{Any}})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_subtrace::Cstring, add_subtrace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_weight_to_trace_ptr = @cfunction(addWeightToTrace, Cvoid, (Ptr{Ptr{Any}}, Ptr{Any}))
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_weight_to_trace::Cstring, add_weight_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    add_retval_to_trace_ptr = @cfunction(
        addRetvalToTrace,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        ),
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_add_retval_to_trace::Cstring, add_retval_to_trace_ptr::Ptr{Cvoid}
    )::Cvoid

    get_sample_from_constraint_ptr = @cfunction(
        getSampleFromConstraint,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        )
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_sample_from_constraint::Cstring,
        get_sample_from_constraint_ptr::Ptr{Cvoid},
    )::Cvoid

    get_subconstraint_ptr = @cfunction(
        getSubconstraint, Cvoid, (Ptr{Ptr{Any}}, Ptr{Ptr{Any}}, Ptr{Ptr{Any}})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_subconstraint::Cstring, get_subconstraint_ptr::Ptr{Cvoid}
    )::Cvoid

    get_sample_from_trace_ptr = @cfunction(
        getSampleFromTrace,
        Cvoid,
        (
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{Ptr{Any}},
            Ptr{UInt64},
            Ptr{UInt64},
            Ptr{Ptr{UInt64}},
            Ptr{UInt64},
        )
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_sample_from_trace::Cstring,
        get_sample_from_trace_ptr::Ptr{Cvoid},
    )::Cvoid

    get_subtrace_ptr = @cfunction(
        getSubtrace, Cvoid, (Ptr{Ptr{Any}}, Ptr{Ptr{Any}}, Ptr{Ptr{Any}})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_subtrace::Cstring, get_subtrace_ptr::Ptr{Cvoid}
    )::Cvoid

    get_weight_from_trace_ptr = @cfunction(
        getWeightFromTrace, Cvoid, (Ptr{Ptr{Any}}, Ptr{Any})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_weight_from_trace::Cstring,
        get_weight_from_trace_ptr::Ptr{Cvoid},
    )::Cvoid

    get_flattened_samples_from_trace_ptr = @cfunction(
        getFlattenedSamplesFromTrace,
        Cvoid,
        (Ptr{Ptr{Any}}, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Ptr{Any})
    )
    @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(
        :enzyme_probprog_get_flattened_samples_from_trace::Cstring,
        get_flattened_samples_from_trace_ptr::Ptr{Cvoid},
    )::Cvoid

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
