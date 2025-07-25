using ..Reactant: MLIR

function initTrace(trace_ptr_ptr::Ptr{Ptr{Any}})
    tr = ProbProgTrace()
    _keepalive!(tr)

    unsafe_store!(trace_ptr_ptr, pointer_from_objref(tr))
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
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
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
            shape = unsafe_wrap(Array, shape_ptr, ndims)
            push!(vals, copy(unsafe_wrap(Array, Ptr{julia_type}(sample_ptr), Tuple(shape))))
        end
    end

    trace.choices[symbol] = tuple(vals...)

    return nothing
end

function addSubtrace(
    trace_ptr_ptr::Ptr{Ptr{Any}},
    symbol_ptr_ptr::Ptr{Ptr{Any}},
    subtrace_ptr_ptr::Ptr{Ptr{Any}},
)
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    subtrace = unsafe_pointer_to_objref(unsafe_load(subtrace_ptr_ptr))::ProbProgTrace

    trace.subtraces[symbol] = subtrace

    return nothing
end

function addWeightToTrace(trace_ptr_ptr::Ptr{Ptr{Any}}, weight_ptr::Ptr{Any})
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace
    trace.weight = unsafe_load(Ptr{Float64}(weight_ptr))
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
    trace = unsafe_pointer_to_objref(unsafe_load(trace_ptr_ptr))::ProbProgTrace

    num_results = unsafe_load(num_results_ptr)

    if num_results == 0
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
            return nothing
        end

        if ndims == 0
            push!(vals, unsafe_load(Ptr{julia_type}(retval_ptr)))
        else
            shape = unsafe_wrap(Array, shape_ptr, ndims)
            push!(vals, copy(unsafe_wrap(Array, Ptr{julia_type}(retval_ptr), Tuple(shape))))
        end
    end

    trace.retval = tuple(vals...)

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
    constraint = unsafe_pointer_to_objref(unsafe_load(constraint_ptr_ptr))::Constraint
    symbol = unsafe_pointer_to_objref(unsafe_load(symbol_ptr_ptr))::Symbol
    num_samples = unsafe_load(num_samples_ptr)
    ndims_array = unsafe_wrap(Array, ndims_array, num_samples)
    width_array = unsafe_wrap(Array, width_array, num_samples)
    shape_ptr_array = unsafe_wrap(Array, shape_ptr_array, num_samples)
    sample_ptr_array = unsafe_wrap(Array, sample_ptr_array, num_samples)

    tostore = get(constraint, symbol, nothing)

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
            return nothing
        end

        if julia_type != eltype(tostore[i])
            @ccall printf(
                "Type mismatch in constrained sample: %s != %s\n"::Cstring,
                string(julia_type)::Cstring,
                string(eltype(tostore[i]))::Cstring,
            )::Cvoid
            return nothing
        end

        if ndims == 0
            unsafe_store!(Ptr{julia_type}(sample_ptr), tostore[i])
        else
            shape = unsafe_wrap(Array, shape_ptr, ndims)
            dest = unsafe_wrap(Array, Ptr{julia_type}(sample_ptr), Tuple(shape))

            if size(dest) != size(tostore[i])
                if length(size(dest)) != length(size(tostore[i]))
                    @ccall printf(
                        "Shape size mismatch in constrained sample: %zd != %zd\n"::Cstring,
                        length(size(dest))::Csize_t,
                        length(size(tostore[i]))::Csize_t,
                    )::Cvoid
                    return nothing
                end
                for i in 1:length(size(dest))
                    d = size(dest)[i]
                    t = size(tostore[i])[i]
                    if d != t
                        @ccall printf(
                            "Shape mismatch in `%zd`th dimension of constrained sample: %zd != %zd\n"::Cstring,
                            i::Csize_t,
                            size(dest)[i]::Csize_t,
                            size(tostore[i])[i]::Csize_t,
                        )::Cvoid
                        return nothing
                    end
                end
            end

            dest .= tostore[i]
        end
    end

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

    return nothing
end
