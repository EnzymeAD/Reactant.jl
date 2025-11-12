Reactant.jax_dtype_struct_type(::Type{T}) where {T} = Py

function Reactant.convert_to_jax_dtype_struct(x::Union{TracedRArray,TracedRNumber})
    JAX_TRACING_SUPPORTED[] || throw("jax could not be loaded.")
    return jaxptr[].ShapeDtypeStruct(
        size(x), jnpptr[].dtype(string(NUMPY_SIMPLE_TYPES[Reactant.unwrapped_eltype(x)]))
    )
end

function pycall_with_jax_tracing(f::Py, args...)
    JAX_TRACING_SUPPORTED[] || throw("jax could not be loaded.")

    seen_args = Reactant.OrderedIdDict()
    jax_inputs = Vector{Any}(undef, length(args))
    static_argnums = ()
    prev_len = 0
    for (i, arg) in enumerate(args)
        jax_inputs[i] = Reactant.make_tracer(seen_args, arg, (), Reactant.TracedToJAX)
        if length(seen_args) == prev_len
            static_argnums = (static_argnums..., i - 1)
        end
        prev_len = length(seen_args)
    end

    linear_args = Reactant.TracedType[]
    for (k, v) in seen_args
        k isa Reactant.TracedType || continue
        push!(linear_args, k)
    end

    lowered = jaxptr[].jit(f; static_argnums).lower(jax_inputs...)
    # To figure out the exact structure of the pyfunc, we need to execute it. Currently,
    # we skip doing that and assume that we are returning nothing, array, or tuple of
    # arrays.
    res = @opcall hlo_call(pyconvert(String, lowered.as_text()), linear_args...)
    return length(res) == 0 ? nothing : (length(res) == 1 ? res[1] : res)
end
