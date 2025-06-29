function PythonCall.pycall(f::Py, arg0::TracedRArray, argNs::TracedRArray...; kwargs...)
    JAX_TRACING_SUPPORTED[] || throw("jax could not be loaded.")

    jax = jaxptr[]
    jnp = jnpptr[]

    inputs = map((arg0, argNs...)) do arg
        jax.ShapeDtypeStruct(
            size(arg),
            jnp.dtype(string(NUMPY_SIMPLE_TYPES[Reactant.unwrapped_eltype(arg)])),
        )
    end

    lowered = jax.jit(f).lower(inputs...)
    res = Reactant.Ops.hlo_call(pyconvert(String, lowered.as_text()), arg0, argNs...)

    return length(res) == 0 ? nothing : res[1]
end
