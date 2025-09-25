@reactant_overlay function PythonCall.pycall(f::Py, args...)
    if Reactant.looped_any(Reactant.use_overlayed_version, args)
        return pycall_with_jax_tracing(f, args...)
    else
        return Base.inferencebarrier(PythonCall.pycall)(f, args...)
    end
end
