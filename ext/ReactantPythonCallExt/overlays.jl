@reactant_overlay function PythonCall.pycall(f::Py, args...)
    if Reactant.use_overlayed_version(args)
        return pycall_with_jax_tracing(f, args...)
    else
        return Reactant.call_with_native(PythonCall.pycall, f, args...)
    end
end
