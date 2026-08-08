@reactant_overlay function PythonCall.pycall(f::Py, args...)
    if Reactant.looped_any(Reactant.use_overlayed_version, args)
        if is_torch_module(f)
            return pycall_with_torch_export(f, args...)
        else
            return pycall_with_jax_tracing(f, args...)
        end
    else
        return Reactant.call_with_native(PythonCall.pycall, f, args...)
    end
end
