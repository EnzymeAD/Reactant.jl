@reactant_overlay function PythonCall.pycall(f::Py, args...)
    if Reactant.looped_any(Reactant.use_overlayed_version, args)
        return overlayed_pycall(f, args...)
    else
        return Base.inferencebarrier(PythonCall.pycall)(f, args...)
    end
end
