@reactant_overlay function PythonCall.pycall(f::Py, args...; kwargs...)
    if Reactant.looped_any(Reactant.use_overlayed_version, args)
        return overlayed_pycall(f, args...; kwargs...)
    else
        return Base.inferencebarrier(PythonCall.pycall)(f, args...; kwargs...)
    end
end
