using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    initialize_dialect()
    @compile_workload begin
        interp = Reactant.ReactantInterpreter()
        Base.code_ircode(sum, (Reactant.TracedRArray{Float64,2},); interp)
    end
    deinitialize_dialect()
    for v in oc_capture_vec
        if v isa Base.RefValue
            p = Ptr{Ptr{Cvoid}}(pointer_from_objref(r))
            Base.atomic_pointerset(p, C_NULL, :monotonic)
        else
            empty!(v)
        end
    end
end
