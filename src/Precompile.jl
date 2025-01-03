using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    initialize_dialect()
    client = XLA.CPUClient(; checkcount=false)
    @compile_workload begin
        x = ConcreteRNumber(2.0; client)
        Reactant.compile(sin, (x,); client)
        # interp = Reactant.ReactantInterpreter()
        # Base.code_ircode(Base.sin, (Reactant.TracedRNumber{Float64},); interp)
    end
    XLA.free_client(client)
    client.client = C_NULL
    deinitialize_dialect()
    for v in oc_capture_vec
        if v isa Base.RefValue
            p = Ptr{Ptr{Cvoid}}(pointer_from_objref(v))
            Base.atomic_pointerset(p, C_NULL, :monotonic)
        else
            empty!(v)
        end
    end
end
