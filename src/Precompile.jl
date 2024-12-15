using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    initialize_dialect()
    @compile_workload begin
        interp = Reactant.ReactantInterpreter()
        Base.code_ircode(sum, (Reactant.TracedRArray{Float64,2},); interp)
    end
    deinitialize_dialect()
end
