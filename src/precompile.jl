using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        Reactant.__init__()
        interp = Reactant.ReactantInterpreter()
        Base.code_ircode(sum, (Reactant.TracedRArray{Float64,2},); interp)
    end
end
