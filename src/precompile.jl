using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    Reactant.__init__()
    XLA.__init__()
    @compile_workload begin
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2))
        @jit sum(x)
    end
end
