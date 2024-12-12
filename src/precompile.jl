using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        Reactant.__init__()
        cpu = XLA.CPUClient()
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2); client=cpu)
        @code_hlo optimize = false sum(x)
    end
end
