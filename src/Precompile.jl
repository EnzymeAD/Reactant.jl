using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        initialize_dialect()
        cpu = XLA.CPUClient()
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2); client=cpu)
        @code_hlo optimize = false sum(x)
        XLA.free_client(cpu)
        deinitialize_dialect()
    end
    XLA.cpuclientcount[] = 0
end
