using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    initialize_dialect()
    cpu = XLA.CPUClient()
    @compile_workload begin
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2); client=cpu)
        @code_hlo optimize = false sum(x)
    end
    XLA.free_client(cpu)
    deinitialize_dialect()
    XLA.cpuclientcount[] = 0
end
