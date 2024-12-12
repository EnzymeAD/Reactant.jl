using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @static if haskey(ENV, "REACTANT_TEST_GROUP")
        return
    end
    @info "enable precompilation" gethostname() Base.active_project()
    @compile_workload begin
        Reactant.__init__()
        cpu = XLA.CPUClient()
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2); client=cpu)
        @code_hlo optimize = false sum(x)
    end
end
