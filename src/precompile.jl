using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    #HACK: check_bounds is 1 with Pkg.test (and is heavely broken see #353).
    #Enable precompilation for normal usage
    Base.JLOptions().check_bounds == 0 || return nothing
    @compile_workload begin
        Reactant.__init__()
        cpu = XLA.CPUClient()
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2); client=cpu)
        @code_hlo optimize = false sum(x)
    end
end
