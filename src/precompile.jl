using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        __init__()
        x = Reactant.ConcreteRArray(randn(Float64, 2, 2))
        @jit sum(x)
    end
end
