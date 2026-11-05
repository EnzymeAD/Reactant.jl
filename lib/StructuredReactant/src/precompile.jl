# One workload function: Julia compiles Reactant's pipeline once per function type.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    function workload(x, n)
        y = sum(x) > 0 ? x .* x : x .- x
        k = 0
        while sum(y) > 1
            y = y ./ 2
            k += 1
        end
        for i in 1:3
            y = y .+ i
        end
        for i in 1:n
            y = y .+ i
        end
        return sum((y .* 2) .- 1), k
    end
    @compile_workload begin
        resolve(
            Tuple{typeof(workload),TracedRArray{Float32,1},TracedRNumber{Int}},
            Base.get_world_counter(),
        )
        if Reactant.Reactant_jll.is_available()
            x = Reactant.to_rarray(Float32[1, 2])
            n = Reactant.ConcreteRNumber(3)
            Reactant.@code_hlo optimize = false structured(workload)(x, n)
        end
    end
    # Resolutions hold this process's worlds; CodeInstances carry their prepared code.
    empty!(RESOLUTIONS)
end
