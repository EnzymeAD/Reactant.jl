function run_brusselator_benchmark!(
    results,
    backend;
    N=8,
    damping=1.0,
    nrepeat=30,
    configurations=ad_configurations("default"),
)
    problem = brusselator_problem(N)
    host_args = (vec(problem.u), problem.coordinates, problem.p)
    cases = (
        ("residual", residual_vector, host_args),
        ("jacobian", residual_jacobian, host_args),
        ("newton_step", newton_step, (host_args..., damping)),
    )
    configs = [
        BenchmarkConfiguration(
            name; compile_options=brusselator_compile_options(ad_options), nrepeat
        ) for (name, ad_options) in configurations
    ]

    for (kind, fn, inputs) in cases
        run_benchmark!(
            results,
            backend,
            "Brusselator [N=$N, damping=$damping]/$kind",
            fn,
            inputs,
            Reactant.to_rarray(inputs);
            configs,
        )
    end
    return nothing
end
