for file in [
    "Builtin.jl",
    "Arith.jl",
    "Affine.jl",
    "Func.jl",
    "Enzyme.jl",
    "StableHLO.jl",
    "CHLO.jl",
    "VHLO.jl",
]
    run(
        `bazel build --action_env=JULIA=$(Base.julia_cmd().exec[1]) --repo_env HERMETIC_PYTHON_VERSION="3.10" --check_visibility=false --verbose_failures //:$file`,
    )
    Base.Filesystem.cp(
        joinpath(@__DIR__, "bazel-bin", file),
        joinpath(dirname(dirname(@__DIR__)), "src", "mlir", "Dialects", file),
    )
end
