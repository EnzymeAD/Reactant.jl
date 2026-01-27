const bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

src_dir = joinpath(dirname(dirname(@__DIR__)), "src")

dialect_files = [
    "Builtin.jl",
    "Arith.jl",
    "Affine.jl",
    "Complex.jl",
    "Func.jl",
    "Enzyme.jl",
    "EnzymeXLA.jl",
    "StableHLO.jl",
    "CHLO.jl",
    "VHLO.jl",
    "Llvm.jl",
    "Nvvm.jl",
    "Gpu.jl",
    "Affine.jl",
    # "TPU.jl", # XXX: currently broken - causes segfault in mlir-jl-tblgen
    "MosaicGPU.jl",
    "Triton.jl",
    "Shardy.jl",
    "MPI.jl",
    "MemRef.jl",
    "SparseTensor.jl",
    "Tensor.jl",
    "Shape.jl",
    "TritonExt.jl",
]

other_files = ["libMLIR_h.jl"]

all_files = vcat(dialect_files, other_files)
bazel_targets = ["//:$f" for f in all_files]

# Build all targets simultaneously
run(
    Cmd(
        `$(bazel_cmd) build --action_env=JULIA=$(Base.julia_cmd().exec[1]) --jobs=$(Threads.nthreads()) --action_env=JULIA_DEPOT_PATH=$(Base.DEPOT_PATH) --repo_env HERMETIC_PYTHON_VERSION="3.12" --check_visibility=false --verbose_failures $bazel_targets`;
        dir=@__DIR__,
    ),
)

# Copy built files to their destinations
for file in dialect_files
    Base.Filesystem.cp(
        joinpath(@__DIR__, "bazel-bin", file),
        joinpath(src_dir, "mlir", "Dialects", file);
        force=true,
    )
end

for file in other_files
    Base.Filesystem.cp(
        joinpath(@__DIR__, "bazel-bin", file), joinpath(src_dir, "mlir", file); force=true
    )
end
