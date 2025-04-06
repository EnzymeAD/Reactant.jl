# Invoke with
# `julia --project=deps deps/build_local.jl [dbg/opt] [auto/cpu/cuda]`

# the pre-built ReactantExtra_jll might not be loadable on this platform
Reactant_jll = Base.UUID("0192cb87-2b54-54ad-80e0-3be72ad8a3c0")

using Pkg, Scratch, Preferences, Libdl

# 1. Get a scratch directory
scratch_dir = get_scratch!(Reactant_jll, "build")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)

source_dir = joinpath(@__DIR__, "ReactantExtra")

# 2. Ensure that an appropriate LLVM_full_jll is installed
Pkg.activate(; temp=true)

# Build!
@info "Building" source_dir scratch_dir
run(`mkdir -p $(scratch_dir)`)
run(
    Cmd(
        `$(Base.julia_cmd().exec[1]) --project=. -e "using Pkg; Pkg.instantiate()"`;
        dir=source_dir,
    ),
)

#--repo_env TF_NEED_ROCM=1
#--define=using_rocm=true --define=using_rocm_hipcc=true
#--action_env TF_ROCM_AMDGPU_TARGETS="gfx900,gfx906,gfx908,gfx90a,gfx1030"

# --repo_env TF_NEED_CUDA=1
# --repo_env TF_NVCC_CLANG=1
# --repo_env TF_NCCL_USE_STUB=1
# --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_50,sm_60,sm_70,sm_80,compute_90"
# --@xla//xla/python:jax_cuda_pip_rpaths=true
# --repo_env=HERMETIC_CUDA_VERSION="12.3.2"
# --repo_env=HERMETIC_CUDNN_VERSION="9.1.1"
# --@local_config_cuda//cuda:include_cuda_libs=true
# --@local_config_cuda//:enable_cuda
# --@local_config_cuda//:cuda_compiler=nvcc
# --crosstool_top="@local_config_cuda//crosstool:toolchain"

build_kind = if length(ARGS) ≥ 1
    kind = ARGS[1]
    if kind ∉ ("dbg", "opt")
        error("Invalid build kind $(kind). Valid options are 'dbg' and 'opt'")
    end
    kind
else
    "dbg"
end

@info "Building JLL with -c $(build_kind)"

build_backend = if length(ARGS) ≥ 2
    backend = ARGS[2]
    if backend ∉ ("auto", "cpu", "cuda")
        error("Invalid build backend $(backend). Valid options are 'auto', 'cpu', and 'cuda'")
    end
    backend
else
    "auto"
end

if build_backend == "auto"
    build_backend = try
        run(Cmd(`nvidia-smi`))
        "cuda"
    catch
        "cpu"
    end
end

arg = if build_backend == "cuda"
    "--config=cuda"
elseif build_backend == "cpu"
    ""
end

@info "Building JLL with backend $(build_backend)"

if isempty(arg)
    run(
        Cmd(
            `bazel build -c $(build_kind) --action_env=JULIA=$(Base.julia_cmd().exec[1])
            --repo_env HERMETIC_PYTHON_VERSION="3.10"
            --check_visibility=false --verbose_failures :libReactantExtra.so @enzyme_ad//:enzymexlamlir-opt`;
            dir=source_dir,
        ),
    )
else
    run(
        Cmd(
            `bazel build $(arg) -c $(build_kind) --action_env=JULIA=$(Base.julia_cmd().exec[1])
            --repo_env HERMETIC_PYTHON_VERSION="3.10"
            --check_visibility=false --verbose_failures :libReactantExtra.so :enzymexlamlir-opt`;
            dir=source_dir,
        ),
    )
end
# env=Dict("HOME"=>ENV["HOME"], "PATH"=>joinpath(source_dir, "..")*":"*ENV["PATH"])))

run(Cmd(`rm -f libReactantExtra.dylib`; dir=joinpath(source_dir, "bazel-bin")))
run(
    Cmd(
        `ln -s libReactantExtra.so libReactantExtra.dylib`;
        dir=joinpath(source_dir, "bazel-bin"),
    ),
)

# Discover built libraries
built_libs = filter(readdir(joinpath(source_dir, "bazel-bin"))) do file
    endswith(file, "Extra.$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(source_dir, "bazel-bin", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

# Tell ReactReactantExtra_jllant_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Reactant_jll",
    "libReactantExtra_path" => lib_path,
    "libReactantDialects_path" => joinpath(source_dir, "bazel-bin");
    force=true,
)
