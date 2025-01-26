# Invoke with
# `julia --project=deps deps/build_local.jl [--debug] [--backend=auto/cpu/cuda]`

# the pre-built ReactantExtra_jll might not be loadable on this platform
Reactant_jll = Base.UUID("0192cb87-2b54-54ad-80e0-3be72ad8a3c0")

using ArgParse

s = ArgParseSettings()
#! format: off
@add_arg_table! s begin
    "--debug"
        help = "Build with debug mode (-c dbg)."
        action = :store_true
    "--backend"
        help = "Build with the specified backend (auto, cpu, cuda)."
        default = "auto"
        arg_type = String
    "--gcc_host_compiler_path"
        help = "Path to the gcc host compiler."
        default = "/usr/bin/gcc"
        arg_type = String
    "--cc"
        default = "/home/wmoses/llvms/llvm16-r/clang+llvm-16.0.2-x86_64-linux-gnu-ubuntu-22.04/bin/clang"
        arg_type = String
    "--hermetic_python_version"
        help = "Hermetic Python version."
        default = "3.10"
        arg_type = String
    # For GCC < 13 we need to disable these flags
    "--xnn_disable_avx512fp16"
        help = "Disable AVX512 FP16 support in XNNPACK."
        action = :store_true
    "--xnn_disable_avxvnniint8"
        help = "Disable AVX VNNI INT8 support in XNNPACK."
        action = :store_true
end
#! format: on
parsed_args = parse_args(ARGS, s)

println("Parsed args:")
for (k, v) in parsed_args
    println("  $k = $v")
end
println()

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

build_kind = parsed_args["debug"] ? "dbg" : "opt"

build_backend = parsed_args["backend"]
@assert build_backend in ("auto", "cpu", "cuda")

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

bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

@info "Building JLL with $(bazel_cmd)"

gcc_host_compiler_path = parsed_args["gcc_host_compiler_path"]
cc = parsed_args["cc"]
hermetic_python_version = parsed_args["hermetic_python_version"]

build_cmd_list = [bazel_cmd, "build"]
!isempty(arg) && push!(build_cmd_list, arg)
append!(build_cmd_list, ["-c", "$(build_kind)"])
push!(build_cmd_list, "--action_env=JULIA=$(Base.julia_cmd().exec[1])")
if parsed_args["xnn_disable_avx512fp16"]
    push!(build_cmd_list, "--define=xnn_enable_avx512fp16=false")
end
if parsed_args["xnn_disable_avxvnniint8"]
    push!(build_cmd_list, "--define=xnn_enable_avxvnniint8=false")
end
push!(build_cmd_list, "--repo_env=HERMETIC_PYTHON_VERSION=$(hermetic_python_version)")
push!(build_cmd_list, "--repo_env=GCC_HOST_COMPILER_PATH=$(gcc_host_compiler_path)")
push!(build_cmd_list, "--repo_env=CC=$(cc)")
push!(build_cmd_list, "--check_visibility=false")
push!(build_cmd_list, "--verbose_failures")
push!(build_cmd_list, ":libReactantExtra.so")

run(Cmd(Cmd(build_cmd_list); dir=source_dir))

# Discover built libraries
built_libs = filter(readdir(joinpath(source_dir, "bazel-bin"))) do file
    endswith(file, "Extra.$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(source_dir, "bazel-bin", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

if build_backend == "cuda"
    if !Base.Filesystem.ispath(joinpath(source_dir, "cuda_nvcc"))
        Base.Filesystem.symlink(joinpath(source_dir, "bazel-bin", "libReactantExtra.so.runfiles", "cuda_nvcc"), joinpath(source_dir, "cuda_nvcc"))
    end
end
# Tell ReactantExtra_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Reactant_jll",
    "libReactantExtra_path" => lib_path,
    "libReactantDialects_path" => joinpath(source_dir, "bazel-bin");
    force=true,
)
