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
        default = something(Sys.which("gcc"), "/usr/bin/gcc")
        arg_type = String
    "--cc"
        default = something(Sys.which("cc"), Sys.which("gcc"), Sys.which("clang"), "/usr/bin/cc")
        arg_type = String
    "--hermetic_python_version"
        help = "Hermetic Python version."
        default = "3.10"
        arg_type = String
    "--jobs"
        help = "Number of parallel jobs."
        default = Sys.CPU_THREADS
        arg_type = Int
    "--copt"
        help = "Options to be passed to the C compiler.  Can be used multiple times."
        action = :append_arg
        arg_type = String
    "--cxxopt"
        help = "Options to be passed to the C++ compiler.  Can be used multiple times."
        action = :append_arg
        arg_type = String
    "--extraopt"
        help = "Extra options to be passed to Bazel.  Can be used multiple times."
        action = :append_arg
        arg_type = String
    "--color"
        help = "Set to `yes` to enable color output, or `no` to disable it. Defaults to same color setting as the Julia process."
        default = something(Base.have_color, false) ? "yes" : "no"
        arg_type = String
end
#! format: on
parsed_args = parse_args(ARGS, s)

println("Parsed args:")
for (k, v) in parsed_args
    println("  $k = $v")
end
println()

source_dir = joinpath(@__DIR__, "ReactantExtra")

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

# Try to guess if `cc` is GCC and get its version number.
cc_is_gcc, gcc_version = let
    io = IOBuffer()
    run(pipeline(ignorestatus(`$(cc) --version`); stdout=io))
    version_string = String(take!(io))
    # Detecing GCC is hard, the name "gcc" may not appear anywhere in the
    # version string, but on the second line there should be FSF.
    m = match(
        r"\([^)]+\) (\d+\.\d+\.\d+).*\n.*Free Software Foundation, Inc\.",
        version_string,
    )
    if !isnothing(m)
        true, VersionNumber(m[1])
    else
        false, v"0"
    end
end

build_cmd_list = [bazel_cmd, "build"]
!isempty(arg) && push!(build_cmd_list, arg)
append!(build_cmd_list, ["-c", "$(build_kind)"])
push!(build_cmd_list, "--action_env=JULIA=$(Base.julia_cmd().exec[1])")
push!(build_cmd_list, "--repo_env=HERMETIC_PYTHON_VERSION=$(hermetic_python_version)")
push!(build_cmd_list, "--repo_env=GCC_HOST_COMPILER_PATH=$(gcc_host_compiler_path)")
push!(build_cmd_list, "--repo_env=CC=$(cc)")
push!(build_cmd_list, "--check_visibility=false")
push!(build_cmd_list, "--verbose_failures")
push!(build_cmd_list, "--jobs=$(parsed_args["jobs"])")
if Sys.isapple()
    push!(build_cmd_list, "--define")
    push!(build_cmd_list, "using_clang=true")
end
for opt in parsed_args["copt"]
    push!(build_cmd_list, "--copt=$(opt)")
end
for opt in parsed_args["cxxopt"]
    push!(build_cmd_list, "--cxxopt=$(opt)")
end
for opt in parsed_args["extraopt"]
    push!(build_cmd_list, opt)
end
# Some versions of GCC can't deal with some components of XLA, disable them if necessary.
if cc_is_gcc
    arch = Base.BinaryPlatforms.arch(Base.BinaryPlatforms.HostPlatform())
    if arch == "x86_64"
        if gcc_version < v"13"
            push!(build_cmd_list, "--define=xnn_enable_avxvnniint8=false")
        end
        if gcc_version < v"12"
            push!(build_cmd_list, "--define=xnn_enable_avx512fp16=false")
        end
        if gcc_version < v"11"
            # TODO: this is not sufficient to complete a build with GCC 10.
            push!(build_cmd_list, "--define=xnn_enable_avxvnni=false")
        end
    end
else
    # Assume the compiler is clang if not GCC. `using_clang` is an option
    # introduced by Enzyme-JAX.
    push!(build_cmd_list, "--define=using_clang=true")
end
push!(build_cmd_list, "--color=$(parsed_args["color"])")
push!(build_cmd_list, ":libReactantExtra.so")

run(Cmd(Cmd(build_cmd_list); dir=source_dir))

# Discover built libraries
built_libs = filter(readdir(joinpath(source_dir, "bazel-bin"))) do file
    endswith(file, "Extra.so") && startswith(file, "lib")
end

lib_path = joinpath(source_dir, "bazel-bin", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

if build_backend == "cuda"
    if !Base.Filesystem.ispath(joinpath(source_dir, "bazel-bin", "cuda", "bin", "ptxas"))
        Base.Filesystem.mkpath(joinpath(source_dir, "bazel-bin", "cuda", "bin"))
        Base.Filesystem.symlink(
            joinpath(
                source_dir,
                "bazel-bin",
                "libReactantExtra.so.runfiles",
                "cuda_nvcc",
                "bin",
                "ptxas",
            ),
            joinpath(source_dir, "bazel-bin", "cuda", "bin", "ptxas"),
        )
    end
end

# Tell ReactantExtra_jll to load our library instead of the default artifact one
using Preferences

set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Reactant_jll",
    "libReactantExtra_path" => lib_path;
    force=true,
)
